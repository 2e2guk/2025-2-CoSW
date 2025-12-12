# finetune_koclip.py
# Plan Q: Memory-Queue Contrastive Fine-tuning for koCLIP (single GPU)
# - memory queue (default K=4096) to keep many negatives
# - batch=64, accum=4, epochs=30, lr=1e-5 (cosine + warmup), grad clip
# - gradient checkpointing (if available), AMP
# - save checkpoint per epoch

import os
import sys
import json
import math
import random
import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from PIL import Image
from tqdm import tqdm

# ───────────── Speed/Determinism Flags ─────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# ───────────── Project Imports ─────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import config
from module1.model_retriever import load_retriever_model  # must return (model, processor)

# ───────────── Defaults ─────────────
DEF_EPOCHS        = 40
DEF_BATCH         = 64
DEF_ACCUM         = 4
DEF_LR            = 5.0e-6
DEF_WD            = 0.1
DEF_WORKERS       = 8
DEF_WARMUP_RATIO  = 0.05
DEF_MAX_GRAD_NORM = 1.0
DEF_QUEUE_SIZE    = 4096
DEF_SEED          = 42

# ───────────── Utils ─────────────
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_device(pref_dev):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        if isinstance(pref_dev, torch.device) and pref_dev.type == "cuda":
            idx = 0 if pref_dev.index is None else int(pref_dev.index)
            if idx < torch.cuda.device_count():
                return torch.device(f"cuda:{idx}")
        return torch.device("cuda:0")
    except Exception:
        return torch.device("cuda:0")

# ───────────── Dataset ─────────────
class AIHubCaptionDataset(Dataset):
    """
    JSON item example:
      {
        "file_path": "train2014/COCO_train2014_000000000009.jpg",
        "caption_ko": ["한국어 캡션1", "한국어 캡션2", ...]
      }
    기본은 이미지당 캡션 1개 랜덤 샘플링(Plan Q에서 queue로 negatives 보완).
    """
    def __init__(self, json_file_path: str, image_root_dir: str, processor, use_all_caps: bool=False):
        self.image_root_dir = image_root_dir
        self.processor = processor
        self.use_all_caps = use_all_caps

        with open(json_file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not use_all_caps:
            self.data = raw
            print(f"[DATA] Loaded {len(self.data)} images (1 caption sampled per item).")
        else:
            # 전개: (image, caption) 페어로 확장
            pairs = []
            for it in raw:
                img_rel = it["file_path"]
                caps = it.get("caption_ko", [])
                for c in caps:
                    pairs.append((img_rel, c))
            self.data = pairs
            print(f"[DATA] Expanded to {len(self.data)} (image, caption) pairs.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.use_all_caps:
            img_rel, cap = self.data[idx]
        else:
            it = self.data[idx]
            img_rel = it["file_path"]
            caps = it.get("caption_ko", [])
            cap = random.choice(caps) if caps else ""

        img_path = os.path.join(self.image_root_dir, img_rel)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            return None
        return image, cap

def create_collate_fn(processor):
    def collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        images = [b[0] for b in batch]
        caps   = [b[1] for b in batch]
        # text
        txt = processor.tokenizer(caps, return_tensors="pt", padding=True, truncation=True)
        # image
        img = processor.image_processor(images, return_tensors="pt")
        return {
            "input_ids": txt.input_ids,               # (B, L)
            "attention_mask": txt.attention_mask,     # (B, L)
            "pixel_values": img.pixel_values          # (B, C, H, W)
        }
    return collate

# ───────────── Memory Queue ─────────────
class FeatureQueue:
    """
    Fixed-size FIFO queue for negatives. Stores L2-normalized fp16 features on device.
    """
    def __init__(self, K: int, dim: int, device: torch.device, dtype=torch.float16):
        self.K = K
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.img = torch.zeros((K, dim), dtype=dtype, device=device)
        self.txt = torch.zeros((K, dim), dtype=dtype, device=device)
        self.ptr = 0
        self.filled = 0

    @torch.no_grad()
    def enqueue(self, img_feats: torch.Tensor, txt_feats: torch.Tensor):
        # img_feats/txt_feats: (B, D), already normalized, on device
        b = img_feats.shape[0]
        if b == 0:
            return
        end = self.ptr + b
        if end <= self.K:
            self.img[self.ptr:end] = img_feats
            self.txt[self.ptr:end] = txt_feats
        else:
            first = self.K - self.ptr
            self.img[self.ptr:] = img_feats[:first]
            self.txt[self.ptr:] = txt_feats[:first]
            remain = b - first
            if remain > 0:
                self.img[:remain] = img_feats[first:]
                self.txt[:remain] = txt_feats[first:]
        self.ptr = (self.ptr + b) % self.K
        self.filled = min(self.K, self.filled + b)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.filled == 0:
            return None, None
        return self.img[:self.filled], self.txt[:self.filled]

# ───────────── Args ─────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=DEF_EPOCHS)
    ap.add_argument("--batch", type=int, default=DEF_BATCH)
    ap.add_argument("--accum", type=int, default=DEF_ACCUM)
    ap.add_argument("--lr", type=float, default=DEF_LR)
    ap.add_argument("--wd", type=float, default=DEF_WD)
    ap.add_argument("--workers", type=int, default=DEF_WORKERS)
    ap.add_argument("--warmup_ratio", type=float, default=DEF_WARMUP_RATIO)
    ap.add_argument("--max_grad_norm", type=float, default=DEF_MAX_GRAD_NORM)
    ap.add_argument("--queue_size", type=int, default=DEF_QUEUE_SIZE)
    ap.add_argument("--use_all_caps", action="store_true", help="Expand (image×all captions).")
    ap.add_argument("--seed", type=int, default=DEF_SEED)
    return ap.parse_args()

# ───────────── Train ─────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    print("\n" + "="*72)
    print(f"[Plan Q] epochs={args.epochs} | batch={args.batch} | accum={args.accum} | "
          f"lr={args.lr} | wd={args.wd} | warmup={args.warmup_ratio*100:.1f}% | "
          f"queue={args.queue_size} | all_caps={args.use_all_caps}")
    DEVICE = resolve_device(getattr(config, "DEVICE_STAGE_1", torch.device("cuda:0")))
    print(f"[Device] {DEVICE}")
    print("="*72)

    # Model / Processor
    model, processor = load_retriever_model()
    model.to(DEVICE).train()
    # gradient checkpointing (if available)
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # Learnable logit_scale
    created_local_logit = False
    if hasattr(model, "logit_scale") and isinstance(model.logit_scale, torch.nn.Parameter):
        logit_scale = model.logit_scale
    else:
        # default: log(1/0.07) ~ 2.659; clamp to ln(100)≈4.605
        logit_scale = torch.nn.Parameter(torch.tensor(math.log(1/0.07), device=DEVICE, dtype=torch.float32))
        created_local_logit = True

    # Data
    train_ds = AIHubCaptionDataset(
        json_file_path=config.TRAIN_CAPTION_JSON_PATH,
        image_root_dir=config.TRAIN_CAPTION_IMAGE_PATH,
        processor=processor,
        use_all_caps=args.use_all_caps
    )
    collate_fn = create_collate_fn(processor)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0)
    )

    # Optim / Scheduler
    param_groups = [{"params": model.parameters(), "lr": args.lr, "weight_decay": args.wd}]
    if created_local_logit:
        param_groups.append({"params": [logit_scale], "lr": args.lr, "weight_decay": 0.0})
    optimizer = AdamW(param_groups, betas=(0.9, 0.98), eps=1e-8)

    steps_per_epoch = max(1, (len(loader) + args.accum - 1) // args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # Save dir
    ckpt_root = os.path.join(config.BASE_DIR, "checkpoints_koclip")
    os.makedirs(ckpt_root, exist_ok=True)

    # Dry probe to get feature dim
    with torch.no_grad():
        sample_img = torch.zeros((1, 3, 224, 224), device=DEVICE)
        sample_txt = processor.tokenizer(["probe"], return_tensors="pt").to(DEVICE)
        d_img = model.get_image_features(pixel_values=sample_img).shape[-1]
        d_txt = model.get_text_features(**sample_txt).shape[-1]
        assert d_img == d_txt, f"Feature dim mismatch: {d_img} vs {d_txt}"
        feat_dim = d_img
    # Memory queue
    q = FeatureQueue(K=args.queue_size, dim=feat_dim, device=DEVICE, dtype=torch.float16)

    # Train
    global_step = 0
    autocast_dtype = torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        accum = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)
        optimizer.zero_grad(set_to_none=True)

        for batch in pbar:
            if batch is None:
                continue
            batch = {k: (v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            with torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=(DEVICE.type=="cuda")):
                img_feats = model.get_image_features(pixel_values=batch["pixel_values"])   # (B, D)
                txt_feats = model.get_text_features(input_ids=batch["input_ids"],
                                                    attention_mask=batch["attention_mask"])  # (B, D)
                # normalize
                img_feats = F.normalize(img_feats, dim=-1)
                txt_feats = F.normalize(txt_feats, dim=-1)

                # in-batch sims
                logits_ii = img_feats @ txt_feats.t()  # (B, B)
                logits_tt = txt_feats @ img_feats.t()  # (B, B) == logits_ii.T

                # queue sims
                q_img, q_txt = q.get()
                if q_img is not None:
                    logits_iq = img_feats @ q_txt.t()  # (B, Kf)
                    logits_tq = txt_feats @ q_img.t()  # (B, Kf)
                    logits_i2t = torch.cat([logits_ii, logits_iq], dim=1)  # (B, B+Kf)
                    logits_t2i = torch.cat([logits_tt, logits_tq], dim=1)  # (B, B+Kf)
                else:
                    logits_i2t = logits_ii
                    logits_t2i = logits_tt

                # scale
                scale = logit_scale.exp() if isinstance(logit_scale, torch.nn.Parameter) else torch.exp(torch.tensor(float(logit_scale), device=DEVICE))
                logits_i2t = logits_i2t * scale
                logits_t2i = logits_t2i * scale

                B = img_feats.size(0)
                targets = torch.arange(B, device=DEVICE)
                loss_i = F.cross_entropy(logits_i2t, targets)
                loss_t = F.cross_entropy(logits_t2i, targets)
                loss = (loss_i + loss_t) * 0.5
                loss = loss / args.accum

            if DEVICE.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item()
            accum += 1

            if accum % args.accum == 0:
                if DEVICE.type == "cuda":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if DEVICE.type == "cuda":
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # clamp logit_scale like CLIP (<= ln 100)
                with torch.no_grad():
                    if isinstance(logit_scale, torch.nn.Parameter):
                        logit_scale.data.clamp_(max=math.log(100))

                # enqueue after step (detach)
                with torch.no_grad():
                    q.enqueue(img_feats.detach().to(DEVICE, dtype=torch.float16),
                              txt_feats.detach().to(DEVICE, dtype=torch.float16))

                lr_now = scheduler.get_last_lr()[0]
                neg_pool = (B + (q.filled if q.filled > 0 else 0))
                pbar.set_description(f"Loss:{running_loss:.4f} | LR:{lr_now:.2e} | Negs:{neg_pool}")
                running_loss = 0.0

        # save epoch
        ep_dir = os.path.join(ckpt_root, f"epoch_{epoch+1:02d}")
        model.save_pretrained(ep_dir)
        processor.save_pretrained(ep_dir)
        print(f"[Checkpoint] saved -> {ep_dir}")

    # final save
    final_dir = os.path.join(config.BASE_DIR, "finetuned_koclip")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"[Final] saved -> {final_dir}")

if __name__ == "__main__":
    main()