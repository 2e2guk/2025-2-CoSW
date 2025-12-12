# finetune_koclip2.py
# -*- coding: utf-8 -*-
# warmup, cosine schedule, gradient clipping 추가 + ckpt_root 외부 경로로 저장
import os
import sys
import json
import random
import argparse
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm

# ── 성능 플래그 ───────────────────────────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# --- 프로젝트 루트 추가 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import config
from module1.model_retriever import load_retriever_model


# -------------------------------
# 유틸: 안전한 디바이스 선택
# -------------------------------
def resolve_device(pref_dev):
    """
    CUDA_VISIBLE_DEVICES로 한 장만 보이게 한 경우에도
    내부 인덱스 0으로 안전하게 매핑
    """
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


# -------------------------------
# 261 캡션 데이터셋
# -------------------------------
class AIHubCaptionDataset(Dataset):
    def __init__(self, json_file_path, image_root_dir, processor):
        print(f"Loading 261-Caption annotations from: {json_file_path}")
        self.image_root_dir = image_root_dir
        self.processor = processor
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} image entries.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_filename = item["file_path"]
        caption = random.choice(item["caption_ko"])
        image_path = os.path.join(self.image_root_dir, image_filename)
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            return None
        return image, caption


def create_caption_collate_fn(processor):
    """
    collate에서는 CPU 텐서만 만들고 반환한다.
    GPU 이동은 학습 루프에서만 수행.
    """
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        text_inputs = processor.tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        image_inputs = processor.image_processor(
            images,
            return_tensors="pt"
        )

        inputs = {
            "input_ids": text_inputs.input_ids,               # CPU
            "attention_mask": text_inputs.attention_mask,     # CPU
            "pixel_values": image_inputs.pixel_values         # CPU
        }
        return inputs
    return collate_fn


# -------------------------------
# 학습 루틴
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64, help="per-iteration batch size")
    ap.add_argument("--accum", type=int, default=4, help="grad accumulation steps")
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--warmup_ratio", type=float, default=0.03, help="warmup ratio of total steps")
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    # ⬇⬇⬇ 추가: 체크포인트/최종 모델 저장 루트 (기본: /home/E/leegw/ckpt_koclip)
    ap.add_argument("--ckpt_root", type=str, default="/home/E/leegw/ckpt_koclip",
                    help="epoch_* 및 최종 모델을 저장할 루트 디렉토리")
    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_stage1_retriever():
    args = parse_args()
    set_seed(args.seed)

    print("\n" + "=" * 50)
    print(f"--- Starting Stage 1 (koCLIP) Fine-tuning (FULL DATA) ---")
    print(f"--- Epochs: {args.epochs}, LR: {args.lr}, BS: {args.batch}, Accum: {args.accum}, "
          f"WD: {args.wd}, Workers: {args.workers}, Warmup: {args.warmup_ratio*100:.1f}% ---")
    print(f"--- CKPT ROOT: {args.ckpt_root} ---")

    # 디바이스 해석(환경에 안전)
    DEVICE = resolve_device(config.DEVICE_STAGE_1)
    print(f"--- Using device: {DEVICE} ---")

    model, processor = load_retriever_model()
    model.train().to(DEVICE)

    dataset = AIHubCaptionDataset(
        json_file_path=config.TRAIN_CAPTION_JSON_PATH,
        image_root_dir=config.TRAIN_CAPTION_IMAGE_PATH,
        processor=processor
    )

    collate_fn = create_caption_collate_fn(processor)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0)
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 스텝/스케줄러: cosine + warmup
    steps_per_epoch = max(1, (len(dataloader) + args.accum - 1) // args.accum)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # 체크포인트 디렉토리: 요청한 외부 경로 사용
    ckpt_dir = os.path.abspath(args.ckpt_root)
    os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        print(f"\nStage 1 - Epoch {epoch + 1}/{args.epochs}")
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        accum = 0

        for batch in progress_bar:
            if batch is None:
                continue

            # GPU로 이동(여기서만)
            batch = {k: (v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            with torch.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                outputs = model(**batch, return_loss=True)
                loss = outputs.loss / args.accum

            if DEVICE.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item()
            accum += 1

            if accum % args.accum == 0:
                if DEVICE.type == "cuda":
                    scaler.unscale_(optimizer)
                # ── Gradient clipping ──
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if DEVICE.type == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                global_step += 1

                lr_now = scheduler.get_last_lr()[0]
                progress_bar.set_description(f"Loss: {running_loss:.4f} | LR: {lr_now:.2e}")
                running_loss = 0.0

        # --- 에폭 종료 시 체크포인트 저장 ---
        epoch_dir = os.path.join(ckpt_dir, f"epoch_{epoch+1:02d}")
        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)
        print(f"[Checkpoint] Saved epoch {epoch+1} to: {epoch_dir}")

    print("--- Stage 1 Fine-tuning Complete ---")

    # 최종 모델 저장도 같은 루트로
    final_dir = os.path.join(ckpt_dir, "finetuned_koclip2")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"--- Stage 1 Model saved to: {final_dir} ---")


if __name__ == "__main__":
    train_stage1_retriever()