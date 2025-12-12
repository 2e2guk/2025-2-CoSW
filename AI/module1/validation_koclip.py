# VLM/module1/validation_koclip.py
import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor

# ──────────────────────────────────────────────────────────────
# 성능/안정 플래그 (낮은 스펙)
# ──────────────────────────────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# ──────────────────────────────────────────────────────────────
# 프로젝트 경로/설정 (고정)
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
import config  # 경로만 사용

CKPT_ROOT = "/home/E/leegw/VLM/checkpoints_koclip"
JSON_PATH = "/home/E/leegw/data/261_mscoco_korean/MSCOCO_train_val_Korean.json"
IMG_ROOT  = "/home/E/leegw/data/261_mscoco_korean"

# ──────────────────────────────────────────────────────────────
# 배치/청크 (낮은 스펙, OOM 회피 우선)
# ──────────────────────────────────────────────────────────────
BATCH_IMAGES = 128
BATCH_TEXTS  = 256

ROW_CHUNK = 2048
COL_CHUNK = 4096

NUM_WORKERS     = 2
PIN_MEMORY      = False
PERSISTENT_WORK = False
PREFETCH_FACTOR = 2

R_AT_K = (1, 5, 10, 30, 50)

def _loader_kwargs():
    """num_workers에 맞춰 안전하게 DataLoader 인자 구성."""
    kw = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    if NUM_WORKERS > 0:
        kw["persistent_workers"] = PERSISTENT_WORK
        kw["prefetch_factor"] = PREFETCH_FACTOR
    return kw

# ──────────────────────────────────────────────────────────────
# 데이터셋 (시작 시 파일 경로 선필터링)
# ──────────────────────────────────────────────────────────────
class AIHubCaptionValDataset(Dataset):
    def __init__(self, json_file_path, image_root_dir, validate_paths=True):
        with open(json_file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if validate_paths:
            data, miss = [], 0
            for it in raw:
                p = os.path.join(image_root_dir, it["file_path"])
                if os.path.exists(p):
                    data.append(it)
                else:
                    miss += 1
            self.data = data
            if miss:
                print(f"[VAL] Skipped {miss} items (file missing).")
        else:
            self.data = raw

        self.image_root_dir = image_root_dir
        print(f"[VAL] Loaded {len(self.data)} validation items.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        it = self.data[idx]
        image_path = os.path.join(self.image_root_dir, it["file_path"])
        img = Image.open(image_path).convert("RGB")
        cap = it["caption_ko"][0] if it.get("caption_ko") else ""
        return img, cap

def collate_fn_val(batch):
    images = [b[0] for b in batch]
    caps   = [b[1] for b in batch]
    return images, caps

# ──────────────────────────────────────────────────────────────
# 임베딩 유틸 (GPU 계산 → CPU(fp16) 보관으로 VRAM 최소화)
# ──────────────────────────────────────────────────────────────
@torch.inference_mode()
def compute_image_embeddings(model, processor, dataset, device, batch_size=BATCH_IMAGES):
    N = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_val, **_loader_kwargs()
    )

    feat_dim = None
    img_emb_cpu = None
    offset = 0

    use_cuda = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    pbar = tqdm(loader, desc="Embed images", dynamic_ncols=True)
    for images, _ in pbar:
        img_inputs = (
            processor.image_processor(images, return_tensors="pt")
            if hasattr(processor, "image_processor")
            else processor.feature_extractor(images, return_tensors="pt")
        )
        pixel_values = img_inputs["pixel_values"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
            feats = model.get_image_features(pixel_values=pixel_values)  # (B, D)

        feats = feats / feats.norm(dim=-1, keepdim=True)

        if feat_dim is None:
            feat_dim = feats.shape[-1]
            img_emb_cpu = torch.empty((N, feat_dim), dtype=torch.float16, device="cpu")

        bs = feats.shape[0]
        img_emb_cpu[offset:offset+bs] = feats.detach().to("cpu", dtype=torch.float16)
        offset += bs

        del feats, img_inputs, pixel_values
        if use_cuda:
            torch.cuda.empty_cache()

    return img_emb_cpu  # (N, D) on CPU fp16

@torch.inference_mode()
def compute_text_embeddings(model, processor, dataset, device, batch_size=BATCH_TEXTS):
    N = len(dataset)

    class _TextOnly(Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i): _, cap = self.base[i]; return cap

    def _collate_txt(batch): return batch

    loader = DataLoader(
        _TextOnly(dataset), batch_size=batch_size, shuffle=False,
        collate_fn=_collate_txt, **_loader_kwargs()
    )

    feat_dim = None
    txt_emb_cpu = None
    offset = 0

    use_cuda = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    pbar = tqdm(loader, desc="Embed texts", dynamic_ncols=True)
    for caps in pbar:
        inputs = processor.tokenizer(caps, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
            feats = model.get_text_features(**inputs)

        feats = feats / feats.norm(dim=-1, keepdim=True)

        if feat_dim is None:
            feat_dim = feats.shape[-1]
            txt_emb_cpu = torch.empty((N, feat_dim), dtype=torch.float16, device="cpu")

        bs = feats.shape[0]
        txt_emb_cpu[offset:offset+bs] = feats.detach().to("cpu", dtype=torch.float16)
        offset += bs

        del feats, inputs
        if use_cuda:
            torch.cuda.empty_cache()

    return txt_emb_cpu  # (N, D) on CPU fp16

# ──────────────────────────────────────────────────────────────
# 스트리밍 top-k 기반 R@K (CPU→GPU 블록 업로드, 매우 낮은 VRAM)
# ──────────────────────────────────────────────────────────────
def _update_running_topk(running_scores, running_indices, block_scores, block_idx_global, k):
    merged_scores = torch.cat([running_scores, block_scores], dim=1)
    merged_idx    = torch.cat([running_indices, block_idx_global], dim=1)
    new_scores, new_pos = merged_scores.topk(k=k, dim=1)
    new_idx = torch.gather(merged_idx, 1, new_pos)
    return new_scores, new_idx

@torch.inference_mode()
def recall_at_k_gpu_streaming_cpuhost(
    image_emb_cpu, text_emb_cpu, ks=R_AT_K, direction="I2T",
    row_chunk=ROW_CHUNK, col_chunk=COL_CHUNK, device=torch.device("cuda:0")
):
    assert image_emb_cpu.shape == text_emb_cpu.shape
    N, D = image_emb_cpu.shape
    ks = tuple(sorted(ks))
    topk_max = max(ks)

    A_cpu = image_emb_cpu if direction == "I2T" else text_emb_cpu
    B_cpu = text_emb_cpu  if direction == "I2T" else image_emb_cpu

    totals = {k: 0 for k in ks}
    use_cuda = (device.type == "cuda")

    for r0 in tqdm(range(0, N, row_chunk), desc=f"Recall {direction}", dynamic_ncols=True):
        r1 = min(r0 + row_chunk, N)

        A_blk = A_cpu[r0:r1].to(device, non_blocking=True)
        neg_inf = float("-inf")
        running_scores = torch.full((r1 - r0, topk_max), neg_inf, device=device, dtype=A_blk.dtype)
        running_indices = torch.full((r1 - r0, topk_max), -1,   device=device, dtype=torch.long)

        for c0 in range(0, N, col_chunk):
            c1 = min(c0 + col_chunk, N)
            B_blk = B_cpu[c0:c1].to(device, non_blocking=True)
            Bt_blk = B_blk.t().contiguous()

            sims_block = A_blk @ Bt_blk
            block_scores, block_idx_local = sims_block.topk(k=topk_max, dim=1)
            block_idx_global = block_idx_local + c0

            running_scores, running_indices = _update_running_topk(
                running_scores, running_indices, block_scores, block_idx_global, topk_max
            )

            del sims_block, block_scores, block_idx_local, block_idx_global, B_blk, Bt_blk
            if use_cuda:
                torch.cuda.empty_cache()

        gt = torch.arange(r0, r1, device=device).unsqueeze(1)
        for k in ks:
            hits = (running_indices[:, :k] == gt).any(dim=1).sum().item()
            totals[k] += hits

        del A_blk, running_scores, running_indices, gt
        if use_cuda:
            torch.cuda.empty_cache()

    return {f"R@{k}": totals[k] / N * 100.0 for k in ks}

# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epochs = sorted([d for d in glob(os.path.join(CKPT_ROOT, "epoch_*")) if os.path.isdir(d)])
    print("\n" + "="*70)
    print("[Validation] epochs found:", ", ".join(os.path.basename(e) for e in epochs))
    print("DATA json: ", JSON_PATH)
    print("DATA root: ", IMG_ROOT)
    print("DEVICE:    ", device)
    print("="*70)

    val_ds = AIHubCaptionValDataset(JSON_PATH, IMG_ROOT, validate_paths=True)

    best = {"epoch": None, "score": -1, "i2t": None, "t2i": None}

    for ep in epochs:
        ep_name = os.path.basename(ep)
        print("\n" + "-"*60)
        print(f"[EVAL] {ep_name}  ({ep})")

        processor = AutoProcessor.from_pretrained(ep, cache_dir=getattr(config, "HF_CACHE_DIR", None))
        model = AutoModel.from_pretrained(ep, cache_dir=getattr(config, "HF_CACHE_DIR", None)).to(device).eval()

        try:
            img_emb_cpu = compute_image_embeddings(model, processor, val_ds, device, batch_size=BATCH_IMAGES)
            txt_emb_cpu = compute_text_embeddings(model, processor, val_ds, device, batch_size=BATCH_TEXTS)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print("[WARN] OOM on embedding. Retrying with half batches.")
                img_emb_cpu = compute_image_embeddings(model, processor, val_ds, device, batch_size=max(1, BATCH_IMAGES // 2))
                txt_emb_cpu = compute_text_embeddings(model, processor, val_ds, device, batch_size=max(1, BATCH_TEXTS // 2))
            else:
                raise

        del model, processor
        if device.type == "cuda":
            torch.cuda.empty_cache()

        row_chunk, col_chunk = ROW_CHUNK, COL_CHUNK
        try:
            i2t = recall_at_k_gpu_streaming_cpuhost(img_emb_cpu, txt_emb_cpu, ks=R_AT_K, direction="I2T",
                                                    row_chunk=row_chunk, col_chunk=col_chunk, device=device)
            t2i = recall_at_k_gpu_streaming_cpuhost(img_emb_cpu, txt_emb_cpu, ks=R_AT_K, direction="T2I",
                                                    row_chunk=row_chunk, col_chunk=col_chunk, device=device)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                row_chunk = max(512, row_chunk // 2)
                col_chunk = max(1024, col_chunk // 2)
                print(f"[WARN] OOM on recall. Retrying with row/col chunk=({row_chunk}, {col_chunk}).")
                i2t = recall_at_k_gpu_streaming_cpuhost(img_emb_cpu, txt_emb_cpu, ks=R_AT_K, direction="I2T",
                                                        row_chunk=row_chunk, col_chunk=col_chunk, device=device)
                t2i = recall_at_k_gpu_streaming_cpuhost(img_emb_cpu, txt_emb_cpu, ks=R_AT_K, direction="T2I",
                                                        row_chunk=row_chunk, col_chunk=col_chunk, device=device)
            else:
                raise

        score = (i2t["R@10"] + t2i["R@10"]) / 2.0
        print("\n--- Validation Retrieval ---")
        print(f"I→T: {i2t}")
        print(f"T→I: {t2i}")

        if score > best["score"]:
            best = {"epoch": ep_name, "score": score, "i2t": i2t, "t2i": t2i}

        del img_emb_cpu, txt_emb_cpu
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("[BEST] by (I→T R@10 + T→I R@10)/2")
    print(f"Epoch: {best['epoch']}  Score: {best['score']:.3f}")
    print(f"I→T: {best['i2t']}")
    print(f"T→I: {best['t2i']}")
    print("="*70)

if __name__ == "__main__":
    main()