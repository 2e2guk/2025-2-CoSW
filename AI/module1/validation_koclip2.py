# -*- coding: utf-8 -*-
import os, sys, json, argparse
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoProcessor

# ──────────────────────────────────────────────────────────────
# 기본 런타임 설정
# ──────────────────────────────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


# ──────────────────────────────────────────────────────────────
# 데이터 묶음: 이미지/텍스트/매핑
# ──────────────────────────────────────────────────────────────
class CaptionRetrievalCorpus:
    """
    JSON 스키마(예: MSCOCO-Korean):
      - item["file_path"]: 이미지 상대경로
      - item["caption_ko"]: 한국어 캡션 리스트(list[str])

    구성:
      - img_paths:     고유 이미지 경로 목록 (길이: M)
      - txts:          모든 캡션 텍스트 (길이: N)
      - img_to_txt:    이미지 idx -> [텍스트 idx 리스트]
      - txt_to_img:    텍스트 idx -> 해당 이미지 idx
    """
    def __init__(self, json_path: str, image_root: str, limit_images: int | None = None):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        path2img_idx = {}
        img_paths, txts, img_to_txt, txt_to_img = [], [], [], []
        missing = 0

        for it in raw:
            rel = it.get("file_path")
            caps = it.get("caption_ko") or []
            if not rel or not isinstance(caps, list) or len(caps) == 0:
                continue
            p = os.path.join(image_root, rel)
            if not os.path.exists(p):
                missing += 1
                continue

            if p not in path2img_idx:
                if limit_images is not None and len(img_paths) >= limit_images:
                    continue
                path2img_idx[p] = len(img_paths)
                img_paths.append(p)
                img_to_txt.append([])

            img_idx = path2img_idx[p]
            for c in caps:
                t_idx = len(txts)
                txts.append(c if isinstance(c, str) else str(c))
                txt_to_img.append(img_idx)
                img_to_txt[img_idx].append(t_idx)

        if missing:
            print(f"[VAL] Skipped {missing} items (file missing).")
        self.img_paths = img_paths
        self.txts = txts
        self.img_to_txt = img_to_txt
        self.txt_to_img = np.asarray(txt_to_img, dtype=np.int64)

        cap_counts = [len(x) for x in img_to_txt]
        mean_caps = (sum(cap_counts) / max(1, len(cap_counts))) if cap_counts else 0.0
        print(f"[VAL] Loaded: images={len(img_paths)}  texts={len(txts)}")
        print(f"[DBG] mean captions / image = {mean_caps:.3f}")

    def image_iter(self, batch_size: int):
        for i in range(0, len(self.img_paths), batch_size):
            paths = self.img_paths[i:i+batch_size]
            imgs = [Image.open(p).convert("RGB") for p in paths]
            yield imgs

    def text_iter(self, batch_size: int):
        for i in range(0, len(self.txts), batch_size):
            yield self.txts[i:i+batch_size]


# ──────────────────────────────────────────────────────────────
# 임베딩 계산
# ──────────────────────────────────────────────────────────────
@torch.inference_mode()
def compute_image_embeddings(model, processor, corpus: CaptionRetrievalCorpus, device,
                             batch_size=256):
    M = len(corpus.img_paths)
    feat_dim, img_cpu = None, None
    use_cuda = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    ip = getattr(processor, "image_processor", getattr(processor, "feature_extractor", None))
    print("[DBG] image_processor:", type(ip).__name__ if ip else None, "| size:", getattr(ip, "size", None))

    pbar = tqdm(total=M, desc="Embed images", dynamic_ncols=True)
    offset = 0
    for images in corpus.image_iter(batch_size):
        if hasattr(processor, "image_processor"):
            pixel = processor.image_processor(images, return_tensors="pt")
        else:
            pixel = processor.feature_extractor(images, return_tensors="pt")

        pixel_values = pixel["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
            feats = model.get_image_features(pixel_values=pixel_values)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        if feat_dim is None:
            feat_dim = feats.shape[-1]
            img_cpu = torch.empty((M, feat_dim), dtype=torch.float16, device="cpu")

        bs = feats.shape[0]
        img_cpu[offset:offset+bs] = feats.detach().to("cpu", dtype=torch.float16)
        offset += bs
        pbar.update(bs)

        del feats, pixel, pixel_values
        if use_cuda: torch.cuda.empty_cache()
    pbar.close()
    return img_cpu  # (M, D) fp16 on CPU


@torch.inference_mode()
def compute_text_embeddings(model, processor, corpus: CaptionRetrievalCorpus, device,
                            batch_size=512):
    N = len(corpus.txts)
    feat_dim, txt_cpu = None, None
    use_cuda = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    pbar = tqdm(total=N, desc="Embed texts", dynamic_ncols=True)
    offset = 0
    max_len = getattr(getattr(processor, "tokenizer", None), "model_max_length", 77)

    for caps in corpus.text_iter(batch_size):
        proc_out = processor(text=caps, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        inputs = {k: v.to(device, non_blocking=True) for k, v in proc_out.items() if k in ("input_ids", "attention_mask")}
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
            feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        if feat_dim is None:
            feat_dim = feats.shape[-1]
            txt_cpu = torch.empty((N, feat_dim), dtype=torch.float16, device="cpu")

        bs = feats.shape[0]
        txt_cpu[offset:offset+bs] = feats.detach().to("cpu", dtype=torch.float16)
        offset += bs
        pbar.update(bs)

        del feats, proc_out, inputs
        if use_cuda: torch.cuda.empty_cache()
    pbar.close()
    return txt_cpu  # (N, D) fp16 on CPU


# ──────────────────────────────────────────────────────────────
# Recall@K (스트리밍)
# ──────────────────────────────────────────────────────────────
def _update_running_topk(running_scores, running_indices, block_scores, block_idx_global, k):
    merged_scores = torch.cat([running_scores, block_scores], dim=1)
    merged_idx    = torch.cat([running_indices, block_idx_global], dim=1)
    new_scores, new_pos = merged_scores.topk(k=k, dim=1)
    new_idx = torch.gather(merged_idx, 1, new_pos)
    return new_scores, new_idx


@torch.inference_mode()
def recall_i2t_stream(image_emb_cpu, text_emb_cpu, img_to_txt_lists,
                      ks=(10, 30, 50), row_chunk=2048, col_chunk=4096,
                      device=torch.device("cuda:0")) -> Dict[str, float]:
    M, D = image_emb_cpu.shape
    N, D2 = text_emb_cpu.shape
    assert D == D2
    ks = tuple(sorted(ks))
    topk_max = max(ks)
    totals = {k: 0 for k in ks}
    use_cuda = (device.type == "cuda")

    for r0 in tqdm(range(0, M, row_chunk), desc="Recall I→T", dynamic_ncols=True):
        r1 = min(r0 + row_chunk, M)
        A_blk = image_emb_cpu[r0:r1].to(device, dtype=torch.float32, non_blocking=True)

        neg_inf = float("-inf")
        running_scores = torch.full((r1 - r0, topk_max), neg_inf, device=device, dtype=torch.float32)
        running_indices = torch.full((r1 - r0, topk_max), -1, device=device, dtype=torch.long)

        for c0 in range(0, N, col_chunk):
            c1 = min(c0 + col_chunk, N)
            B_blk = text_emb_cpu[c0:c1].to(device, dtype=torch.float32, non_blocking=True)
            sims_block = A_blk @ B_blk.t()

            block_scores, block_idx_local = sims_block.topk(k=topk_max, dim=1)
            block_idx_global = block_idx_local + c0

            running_scores, running_indices = _update_running_topk(
                running_scores, running_indices, block_scores, block_idx_global, topk_max
            )

            del sims_block, block_scores, block_idx_local, block_idx_global, B_blk
            if use_cuda: torch.cuda.empty_cache()

        preds = running_indices
        hits_per_k = {k: 0 for k in ks}
        for i in range(r1 - r0):
            gt_list = img_to_txt_lists[r0 + i]
            if not gt_list:
                continue
            gt = torch.tensor(gt_list, device=device, dtype=torch.long)
            for k in ks:
                topk_ids = preds[i, :k]
                if torch.isin(topk_ids, gt).any():
                    hits_per_k[k] += 1

        for k in ks:
            totals[k] += hits_per_k[k]

        del A_blk, running_scores, running_indices
        if use_cuda: torch.cuda.empty_cache()

    return {f"I2T_R@{k}": totals[k] / M * 100.0 for k in ks}


@torch.inference_mode()
def recall_t2i_stream(text_emb_cpu, image_emb_cpu, txt_to_img_np,
                      ks=(10, 30, 50), row_chunk=4096, col_chunk=2048,
                      device=torch.device("cuda:0")) -> Dict[str, float]:
    N, D = text_emb_cpu.shape
    M, D2 = image_emb_cpu.shape
    assert D == D2
    ks = tuple(sorted(ks))
    topk_max = max(ks)
    totals = {k: 0 for k in ks}
    use_cuda = (device.type == "cuda")

    for r0 in tqdm(range(0, N, row_chunk), desc="Recall T→I", dynamic_ncols=True):
        r1 = min(r0 + row_chunk, N)
        A_blk = text_emb_cpu[r0:r1].to(device, dtype=torch.float32, non_blocking=True)

        neg_inf = float("-inf")
        running_scores = torch.full((r1 - r0, topk_max), neg_inf, device=device, dtype=torch.float32)
        running_indices = torch.full((r1 - r0, topk_max), -1, device=device, dtype=torch.long)

        for c0 in range(0, M, col_chunk):
            c1 = min(c0 + col_chunk, M)
            B_blk = image_emb_cpu[c0:c1].to(device, dtype=torch.float32, non_blocking=True)
            sims_block = A_blk @ B_blk.t()

            block_scores, block_idx_local = sims_block.topk(k=topk_max, dim=1)
            block_idx_global = block_idx_local + c0

            running_scores, running_indices = _update_running_topk(
                running_scores, running_indices, block_scores, block_idx_global, topk_max
            )

            del sims_block, block_scores, block_idx_local, block_idx_global, B_blk
            if use_cuda: torch.cuda.empty_cache()

        gt_imgs = torch.tensor(txt_to_img_np[r0:r1], device=device, dtype=torch.long)
        preds = running_indices
        for k in ks:
            hits = (preds[:, :k] == gt_imgs.unsqueeze(1)).any(dim=1).sum().item()
            totals[k] += hits

        del A_blk, running_scores, running_indices, gt_imgs
        if use_cuda: torch.cuda.empty_cache()

    return {f"T2I_R@{k}": totals[k] / N * 100.0 for k in ks}


@torch.inference_mode()
def recall_t2t_stream(text_emb_cpu, img_to_txt_lists: List[List[int]], txt_to_img_np: np.ndarray,
                      ks=(10, 30, 50), row_chunk=4096, col_chunk=4096,
                      device=torch.device("cuda:0")) -> Dict[str, float]:
    """
    텍스트→텍스트: 쿼리 캡션과 같은 이미지에 속한 '다른' 캡션을 상위 K 내에 하나라도 찾으면 hit.
    자기 자신은 정답에서 제외한다.
    분모: '같은 이미지 내 캡션이 2개 이상'인 쿼리만 집계.
    """
    N, D = text_emb_cpu.shape
    ks = tuple(sorted(ks))
    topk_max = max(ks)
    totals = {k: 0 for k in ks}
    valid = 0
    use_cuda = (device.type == "cuda")

    # 미리 전체 텍스트 인덱스 벡터
    all_idx = torch.arange(N, device=device)

    for r0 in tqdm(range(0, N, row_chunk), desc="Recall T→T", dynamic_ncols=True):
        r1 = min(r0 + row_chunk, N)
        A_blk = text_emb_cpu[r0:r1].to(device, dtype=torch.float32, non_blocking=True)
        q_idx_global = torch.arange(r0, r1, device=device)  # (R,)

        neg_inf = float("-inf")
        running_scores = torch.full((r1 - r0, topk_max), neg_inf, device=device, dtype=torch.float32)
        running_indices = torch.full((r1 - r0, topk_max), -1, device=device, dtype=torch.long)

        for c0 in range(0, N, col_chunk):
            c1 = min(c0 + col_chunk, N)
            B_blk = text_emb_cpu[c0:c1].to(device, dtype=torch.float32, non_blocking=True)
            sims_block = A_blk @ B_blk.t()  # (R, C)

            # 동일 인덱스(자기 자신) 마스킹: (q_global == col_global) 위치 -inf
            col_idx_global = torch.arange(c0, c1, device=device)  # (C,)
            # broadcast compare -> mask
            mask_self = (q_idx_global.unsqueeze(1) == col_idx_global.unsqueeze(0))
            sims_block = sims_block.masked_fill(mask_self, neg_inf)

            block_scores, block_idx_local = sims_block.topk(k=topk_max, dim=1)
            block_idx_global = block_idx_local + c0

            running_scores, running_indices = _update_running_topk(
                running_scores, running_indices, block_scores, block_idx_global, topk_max
            )

            del sims_block, block_scores, block_idx_local, block_idx_global, B_blk, mask_self
            if use_cuda: torch.cuda.empty_cache()

        # 평가
        preds = running_indices  # (R, topk)
        hits_per_k = {k: 0 for k in ks}
        local_valid = 0
        for i in range(r1 - r0):
            qg = r0 + i
            img_id = int(txt_to_img_np[qg])
            pos_all = img_to_txt_lists[img_id]
            # 자기 자신 제외
            pos_set = [t for t in pos_all if t != qg]
            if not pos_set:
                continue  # 유효 쿼리 아님
            local_valid += 1
            pos_t = torch.tensor(pos_set, device=device, dtype=torch.long)
            for k in ks:
                if torch.isin(preds[i, :k], pos_t).any():
                    hits_per_k[k] += 1

        for k in ks:
            totals[k] += hits_per_k[k]
        valid += local_valid

        del A_blk, running_scores, running_indices
        if use_cuda: torch.cuda.empty_cache()

    # 분모는 valid(유효 쿼리 수)
    return {f"T2T_R@{k}": (totals[k] / max(1, valid) * 100.0) for k in ks}


# ──────────────────────────────────────────────────────────────
# 유틸: 에폭 디렉토리 찾기
# ──────────────────────────────────────────────────────────────
def discover_epoch_dirs(ckpt_root: str | None,
                        ckpt_globs: List[str] | None,
                        ckpt_list: List[str] | None,
                        only_epochs: str | None) -> List[str]:
    dirs = set()

    if ckpt_root:
        for d in glob(os.path.join(ckpt_root, "epoch_*")):
            if os.path.isdir(d):
                dirs.add(os.path.abspath(d))

    if ckpt_globs:
        for pat in ckpt_globs:
            for d in glob(pat):
                if os.path.isdir(d) and os.path.basename(d).startswith("epoch_"):
                    dirs.add(os.path.abspath(d))

    if ckpt_list:
        for d in ckpt_list:
            d = d.strip()
            if d and os.path.isdir(d):
                dirs.add(os.path.abspath(d))

    dirs = list(dirs)

    if only_epochs:
        wanted = set()
        for tok in only_epochs.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                a, b = tok.split("-", 1)
                a, b = int(a), int(b)
                for n in range(a, b + 1):
                    wanted.add(f"epoch_{n:02d}")
            else:
                wanted.add(f"epoch_{int(tok):02d}")
        dirs = [d for d in dirs if os.path.basename(d) in wanted]

    def ep_key(p):
        base = os.path.basename(p)
        import re
        m = re.search(r"(\d+)$", base)
        return int(m.group(1)) if m else 10**9

    return sorted(dirs, key=ep_key)


# ──────────────────────────────────────────────────────────────
# 로더: HF 모델 ID 또는 로컬 경로
# ──────────────────────────────────────────────────────────────
def load_model_and_processor(id_or_path: str, device: torch.device, trust_remote_code=True):
    """
    id_or_path가 디렉토리면 로컬 체크포인트로 간주.
    그렇지 않으면 HF 모델 ID로 간주.
    """
    if os.path.isdir(id_or_path):
        src = id_or_path
    else:
        src = id_or_path  # HF Model ID
    processor = AutoProcessor.from_pretrained(src, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(src, trust_remote_code=trust_remote_code).to(device).eval()
    return model, processor


# ──────────────────────────────────────────────────────────────
# 평가 루틴: 한 모델(=베이스라인 또는 한 에폭)에 대해 일괄 계산
# ──────────────────────────────────────────────────────────────
@torch.inference_mode()
def evaluate_one(id_or_path: str, corpus: CaptionRetrievalCorpus, device: torch.device,
                 ks: Tuple[int, ...], batch_images=256, batch_texts=512,
                 row_chunk=2048, col_chunk=4096, trust_remote_code=True) -> Dict[str, float]:
    print(f"\n[EVAL] {id_or_path}")
    model, processor = load_model_and_processor(id_or_path, device, trust_remote_code=trust_remote_code)

    # 임베딩
    try:
        img_emb = compute_image_embeddings(model, processor, corpus, device, batch_size=batch_images)
        txt_emb = compute_text_embeddings(model, processor, corpus, device, batch_size=batch_texts)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            if device.type == "cuda": torch.cuda.empty_cache()
            print("[WARN] OOM on embedding. Retrying with half batches.")
            img_emb = compute_image_embeddings(model, processor, corpus, device, batch_size=max(1, batch_images // 2))
            txt_emb = compute_text_embeddings(model, processor, corpus, device, batch_size=max(1, batch_texts // 2))
        else:
            raise

    del model, processor
    if device.type == "cuda": torch.cuda.empty_cache()

    # 리콜 계산
    try:
        i2t = recall_i2t_stream(img_emb, txt_emb, corpus.img_to_txt, ks=ks,
                                row_chunk=row_chunk, col_chunk=col_chunk, device=device)
        t2i = recall_t2i_stream(txt_emb, img_emb, corpus.txt_to_img, ks=ks,
                                row_chunk=col_chunk, col_chunk=row_chunk, device=device)
        t2t = recall_t2t_stream(txt_emb, corpus.img_to_txt, corpus.txt_to_img, ks=ks,
                                row_chunk=col_chunk, col_chunk=col_chunk, device=device)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            if device.type == "cuda": torch.cuda.empty_cache()
            rch = max(512, row_chunk // 2)
            cch = max(1024, col_chunk // 2)
            print(f"[WARN] OOM on recall. Retrying with row/col chunk=({rch}, {cch}).")
            i2t = recall_i2t_stream(img_emb, txt_emb, corpus.img_to_txt, ks=ks,
                                    row_chunk=rch, col_chunk=cch, device=device)
            t2i = recall_t2i_stream(txt_emb, img_emb, corpus.txt_to_img, ks=ks,
                                    row_chunk=cch, col_chunk=rch, device=device)
            t2t = recall_t2t_stream(txt_emb, corpus.img_to_txt, corpus.txt_to_img, ks=ks,
                                    row_chunk=cch, col_chunk=cch, device=device)
        else:
            raise

    # 결과 합치기
    out = {}
    out.update(i2t); out.update(t2i); out.update(t2t)

    # 메모리 정리
    del img_emb, txt_emb
    if device.type == "cuda": torch.cuda.empty_cache()
    return out


# ──────────────────────────────────────────────────────────────
# CSV 유틸
# ──────────────────────────────────────────────────────────────
def append_csv(path: str, header_fields: List[str], row_values: List):
    import csv, os
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header_fields)
        w.writerow(row_values)


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    # 데이터
    ap.add_argument("--json",      type=str, default="/home/E/leegw/data/261_mscoco_korean/MSCOCO_train_val_Korean.json")
    ap.add_argument("--img_root",  type=str, default="/home/E/leegw/data/261_mscoco_korean")
    ap.add_argument("--limit_images", type=int, default=None)

    # 장치/배치
    ap.add_argument("--device",    type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_images", type=int, default=256)
    ap.add_argument("--batch_texts",  type=int, default=512)
    ap.add_argument("--row_chunk",    type=int, default=2048)
    ap.add_argument("--col_chunk",    type=int, default=4096)

    # 리콜 K
    ap.add_argument("--ks", type=str, default="10,30,50")

    # 베이스라인 & 파인튜닝 체크포인트들
    ap.add_argument("--baseline_model_id", type=str, default="koclip/koclip-base-pt",
                    help="미세조정 전 KoCLIP 모델(HF ID 또는 로컬 경로)")
    ap.add_argument("--no_baseline", action="store_true", help="베이스라인 생략")

    ap.add_argument("--ckpt_root", type=str, default="/home/E/leegw/ckpt_koclip",
                    help="이 루트 아래 epoch_* 디렉터리를 자동 탐색")
    ap.add_argument("--ckpt_glob", action="append", default=None,
                    help='글롭 패턴 추가. 예: --ckpt_glob "/path/to/epoch_*" (여러 번 지정 가능)')
    ap.add_argument("--ckpt_list", type=str, default=None,
                    help="콤마로 구분 경로 목록. 예: /p/ep_01,/q/ep_10")
    ap.add_argument("--only_epochs", type=str, default=None, help="평가할 에폭. 예: '1-5,10,14'")

    # 기타
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--save_csv", type=str, default="/home/E/leegw/koclip_compare.csv")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    ks = tuple(sorted(set(int(x) for x in args.ks.split(","))))
    print("\n" + "="*72)
    print("[KoCLIP Validation: Baseline vs Fine-tuned epochs]")
    print("JSON:       ", args.json)
    print("IMG root:   ", args.img_root)
    print("DEVICE:     ", device)
    print("Ks:         ", ks)
    print("="*72)

    # 데이터 적재
    corpus = CaptionRetrievalCorpus(args.json, args.img_root, limit_images=args.limit_images)

    # 타깃 모델 리스트 구성
    targets: List[Tuple[str, str]] = []  # (표시명, 경로 또는 ID)

    if not args.no_baseline:
        targets.append(("baseline", args.baseline_model_id))

    ckpt_list = []
    if args.ckpt_list:
        ckpt_list = [s for s in args.ckpt_list.split(",") if s.strip()]
    epochs = discover_epoch_dirs(args.ckpt_root, args.ckpt_glob, ckpt_list, args.only_epochs)
    for ep in epochs:
        targets.append((os.path.basename(ep), ep))

    if not targets:
        print("[ERR] No models to evaluate. Check --baseline_model_id or --ckpt_root/--ckpt_glob/--ckpt_list.")
        return

    # CSV 헤더
    header = ["model"]
    for p in ("I2T", "T2I", "T2T"):
        for k in ks:
            header.append(f"{p}_R@{k}")

    # 평가 루프
    for name, src in targets:
        scores = evaluate_one(
            src, corpus, device, ks,
            batch_images=args.batch_images, batch_texts=args.batch_texts,
            row_chunk=args.row_chunk, col_chunk=args.col_chunk,
            trust_remote_code=args.trust_remote_code
        )

        # 출력 요약
        def fmt(prefix): return "  " + prefix + " " + ", ".join([f"R@{k}={scores[f'{prefix}_R@{k}']:.2f}" for k in ks])
        print(f"\n[{name}]")
        print(fmt("I2T"))
        print(fmt("T2I"))
        print(fmt("T2T"))

        # CSV 저장
        row = [name] + [scores[f"I2T_R@{k}"] for k in ks] + \
              [scores[f"T2I_R@{k}"] for k in ks] + \
              [scores[f"T2T_R@{k}"] for k in ks]
        if args.save_csv:
            append_csv(args.save_csv, header, row)

    print("\n[Done] CSV ->", args.save_csv if args.save_csv else "(not saved)")


if __name__ == "__main__":
    main()