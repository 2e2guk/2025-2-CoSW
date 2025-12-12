# -*- coding: utf-8 -*-
import os, re, json, argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import torch, faiss
from transformers import AutoModel, AutoProcessor

# -----------------------------
# 공용 유틸
# -----------------------------
def load_index(dir_):
    # faiss index
    cand = [os.path.join(dir_, n) for n in ("faiss.index", "index.faiss")]
    idx_path = next((p for p in cand if os.path.isfile(p)), None)
    if idx_path is None:
        raise FileNotFoundError(f"FAISS index not found in {dir_}")
    index = faiss.read_index(idx_path)

    # image paths in index order
    paths = None
    for name in ("index_img_paths.npy", "index_img_paths.json", "img_paths.txt"):
        p = os.path.join(dir_, name)
        if os.path.isfile(p):
            if p.endswith(".npy"):
                paths = np.load(p, allow_pickle=True).tolist()
            elif p.endswith(".json"):
                with open(p, "r", encoding="utf-8") as f:
                    paths = json.load(f)
            else:
                with open(p, "r", encoding="utf-8") as f:
                    paths = [line.strip() for line in f if line.strip()]
            break
    if paths is None:
        raise FileNotFoundError(f"index image paths file not found in {dir_}")
    return index, paths

def build_text_corpus(json_path, img_root, index_paths):
    # map path -> index id
    path2id = {os.path.abspath(p): i for i, p in enumerate(index_paths)}
    texts, gt_img = [], []
    miss = 0
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for it in data:
        rel = it.get("file_path")
        caps = it.get("caption_ko") or []
        if not rel or not isinstance(caps, list) or not caps:
            continue
        abs_path = os.path.abspath(os.path.join(img_root, rel))
        img_id = path2id.get(abs_path, None)
        if img_id is None:
            miss += 1
            continue
        for c in caps:
            texts.append(str(c))
            gt_img.append(img_id)
    if miss:
        print(f"[WARN] {miss} samples skipped (image not in index).")
    return texts, np.asarray(gt_img, dtype=np.int64)

@torch.inference_mode()
def encode_texts(model, processor, texts, device, bs, max_len=None):
    model.eval().to(device)
    if max_len is None:
        tok = getattr(processor, "tokenizer", None)
        max_len = getattr(tok, "model_max_length", 77) if tok else 77
    feats = []
    for s in tqdm(range(0, len(texts), bs), desc="Encode texts"):
        batch = texts[s:s+bs]
        proc = processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        proc = {k: v.to(device) for k, v in proc.items() if isinstance(v, torch.Tensor)}
        z = model.get_text_features(**proc)  # (B, D)
        z = torch.nn.functional.normalize(z, dim=-1)
        feats.append(z.detach().to("cpu", dtype=torch.float32))
        del z, proc
        if device.type == "cuda": torch.cuda.empty_cache()
    return torch.cat(feats, dim=0).numpy()  # (N, D) float32

def discover_epoch_dirs(root: str, only_epochs: str | None):
    """
    root/epoch_* 디렉터리 나열 후 only_epochs(예: '15-17,20')로 필터링
    """
    dirs = [d for d in glob(os.path.join(root, "epoch_*")) if os.path.isdir(d)]
    def ep_num(p):
        m = re.search(r"(\d+)$", os.path.basename(p))
        return int(m.group(1)) if m else 10**9
    dirs = sorted(dirs, key=ep_num)

    if not only_epochs:
        return dirs

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

    return [d for d in dirs if os.path.basename(d) in wanted]

# -----------------------------
# 단일 에폭 실행
# -----------------------------
def run_single(json_path, img_root, index_dir, koclip_ckpt, topk, bs_text, device, out_npz, max_queries, trust_remote_code):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[cfg] device={dev} | topk={topk} | bs_text={bs_text}")

    # 1) index 로드
    index, index_paths = load_index(index_dir)
    print(f"[index] {len(index_paths)} images in FAISS")

    # 2) 텍스트/GT 구축
    texts, gt_img = build_text_corpus(json_path, img_root, index_paths)
    if max_queries:
        texts = texts[:max_queries]
        gt_img = gt_img[:max_queries]
    print(f"[corpus] N_texts={len(texts)}")

    # 3) KoCLIP 로드
    processor = AutoProcessor.from_pretrained(koclip_ckpt, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(koclip_ckpt, trust_remote_code=trust_remote_code)

    # 4) 텍스트 임베딩
    X = encode_texts(model, processor, texts, dev, bs_text)

    # 5) 검색 (내적 기반; 이미 정규화 가정)
    D, I = index.search(X, topk)
    print(f"[search] cand shape = {I.shape}")

    # 6) 저장
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(
        out_npz,
        cand_img=I.astype(np.int32),
        gt_img=gt_img.astype(np.int32),
        texts=np.array(texts, dtype=object),
        index_img_paths=np.array(index_paths, dtype=object),
    )
    print(f"[saved] {out_npz}")

# -----------------------------
# 배치(범위) 실행
# -----------------------------
def run_batch(json_path, img_root, index_root, ckpt_root, only_epochs, out_dir,
              topk, bs_text, device, max_queries, trust_remote_code, overwrite=False):
    idx_dirs = discover_epoch_dirs(index_root, only_epochs)
    ckpt_dirs = {os.path.basename(p): p for p in discover_epoch_dirs(ckpt_root, only_epochs)}

    if not idx_dirs:
        raise FileNotFoundError(f"No epoch_* found under index_root={index_root} (filter={only_epochs})")

    print("[batch] epochs:", ", ".join(os.path.basename(d) for d in idx_dirs))
    os.makedirs(out_dir, exist_ok=True)

    for idx_dir in idx_dirs:
        ep = os.path.basename(idx_dir)  # e.g., epoch_15
        ckpt_dir = ckpt_dirs.get(ep, None)
        if ckpt_dir is None:
            print(f"[SKIP] missing KoCLIP ckpt for {ep} under {ckpt_root}")
            continue

        out_npz = os.path.join(out_dir, f"cands_{ep[-2:]}_top{topk}.npz")
        if (not overwrite) and os.path.isfile(out_npz):
            print(f"[SKIP] exists: {out_npz}")
            continue

        print(f"\n[RUN] {ep} -> {out_npz}")
        run_single(
            json_path=json_path,
            img_root=img_root,
            index_dir=idx_dir,
            koclip_ckpt=ckpt_dir,
            topk=topk,
            bs_text=bs_text,
            device=device,
            out_npz=out_npz,
            max_queries=max_queries,
            trust_remote_code=trust_remote_code,
        )

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Prepare top-K image candidates per text using KoCLIP + FAISS index.")

    # 공통
    ap.add_argument("--json", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--bs_text", type=int, default=8192)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max_queries", type=int, default=None)
    ap.add_argument("--trust_remote_code", action="store_true", default=True)

    # 단일 실행 모드
    ap.add_argument("--index_dir", type=str, help="build_koclip_index.py가 만든 단일 epoch_* 디렉터리")
    ap.add_argument("--koclip_ckpt", type=str, help="동일 epoch의 KoCLIP 체크포인트 디렉터리")
    ap.add_argument("--out_npz", type=str, help="단일 실행 결과 저장 경로")

    # 배치 실행 모드
    ap.add_argument("--index_root", type=str, help="여러 epoch_*을 포함하는 인덱스 루트")
    ap.add_argument("--ckpt_root", type=str, help="여러 epoch_*을 포함하는 KoCLIP 체크포인트 루트")
    ap.add_argument("--only_epochs", type=str, default=None, help="예: '15-17,20'")
    ap.add_argument("--out_dir", type=str, help="배치 결과 저장 디렉터리")
    ap.add_argument("--overwrite", action="store_true", help="기존 결과 덮어쓰기")

    args = ap.parse_args()

    # 모드 판별
    single_ok = args.index_dir and args.koclip_ckpt and args.out_npz
    batch_ok  = args.index_root and args.ckpt_root and args.out_dir

    if single_ok and batch_ok:
        raise ValueError("단일 모드와 배치 모드를 동시에 지정할 수 없습니다.")
    if not single_ok and not batch_ok:
        raise ValueError("단일 모드(index_dir, koclip_ckpt, out_npz) 또는 배치 모드(index_root, ckpt_root, out_dir) 중 하나를 지정하세요.")

    return args, single_ok

def main():
    args, single_ok = parse_args()

    if single_ok:
        run_single(
            json_path=args.json,
            img_root=args.img_root,
            index_dir=args.index_dir,
            koclip_ckpt=args.koclip_ckpt,
            topk=args.topk,
            bs_text=args.bs_text,
            device=args.device,
            out_npz=args.out_npz,
            max_queries=args.max_queries,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        run_batch(
            json_path=args.json,
            img_root=args.img_root,
            index_root=args.index_root,
            ckpt_root=args.ckpt_root,
            only_epochs=args.only_epochs,
            out_dir=args.out_dir,
            topk=args.topk,
            bs_text=args.bs_text,
            device=args.device,
            max_queries=args.max_queries,
            trust_remote_code=args.trust_remote_code,
            overwrite=args.overwrite,
        )

if __name__ == "__main__":
    main()