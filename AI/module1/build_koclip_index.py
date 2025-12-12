# VLM/module1/build_koclip_index.py
import os, json, re
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch, faiss
from transformers import AutoModel, AutoProcessor

def discover_epoch_dirs(ckpt_root: str, only_epochs: str | None):
    dirs = []
    for name in sorted(os.listdir(ckpt_root)):
        p = os.path.join(ckpt_root, name)
        if os.path.isdir(p) and name.startswith("epoch_"):
            dirs.append(p)
    if only_epochs:
        wanted = set()
        for tok in only_epochs.split(","):
            tok = tok.strip()
            if not tok: continue
            if "-" in tok:
                a,b = map(int, tok.split("-",1))
                for n in range(a, b+1): wanted.add(f"epoch_{n:02d}")
            else:
                wanted.add(f"epoch_{int(tok):02d}")
        dirs = [p for p in dirs if os.path.basename(p) in wanted]
    def ep_key(p):
        m = re.search(r"(\d+)$", os.path.basename(p))
        return int(m.group(1)) if m else 10**9
    return sorted(dirs, key=ep_key)

@torch.inference_mode()
def encode_and_build(model_path, json_path, image_root, out_dir, device="cuda:0", bs=256):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[KoCLIP] loading: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(device)
    for p in model.parameters(): p.requires_grad = False

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    paths, seen = [], set()
    for it in raw:
        p = os.path.join(image_root, it["file_path"])
        if os.path.exists(p) and p not in seen:
            seen.add(p); paths.append(p)
    print(f"[KoCLIP] #unique images = {len(paths)}")

    embs = []
    for i in tqdm(range(0, len(paths), bs), desc="Encode images"):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+bs]]
        ip = getattr(processor, "image_processor", getattr(processor, "feature_extractor", None))
        pixel = ip(imgs, return_tensors="pt")
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            z = model.get_image_features(pixel_values=pixel["pixel_values"].to(device))
            z = z / z.norm(dim=-1, keepdim=True)
        embs.append(z.float().cpu().numpy())
        del imgs, ip, pixel, z
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    X = np.concatenate(embs, 0).astype("float32")
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X)

    faiss.write_index(index, os.path.join(out_dir, "koclip_flatip.index"))
    with open(os.path.join(out_dir, "images.json"), "w", encoding="utf-8") as f:
        json.dump(paths, f, ensure_ascii=False, indent=2)
    print(f"[KoCLIP] saved -> {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--out_dir", required=True)
    # 단일 체크포인트 또는…
    ap.add_argument("--model", dest="model_path", default="")
    # …루트+범위로 여러 에폭 처리
    ap.add_argument("--ckpt_root", default="")
    ap.add_argument("--only_epochs", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bs", type=int, default=1024)
    args = ap.parse_args()

    if args.ckpt_root:
        eps = discover_epoch_dirs(args.ckpt_root, args.only_epochs or None)
        assert eps, f"No epoch_* in {args.ckpt_root} matching '{args.only_epochs}'"
        for ep in eps:
            tag = os.path.basename(ep)           # epoch_15, …
            out = os.path.join(args.out_dir, tag)
            encode_and_build(ep, args.json, args.img_root, out,
                             device=args.device, bs=args.bs)
    else:
        assert args.model_path, "Provide --model or --ckpt_root"
        encode_and_build(args.model_path, args.json, args.img_root, args.out_dir,
                         device=args.device, bs=args.bs)