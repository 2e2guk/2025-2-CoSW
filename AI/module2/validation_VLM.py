# VLM/module2/validation_VLM.py
import os, sys, json, argparse, random, re
from glob import glob
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
import config

LOCAL_MODEL_DIR_FALLBACK = getattr(config, "RERANKER_MODEL_ID", os.path.join(BASE_DIR, "module2", "ax_model_local"))
HF_CACHE = getattr(config, "HF_CACHE_DIR", None)

_YES_SET = {"예", "네", "맞습니다", "맞아요", "예요", "그렇다", "그렇습니다"}
_NO_SET  = {"아니요", "아니오", "아니에요", "아닙니다", "그렇지 않다"}

def normalize_text(s: str) -> str:
    if s is None: return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[“”\"\'\(\)\[\]\{\}<>]", "", s)
    s = s.rstrip(".!?…~ ")
    return s

def tokens(s: str) -> List[str]:
    return normalize_text(s).split()

def f1_score(pred: str, gold: str) -> float:
    p_toks, g_toks = tokens(pred), tokens(gold)
    if len(p_toks) == 0 and len(g_toks) == 0: return 1.0
    if len(p_toks) == 0 or len(g_toks) == 0: return 0.0
    common = {}
    for t in p_toks:
        common[t] = min(p_toks.count(t), g_toks.count(t))
    overlap = sum(common.values())
    if overlap == 0: return 0.0
    prec = overlap / len(p_toks)
    rec  = overlap / len(g_toks)
    return 2 * prec * rec / (prec + rec + 1e-12)

def yn_label(s: str):
    s = normalize_text(s)
    if s in _YES_SET: return 1
    if s in _NO_SET:  return 0
    return None

class AIHubVQADataset(Dataset):
    def __init__(self, root_dir: str, split="auto", max_samples=None, seed=42):
        self.samples: List[Tuple[str, str, str]] = []

        if split == "auto":
            if os.path.isdir(os.path.join(root_dir, "2.Validation")):
                split = "2.Validation"
            else:
                split = "1.Training"

        label_root = os.path.join(root_dir, split, "라벨링데이터")
        img_root   = os.path.join(root_dir, split, "원천데이터")

        for cat in sorted(glob(os.path.join(label_root, "*"))):
            for sub in sorted(glob(os.path.join(cat, "*"))):
                if not os.path.isdir(sub): continue
                rel = os.path.relpath(sub, label_root)
                img_dir = os.path.join(img_root, rel)

                j_images = os.path.join(sub, "images.json")
                j_q      = os.path.join(sub, "question.json")
                j_a      = os.path.join(sub, "annotation.json")
                if not (os.path.exists(j_images) and os.path.exists(j_q) and os.path.exists(j_a) and os.path.isdir(img_dir)):
                    continue

                with open(j_images, 'r', encoding='utf-8') as f: images_data = json.load(f)["images"]
                with open(j_q, 'r', encoding='utf-8') as f: question_data = json.load(f)["questions"]
                with open(j_a, 'r', encoding='utf-8') as f: annotation_data = json.load(f)["annotations"]

                image_id2file = {x["image_id"]: x["image"] for x in images_data}
                qid2pair = {q["question_id"]: (q["image_id"], q["question"]) for q in question_data}

                for anno in annotation_data:
                    qid = anno["question_id"]
                    if qid not in qid2pair: continue
                    img_id, q_text = qid2pair[qid]
                    if img_id not in image_id2file: continue
                    a_text = anno.get("multiple_choice_answer", "")
                    img_path = os.path.join(img_dir, image_id2file[img_id])
                    if os.path.exists(img_path):
                        self.samples.append((img_path, q_text, a_text))

        # ★ Validation/Training 구분 없이 max_samples 적용
        if max_samples is not None and len(self.samples) > max_samples:
            random.seed(seed)
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"[VQA] Loaded {len(self.samples)} samples from {root_dir}/{split}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, q, a = self.samples[i]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            return None
        return img, q, a

def build_collate(processor, max_seq_len=256, target_res=224):
    if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id

    def collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        imgs, qs, ans = zip(*batch)
        imgs = [im.resize((target_res, target_res), Image.BICUBIC) for im in imgs]

        prompts = []
        for q in qs:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"질문: {q}"}]}]
            prompts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

        model_inputs = processor(
            images=imgs, text=prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=max_seq_len
        )
        model_inputs["pad_token_id"] = torch.tensor([pad_id], dtype=torch.long)
        model_inputs["answers"] = list(ans)
        return model_inputs
    return collate

@torch.inference_mode()
def evaluate_ckpt(ckpt_dir: str, ds: Dataset, device: torch.device,
                  batch_size=2, num_workers=4, max_seq_len=256, target_res=224,
                  max_new_tokens=16, top_k=None, top_p=None):

    # 1) dtype을 먼저 정한다
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    # 2) processor / tokenizer
    proc_dir = ckpt_dir if os.path.isdir(ckpt_dir) else LOCAL_MODEL_DIR_FALLBACK
    processor = AutoProcessor.from_pretrained(proc_dir, trust_remote_code=True, cache_dir=HF_CACHE)

    tok = processor.tokenizer
    tok.padding_side = "left"                 # decoder-only 권장
    # tok.truncation_side = "right"           # 보통은 기본값 유지 권장
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # 3) model (한 번만 로드)
    model = AutoModelForCausalLM.from_pretrained(
        proc_dir, trust_remote_code=True, torch_dtype=dtype, cache_dir=HF_CACHE
    ).to(device).eval()

    # 4) generate()와 config에 pad/eos/use_cache 반영
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id
    if hasattr(model, "config"):
        model.config.use_cache = True
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tok.pad_token_id

    collate = build_collate(processor, max_seq_len=max_seq_len, target_res=target_res)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0),
        prefetch_factor=(2 if num_workers>0 else None)
    )

    total = 0
    em_sum = 0.0
    f1_sum = 0.0
    yn_total = 0
    yn_correct = 0
    samples_preview = []

    for batch in tqdm(loader, desc=f"Eval {os.path.basename(ckpt_dir)}", dynamic_ncols=True):
        if batch is None: continue
        answers = batch.pop("answers")
        pad_id = int(batch.pop("pad_token_id").item())
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            use_cache=True,
            return_dict_in_generate=True,  # ★ 객체 반환
        )
        if top_k is not None or top_p is not None:
            gen_kwargs.update(dict(do_sample=True))
            if top_k is not None: gen_kwargs["top_k"] = int(top_k)
            if top_p is not None: gen_kwargs["top_p"] = float(top_p)

        out = model.generate(**batch, **gen_kwargs)
        seqs = out.sequences if hasattr(out, "sequences") else out  # ★ 안전 처리

        input_ids = batch["input_ids"]
        not_pad = (input_ids != pad_id).int()
        prompt_lens = not_pad.sum(dim=1).tolist()

        decoded_pred = []
        for i in range(seqs.size(0)):
            gen_ids = seqs[i, prompt_lens[i]:]
            text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
            text = normalize_text(text)
            text = re.sub(r"^(답변[:：]\s*)", "", text)
            text = text.split("\n")[0]
            decoded_pred.append(text)

        for pred, gold in zip(decoded_pred, answers):
            total += 1
            gold_n = normalize_text(gold)
            em_sum += float(pred == gold_n)
            f1_sum += f1_score(pred, gold_n)
            y_pred, y_gold = yn_label(pred), yn_label(gold_n)
            if y_pred is not None and y_gold is not None:
                yn_total += 1
                yn_correct += int(y_pred == y_gold)

        if len(samples_preview) < 10:
            for p,g in zip(decoded_pred, answers):
                if len(samples_preview) >= 10: break
                samples_preview.append((p, g))

    metrics = {
        "count": total,
        "EM": (em_sum / max(1,total)) * 100.0,
        "F1": (f1_sum / max(1,total)) * 100.0,
        "YN_acc": (yn_correct / max(1,yn_total)) * 100.0,
        "YN_count": yn_total,
    }
    return metrics, samples_preview

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", type=str, default=os.path.join(BASE_DIR, "checkpoints_VLM"))
    ap.add_argument("--vqa_root",  type=str, default=config.TRAIN_VQA_DATA_PATH)
    ap.add_argument("--split",     type=str, choices=["auto","train","val"], default="auto")
    ap.add_argument("--max_samples", type=int, default=10000)   # ★ 어떤 split에도 적용
    ap.add_argument("--batch_size",  type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--target_res",  type=int, default=224)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    split = {"auto":"auto","train":"1.Training","val":"2.Validation"}[args.split]
    ds = AIHubVQADataset(args.vqa_root, split=split, max_samples=args.max_samples)

    ckpts = sorted([d for d in glob(os.path.join(args.ckpt_root, "epoch_*")) if os.path.isdir(d)])
    assert ckpts, f"No checkpoints under {args.ckpt_root}"
    print("\n" + "="*70)
    print("[Validation] checkpoints:", ", ".join(os.path.basename(c) for c in ckpts))
    print(f"DATA: {args.vqa_root} | split={args.split} | device={device}")
    print("="*70)

    best = {"ckpt": None, "EM": -1.0, "F1": -1.0, "metrics": None}
    for ck in ckpts:
        metrics, preview = evaluate_ckpt(
            ck, ds, device,
            batch_size=args.batch_size, num_workers=args.num_workers,
            max_seq_len=args.max_seq_len, target_res=args.target_res,
            max_new_tokens=args.max_new_tokens, top_k=args.top_k, top_p=args.top_p
        )
        name = os.path.basename(ck)
        print(f"\n--- {name} ---")
        print(f"Count={metrics['count']} | EM={metrics['EM']:.2f} | F1={metrics['F1']:.2f} | YN_acc={metrics['YN_acc']:.2f} (N={metrics['YN_count']})")
        print("Samples (pred ↔ gold):")
        for p,g in preview[:5]:
            print(f"  P: {p}\n  G: {normalize_text(g)}\n")

        if metrics["EM"] > best["EM"]:
            best = {"ckpt": name, "EM": metrics["EM"], "F1": metrics["F1"], "metrics": metrics}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("[BEST] by EM")
    print(f"Checkpoint: {best['ckpt']} | EM={best['EM']:.2f} | F1={best['F1']:.2f}")
    print(best["metrics"])
    print("="*70)

if __name__ == "__main__":
    main()