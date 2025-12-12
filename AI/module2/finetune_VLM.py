# VLM/module2/finetune_VLM.py
import os, sys, json, argparse, re
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch import distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM, get_scheduler

# ── 런타임/메모리 최적화 ─────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# ── 프로젝트 경로/설정 ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
import config  # config.RERANKER_MODEL_ID = 로컬 모델 폴더 경로

LOCAL_MODEL_DIR = config.RERANKER_MODEL_ID
assert os.path.isdir(LOCAL_MODEL_DIR), f"LOCAL_MODEL_DIR not found: {LOCAL_MODEL_DIR}"

# ── 기본 하이퍼파라미터(필요시 CLI로 override) ─────────────────────────────
DEFAULT_TRAIN_EPOCHS           = 4                 # 재개 시 추가로 돌릴 에폭 수
MAX_SEQ_LEN                    = 256
TARGET_RES                     = 224               # 이미지 다운스케일(OOM 완화)
NUM_WORKERS                    = 8
PIN_MEMORY                     = True
PREFETCH_FACTOR                = 2
MAX_SAMPLES_PER_EPOCH          = 100000          # 에폭마다 랜덤 샘플 상한

# 스케줄러
DEFAULT_WARMUP_RATIO           = 0.02             # 재개 시 살짝 짧게
DEFAULT_SCHEDULER_TYPE         = "cosine"         # "linear" or "cosine"
MAX_GRAD_NORM                  = 1.0

# 배치/누적
TRAIN_BATCH_SIZE_PER_GPU       = 1                # 7B + 이미지이므로 작게 유지
TRAIN_GRAD_ACCUM_STEPS         = 16               # 유효 배치 = world * 1 * 16

# ── 전역 컨텍스트(워커에서 Processor 접근) ─────────────────────────────────
_GLOBAL_PROCESSOR = None
def set_collate_context(processor):
    global _GLOBAL_PROCESSOR
    _GLOBAL_PROCESSOR = processor

def _worker_init_fn(_):
    proc = AutoProcessor.from_pretrained(LOCAL_MODEL_DIR, trust_remote_code=True)
    set_collate_context(proc)

# ── collate_fn (이미지+채팅 템플릿, 라벨 마스킹) ───────────────────────────
def vlm_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    imgs  = [b[0] for b in batch]
    qs    = [b[1] for b in batch]
    ans   = [b[2] for b in batch]

    # 이미지 다운스케일(속도/메모리)
    imgs = [im.resize((TARGET_RES, TARGET_RES), Image.BICUBIC) for im in imgs]

    proc = _GLOBAL_PROCESSOR
    assert proc is not None, "Processor is not set in collate context"
    assert hasattr(proc, "apply_chat_template"), "Processor requires apply_chat_template"

    # pad 토큰 안전장치 (학습은 기존과 동일 패딩 정책 유지)
    if proc.tokenizer.pad_token_id is None and proc.tokenizer.eos_token_id is not None:
        proc.tokenizer.pad_token_id = proc.tokenizer.eos_token_id
    pad_id = proc.tokenizer.pad_token_id

    # user-only / with-answer 템플릿
    prompt_texts, target_texts = [], []
    for q, a in zip(qs, ans):
        user_only = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"질문: {q}"}]}
        ]
        with_answer = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"질문: {q}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"{a}"}]},
        ]
        prompt_texts.append(proc.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True))
        target_texts.append(proc.apply_chat_template(with_answer, tokenize=False, add_generation_prompt=False))

    prompt_inputs = proc(
        images=imgs, text=prompt_texts, return_tensors="pt",
        padding=True, truncation=True, max_length=MAX_SEQ_LEN
    )
    inputs = proc(
        images=imgs, text=target_texts, return_tensors="pt",
        padding=True, truncation=True, max_length=MAX_SEQ_LEN
    )

    input_ids = inputs["input_ids"]
    labels = input_ids.clone()

    # 프롬프트 길이까지 -100 마스킹 + pad도 -100
    for i in range(labels.size(0)):
        prefix_len = (prompt_inputs["input_ids"][i] != pad_id).sum().item()
        labels[i, :prefix_len] = -100
    labels[labels == pad_id] = -100
    inputs["labels"] = labels

    return inputs  # (GPU 이동은 학습 루프에서)

# ── VQA 데이터셋 ────────────────────────────────────────────────────────────
class AIHubVQADataset(Dataset):
    """
    TRAIN_VQA_DATA_PATH/
      1.Training/
        라벨링데이터/<cat>/<subdir>/{images,question,annotation}.json
        원천데이터/<cat>/<subdir>/*.jpg
    """
    def __init__(self, root_dir, split="1.Training"):
        self.samples = []
        label_root = os.path.join(root_dir, split, "라벨링데이터")
        img_root   = os.path.join(root_dir, split, "원천데이터")
        for cat_sub in sorted(glob(os.path.join(label_root, "*", "*"))):
            if not os.path.isdir(cat_sub): continue
            rel = os.path.relpath(cat_sub, label_root)
            img_dir = os.path.join(img_root, rel)

            paths = {n: os.path.join(cat_sub, f"{n}.json") for n in ["images","question","annotation"]}
            if not (all(os.path.exists(p) for p in paths.values()) and os.path.isdir(img_dir)):
                continue

            with open(paths["images"], 'r', encoding='utf-8') as f: images_data = json.load(f)["images"]
            with open(paths["question"], 'r', encoding='utf-8') as f: question_data = json.load(f)["questions"]
            with open(paths["annotation"], 'r', encoding='utf-8') as f: annotation_data = json.load(f)["annotations"]

            image_id2file = {x["image_id"]: x["image"] for x in images_data}
            qid2pair = {q["question_id"]: (q["image_id"], q["question"]) for q in question_data}

            for anno in annotation_data:
                qid = anno["question_id"]
                if qid not in qid2pair: continue
                img_id, q_text = qid2pair[qid]
                if img_id not in image_id2file: continue
                a_text = anno.get("multiple_choice_answer", "")
                img_path = os.path.join(img_dir, image_id2file[img_id])
                self.samples.append((img_path, q_text, a_text))
        print(f"[Stage2] Loaded {len(self.samples)} VQA pairs from {root_dir}/{split}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, q, a = self.samples[i]
        try:
            img = Image.open(p).convert("RGB")
        except FileNotFoundError:
            return None
        return img, q, a

# ── Subset 뷰 ───────────────────────────────────────────────────────────────
class SubsetDataset(Dataset):
    def __init__(self, base: Dataset, indices: list[int]):
        self.base = base
        self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): return self.base[self.indices[i]]

# ── 에폭별 샘플링(전역 cap → 브로드캐스트 → rank 분배) ────────────────────
def epoch_indices_ddp(total_size: int, cap_per_epoch: int, rank: int, world_size: int, device: torch.device):
    real_total = min(cap_per_epoch, total_size)
    len_tensor = torch.tensor([real_total], device=device, dtype=torch.long) if rank == 0 else torch.empty(1, device=device, dtype=torch.long)
    if world_size > 1:
        dist.broadcast(len_tensor, src=0)
    real_total = int(len_tensor.item())
    if rank == 0:
        idx_all = torch.randperm(total_size, device=device, dtype=torch.long)[:real_total]
    else:
        idx_all = torch.empty(real_total, device=device, dtype=torch.long)
    if world_size > 1:
        dist.broadcast(idx_all, src=0)
    per_rank = real_total // world_size
    use_cnt = per_rank * world_size
    idx_all = idx_all[:use_cnt]
    if world_size > 1:
        idx_rank = idx_all.view(world_size, -1)[rank].cpu().tolist()
    else:
        idx_rank = idx_all.cpu().tolist()
    return idx_rank, per_rank

# ── LoRA 대상 모듈 자동 수집 ────────────────────────────────────────────────
def collect_lora_targets(model: nn.Module) -> list[str]:
    allow_regex = [
        r"\bself_attn\.(q_proj|k_proj|v_proj|out_proj)\b",
        r"\bmlp\.(fc1|fc2|gate_proj|up_proj|down_proj)\b",
        r"\b(attn|attention)\.(q_proj|k_proj|v_proj|o_proj)\b",
    ]
    allow = [re.compile(p) for p in allow_regex]
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(p.search(name) for p in allow):
                targets.append(name)
    return sorted(set(targets))

def infer_epoch_offset(resume_ckpt: str | None) -> int:
    if not resume_ckpt:
        return 0
    base = os.path.basename(resume_ckpt.rstrip("/"))
    m = re.match(r"epoch_(\d+)", base)
    return int(m.group(1)) if m else 0

# ── 학습 워커(랭크별) ───────────────────────────────────────────────────────
def train_worker(rank, world_size, args):
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{args.port}",
            world_size=world_size,
            rank=rank,
        )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_rank0 = (rank == 0)

    # dtype 선택
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    if is_rank0:
        eff_bs = TRAIN_BATCH_SIZE_PER_GPU * TRAIN_GRAD_ACCUM_STEPS * world_size
        print("\n" + "="*60)
        print(f"[Stage2] world_size={world_size} | rank={rank}")
        print(f"BASE MODEL DIR: {LOCAL_MODEL_DIR}")
        print(f"RESUME FROM: {args.resume_ckpt if args.resume_ckpt else '(fresh)'}")
        print(f"perGPU BS={TRAIN_BATCH_SIZE_PER_GPU} | accum={TRAIN_GRAD_ACCUM_STEPS} | add-epochs={args.epochs}")
        print(f"Effective batch size ≈ {eff_bs}")
        print(f"MAX_SEQ_LEN={MAX_SEQ_LEN}, TARGET_RES={TARGET_RES}, MAX_SAMPLES_PER_EPOCH={args.max_samples}")
        print(f"Scheduler={args.scheduler}, warmup={int(args.warmup_ratio*100)}%")
        print(f"LRs -> proj:{args.lr_proj} | head:{args.lr_head} | lora:{args.lr_lora}")

    # ── Processor 로드(재개시에도 동일 템플릿 유지) ────────────────────────
    proc_from = args.resume_ckpt if (args.resume_ckpt and os.path.isdir(args.resume_ckpt)) else LOCAL_MODEL_DIR
    processor = AutoProcessor.from_pretrained(proc_from, trust_remote_code=True)

    # ── Model 로드 ─────────────────────────────────────────────────────────
    # 1) 항상 base를 로드
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_DIR, torch_dtype=dtype, trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
    except Exception:
        base_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_DIR, torch_dtype=dtype, trust_remote_code=True
        )
    base_model.to(device)
    base_model.train()
    if hasattr(base_model, "config"):
        base_model.config.use_cache = False
    try:
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
    except Exception:
        pass

    # (1) 모든 파라미터 동결
    for p in base_model.parameters():
        p.requires_grad = False

    # (2) projector/connector/lm_head 학습 허용
    for name, p in base_model.named_parameters():
        if any(k in name for k in ["multi_modal_projector", "mm_projector", "connector", "lm_head"]):
            if p.dtype.is_floating_point:
                p.requires_grad = True

    # (3) LoRA 적용: 재개 시에는 어댑터 가중치 불러오기
    from peft import LoraConfig, get_peft_model, PeftModel
    targets = collect_lora_targets(base_model)
    if is_rank0:
        print(f"[LoRA] target linear modules found: {len(targets)}")
        for t in targets[:8]:
            print(f"  - {t}")
        if len(targets) > 8:
            print("  ...")

    if args.resume_ckpt:
        # base + resume 어댑터 로드
        model = PeftModel.from_pretrained(base_model, args.resume_ckpt, is_trainable=True)
    else:
        # 새 LoRA 시작
        lora_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
            target_modules=targets, task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_cfg)

    # DDP 래핑
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank], output_device=rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=25,
            static_graph=True,
            broadcast_buffers=False
        )

    # 전체 메타만 로드(실제 샘플은 에폭마다 서브셋 구성)
    full_ds = AIHubVQADataset(config.TRAIN_VQA_DATA_PATH, split="1.Training")
    total_size = len(full_ds)

    # ── 파라미터 그룹(다른 LR) ────────────────────────────────────────────
    named_params = dict(model.named_parameters())
    proj = [p for n,p in named_params.items() if getattr(p, "requires_grad", False) and any(k in n for k in ["multi_modal_projector","mm_projector","connector"])]
    head = [p for n,p in named_params.items() if getattr(p, "requires_grad", False) and "lm_head" in n]
    lora = [p for n,p in named_params.items() if getattr(p, "requires_grad", False) and "lora_" in n]

    param_groups = []
    if proj: param_groups.append({"params": proj, "lr": args.lr_proj, "weight_decay": 0.0})
    if head: param_groups.append({"params": head, "lr": args.lr_head, "weight_decay": 0.0})
    if lora: param_groups.append({"params": lora, "lr": args.lr_lora, "weight_decay": 0.0})

    if param_groups:
        optimizer = AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    else:
        optimizer = AdamW((p for p in named_params.values() if getattr(p, "requires_grad", False)),
                          lr=3e-5, weight_decay=0.01, betas=(0.9,0.95), eps=1e-8)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    scheduler = None

    ckpt_root = os.path.join(BASE_DIR, "checkpoints_VLM")
    if is_rank0: os.makedirs(ckpt_root, exist_ok=True)

    epoch_offset = infer_epoch_offset(args.resume_ckpt)  # e.g., epoch_04 -> 4

    for e in range(args.epochs):
        if world_size > 1:
            dist.barrier()
        idx_rank, per_rank = epoch_indices_ddp(total_size, args.max_samples, rank, world_size, device)
        subset_ds = SubsetDataset(full_ds, idx_rank)

        # 이 에폭의 DataLoader (persistent_workers=False: 에폭마다 새로 만듦)
        set_collate_context(processor)
        pf_kw = {}
        if NUM_WORKERS > 0 and PREFETCH_FACTOR:
            pf_kw["prefetch_factor"] = PREFETCH_FACTOR
        loader = DataLoader(
            subset_ds,
            batch_size=TRAIN_BATCH_SIZE_PER_GPU,
            shuffle=False,
            collate_fn=vlm_collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=False,
            worker_init_fn=_worker_init_fn if NUM_WORKERS > 0 else None,
            **pf_kw,
        )

        # 스케줄러 초기화(첫 에폭에서 한 번만)
        if scheduler is None:
            steps_per_epoch = max(1, (len(loader) + TRAIN_GRAD_ACCUM_STEPS - 1) // TRAIN_GRAD_ACCUM_STEPS)
            total_steps = args.epochs * steps_per_epoch
            warmup = max(10, int(args.warmup_ratio * total_steps))
            scheduler = get_scheduler(args.scheduler, optimizer=optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)
            if is_rank0:
                eff_bs = TRAIN_BATCH_SIZE_PER_GPU * TRAIN_GRAD_ACCUM_STEPS * world_size
                print(f"[steps] per-epoch steps≈{steps_per_epoch}, total_steps≈{total_steps}, eff_batch={eff_bs}, per-rank samples={per_rank}")

        # ── 학습 루프 ───────────────────────────────────────────────────────
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        epoch_idx = epoch_offset + e + 1
        iterator = loader if not is_rank0 else tqdm(loader, desc=f"Epoch {epoch_idx}")
        optimizer.zero_grad(set_to_none=True)
        running_loss, accum = 0.0, 0

        for batch in iterator:
            if batch is None:
                continue
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            with torch.cuda.amp.autocast(dtype=dtype):
                out = model(**batch)
                loss = out.loss / TRAIN_GRAD_ACCUM_STEPS

            if dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item()
            accum += 1

            if accum % TRAIN_GRAD_ACCUM_STEPS == 0:
                if dtype == torch.float16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if getattr(p, "requires_grad", False)), MAX_GRAD_NORM
                )

                if dtype == torch.float16:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if is_rank0:
                    iterator.set_description(f"Epoch {epoch_idx} | Loss: {running_loss:.4f}")
                running_loss = 0.0

        # 에폭 체크포인트 (rank0만)
        if is_rank0:
            to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            epoch_dir = os.path.join(ckpt_root, f"epoch_{epoch_idx:02d}")
            to_save.save_pretrained(epoch_dir)   # PEFT 어댑터/프로젝터 저장
            processor.save_pretrained(epoch_dir)
            print(f"[Checkpoint] saved -> {epoch_dir}")

        if world_size > 1:
            dist.barrier()

    # 최종 저장 (rank0만)
    if is_rank0:
        final_dir = os.path.join(BASE_DIR, "checkpoints_vlm_rand")
        to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        to_save.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        print(f"[Final] saved -> {final_dir}")

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()

# ── 런처 ────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=1, help="사용할 GPU 개수")
    ap.add_argument("--port", type=str, default=os.getenv("MASTER_PORT", "29500"), help="DDP init port")
    ap.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS, help="추가로 돌릴 에폭 수")
    ap.add_argument("--resume_ckpt", type=str, default="", help="재개할 체크포인트 디렉토리 (e.g., .../checkpoints_VLM/epoch_04)")
    ap.add_argument("--scheduler", type=str, default=DEFAULT_SCHEDULER_TYPE, choices=["linear","cosine"])
    ap.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    ap.add_argument("--max_samples", type=int, default=MAX_SAMPLES_PER_EPOCH)
    # param-group LRs (재개라서 더 작게)
    ap.add_argument("--lr_proj", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=3e-6)
    ap.add_argument("--lr_lora", type=float, default=2e-5)
    return ap.parse_args()

def main():
    args = parse_args()

    # torchrun 모드
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"]); rank = int(os.environ["RANK"])
        train_worker(rank, world_size, args); return

    # 내부 spawn 모드 (--gpus N)
    requested = max(1, args.gpus)
    available = torch.cuda.device_count()
    world_size = min(requested, available)
    if world_size <= 1:
        train_worker(rank=0, world_size=1, args=args)
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", args.port)
        mp.spawn(train_worker, nprocs=world_size, args=(world_size, args), join=True)

if __name__ == "__main__":
    main()