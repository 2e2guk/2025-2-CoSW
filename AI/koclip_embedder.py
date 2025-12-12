# koclip_embedder.py
import os
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoProcessor, AutoModel, CLIPModel

log = logging.getLogger("koclip_embedder")

# ─────────────────────────────────────────────────────────────────────────────
# 환경 변수
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = os.getenv("KOCLIP_MODEL_ID", "/home/elicer/work/ckpt_koclip/epoch_17")
DEVICE_STR = os.getenv("KOCLIP_DEVICE", "cpu").strip()  # "cuda:0" or "cpu"
DEVICE = torch.device(DEVICE_STR if torch.cuda.is_available() or "cpu" in DEVICE_STR else "cpu")

# CUDA면 fp16/bf16, 아니면 fp32
if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported():
    DTYPE = torch.bfloat16
elif DEVICE.type == "cuda":
    DTYPE = torch.float16
else:
    DTYPE = torch.float32

# 전역 싱글톤
_MODEL = None
_PROC = None
_DIM = 512  # KoCLIP 계열 기본 512. 로드 후 실제 값으로 보정.

# ─────────────────────────────────────────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _load_model_once():
    """모델/프로세서를 한 번만 로드."""
    global _MODEL, _PROC, _DIM
    if _MODEL is not None and _PROC is not None:
        return

    # Processor
    _PROC = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Model: AutoModel 우선, 실패시 CLIPModel 시도
    try:
        _MODEL = AutoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True)
    except Exception:
        _MODEL = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE)

    _MODEL.eval().to(DEVICE)
    try:
        if hasattr(_MODEL, "gradient_checkpointing_enable"):
            _MODEL.gradient_checkpointing_enable = lambda *a, **k: None  # 안전장치
    except Exception:
        pass

    # 출력 차원 추정
    try:
        # 가장 확실: 더미 토큰 통과시켜 shape 체크
        toks = _PROC(text=["probe"], return_tensors="pt", padding=True, truncation=True)
        toks = {k: v.to(DEVICE) for k, v in toks.items()}
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=DTYPE, enabled=(DEVICE.type=="cuda")):
            if hasattr(_MODEL, "get_text_features"):
                z = _MODEL.get_text_features(**toks)
            else:
                out = _MODEL(**toks)
                z = getattr(out, "text_embeds", None)
                if z is None:
                    # CLIPModel forward(text) 경로 대비
                    z = _MODEL.get_text_features(**toks)
        _DIM = int(z.shape[-1])
    except Exception:
        # 실패해도 기본 512 유지
        _DIM = 512

    log.info("[koclip] model loaded: %s | device=%s dtype=%s dim=%d",
             MODEL_ID, DEVICE, str(DTYPE).replace("torch.", ""), _DIM)

def embedding_dim() -> int:
    _load_model_once()
    return _DIM

def _encode_text(texts: List[str]) -> torch.Tensor:
    """Text -> (B, D) tensor (device) 반환. L2 normalize는 하지 않음."""
    _load_model_once()
    if len(texts) == 0:
        return torch.empty((0, _DIM), dtype=torch.float32, device=DEVICE)

    toks = _PROC(text=texts, return_tensors="pt", padding=True, truncation=True)
    toks = {k: v.to(DEVICE) for k, v in toks.items()}

    with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=DTYPE, enabled=(DEVICE.type=="cuda")):
        if hasattr(_MODEL, "get_text_features"):
            feats = _MODEL.get_text_features(**toks)  # (B, D)
        else:
            out = _MODEL(**toks)
            feats = getattr(out, "text_embeds", None)
            if feats is None:
                feats = _MODEL.get_text_features(**toks)

    # 계산은 dtype 혼재 가능성 → float32 고정 변환
    return feats.to(dtype=torch.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────────────────────
def embed_text_batch(texts: List[str], batch_size: int = 128) -> np.ndarray:
    """
    입력: List[str]
    출력: np.ndarray [B, D], float32, L2-normalized
    """
    _load_model_once()
    if not texts:
        return np.zeros((0, _DIM), dtype=np.float32)

    vecs = []
    N = len(texts)
    for s in range(0, N, batch_size):
        chunk = texts[s:s + batch_size]
        t = _encode_text(chunk)               # (b, D) float32
        t = F.normalize(t, dim=-1)            # 코사인용 L2 정규화
        vecs.append(t)

    z = torch.cat(vecs, dim=0) if len(vecs) > 1 else vecs[0]
    return z.cpu().numpy().astype(np.float32)

# 선택: 단건 래퍼(디버그용)
def embed_text_one(text: str) -> np.ndarray:
    return embed_text_batch([text])