# -*- coding: utf-8 -*-
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

import config

# --- 디바이스 안전 해석 ---
def _resolve_device(pref_dev):
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

DEVICE = _resolve_device(config.DEVICE_STAGE_1)
RETRIEVER_MODEL_ID = config.RETRIEVER_MODEL_ID
W_TEXT = float(config.RETRIEVER_W_TEXT)
W_IMAGE = float(config.RETRIEVER_W_IMAGE)

def load_retriever_model():
    """koCLIP (processor, model) 로드"""
    print(f"[Retriever] model={RETRIEVER_MODEL_ID}, device={DEVICE}")
    processor = AutoProcessor.from_pretrained(
        RETRIEVER_MODEL_ID, cache_dir=str(config.HF_CACHE_DIR), trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        RETRIEVER_MODEL_ID, cache_dir=str(config.HF_CACHE_DIR), trust_remote_code=False
    )
    model.to(DEVICE).eval()
    return model, processor

@torch.no_grad()
def get_embedding(model: AutoModel,
                  processor: AutoProcessor,
                  text: str | None = None,
                  image: Image.Image | None = None) -> np.ndarray:
    """텍스트/이미지 koCLIP 임베딩 → 가중합 → L2 정규화"""
    emb_dim = getattr(getattr(model, "config", None), "projection_dim", config.RETRIEVER_EMBEDDING_DIM)
    v_text = np.zeros(emb_dim, dtype=np.float32)
    v_image = np.zeros(emb_dim, dtype=np.float32)

    use_cuda = (DEVICE.type == "cuda")
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    if text:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
            tfeat = model.get_text_features(**inputs)  # (1, D)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        v_text = tfeat.detach().float().cpu().numpy().reshape(-1)

    if image:
        img_inputs = processor(images=[image], return_tensors="pt")
        pix = img_inputs["pixel_values"].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
            image_features = model.get_image_features(pixel_values=pix)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        v_image = image_features.detach().float().cpu().numpy().reshape(-1)

    if text and not image:
        final_vec = v_text
    elif image and not text:
        final_vec = v_image
    elif text and image:
        final_vec = (W_TEXT * v_text) + (W_IMAGE * v_image)
    else:
        final_vec = np.zeros(emb_dim, dtype=np.float32)

    norm = np.linalg.norm(final_vec)
    return final_vec / norm if norm > 0 else final_vec