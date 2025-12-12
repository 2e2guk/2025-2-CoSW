# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

import config

DEVICE = config.DEVICE_STAGE_2
MODEL_ID = config.RERANKER_MODEL_ID
PROMPT_TEMPLATE = config.RERANKER_PROMPT_TEMPLATE
YES_STRS = tuple(config.RERANKER_YES_STRINGS)
NO_STRS  = tuple(config.RERANKER_NO_STRINGS)


# ------------------ 모델/프로세서 로더 ------------------

def load_reranker_model():
    print(f"[Re-ranker] model={MODEL_ID}, device={DEVICE}")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=dtype, trust_remote_code=True, attn_implementation="flash_attention_2"
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=dtype, trust_remote_code=True
        )
    model.to(DEVICE).eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model, processor


# ------------------ 유틸 ------------------

def _load_image(path: Optional[str]) -> Optional[Image.Image]:
    if not path:
        return None
    return Image.open(path).convert("RGB")


def _find_image_token(processor, model) -> str:
    # 1) processor.image_processor.image_token
    t = getattr(getattr(processor, "image_processor", None), "image_token", None)
    if isinstance(t, str):
        return t
    # 2) 모델 config에 설정되어 있으면 우선
    t = getattr(getattr(model, "config", None), "image_token", None)
    if isinstance(t, str):
        return t
    # 3) AX4VL은 보통 <|extra_id_11|> 이므로 이것을 우선 시도, 없으면 <|image|>
    specials = set(processor.tokenizer.all_special_tokens)
    for cand in ("<|extra_id_11|>", "<|image|>"):
        if cand in specials:
            return cand
    # 4) 마지막 폴백
    for s in specials:
        if ("image" in s) or ("extra_id" in s):
            return s
    raise RuntimeError("이미지 토큰 문자열을 찾지 못했습니다.")


def _num_tokens_per_tile(processor, model) -> int:
    n = getattr(getattr(processor, "image_processor", None), "num_tokens_per_tile", None)
    if n is None:
        n = getattr(getattr(model, "config", object()), "num_tokens_per_tile", None)
    return int(n if n is not None else 144)


def _preprocess_image(processor, img: Image.Image, device: torch.device):
    out = processor.image_processor(images=img, return_tensors="pt")
    pixel_values = out["pixel_values"].to(device)  # (tiles, 3, H, W)
    image_sizes = out.get("image_sizes", None)
    if image_sizes is None:
        image_sizes = torch.tensor(
            [[pixel_values.shape[-2], pixel_values.shape[-1]]],
            dtype=torch.long, device=device
        )
    else:
        image_sizes = image_sizes.to(device)
    n_tiles = int(pixel_values.shape[0])
    return pixel_values, image_sizes, n_tiles


def _build_chat_with_image_tokens(image_token: str, repeat: int, prompt: str) -> str:
    img_block = image_token * repeat
    return (
        "<|im_start|><|user|>\n"
        f"{img_block}\n"
        f"{prompt}\n"
        "<|im_end|>\n"
        "<|im_start|><|assistant|>"
    )


# ------------------ 점수 계산 ------------------

@torch.inference_mode()
def get_rerank_score(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    user_text: str = "",
    user_image: Image.Image | None = None,
    police_text: str = "",
    police_image: Image.Image | None = None,
) -> float:
    base_image = police_image if police_image is not None else user_image
    if base_image is None:
        return 0.0

    # 1) 프롬프트
    u_txt = user_text if user_text else "없음"
    p_txt = police_text if police_text else "없음"
    prompt = PROMPT_TEMPLATE.format(user_text=u_txt, police_text=p_txt)

    # 2) 이미지 전처리 및 토큰 개수 산정
    device = next(model.parameters()).device
    pixel_values, image_sizes, n_tiles = _preprocess_image(processor, base_image, device)
    tpt = _num_tokens_per_tile(processor, model)  # e.g., 144
    image_token_str = _find_image_token(processor, model)
    image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token_str)
    repeat = n_tiles * tpt

    # 3) 모델 config에 명시적으로 심기(AX4VL forward가 여기 값을 사용)
    if hasattr(model, "config"):
        model.config.image_token_id = int(image_token_id)
        model.config.image_token = image_token_str
        model.config.num_tokens_per_tile = int(tpt)

    # 4) 이미지 토큰 반복 포함한 RAW 텍스트
    chat_text = _build_chat_with_image_tokens(image_token_str, repeat, prompt)

    # 5) 텍스트 토크나이즈 (자르지 않음)
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "right"

    enc = tok(chat_text, return_tensors="pt", padding=False, truncation=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # 6) 방어적 검증: 이미지 토큰 개수 확인
    got = int((input_ids == image_token_id).sum().item())
    if got != repeat:
        raise RuntimeError(
            f"image tokens count mismatch: got={got}, expected={repeat}, "
            f"tiles={n_tiles}, tpt={tpt}, token='{image_token_str}', id={image_token_id}"
        )

    # 7) 생성 1토큰 → '예/아니오' 확률로 점수화
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = out.scores[0][0]  # (V,)

    def first_token_ids(strings: tuple[str, ...]) -> List[int]:
        ids: List[int] = []
        for s in strings:
            enc_ = tok.encode(s, add_special_tokens=False)
            if enc_:
                ids.append(enc_[0])
        return ids

    yes_ids = first_token_ids(YES_STRS)
    no_ids  = first_token_ids(NO_STRS)

    probs = torch.softmax(logits.float(), dim=-1)
    yes_p = probs[yes_ids].sum().item() if yes_ids else 0.0
    no_p  = probs[no_ids].sum().item()  if no_ids  else 0.0
    denom = yes_p + no_p
    return (yes_p / denom) if denom > 0 else 0.0


# ------------------ 배치 재정렬 ------------------

@dataclass
class Candidate:
    id: int
    text: str
    image_path: Optional[str] = None


@torch.inference_mode()
def rerank_candidates(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    user_text: str,
    user_image_path: Optional[str],
    candidates: List[Candidate],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    user_img = _load_image(user_image_path)

    scored: List[Dict[str, Any]] = []
    for c in candidates:
        police_img = _load_image(c.image_path) if c.image_path else None
        s = get_rerank_score(
            model, processor,
            user_text=user_text, user_image=user_img,
            police_text=c.text, police_image=police_img
        )
        scored.append({"id": c.id, "text": c.text, "score": float(s)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    out = []
    for i, item in enumerate(scored[:top_n], start=1):
        out.append({
            "rank": i,
            "id": item["id"],
            "score": round(item["score"], 6),
            "text": item["text"],
        })
    return out