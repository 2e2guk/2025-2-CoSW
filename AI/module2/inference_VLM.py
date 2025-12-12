# -*- coding: utf-8 -*-
import os
import sys
import argparse
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

DEFAULT_CKPT = os.environ.get("VLM_CKPT", os.path.expanduser("~/work/VLM/module2/ax_model_local"))
DEFAULT_IMG  = os.environ.get("VLM_IMG",  os.path.expanduser("~/work/VLM/data/test_image2.jpg"))


def load_image(path: Optional[str]) -> Optional[Image.Image]:
    if not path:
        return None
    return Image.open(path).convert("RGB")


def get_image_feats(image_processor, img: Image.Image, device: torch.device):
    """
    이미지를 먼저 전처리해서 타일 개수를 알아낸다.
    결과: pixel_values: (tiles, 3, H, W), image_sizes: (tiles, 2), n_tiles: int
    """
    out = image_processor(images=img, return_tensors="pt")
    pixel_values = out["pixel_values"]  # (tiles, 3, 384, 384)
    image_sizes = out.get("image_sizes", None)
    if image_sizes is None:
        # 일부 버전은 image_sizes를 안 줄 수 있어요. 그 경우 타일만 맞춰서 빈 텐서라도 준다.
        image_sizes = torch.tensor([[pixel_values.shape[-2], pixel_values.shape[-1]]]*pixel_values.shape[0])
    n_tiles = int(pixel_values.shape[0])
    pixel_values = pixel_values.to(device)
    image_sizes = image_sizes.to(device)
    return pixel_values, image_sizes, n_tiles


def find_image_token(processor, fallback_candidates=("<|extra_id_11|>", "<|image|>")) -> str:
    """
    토크나이저의 special tokens 중 이미지 토큰 후보를 찾는다.
    우선순위: processor.image_processor.image_token -> '<|image|>' -> '<|extra_id_11|>' 등
    """
    # 1) image_processor 정의 우선
    t = getattr(getattr(processor, "image_processor", None), "image_token", None)
    if isinstance(t, str):
        return t

    # 2) special tokens에서 후보 우선 선택
    specials = set(processor.tokenizer.all_special_tokens)
    for cand in fallback_candidates:
        if cand in specials:
            return cand
    # 마지막 fallback: 첫 번째 special 중 "image"나 "extra_id" 포함된 것
    for s in specials:
        if ("image" in s) or ("extra_id" in s):
            return s
    # 정말 없으면 에러
    raise RuntimeError("이미지 토큰 문자열을 찾지 못했습니다. tokenizer의 special tokens를 확인해주세요.")


def build_raw_chat_with_repeated_image_tokens(image_token: str, repeat: int, prompt: str) -> str:
    """
    A.X-4.0-VL 류의 심플한 raw 템플릿을 직접 만든다.
    이미지 토큰을 repeat 번 연달아 붙임.
    """
    img_block = image_token * repeat
    # 줄바꿈으로 구분해 주는 게 안전
    chat = (
        "<|im_start|><|user|>\n"
        f"{img_block}\n"
        f"{prompt}\n"
        "<|im_end|>\n"
        "<|im_start|><|assistant|>"
    )
    return chat


def decode_assistant_only(full_text: str) -> str:
    anchors = ["<|im_start|><|assistant|>", "<|assistant|>", "assistant:"]
    for a in anchors:
        if a in full_text:
            return full_text.split(a)[-1].strip()
    return full_text.strip().splitlines()[-1].strip()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--max_input_len", type=int, default=2048)  # 1872 토큰(+텍스트)도 감당되게 넉넉히
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    if args.debug:
        print("\n============================== LOAD ==============================")
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    if args.debug:
        print(f"[tokenizer] pad_id={tok.pad_token_id}, eos_id={tok.eos_token_id}, padding_side={tok.padding_side}")

    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id
    if hasattr(model, "config"):
        model.config.use_cache = True
    if args.debug:
        print(f"[model] dtype={dtype}, use_cache={getattr(model.config, 'use_cache', None)}")

    img = load_image(args.image)

    # 1) 이미지 먼저 전처리해서 타일 수 파악
    pixel_values = None
    image_sizes = None
    n_tiles = 0
    if img is not None:
        pixel_values, image_sizes, n_tiles = get_image_feats(processor.image_processor, img, device)
        if args.debug:
            print(f"[image] pixel_values={tuple(pixel_values.shape)}, n_tiles={n_tiles}")

    # 2) 이미지 토큰 문자열과 타일당 토큰 수 탐지
    image_token_str = None
    image_token_id = None
    num_tokens_per_tile = 1
    if img is not None:
        image_token_str = find_image_token(processor)
        image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token_str)
        # 보통 AX4VL은 144
        num_tokens_per_tile = getattr(processor.image_processor, "num_tokens_per_tile", None)
        if num_tokens_per_tile is None:
            # 혹시 없으면 모델 config 쪽에서도 시도
            num_tokens_per_tile = getattr(getattr(model, "config", object()), "num_tokens_per_tile", None)
        if num_tokens_per_tile is None:
            # 마지막 안전망: 144로 가정
            num_tokens_per_tile = 144
        # 모델 config에 주입(내부에서 참조할 수 있음)
        setattr(model.config, "image_token_id", image_token_id)
        setattr(model.config, "image_token", image_token_str)
        setattr(model.config, "num_tokens_per_tile", num_tokens_per_tile)

    # 3) RAW 텍스트 구성: 이미지 토큰을 tiles * tokens_per_tile 만큼 반복
    if img is not None:
        repeat = n_tiles * num_tokens_per_tile
        chat_text = build_raw_chat_with_repeated_image_tokens(image_token_str, repeat, args.prompt)
    else:
        # 텍스트만
        chat_text = (
            "<|im_start|><|user|>\n"
            f"{args.prompt}\n"
            "<|im_end|>\n"
            "<|im_start|><|assistant|>"
        )

    # 4) 텍스트만 토크나이즈(이미지는 따로 줌: 이미지 토큰이 사라지지 않게)
    enc = processor.tokenizer(
        chat_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_len
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    if args.debug and img is not None:
        got = int((input_ids == image_token_id).sum().item())
        expected = n_tiles * num_tokens_per_tile
        print(f"[image_token] str={image_token_str} id={image_token_id} "
              f"num_tokens_per_tile={num_tokens_per_tile} "
              f"count_in_ids={got}, expected={expected}")

    if args.debug:
        print("\n============================== CHAT (PREVIEW) ==============================")
        preview = chat_text
        if len(preview) > 600:
            preview = preview[:600] + " ... (truncated)"
        print(preview)

    # 5) 최종 입력 묶어서 generate
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if img is not None:
        inputs["pixel_values"] = pixel_values
        inputs["image_sizes"] = image_sizes

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0.0),
        temperature=args.temperature,
        top_p=args.top_p,
    )
    if args.top_k is not None:
        gen_kwargs["top_k"] = args.top_k

    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
        out = model.generate(**inputs, **gen_kwargs)

    # 6) 디코드
    if hasattr(out, "sequences"):
        seq = out.sequences[0]
    else:
        seq = out[0]
    full_text = processor.tokenizer.decode(seq, skip_special_tokens=False)
    answer = decode_assistant_only(full_text)

    print("\n================= MODEL OUTPUT =================")
    print(answer.strip())
    print("================================================")


if __name__ == "__main__":
    main()