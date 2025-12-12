# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from typing import Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


def load_image(path: Optional[str]) -> Optional[Image.Image]:
    if not path:
        return None
    return Image.open(path).convert("RGB")


@torch.no_grad()
def tiles_and_feats(image_processor, img: Image.Image, device: torch.device):
    out = image_processor(images=img, return_tensors="pt")
    pixel_values = out["pixel_values"].to(device)
    image_sizes = out.get("image_sizes", None)
    if image_sizes is None:
        image_sizes = torch.tensor([[pixel_values.shape[-2], pixel_values.shape[-1]]]*pixel_values.shape[0],
                                   device=device)
    n_tiles = int(pixel_values.shape[0])
    return pixel_values, image_sizes, n_tiles


def prefer_expected_image_token(processor, model) -> Tuple[str, int]:
    """
    1순위: model.config.image_token / image_token_id
    2순위: processor.image_processor.image_token
    3순위: 토크나이저에 존재하는 '<|extra_id_11|>'  (AX4VL류에서 흔함)
    4순위: '<|image|>' 등 기타 후보
    """
    tok = processor.tokenizer

    # 1) 모델 config 선호
    t_cfg = getattr(getattr(model, "config", object()), "image_token", None)
    if isinstance(t_cfg, str):
        tid = tok.convert_tokens_to_ids(t_cfg)
        if tid is not None and tid >= 0 and tid != tok.unk_token_id:
            return t_cfg, int(tid)

    # 2) 프로세서 image_processor
    ip = getattr(processor, "image_processor", None)
    t_ip = getattr(ip, "image_token", None)
    if isinstance(t_ip, str):
        tid = tok.convert_tokens_to_ids(t_ip)
        if tid is not None and tid >= 0 and tid != tok.unk_token_id:
            try:
                model.config.image_token = t_ip
                model.config.image_token_id = int(tid)
            except Exception:
                pass
            return t_ip, int(tid)

    # 3) 흔한 기대 토큰 강제 우선: <|extra_id_11|>
    for cand in ("<|extra_id_11|>", "<|image|>"):
        tid = tok.convert_tokens_to_ids(cand)
        if tid is not None and tid >= 0 and tid != tok.unk_token_id:
            try:
                model.config.image_token = cand
                model.config.image_token_id = int(tid)
            except Exception:
                pass
            return cand, int(tid)

    # 4) 정말 없으면 추가 등록
    fallback = "<|image|>"
    tok.add_special_tokens({"additional_special_tokens": [fallback]})
    model.resize_token_embeddings(len(tok))
    tid = tok.convert_tokens_to_ids(fallback)
    try:
        model.config.image_token = fallback
        model.config.image_token_id = int(tid)
    except Exception:
        pass
    return fallback, int(tid)


def tokens_per_tile(processor, model) -> int:
    for src in (processor.image_processor, getattr(model, "config", object())):
        val = getattr(src, "num_tokens_per_tile", None)
        if isinstance(val, int) and val > 0:
            try:
                model.config.num_tokens_per_tile = int(val)
            except Exception:
                pass
            return val
    try:
        model.config.num_tokens_per_tile = 144
    except Exception:
        pass
    return 144


def build_raw_chat(image_token: str, repeat: int, prompt: str) -> str:
    img_block = image_token * max(0, repeat)
    return (
        "<|im_start|><|user|>\n"
        f"{img_block}\n"
        f"{prompt}\n"
        "<|im_end|>\n"
        "<|im_start|><|assistant|>"
    )


def count_token(input_ids: torch.Tensor, token_id: int) -> int:
    return int((input_ids == token_id).sum().item())


def print_head(seq: torch.Tensor, n: int = 40):
    s = seq.flatten().tolist()[:n]
    print(f"  head_ids({n}): {s}")


def safe_generate(model, inputs: dict, label: str):
    print(f"\n[generate:{label}] start")
    try:
        out = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )
        print(f"[generate:{label}] OK")
        return out
    except Exception as e:
        print(f"[generate:{label}] ERROR: {type(e).__name__}: {e}")
        return None


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--user_text", default="검은색 가방을 잃어버렸습니다. 앞쪽 흰색 지퍼가 특징입니다.")
    ap.add_argument("--police_text", default="검정 백팩. 전면 흰색 지퍼.")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_input_len", type=int, default=2048)
    ap.add_argument("--try_manual", action="store_true")
    ap.add_argument("--try_template", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"[env] device={device}, dtype={dtype}")
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt, dtype=dtype, trust_remote_code=True, attn_implementation="flash_attention_2"
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=dtype, trust_remote_code=True)
    model.to(device).eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    img = load_image(args.image)
    if img is None:
        print("[error] image not found/failed to load")
        return

    pixel_values, image_sizes, n_tiles = tiles_and_feats(processor.image_processor, img, device)
    print(f"[image] tiles={n_tiles}, pixel_values={tuple(pixel_values.shape)}, image_sizes={tuple(image_sizes.shape)}")

    # 올바른 이미지 토큰/ID 및 타일당 토큰수
    img_tok_str, img_tok_id = prefer_expected_image_token(processor, model)
    tpt = tokens_per_tile(processor, model)
    expected = n_tiles * tpt
    print(f"[token] image_token_str={repr(img_tok_str)}, image_token_id={img_tok_id}")
    print(f"[token] num_tokens_per_tile={tpt}, expected_tokens={expected}")

    # 안내용: 토크나이저 상태
    try:
        enc_chk = tok(img_tok_str, add_special_tokens=False)
        print(f"[sanity] encode(image_token) -> {enc_chk}")
    except Exception:
        pass

    prompt = (
        "아래 사용자 진술과 경찰 기록이 같은 물건을 설명하는지 '예' 또는 '아니오'로만 답하세요.\n\n"
        f"사용자 진술: {args.user_text}\n경찰 기록: {args.police_text}\n\n답변:"
    )

    # ===== A. 수동 경로(문자열에 이미지토큰 expected번 삽입) =====
    if args.try_manual:
        chat_text = build_raw_chat(img_tok_str, expected, prompt)
        enc = tok(chat_text, return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_input_len)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        got = count_token(input_ids, img_tok_id)
        print("\n[manual] ------")
        print(f"[manual] got_image_token_count={got} (expected {expected})")
        print_head(input_ids)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
        }
        safe_generate(model, inputs, "manual")

        # ===== B. 템플릿 경로(apply_chat_template: 이미지 객체를 content에 직접 넣고 images 인자 미전달) =====
    if args.try_template:
        print("\n[template] ------")
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},  # 이미지를 content에 직접 넣기
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            enc2 = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            # --- 반환값 정규화: Tensor(=input_ids만) 이면 dict로 변환
            if isinstance(enc2, torch.Tensor):
                input_ids = enc2.to(device)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                enc2 = {"input_ids": input_ids, "attention_mask": attention_mask}
            elif hasattr(enc2, "to"):
                # BatchEncoding 등 .to() 지원 시
                enc2 = enc2.to(device)
            else:
                # 일반 dict
                enc2 = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc2.items()}

            # pixel_values / image_sizes 보강 (없는 경우에만 채움)
            enc2.setdefault("pixel_values", pixel_values)
            enc2.setdefault("image_sizes", image_sizes)

            got2 = count_token(enc2["input_ids"], img_tok_id)
            print(f"[template] got_image_token_count={got2} (ideal >= {tpt}, often == {expected})")
            print_head(enc2["input_ids"])

            safe_generate(model, enc2, "template")
        except Exception as e:
            print(f"[template] ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()