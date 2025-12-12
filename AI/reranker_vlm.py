# reranker_vlm.py
import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F

logger = logging.getLogger("reranker_vlm")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

# ----------------- helpers -----------------
def _pick_device(name: Optional[str]) -> torch.device:
    if not name:
        return torch.device("cpu")
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")

def _pick_dtype_auto(device: torch.device, prefer: Optional[str]) -> torch.dtype:
    """
    - prefer: "auto"|"bf16"|"fp16"|"float16"|"half"
    - CPU면 반드시 float32
    - CUDA에서 bf16 미지원이면 fp16
    - 그 외는 요청값 존중
    """
    req = (prefer or "auto").lower()
    if device.type == "cpu":
        return torch.float32

    # CUDA
    if req in ("bf16", "bfloat16", "auto"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # bf16 미지원 → fp16
        return torch.float16
    if req in ("fp16", "float16", "half"):
        return torch.float16
    # 기본
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def _register_remote_module(base_model: str):
    from transformers import AutoConfig
    try:
        _ = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    except Exception as e:
        logger.warning("[rerank] AutoConfig registration warning: %r", e)

def _try_import(module_names: List[str]):
    import importlib
    last = None
    for m in module_names:
        try:
            return importlib.import_module(m)
        except Exception as e:
            last = e
    if last:
        logger.debug("[rerank] import failed for %s: %r", module_names, last)
    return None

def _find_model_class(mod, prefer: Tuple[str, ...]) -> Optional[Type]:
    import inspect
    cands = []
    for name, obj in vars(mod).items():
        if inspect.isclass(obj) and hasattr(obj, "from_pretrained") and callable(getattr(obj, "from_pretrained")):
            cands.append((name, obj))
    if not cands:
        return None
    def score(name: str) -> int:
        s = 0
        low = name.lower()
        for i, kw in enumerate(prefer):
            if kw.lower() in low:
                s += (len(prefer) - i) * 10
        if "ax4vl" in low: s += 5
        if "conditional" in low: s += 2
        return s
    cands.sort(key=lambda kv: score(kv[0]), reverse=True)
    return cands[0][1]

def _get_arch_from_config(base_model: str) -> Optional[str]:
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        archs = getattr(cfg, "architectures", None)
        if isinstance(archs, (list, tuple)) and len(archs) > 0:
            return str(archs[0])
    except Exception as e:
        logger.warning("[rerank] read config.architectures failed: %r", e)
    return None

def _load_model_processor_tokenizer(base_model: str,
                                    device: torch.device,
                                    torch_dtype: torch.dtype):
    """
    1) 로컬 modeling 모듈에서 클래스 찾아 로드
    2) 실패 시 AutoModel(ForCausalLM) 폴백
    3) Processor/Tokenizer 로드
    """
    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    repo = os.path.basename(os.path.abspath(base_model))
    _register_remote_module(base_model)

    prefer_class_name = _get_arch_from_config(base_model)
    modeling_mod = _try_import([
        f"transformers_modules.{repo}.modeling_ax4vl",
        f"transformers_modules.{repo}.modeling_{repo}",
        f"transformers_modules.{repo}.modeling",
    ])
    processor_mod = _try_import([
        f"transformers_modules.{repo}.processing_ax4vl",
        f"transformers_modules.{repo}.processing_{repo}",
        f"transformers_modules.{repo}.processing",
    ])

    model = None
    if modeling_mod is not None and prefer_class_name:
        cls = getattr(modeling_mod, prefer_class_name, None)
        if cls is not None and hasattr(cls, "from_pretrained"):
            logger.info("[rerank] modeling class (by config.architectures): %s.%s",
                        modeling_mod.__name__, prefer_class_name)
            model = cls.from_pretrained(
                base_model, trust_remote_code=True,
                torch_dtype=torch_dtype, low_cpu_mem_usage=True
            )

    if model is None and modeling_mod is not None:
        cls = _find_model_class(
            modeling_mod,
            prefer=("ForConditionalGeneration", "ForCausalLM", "Model")
        )
        if cls is not None:
            logger.info("[rerank] modeling class (auto discovered): %s.%s",
                        modeling_mod.__name__, cls.__name__)
            model = cls.from_pretrained(
                base_model, trust_remote_code=True,
                torch_dtype=torch_dtype, low_cpu_mem_usage=True
            )

    if model is None:
        for auto in (AutoModelForCausalLM, AutoModel):
            try:
                model = auto.from_pretrained(
                    base_model, trust_remote_code=True,
                    torch_dtype=torch_dtype, low_cpu_mem_usage=True
                )
                logger.info("[rerank] model loaded via %s", auto.__name__)
                break
            except Exception as e:
                logger.warning("[rerank] %s path failed: %r", auto.__name__, e)

    if model is None:
        raise RuntimeError("VLM load failed: no suitable class found that supports from_pretrained")

    # Processor
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        logger.info("[rerank] processor loaded via AutoProcessor")
    except Exception as e:
        logger.warning("[rerank] AutoProcessor failed: %r", e)
        if processor_mod is not None:
            import inspect
            for name, obj in vars(processor_mod).items():
                if inspect.isclass(obj) and name.endswith("Processor") and hasattr(obj, "from_pretrained"):
                    processor = obj.from_pretrained(base_model, trust_remote_code=True)
                    logger.info("[rerank] processor class resolved: %s.%s", processor_mod.__name__, name)
                    break

    # Tokenizer
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    except Exception as e:
        logger.warning("[rerank] AutoTokenizer failed: %r; try processor.tokenizer", e)
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            tokenizer = tok

    if tokenizer is None:
        raise RuntimeError("Tokenizer load failed")

    # pad 토큰 안전장치(없으면 eos로 대체)
    try:
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception:
        pass

    model.to(device)
    model.eval()
    return model, processor, tokenizer


# ----------------- Reranker -----------------
class _VLMReranker:
    backend = "vlm"

    def __init__(self):
        self.ready = False
        self.device = _pick_device(os.getenv("VLM_RERANKER_DEVICE"))
        self.torch_dtype = _pick_dtype_auto(self.device, os.getenv("VLM_RERANKER_DTYPE"))
        self.batch = int(os.getenv("VLM_RERANK_BATCH", "8"))

        self.base_model_id = os.getenv("VLM_BASE_MODEL_ID")
        self.adapter_id = os.getenv("VLM_ADAPTER_ID")

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.hidden_size = None

    def ensure_ready(self):
        if self.ready:
            return
        if not self.base_model_id:
            raise RuntimeError("VLM_BASE_MODEL_ID is not set")

        logger.info("[rerank] loading base VLM: %s, device=%s dtype=%s",
                    self.base_model_id, self.device, self.torch_dtype)

        model, processor, tokenizer = _load_model_processor_tokenizer(
            self.base_model_id, self.device, self.torch_dtype
        )

        # PEFT 어댑터
        if self.adapter_id and os.path.exists(self.adapter_id):
            try:
                from peft import PeftModel
                logger.info("[rerank] applying PEFT adapter: %s", self.adapter_id)
                model = PeftModel.from_pretrained(model, self.adapter_id, is_trainable=False)
            except Exception as e:
                logger.warning("[rerank] PEFT load failed: %r", e)

        # hidden size 추정
        try:
            dummy = self._encode_text(model, tokenizer, ["dummy"])
            self.hidden_size = int(dummy.shape[-1])
        except Exception as e:
            logger.warning("[rerank] warmup encode failed (will still proceed): %r", e)

        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.ready = True
        logger.info("[rerank] ready. base='%s', adapter='%s', device=%s, dtype=%s",
                    self.base_model_id, self.adapter_id, str(self.device), str(self.torch_dtype))

    def status(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "ready": self.ready,
            "base_model": self.base_model_id,
            "adapter": self.adapter_id,
            "device": str(self.device),
            "dtype": str(self.torch_dtype),
        }

    @torch.inference_mode()
    def _encode_text(self, model, tokenizer, texts: List[str]) -> torch.Tensor:
        """마지막 hidden_state 평균 풀링 → L2 normalize"""
        tok = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        tok = {k: v.to(self.device) for k, v in tok.items()}
        out = model(**tok, output_hidden_states=True, use_cache=False, return_dict=True)
        h = out.hidden_states[-1]        # (B, T, H)
        emb = h.mean(dim=1)              # (B, H)
        emb = F.normalize(emb.float(), dim=-1)
        return emb

    def _item_text(self, it: Dict[str, Any]) -> str:
        """
        리랭크 품질을 위해 텍스트 힌트를 최대한 포함.
        """
        parts = []
        for k in (
            "fdPrdtNm",   # 물품명
            "prdtClNm",   # 분류
            "clrNm",      # 색상
            "fdSbjt",     # 제목/요약
            "fdPlace",    # 습득 장소
            "depPlace",   # 보관 장소
            "fdYmd",      # 날짜
            "atcId",      # ID
            "ttl", "prdtnm"  # 혹시 들어올 수 있는 다른 키
        ):
            v = it.get(k)
            if v:
                parts.append(str(v))
        return " ".join(parts)

    @torch.inference_mode()
    def rerank(self, query: str, items: List[Dict[str, Any]], top_n: int):
        """
        텍스트 기반 코사인 유사도 리랭크.
        (이미지 사용은 현재 미지원. 사용하려면 업서트 단계에서 이미지 임베딩을 함께 저장하거나,
         여기서 이미지 URL 다운로드+전처리를 추가해야 함)
        """
        if not items:
            return []

        if not self.ready or self.model is None or self.tokenizer is None:
            return items[:top_n]

        # 1) 쿼리 임베딩
        try:
            q_vec = self._encode_text(self.model, self.tokenizer, [str(query)])  # (1, H)
        except Exception as e:
            logger.warning("[rerank] encode query failed (%r). fallback to base order", e)
            return items[:top_n]

        # 2) 아이템 임베딩 (마이크로 배치)
        texts = [self._item_text(it) for it in items]
        scores: List[float] = []

        bs = max(1, int(self.batch))
        for s in range(0, len(texts), bs):
            chunk = texts[s: s+bs]
            try:
                emb = self._encode_text(self.model, self.tokenizer, chunk)  # (b, H)
                sim = torch.mm(emb, q_vec.T).squeeze(-1)  # (b,)
                scores.extend(sim.tolist())
            except Exception as e:
                logger.warning("[rerank] encode batch %d~%d failed: %r (zeros filled)", s, s+len(chunk)-1, e)
                scores.extend([0.0] * len(chunk))

        # 3) 정렬
        order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)

        ret: List[Dict[str, Any]] = []
        for i in order[:top_n]:
            obj = dict(items[i])
            obj["score_vlm"] = float(scores[i])
            ret.append(obj)
        return ret


RERANKER = _VLMReranker()

def get_reranker(force: bool = False) -> _VLMReranker:
    if force or not RERANKER.ready:
        try:
            RERANKER.ensure_ready()
        except Exception as e:
            logger.warning("[rerank] init failed: %r", e)
    return RERANKER

def load_from_env() -> _VLMReranker:
    return get_reranker(force=True)