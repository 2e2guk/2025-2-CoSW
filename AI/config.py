import os
from typing import Optional

def getenv_str(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if (v is not None and len(v) > 0) else default

def getenv_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y")

def getenv_dtype(key: str, default: str = "auto") -> str:
    v = os.getenv(key)
    return v if v else default

# index(KoCLIP)
KOCLIP_MODEL_ID = getenv_str("KOCLIP_MODEL_ID", "/home/elicer/work/ckpt_koclip/epoch_17")

# reranker(VLM)
RERANKER_BACKEND = getenv_str("RERANKER_BACKEND", "none")

VLM_BASE_MODEL_ID = getenv_str("VLM_BASE_MODEL_ID", None)        # e.g. /home/elicer/work/VLM/module2/ax_model_local
VLM_ADAPTER_ID    = getenv_str("VLM_ADAPTER_ID", None)           # e.g. /home/elicer/work/VLM/checkpoints_VLM/epoch_06

VLM_RERANKER_DEVICE = getenv_str("VLM_RERANKER_DEVICE", None)    # e.g. "cuda:0" or "cpu"
VLM_RERANKER_DTYPE  = getenv_dtype("VLM_RERANKER_DTYPE", "auto") # "auto"|"float16"|"bfloat16"|"float32"