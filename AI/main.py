# main.py
import os, re, time, logging
from typing import Any, Dict, List, Optional
from importlib import import_module

from fastapi import FastAPI, Body, Header, HTTPException
from pydantic import BaseModel

from embeddings_index import (
    index_stats, clear_index, upsert_items, search_topk, meta_lookup,
    backfill_meta, meta_coverage_stats,
)

log = logging.getLogger("main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

app = FastAPI(title="Police Lost&Found – Retrieval API")

# ─────────────────────────────────────────────────────────────────────────────
# 환경변수 / 플래그
# ─────────────────────────────────────────────────────────────────────────────
BACKEND_UPSERT_KEY = os.getenv("BACKEND_UPSERT_KEY", "").strip()
POLICE_FETCH_ENABLED = os.getenv("POLICE_FETCH_ENABLED", "0") in ("1","true","True")
MODEL_WARMUP = os.getenv("MODEL_WARMUP", "1")

# infer 기본값(ENV로 커스터마이즈 가능)
DEF_RETRIEVAL_K = int(os.getenv("INFER_DEFAULT_RETRIEVAL_K", "3000"))
DEF_RETURN_N     = int(os.getenv("INFER_DEFAULT_RETURN_N", "3"))
DEF_DO_RERANK    = str(os.getenv("INFER_DEFAULT_DO_RERANK", "1")).lower() in ("1","true","yes","on")
DEF_W_TEXT       = float(os.getenv("INFER_DEFAULT_W_TEXT", "0.7"))
DEF_W_IMAGE      = float(os.getenv("INFER_DEFAULT_W_IMAGE", "0.3"))
DEF_PREVIEW_N    = int(os.getenv("INFER_DEFAULT_PREVIEW_N", "800"))

def _truthy(x: Optional[str]) -> bool:
    return str(x or "").lower() in ("1","true","yes","on")

# ─────────────────────────────────────────────────────────────────────────────
# 유틸: 날짜/문자 정규화
# ─────────────────────────────────────────────────────────────────────────────
def _ymd_normalize(s: Optional[str]) -> str:
    if not s: return ""
    digits = re.sub(r"\D", "", str(s))
    return digits[:8] if len(digits) >= 8 else digits

def _strip(x: Optional[str]) -> str:
    return (x or "").strip()

# ─────────────────────────────────────────────────────────────────────────────
# 유입 스키마 → 내부 표준 스키마
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_one(d: Dict[str, Any]) -> Dict[str, str]:
    if "atcId" in d and "fdPrdtNm" in d:
        return {
            "atcId": _strip(d.get("atcId")),
            "fdPrdtNm": _strip(d.get("fdPrdtNm")),
            "fdSbjt": _strip(d.get("fdSbjt")),
            "depPlace": _strip(d.get("depPlace")),
            "fdYmd": _ymd_normalize(d.get("fdYmd")),
            "prdtClNm": _strip(d.get("prdtClNm")),
            "clrNm": _strip(d.get("clrNm")),
            "fdFilePathImg": _strip(d.get("fdFilePathImg")),
            "fdPlace": _strip(d.get("fdPlace")),
        }
    return {
        "atcId": _strip(d.get("id") or d.get("atcId")),
        "fdPrdtNm": _strip(d.get("name") or d.get("fdPrdtNm")),
        "fdSbjt": _strip(d.get("subject") or d.get("fdSbjt") or (d.get("raw") or {}).get("fdSbjt", "")),
        "depPlace": _strip(d.get("custody_place") or d.get("depPlace")),
        "fdYmd": _ymd_normalize(d.get("date") or d.get("fdYmd")),
        "prdtClNm": _strip(d.get("category") or d.get("prdtClNm")),
        "clrNm": _strip(d.get("color") or d.get("clrNm")),
        "fdFilePathImg": _strip(d.get("image") or d.get("fdFilePathImg")),
        "fdPlace": _strip(d.get("found_place") or d.get("fdPlace")),
    }

def _normalize_list(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for d in items:
        nd = _normalize_one(d)
        if not nd["atcId"] or not nd["fdPrdtNm"]:
            continue
        out.append(nd)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic 모델
# ─────────────────────────────────────────────────────────────────────────────
class UpsertReq(BaseModel):
    items: List[Dict[str, Any]]
    clear: Optional[bool] = False

class RefreshReq(BaseModel):
    mode: Optional[str] = "limit"
    target_n: Optional[int] = 1500
    rows: Optional[int] = 100
    clear: Optional[bool] = False
    op: Optional[str] = "op2"

class InferParams(BaseModel):
    retrieval_k: Optional[int] = None
    return_n:  Optional[int] = None
    do_rerank: Optional[bool] = None
    w_text:    Optional[float] = None
    w_image:   Optional[float] = None
    retrieval_preview_n: Optional[int] = None

class InferReq(BaseModel):
    text: str
    # 최상위(편의) 필드 — 백엔드가 text + return_n 만 보내도 동작
    return_n: Optional[int] = None
    retrieval_k: Optional[int] = None
    do_rerank: Optional[bool] = None
    w_text: Optional[float] = None
    w_image: Optional[float] = None
    retrieval_preview_n: Optional[int] = None
    # 혹시 params 형태로 올 때도 지원
    params: Optional[InferParams] = None

def _resolve_infer_params(req: InferReq):
    """우선순위: 최상위 필드 > params > ENV 기본값"""
    p = req.params or InferParams()

    def pick(top, sub, default):
        return default if top is None and sub is None else (top if top is not None else sub)

    k  = pick(req.retrieval_k,       p.retrieval_k,       DEF_RETRIEVAL_K)
    n  = pick(req.return_n,          p.return_n,          DEF_RETURN_N)
    rr = pick(req.do_rerank,         p.do_rerank,         DEF_DO_RERANK)
    wt = pick(req.w_text,            p.w_text,            DEF_W_TEXT)
    wi = pick(req.w_image,           p.w_image,           DEF_W_IMAGE)
    pn = pick(req.retrieval_preview_n, p.retrieval_preview_n, DEF_PREVIEW_N)

    # 정합성/정규화
    k = max(1, int(k))
    n = max(1, int(n))
    wt = float(wt); wi = float(wi)
    if wt + wi <= 0:
        wt, wi = 1.0, 0.0
    s = wt + wi
    wt, wi = wt / s, wi / s
    pn = max(0, int(pn))

    return k, n, rr, wt, wi, pn

# ─────────────────────────────────────────────────────────────────────────────
# Startup: KoCLIP/VLM warmup
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def _warmup_models():
    if not _truthy(MODEL_WARMUP):
        log.info("[startup] MODEL_WARMUP disabled")
        return
    try:
        from koclip_embedder import embedding_dim, embed_text_batch
        dim = embedding_dim()
        embed_text_batch(["warmup"], batch_size=1)
        log.info("[startup] KoCLIP warmup done. dim=%s device=%s model=%s",
                 dim, os.getenv("KOCLIP_DEVICE","cpu"), os.getenv("KOCLIP_MODEL_ID",""))
    except Exception as e:
        log.warning("[startup] KoCLIP warmup failed: %r", e)
    try:
        mod_name = os.getenv("RERANKER_MODULE", "reranker_vlm")
        reranker = import_module(mod_name).load_from_env()
        log.info("[startup] reranker ready: %s", reranker.status())
    except Exception as e:
        log.warning("[startup] reranker warmup failed: %r", e)

# ─────────────────────────────────────────────────────────────────────────────
# 헬스/디버그
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/health")
def api_health():
    return {"ok": True, "time": int(time.time())}

@app.get("/api/v1/index/stats")
def api_index_stats():
    return {"status": "success", "meta": index_stats()}

@app.get("/api/v1/debug/models")
def api_debug_models():
    out: Dict[str, Any] = {"koclip": {}, "reranker": {}}
    try:
        from koclip_embedder import embedding_dim
        out["koclip"]["dim"] = embedding_dim()
        out["koclip"]["device"] = os.getenv("KOCLIP_DEVICE","cpu")
        out["koclip"]["model"] = os.getenv("KOCLIP_MODEL_ID","")
    except Exception as e:
        out["koclip"]["error"] = repr(e)
    try:
        mod_name = os.getenv("RERANKER_MODULE", "reranker_vlm")
        reranker = import_module(mod_name).load_from_env()
        out["reranker"] = reranker.status()
    except Exception as e:
        out["reranker"]["error"] = repr(e)
    return out

@app.get("/api/v1/index/meta/stats")
def api_meta_stats():
    return meta_coverage_stats()

# ─────────────────────────────────────────────────────────────────────────────
# 업서트/백필
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/index/backfill_meta")
def api_backfill_meta(req: UpsertReq, x_backend_key: Optional[str] = Header(None, convert_underscores=False)):
    if BACKEND_UPSERT_KEY:
        if not x_backend_key or x_backend_key != BACKEND_UPSERT_KEY:
            raise HTTPException(status_code=401, detail="invalid backend key")
    items_norm = _normalize_list(req.items)
    if not items_norm:
        raise HTTPException(status_code=422, detail="no valid items after normalization")
    res = backfill_meta(items_norm)
    stats = meta_coverage_stats()
    return {"status": "success", "updated": res["updated"], "skipped": res["skipped"], "meta_stats": stats}

@app.post("/api/v1/index/upsert")
def api_index_upsert(req: UpsertReq, x_backend_key: Optional[str] = Header(None, convert_underscores=False)):
    if BACKEND_UPSERT_KEY:
        if not x_backend_key or x_backend_key != BACKEND_UPSERT_KEY:
            raise HTTPException(status_code=401, detail="invalid backend key")
    t0 = time.time()
    items_norm = _normalize_list(req.items)
    if req.clear:
        clear_index()
    if not items_norm:
        raise HTTPException(status_code=422, detail="no valid items after normalization")
    n_added = upsert_items(items_norm)
    meta = index_stats()
    meta["last_upsert_took_s"] = round(time.time() - t0, 3)
    meta["last_upsert_added"] = int(n_added)
    return {"status": "success", "meta": meta}

@app.post("/api/v1/index/refresh")
def api_index_refresh(req: RefreshReq):
    log.info("[index.refresh] allow_crawl=%s clear=%s mode=%s target_n=%s rows=%s op=%s",
             POLICE_FETCH_ENABLED, req.clear, req.mode, req.target_n, req.rows, req.op)
    if not POLICE_FETCH_ENABLED:
        return {"status":"success", "meta": index_stats()}
    return {"status":"success", "meta": index_stats()}

# ─────────────────────────────────────────────────────────────────────────────
# 파이프라인 추론
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/pipeline/infer")
def api_pipeline_infer(req: InferReq):
    k, n, do_rerank, w_text, w_image, preview_n = _resolve_infer_params(req)

    base_hits = search_topk(req.text, k)
    if not base_hits:
        return {"top": []}

    ids = [h["atcId"] for h in base_hits]
    meta_map = {m["atcId"]: m for m in meta_lookup(ids)}

    items: List[Dict[str, Any]] = []
    for h in base_hits:
        m = meta_map.get(h["atcId"], {})
        obj = {"atcId": h["atcId"], "score_base": h["score_base"]}
        obj.update(m)
        items.append(obj)

    if do_rerank:
        try:
            mod_name = os.getenv("RERANKER_MODULE", "reranker_vlm")
            reranker = import_module(mod_name).get_reranker(force=True)
            ranked = reranker.rerank(req.text, items, top_n=min(n, len(items)))
        except Exception as e:
            log.warning("[infer] rerank failed: %r", e)
            ranked = items[:n]
    else:
        ranked = items[:n]

    def _short(s: str, L: int) -> str:
        s = s or "";  return s if len(s) <= L else (s[:L] + "…")

    out: List[Dict[str, Any]] = []
    for obj in ranked:
        sb = float(obj.get("score_base", 0.0))
        sv = float(obj.get("score_vlm", 0.0)) if "score_vlm" in obj else 0.0
        m  = meta_map.get(obj["atcId"], {})
        out.append({
            "atcId": obj["atcId"],
            "score_base": sb,
            "score_vlm": sv if "score_vlm" in obj else None,
            "score": w_text*sb + w_image*sv,
            "preview": {
                "fdPrdtNm": _short(str(m.get("fdPrdtNm","")), preview_n),
                "prdtClNm": _short(str(m.get("prdtClNm","")), preview_n),
                "clrNm": _short(str(m.get("clrNm","")), preview_n),
                "depPlace": _short(str(m.get("depPlace","")), preview_n),
                "fdYmd": _short(str(m.get("fdYmd","")), preview_n),
                "fdPlace": _short(str(m.get("fdPlace","")), preview_n),
                "fdFilePathImg": _short(str(m.get("fdFilePathImg","")), preview_n),
            }
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return {"top": out[:n]}