# embeddings_index.py
import os, json, time, logging
from typing import Dict, List, Tuple
import numpy as np
import faiss

from koclip_embedder import embed_text_batch, embedding_dim

log = logging.getLogger("emb_index")

# ─────────────────────────────────────────────────────────────────────────────
# 파일 경로 (ENV 우선)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.getenv("INDEX_FAISS_PATH", os.path.join(BASE_DIR, "police_db.faiss"))
META_PATH  = os.getenv("INDEX_META_PATH",  os.path.join(BASE_DIR, "police_db.meta.json"))

# 구형 .npy(id 리스트) 호환 경로(있으면 1회 마이그레이션)
LEGACY_META_NPY = os.path.join(BASE_DIR, "police_meta.npy")

# 차원: KoCLIP에서 읽되 실패시 512
try:
    DIM = int(embedding_dim())
except Exception:
    DIM = 512

# ─────────────────────────────────────────────────────────────────────────────
# 전역 상태
# ─────────────────────────────────────────────────────────────────────────────
_index: faiss.IndexFlatIP | None = None     # 벡터 인덱스 (IP; L2-normalized 가정)
_ids: List[str] = []                        # 인덱스 순서와 동일한 atcId 리스트
_meta: Dict[str, Dict] = {}                 # atcId -> 메타(미리보기 필드)
_last_persist_t: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _faiss_new() -> faiss.IndexFlatIP:
    return faiss.IndexFlatIP(DIM)

def _read_meta_file(path: str) -> Tuple[List[str], Dict[str, Dict]]:
    if not os.path.exists(path):
        # 구형 포맷 마이그레이션 시도
        if os.path.exists(LEGACY_META_NPY):
            try:
                arr = np.load(LEGACY_META_NPY, allow_pickle=True).tolist()
                if isinstance(arr, list):
                    log.warning("[emb_index] migrate legacy ids from .npy (len=%d)", len(arr))
                    return list(arr), {}
            except Exception as e:
                log.warning("[emb_index] legacy npy read failed: %r", e)
        return [], {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        ids = obj.get("ids", [])
        meta = obj.get("meta", {})
        if not isinstance(ids, list): ids = []
        if not isinstance(meta, dict): meta = {}
        return ids, meta
    except Exception as e:
        log.warning("[emb_index] meta json read failed: %r", e)
        return [], {}

def _write_meta_file():
    global _last_persist_t
    tmp = META_PATH + ".tmp"
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"ids": _ids, "meta": _meta}, f, ensure_ascii=False)
    os.replace(tmp, META_PATH)
    _last_persist_t = time.time()

def _persist_all():
    os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
    faiss.write_index(_index, FAISS_PATH)
    _write_meta_file()

def _load_or_init():
    global _index, _ids, _meta
    if _index is not None:
        return
    if os.path.exists(FAISS_PATH):
        _index = faiss.read_index(FAISS_PATH)
        # 차원 불일치 방어
        if hasattr(_index, "d") and int(_index.d) != DIM:
            log.warning("[emb_index] FAISS dim %d != expected %d; continuing anyway", int(_index.d), DIM)
    else:
        _index = _faiss_new()
    _ids, _meta = _read_meta_file(META_PATH)

    n = int(_index.ntotal)
    if len(_ids) != n:
        log.warning("[emb_index] index/meta length mismatch: %d vs %d. unmapped tails will be ignored.",
                    n, len(_ids))
        # 매칭 안되는 벡터 구간은 검색 결과에서 스킵(인덱스 truncate 불가)

    log.info("[emb_index] ready: N=%d dim=%d ids=%d meta=%d",
             int(_index.ntotal), int(getattr(_index, "d", DIM)), len(_ids), len(_meta))

def _make_text_fields(it: Dict[str, str]) -> str:
    parts = [
        it.get("fdPrdtNm", ""),
        it.get("prdtClNm", ""),
        it.get("clrNm", ""),
        it.get("fdSbjt", ""),
        it.get("fdPlace", ""),
        it.get("depPlace", ""),
        it.get("fdYmd", ""),
    ]
    return " ".join([str(p) for p in parts if p])

def _ensure_loaded():
    if _index is None:
        _load_or_init()

# ─────────────────────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────────────────────
def index_stats() -> Dict:
    _ensure_loaded()
    return {
        "N": int(_index.ntotal),
        "dim": int(getattr(_index, "d", DIM)),
        "ready": bool(_index.ntotal > 0 and len(_ids) > 0),
        "model_id": os.getenv("KOCLIP_MODEL_ID", ""),
        "last_built_at": None,
        "paths": {"faiss": FAISS_PATH, "meta": META_PATH},
        "ids_cached": len(_ids),
        "meta_cached": len(_meta),
        "last_persist": _last_persist_t or None,
    }

def clear_index():
    global _index, _ids, _meta
    _index = _faiss_new()
    _ids = []
    _meta = {}
    _persist_all()
    log.info("[emb_index] cleared")

def upsert_items(items: List[Dict[str, str]]) -> int:
    """
    append-only 정책: 이미 존재하는 atcId는 스킵.
    메타도 함께 저장하여 추후 preview 제공.
    """
    global _index, _ids, _meta
    _ensure_loaded()

    exist = set(_ids)
    to_add_ids: List[str] = []
    texts: List[str] = []
    to_add_meta: Dict[str, Dict] = {}

    for it in items:
        aid = (it.get("atcId") or "").strip()
        if not aid or aid in exist:
            # 존재하면 메타만 최신화(백엔드 보정사항 반영)
            if aid and it:
                _meta[aid] = {
                    "fdPrdtNm": it.get("fdPrdtNm", ""),
                    "prdtClNm": it.get("prdtClNm", ""),
                    "clrNm": it.get("clrNm", ""),
                    "depPlace": it.get("depPlace", ""),
                    "fdYmd": it.get("fdYmd", ""),
                    "fdPlace": it.get("fdPlace", ""),
                    "fdFilePathImg": it.get("fdFilePathImg", ""),
                }
            continue
        to_add_ids.append(aid)
        texts.append(_make_text_fields(it))
        to_add_meta[aid] = {
            "fdPrdtNm": it.get("fdPrdtNm", ""),
            "prdtClNm": it.get("prdtClNm", ""),
            "clrNm": it.get("clrNm", ""),
            "depPlace": it.get("depPlace", ""),
            "fdYmd": it.get("fdYmd", ""),
            "fdPlace": it.get("fdPlace", ""),
            "fdFilePathImg": it.get("fdFilePathImg", ""),
        }

    if not to_add_ids:
        _write_meta_file()  # 메타만 갱신된 경우
        log.info("[emb_index] nothing to add. total=%d (meta updated=%d)", int(_index.ntotal), len(items))
        return 0

    vecs = embed_text_batch(texts)  # (B, D) float32 normalized
    if vecs.shape[1] != DIM:
        raise RuntimeError(f"embedding dim mismatch: {vecs.shape[1]} vs {DIM}")

    _index.add(vecs)
    _ids.extend(to_add_ids)
    _meta.update(to_add_meta)
    _persist_all()

    log.info("[emb_index] added=%d total=%d", len(to_add_ids), int(_index.ntotal))
    return len(to_add_ids)

def backfill_meta(items: List[Dict[str, str]]) -> Dict[str, int]:
    """
    인덱스 추가 없이 메타만 보강/수정.
    """
    global _meta
    _ensure_loaded()
    updated = 0
    skipped = 0
    for it in items:
        aid = (it.get("atcId") or "").strip()
        if not aid:
            skipped += 1
            continue
        before = _meta.get(aid, {})
        now = {
            "fdPrdtNm": it.get("fdPrdtNm", "") or before.get("fdPrdtNm", ""),
            "prdtClNm": it.get("prdtClNm", "") or before.get("prdtClNm", ""),
            "clrNm": it.get("clrNm", "") or before.get("clrNm", ""),
            "depPlace": it.get("depPlace", "") or before.get("depPlace", ""),
            "fdYmd": it.get("fdYmd", "") or before.get("fdYmd", ""),
            "fdPlace": it.get("fdPlace", "") or before.get("fdPlace", ""),
            "fdFilePathImg": it.get("fdFilePathImg", "") or before.get("fdFilePathImg", ""),
        }
        _meta[aid] = now
        updated += 1
    _write_meta_file()
    return {"updated": updated, "skipped": skipped}

def meta_lookup(ids: List[str]) -> List[Dict]:
    _ensure_loaded()
    out = []
    for aid in ids:
        m = _meta.get(aid, {})
        out.append({
            "atcId": aid,
            "fdPrdtNm": m.get("fdPrdtNm", ""),
            "prdtClNm": m.get("prdtClNm", ""),
            "clrNm": m.get("clrNm", ""),
            "depPlace": m.get("depPlace", ""),
            "fdYmd": m.get("fdYmd", ""),
            "fdPlace": m.get("fdPlace", ""),
            "fdFilePathImg": m.get("fdFilePathImg", ""),
        })
    return out

def meta_coverage_stats() -> Dict[str, int]:
    _ensure_loaded()
    n_ids = len(_ids)
    n_meta = sum(1 for aid in _ids if aid in _meta)
    return {"ids": n_ids, "meta_has": n_meta, "meta_missing": n_ids - n_meta}

def search_topk(query: str, k: int) -> List[Dict[str, float]]:
    """
    KoCLIP 텍스트 임베딩으로 코사인 유사도 검색(IP).
    meta/ids 불일치 구간은 스킵.
    반환: [{"atcId": "...", "score_base": float}, ...]
    """
    _ensure_loaded()
    if _index.ntotal == 0 or len(_ids) == 0:
        return []

    k = max(1, int(k))
    q = embed_text_batch([str(query)])  # (1, D)
    D, I = _index.search(q, min(k, int(_index.ntotal)))  # (1, k)
    I = I[0].tolist()
    D = D[0].tolist()

    out: List[Dict[str, float]] = []
    for idx, score in zip(I, D):
        if idx < 0:  # faiss 패딩
            continue
        if idx >= len(_ids):
            # ids 리스트가 더 짧은 경우(구형 상태) → 매핑 불가 항목은 건너뜀
            continue
        aid = _ids[idx]
        out.append({"atcId": aid, "score_base": float(score)})
    return out