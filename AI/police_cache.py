# police_cache.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

try:
    import faiss  # pip install faiss-cpu  (또는 faiss-gpu)
except Exception as e:
    raise RuntimeError("faiss가 필요합니다. `pip install faiss-cpu` 로 설치하세요.") from e

from police_api import get_by_name_place
from koclip_embedder import KoCLIPEmbedder

# 저장 경로
CACHE_DIR = os.getenv("POLICE_CACHE_DIR", "./police_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
INDEX_PATH = os.path.join(CACHE_DIR, "index.faiss")
META_PATH  = os.path.join(CACHE_DIR, "meta.json")

def _hash_to_int64(s: str) -> int:
    # atcId(문자열)를 int64로 안정적으로 매핑
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)

def _pack_text_for_embed(item: Dict[str, Any]) -> str:
    # KoCLIP 텍스트 임베딩으로 사용할 필드 합성
    # title + category + place + (설명/주제)
    name = item.get("name") or ""
    cat  = item.get("category") or ""
    plc  = item.get("custody_place") or ""
    sbjt = item.get("raw", {}).get("fdSbjt") or ""
    return " ".join([name, cat, plc, sbjt]).strip()

class PoliceCache:
    def __init__(self):
        self.dim = None        # 인덱스 차원
        self.index = None      # faiss index
        self.id_map = {}       # int64 -> packed item
        self.last_refresh_ts = None
        self.embedder: Optional[KoCLIPEmbedder] = None

    def _ensure_embedder(self):
        if self.embedder is None:
            self.embedder = KoCLIPEmbedder()

    def _new_index(self, dim: int):
        self.dim = dim
        # inner-product 기반(코사인 유사도용 L2정규화 전제로)
        self.index = faiss.IndexFlatIP(dim)
        # id 매핑
        self.index = faiss.IndexIDMap2(self.index)

    def save(self):
        if self.index is None or self.dim is None:
            return
        faiss.write_index(self.index, INDEX_PATH)
        meta = {
            "dim": self.dim,
            "last_refresh_ts": self.last_refresh_ts,
            "count": len(self.id_map),
            "ids": list(map(str, self.id_map.keys())),  # 디버그용(필요없으면 제거)
            "items": self.id_map,
        }
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def load(self) -> bool:
        if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
            return False
        try:
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.dim = int(meta["dim"])
            self.last_refresh_ts = meta.get("last_refresh_ts")
            self.id_map = {int(k): v for k, v in meta.get("items", {}).items()}
            return True
        except Exception:
            return False

    async def refresh_by_queries(
        self,
        queries: List[str],
        place: Optional[str] = None,
        pages: int = 5,
        rows_per_page: int = 50,
    ) -> Dict[str, Any]:
        """
        간단 구현:
        - 질의어(예: ["노트북","지갑","휴대폰"])별로 다건 조회 -> items merge
        - place(CSTDY_PLACE)가 주어지면 필터로 사용
        - 가져온 items에 대해 KoCLIP 임베딩 계산→FAISS 구축
        """
        # 1) 수집
        all_items: Dict[str, Dict[str, Any]] = {}
        for q in queries:
            for page in range(1, pages + 1):
                got = await get_by_name_place(
                    PRDT_NM=q,
                    CSTDY_PLACE=place,
                    pageNo=page,
                    numOfRows=rows_per_page,
                )
                if not got:
                    break
                for it in got:
                    all_items[it["id"]] = it

        items = list(all_items.values())
        if not items:
            # 빈 인덱스 초기화
            self._new_index(dim=512)
            self.id_map = {}
            self.last_refresh_ts = time.time()
            self.save()
            return {"collected": 0, "indexed": 0, "dim": 512}

        # 2) 임베딩
        self._ensure_embedder()
        texts = [_pack_text_for_embed(x) for x in items]
        embs  = self.embedder.encode_texts(texts)  # (N, D) L2 normalized
        D = int(embs.shape[1])
        self._new_index(D)

        # 3) 인덱싱
        ids = np.array([_hash_to_int64(x["id"]) for x in items], dtype=np.int64)
        self.index.add_with_ids(embs, ids)

        # 4) 메타 기록
        self.id_map = {int(i): it for i, it in zip(ids.tolist(), items)}
        self.last_refresh_ts = time.time()

        # 5) 저장
        self.save()
        return {"collected": len(items), "indexed": len(items), "dim": D}

    def search_text(self, q: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.index is None or self.dim is None or self.index.ntotal == 0:
            return []
        self._ensure_embedder()
        qv = self.embedder.encode_query(q).reshape(1, -1)  # (1, D)
        sims, ids = self.index.search(qv, top_k)  # sims: (1, k), ids: (1, k)
        out: List[Dict[str, Any]] = []
        for sid, sim in zip(ids[0].tolist(), sims[0].tolist()):
            if sid == -1:
                continue
            item = self.id_map.get(int(sid))
            if not item:
                continue
            out.append({
                "score": float(sim),
                **item
            })
        return out

# 전역 싱글톤
POLICE_CACHE = PoliceCache()