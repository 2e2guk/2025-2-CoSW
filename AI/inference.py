import torch
import faiss
import numpy as np
from PIL import Image
import os
import json

# --- 1. config 및 모듈 임포트 ---
import config
from stage1_retriever.model_retriever import (
    load_retriever_model,
    get_embedding as get_stage1_embedding
)
from module2.model_precise import (
    load_reranker_model,
    get_rerank_score as get_stage2_score
)

# --- 2. 전역 변수 초기화 ---
# (이 스크립트가 서버로 실행될 때, 모델과 DB는 딱 한 번만 로드되어야 합니다)
RETRIEVER_MODEL = None
RETRIEVER_PROCESSOR = None
RERANKER_MODEL = None
RERANKER_PROCESSOR = None
VECTOR_DB = None
ID_MAPPING = None


def load_all_models_and_db():
    """
    추론에 필요한 모든 모델(1단계, 2단계)과 Vector DB를
    VRAM과 메모리에 로드합니다.
    """
    global RETRIEVER_MODEL, RETRIEVER_PROCESSOR, RERANKER_MODEL, RERANKER_PROCESSOR, VECTOR_DB, ID_MAPPING

    print("--- [Inference Server] Loading all models... ---")

    # 1. Stage 1 (koCLIP) 로드 (cuda:2에 로드됨)
    RETRIEVER_MODEL, RETRIEVER_PROCESSOR = load_retriever_model()

    # 2. Stage 2 (BLIP-2) 로드 (cuda:2에 로드됨)
    # (VRAM이 부족하다면, 1단계와 2단계 DEVICE를 config.py에서 분리해야 함)
    try:
        RERANKER_MODEL, RERANKER_PROCESSOR = load_reranker_model()
    except torch.cuda.OutOfMemoryError:
        print("\n--- FATAL ERROR: CUDA Out of Memory ---")
        print("--- 1단계와 2단계 모델을 2번 GPU에 동시에 올릴 VRAM이 부족합니다. ---")
        print("--- config.py에서 DEVICE_STAGE_1을 'cuda:0' 등으로 변경하세요. ---")
        exit(1)

    # 3. Stage 1 Vector DB (FAISS) 로드
    print(f"Loading Vector DB from {config.INDEX_FILE_PATH}")
    if not os.path.exists(config.INDEX_FILE_PATH):
        print(f"--- FATAL ERROR: Vector DB file not found. ---")
        print(f"--- 'stage1_retriever/vectorDB.py'를 먼저 실행하여 'police_db.faiss'를 생성해야 합니다. ---")
        exit(1)

    VECTOR_DB = faiss.read_index(config.INDEX_FILE_PATH)

    # 4. ID 매핑 파일 로드
    print(f"Loading ID Mapping from {config.ID_MAPPING_FILE_PATH}")
    with open(config.ID_MAPPING_FILE_PATH, 'r') as f:
        ID_MAPPING = json.load(f)  # [1001, 1002, 1003, ...] (faiss_id -> real_db_id)

    print("--- [Inference Server] All models and DB loaded. Ready. ---")


def get_real_police_item_data(db_id) -> dict:
    """
    (TODO) 이 함수는 실제 DB(PostgreSQL 등)에 연결되어야 합니다.

    ID_MAPPING에서 찾은 실제 DB ID (예: 1002)를 기반으로,
    DB에서 해당 아이템의 (text, image_path)를 가져옵니다.
    """
    # --- 가짜 DB 구현 (테스트용) ---
    if db_id == 1001:
        return {"text": "검은색 아이폰 12, 서울역에서 습득", "image_path": None}
    elif db_id == 1002:
        return {"text": "갈색 MCM 지갑, 토끼 로고 있음", "image_path": os.path.join(config.DATA_DIR, "test_image.jpg")}
    elif db_id == 1003:
        return {"text": "회색 털뭉치 키링", "image_path": None}
    else:
        return {"text": None, "image_path": os.path.join(config.DATA_DIR, "test_image.jpg")}  # 1004번 등


def search_pipeline(user_text: str = None, user_image_path: str = None):
    """
    사용자 입력을 받아 2-Step 검색 파이프라인을 실행합니다.
    """
    print("\n" + "=" * 50)
    print(f"[Request] User Text: '{user_text}'")
    print(f"[Request] User Image: '{user_image_path}'")

    # --- 1. 사용자 입력 전처리 ---
    user_image = None
    if user_image_path:
        try:
            user_image = Image.open(user_image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: User image not found at {user_image_path}")

    # --- 2. Stage 1: Retriever (Vector DB 검색) ---
    print("--- [Stage 1] Generating query vector...")
    v_query = get_stage1_embedding(
        model=RETRIEVER_MODEL,
        processor=RETRIEVER_PROCESSOR,
        text=user_text,
        image=user_image
    )

    # FAISS 검색 (Top-K)
    k = config.INFERENCE_TOP_K_RETRIEVER  # (config.py에서 50으로 설정)
    query_vector_2d = np.array([v_query]).astype(np.float32)

    print(f"--- [Stage 1] Searching Vector DB for Top-{k} candidates...")
    distances, faiss_ids = VECTOR_DB.search(query_vector_2d, k)

    candidate_ids = [ID_MAPPING[i] for i in faiss_ids[0]]  # [0] -> 실제 DB ID로 변환
    stage1_scores = distances[0]

    print(f"--- [Stage 1] Candidates found: {candidate_ids}")

    # --- 3. Stage 2: Re-ranker (정밀 점수 계산) ---
    print(f"--- [Stage 2] Re-ranking {len(candidate_ids)} candidates...")
    reranked_results = []

    for i, real_db_id in enumerate(candidate_ids):

        # 3-1. 후보의 원본 데이터 가져오기 (TODO: DB 연결 필요)
        item_data = get_real_police_item_data(real_db_id)
        police_text = item_data.get("text")
        police_image_path = item_data.get("image_path")
        police_image = None
        if police_image_path:
            try:
                police_image = Image.open(police_image_path).convert('RGB')
            except FileNotFoundError:
                pass  # 이미지는 없을 수 있음

        # 3-2. 2단계 (BLIP-2) 정밀 점수 계산
        stage2_score = get_stage2_score(
            model=RERANKER_MODEL,
            processor=RERANKER_PROCESSOR,
            user_text=user_text,
            user_image=user_image,
            police_text=police_text,
            police_image=police_image
        )

        # (임시) 1단계 점수와 2단계 점수를 결합 (가중치는 임의)
        # (우선은 2단계 점수만 사용)
        final_score = stage2_score

        reranked_results.append({
            "db_id": real_db_id,
            "stage1_score": float(stage1_scores[i]),  # (Cosine Sim)
            "stage2_score": float(final_score),
            "details": item_data
        })

        print(f"  > Re-ranked ID {real_db_id}: Score = {final_score:.4f}")

    # --- 4. 최종 정렬 ---
    reranked_results.sort(key=lambda x: x['stage2_score'], reverse=True)

    print("--- [Complete] Final Results: ---")
    for i, result in enumerate(reranked_results[:5]):  # 상위 5개만 출력
        print(
            f"  Rank {i + 1}: ID={result['db_id']}, Score={result['stage2_score']:.4f}, Text='{result['details']['text']}'")

    return reranked_results


# --- 5. 메인 실행 (테스트용) ---
if __name__ == "__main__":
    # (서버 시작 시 1회 실행)
    load_all_models_and_db()

    # (테스트 쿼리 1)
    search_pipeline(
        user_text="갈색 MCM 지갑",
        user_image_path=None  # 사용자는 텍스트로만 검색
    )

    # (테스트 쿼리 2)
    search_pipeline(
        user_text="검은색 휴대폰",
        user_image_path=None
    )

    # (테스트 쿼리 3)
    search_pipeline(
        user_text=None,  # 사용자는 이미지로만 검색
        user_image_path=os.path.join(config.DATA_DIR, "test_image.jpg")
    )
