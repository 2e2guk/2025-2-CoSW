# vectorDB 에 경찰이 올리는 데이터를 저장하는 코드
import faiss
import numpy as np
from PIL import Image
import os

# 1단계(Retriever) 모듈 임포트
from model_retriever import load_retriever_model, get_embedding

# --- 0. 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")  # 이미지 파일이 저장된 폴더 (예시)
INDEX_FILE_PATH = os.path.join(BASE_DIR, "police_db.faiss")

# (중요) ID 매핑 파일.
# FAISS의 0, 1, 2... 인덱스가 실제 DB의 어떤 ID인지를 저장합니다.
ID_MAPPING_FILE_PATH = os.path.join(BASE_DIR, "police_db_id_mapping.json")


def load_real_police_data():
    """
    (TODO) 이 함수를 실제 데이터베이스(PostgreSQL, MongoDB 등)에 연결해야 합니다.

    데이터베이스에서 모든 유실물 정보를 다음 형식의 리스트로 반환합니다:
    [
        {"db_id": "item_uuid_1", "text": "검은색 아이폰", "image_path": "/path/to/img1.jpg"},
        {"db_id": "item_uuid_2", "text": "갈색 지갑", "image_path": None},
        {"db_id": "item_uuid_3", "text": None, "image_path": "/path/to/img3.jpg"},
        ...
    ]
    """
    print("--- (DUMMY) Loading data from database... ---")

    # 지금은 테스트용 'test_image.jpg'를 사용하는 더미 데이터를 반환합니다.
    # 나중에 이 부분을 실제 DB 로직으로 교체해야 합니다.
    dummy_data = [
        {
            "db_id": 1001,
            "text": "검은색 아이폰 12, 서울역에서 습득",
            "image_path": None
        },
        {
            "db_id": 1002,
            "text": "갈색 MCM 지갑, 토끼 로고 있음",
            "image_path": os.path.join(DATA_DIR, "test_image.jpg")
        },
        {
            "db_id": 1003,
            "text": "회색 털뭉치 키링",
            "image_path": None
        }
    ]
    print(f"--- (DUMMY) Loaded {len(dummy_data)} items. ---")
    return dummy_data


def build_and_save_index():
    print("--- [Job Start] Building Vector DB Index ---")

    # 1. 1단계 koCLIP 모델 로드 (VRAM에 올리기)
    model, processor = load_retriever_model()

    # 2. 실제 경찰 DB 데이터 로드 (TODO: 이 함수를 수정해야 함)
    police_items = load_real_police_data()

    if not police_items:
        print("Error: No items loaded from database. Aborting index build.")
        return

    # 3. FAISS 인덱스 초기화
    embedding_dim = model.config.projection_dim
    index = faiss.IndexFlatIP(embedding_dim)

    faiss_id_to_real_db_id = []
    all_embeddings_list = []

    print(f"Processing and embedding {len(police_items)} items...")

    # 4. DB를 순회하며 모든 아이템을 벡터로 변환
    for item in police_items:
        item_text = item.get("text")
        item_image_path = item.get("image_path")
        item_image = None

        # 이미지가 있다면 로드
        if item_image_path:
            try:
                item_image = Image.open(item_image_path).convert('RGB')
            except FileNotFoundError:
                print(f"Warning: Image {item_image_path} not found. Skipping image for item {item.get('db_id')}.")
                item_image = None

        # 'model.py'의 함수를 사용해 임베딩 생성
        embedding = get_embedding(
            model=model,
            processor=processor,
            text=item_text,
            image=item_image
        )

        all_embeddings_list.append(embedding)
        faiss_id_to_real_db_id.append(item.get('db_id'))

    # 5. 모든 벡터를 numpy 배열로 변환
    all_embeddings_np = np.array(all_embeddings_list).astype(np.float32)

    # 6. FAISS 인덱스에 모든 벡터를 '추가'
    index.add(all_embeddings_np)

    print(f"Successfully added {index.ntotal} vectors to the index.")

    # 7. 완성된 인덱스를 파일로 '저장'
    faiss.write_index(index, INDEX_FILE_PATH)
    print(f"--- Vector DB Index saved to: {INDEX_FILE_PATH} ---")

    # 8. (중요) ID 매핑 파일 저장
    # (실제 서비스에서는 이 파일을 읽어서 FAISS ID를 DB ID로 변환해야 함)
    import json
    with open(ID_MAPPING_FILE_PATH, 'w') as f:
        json.dump(faiss_id_to_real_db_id, f)
    print(f"--- ID Mapping file saved to: {ID_MAPPING_FILE_PATH} ---")
    print("--- [Job End] Index build complete. ---")


if __name__ == "__main__":
    build_and_save_index()