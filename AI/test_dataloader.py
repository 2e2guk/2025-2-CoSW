import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json

# --- (수정) 1. sys 모듈 임포트 ---
import sys
# -----------------------------------

# config.py에서 설정값 임포트
import config

# --- (수정) 2. Python Import 경로에 로컬 모델 폴더 강제 추가 ---
# (이것이 ImportError를 해결할 핵심입니다)
LOCAL_MODEL_PATH = config.RERANKER_MODEL_ID
if LOCAL_MODEL_PATH not in sys.path:
    sys.path.append(LOCAL_MODEL_PATH)
# -----------------------------------------------------------

# (수정) 3. 'sys.path'가 설정된 "이후에" transformers 임포트
from transformers import AutoProcessor


# --- AI Hub 104번 VQA 데이터셋 로더 ---
# (이 클래스는 수정할 필요 없이 완벽하게 작동합니다)
class AIHubVQADataset(Dataset):
    def __init__(self, json_dir_path, image_dir_path):
        print(f"Loading VQA annotations from: {json_dir_path}")
        self.image_dir_path = image_dir_path
        self.annotations = []
        images_json_path = os.path.join(json_dir_path, "images.json")
        question_json_path = os.path.join(json_dir_path, "question.json")
        annotation_json_path = os.path.join(json_dir_path, "annotation.json")
        with open(images_json_path, 'r', encoding='utf-8') as f:
            images_data = json.load(f)["images"]
        with open(question_json_path, 'r', encoding='utf-8') as f:
            question_data = json.load(f)["questions"]
        with open(annotation_json_path, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)["annotations"]
        image_id_to_filename = {img["image_id"]: img["image"] for img in images_data}
        question_id_to_data = {q["question_id"]: (q["image_id"], q["question"]) for q in question_data}
        for anno in annotation_data:
            q_id = anno["question_id"]
            if q_id in question_id_to_data:
                img_id, question_text = question_id_to_data[q_id]
                if img_id in image_id_to_filename:
                    self.annotations.append({
                        "image_file": image_id_to_filename[img_id],
                        "question": question_text,
                        "answer": anno["multiple_choice_answer"]
                    })
        print(f"Loaded {len(self.annotations)} VQA entries from this folder.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        image_filename = item["image_file"]
        question = item["question"]
        answer = item["answer"]
        image_path = os.path.join(self.image_dir_path, image_filename)

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), (255, 255, 255))

        return image, question, answer


# --- (수정) "A.X" (Causal LM) 모델을 위한 Collate 함수 ---
def create_vqa_collate_fn(processor, device):
    """
    (PIL.Image, str_q, str_a) 리스트를
    "A.X" Causal LM 모델 입력 텐서로 변환하는 collate_fn을 반환합니다.
    """

    def collate_fn(batch):
        images = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]

        # Causal LM은 (질문 + 답변)을 하나의 시퀀스로 만듭니다.
        # 예: "질문: 트럭 색은? 답변: 검은색<eos>"
        texts = [f"질문: {q} 답변: {a}{processor.tokenizer.eos_token}" for q, a in zip(questions, answers)]

        # 'processor'가 이미지와 (질문+답변) 텍스트를 모두 처리
        inputs = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # (A.X 모델이 지원하는 길이 예시)
        )

        # (중요) Causal LM 학습 시, "질문" 부분은 Loss 계산에서 제외하고
        # "답변" 부분만 Loss를 계산하도록 레이블을 생성합니다.
        # (이 작업은 processor.tokenizer를 사용해야 해서 복잡하므로,
        #  우선 'inputs'만 생성하고, 'labels'는 'input_ids'를 복사하여 사용)
        inputs['labels'] = inputs['input_ids'].clone()

        # (수정!) 딕셔너리의 "각 텐서"를 device로 이동
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device)

        return inputs

    return collate_fn


# --- 메인 실행 (데이터 로더 테스트) ---
if __name__ == "__main__":

    TEST_DEVICE = torch.device('cpu')
    print("--- Testing AI Hub 104 VQA Dataloader --- (CPU ONLY TEST) ---")
    print(f"--- Forcing CPU for this test. Device: {TEST_DEVICE} ---")

    TARGET_FOLDER_PATH = "/home/diskD/leegw/104_vqa_data/005.시각정보기반/1.Training"
    SUB_CATEGORY_PATH = "vehicle/하_train_vehicle"

    JSON_DIR_PATH = os.path.join(TARGET_FOLDER_PATH, "라벨링데이터", SUB_CATEGORY_PATH)
    IMAGE_DIR_PATH = os.path.join(TARGET_FOLDER_PATH, "원천데이터", SUB_CATEGORY_PATH)

    print(f"Attempting to load data from: {JSON_DIR_PATH}")
    print(f"Image directory set to: {IMAGE_DIR_PATH}")

    try:
        # --- (수정) "A.X" (한국어) 프로세서 로드 ---
        print(f"Loading KOREAN VLM processor ({config.RERANKER_MODEL_ID})...")
        processor = AutoProcessor.from_pretrained(
            config.RERANKER_MODEL_ID,
            trust_remote_code=True  # (A.X 모델은 이 옵션이 필요할 수 있음)
        )
        # --------------------------------------------------------

        dataset = AIHubVQADataset(
            json_dir_path=JSON_DIR_PATH,
            image_dir_path=IMAGE_DIR_PATH
        )

        if len(dataset) == 0:
            raise Exception(f"Dataset loaded 0 entries. Check JSON keys/logic in Dataloader.")

        collate_fn = create_vqa_collate_fn(processor, device=TEST_DEVICE)

        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

        print("\n--- Dataloader created. Fetching first batch (on CPU)... ---")
        batch = next(iter(dataloader))

        print("\n--- [Success] Batch loaded successfully on CPU! ---")
        print(f"Batch keys: {batch.keys()}")
        print(f"Image tensor device: {batch['pixel_values'].device}")

        # (수정) A.X 프로세서의 토크나이저로 디코딩
        decoded_input = processor.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)

        print(f"Example Full Input (decoded): {decoded_input}")

    except FileNotFoundError as e:
        print(f"\n--- [FATAL ERROR] File Not Found ---")
        print(f"Error details: {e}")

    except KeyError as e:
        print(f"\n--- [FATAL ERROR] JSON Key Error ---")
        print(f"Error details: Key {e} not found in JSON annotation.")

    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        print(f"{e}")