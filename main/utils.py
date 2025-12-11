import requests
import urllib3
import json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


AI_BASE = "https://jzybnzxdkntxgmhx.tunnel.elice.io/api/v1"

AI_SYNC_URL = f"{AI_BASE}/index/refresh"


def ai_infer(description, top_k=4):
    """
    /pipeline/infer 는 즉시 결과(top 리스트)를 반환
    Django에서는 이 결과를 그대로 formatted matches 로 변환
    """
    payload = {"text": description, "top_k": top_k}

    try:
        res = requests.post(
            f"{AI_BASE}/pipeline/infer",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        print("🔥 [infer raw]", res.text)
        res.raise_for_status()
    except Exception as e:
        print("🔥 infer 호출 실패:", e)
        return []

    try:
        data = res.json()
    except:
        print("🔥 infer 응답 JSON 파싱 실패")
        return []

    return data.get("top", [])


# AI Sync 함수
def send_police_data_to_ai(police_dict):
    try:
        # AI Refresh API가 요구하는 데이터 포맷
        payload = {
            "id": police_dict.get("atcId"),
            "name": police_dict.get("fdPrdtNm"),
            "category": police_dict.get("prdtClNm"),
            "custody_place": police_dict.get("depPlace"),
            "date": police_dict.get("fdYmd"),
            "found_place": police_dict.get("fdPlace"),
            "image": police_dict.get("fdFilePathImg"),
            "raw": police_dict,
        }

        # AI 서버로 POST 전송
        response = requests.post(AI_SYNC_URL, json=payload, timeout=5)

        if response.status_code == 200:
            return True
        else:
            print(f"[AI Sync] 실패 (Code {response.status_code}): {response.text}")
            return False

    except Exception as e:
        print(f"[AI Sync] 전송 중 에러: {e}")
        return False
