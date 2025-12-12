# main/utils.py

import requests
import urllib3
import json
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

AI_BASE = "https://jzybnzxdkntxgmhx.tunnel.elice.io/api/v1"

# 🔹 infer는 그대로
AI_INFER_URL = f"{AI_BASE}/pipeline/infer"

# 🔹 업서트 엔드포인트 (새로 고정)
AI_UPSERT_URL = f"{AI_BASE}/index/upsert"

# (선택) 업서트용 키 있으면 헤더에 추가
BACKEND_UPSERT_KEY = os.getenv("BACKEND_UPSERT_KEY", "")


def ai_infer(description, top_k=4):
    """
    /pipeline/infer 는 즉시 결과(top 리스트)를 반환
    Django에서는 이 결과를 그대로 formatted matches 로 변환
    """
    payload = {"text": description, "top_k": top_k}

    try:
        res = requests.post(
            AI_INFER_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        print("🔥 [infer raw]", res.text)
        res.raise_for_status()
    except Exception as e:
        print("🔥 infer 호출 실패:", e)
        return []

    try:
        data = res.json()
    except Exception:
        print("🔥 infer 응답 JSON 파싱 실패")
        return []

    return data.get("top", [])


# 🔹 Django 쪽 dict → AI 서버가 요구하는 "경찰청 필드명" 스키마로 변환
def _to_ai_item(police_dict: dict) -> dict:
    """
    police_dict: xmltodict로 받은 원본 item 이거나,
                 우리가 가공한 dict (id/name/... 가 섞여 있을 수도 있음)

    AI 서버 스펙:
    {
      "atcId": "...",
      "fdPrdtNm": "...",
      "fdSbjt": "...",
      "depPlace": "...",
      "fdYmd": "YYYYMMDD",
      "prdtClNm": "...",
      "clrNm": "...",
      "fdFilePathImg": "...",
      "fdPlace": "..."
    }
    """
    # 날짜 처리: date / fdYmd 둘 다 지원, 2025-12-11 → 20251211 로 정규화
    date = police_dict.get("fdYmd") or police_dict.get("date")
    if date and "-" in date:
        date = date.replace("-", "")

    return {
        # id / atcId 둘 다 지원 (향후 정규화 스키마도 쓸 수 있게)
        "atcId": police_dict.get("atcId") or police_dict.get("id"),
        "fdPrdtNm": police_dict.get("fdPrdtNm") or police_dict.get("name") or "",
        "fdSbjt": police_dict.get("fdSbjt") or police_dict.get("subject") or "",
        "depPlace": police_dict.get("depPlace")
        or police_dict.get("custody_place")
        or "",
        "fdYmd": date or "",
        "prdtClNm": police_dict.get("prdtClNm") or police_dict.get("category") or "",
        "clrNm": police_dict.get("clrNm") or police_dict.get("color") or "",
        "fdFilePathImg": police_dict.get("fdFilePathImg")
        or police_dict.get("image")
        or "",
        "fdPlace": police_dict.get("fdPlace") or police_dict.get("found_place") or "",
    }


def send_police_data_to_ai(police_dict: dict) -> bool:
    """
    경찰청 1개 row(or 그에 대응하는 dict)를
    AI 서버 /index/upsert 스펙에 맞게 변환해서 전송.
    - items: [ ... ] 배열 안에 1개만 넣어서 보냄
    - clear: 항상 False (전체 재구축이 필요할 때만 True로 사용)
    """
    try:
        item = _to_ai_item(police_dict)

        payload = {
            "items": [item],
            "clear": False,
        }

        headers = {"Content-Type": "application/json"}
        if BACKEND_UPSERT_KEY:
            headers["X-Backend-Key"] = BACKEND_UPSERT_KEY

        # 디버그용 로그 (원하면 주석 처리 가능)
        print("📡 [AI UPSERT 요청]", json.dumps(payload, ensure_ascii=False)[:500])

        res = requests.post(
            AI_UPSERT_URL,
            json=payload,
            headers=headers,
            timeout=120,
        )

        print("📨 [AI UPSERT 응답]", res.status_code, res.text[:300])

        if res.status_code == 200:
            return True
        else:
            # FastAPI 422 등 실패 사유 로그
            try:
                print("❗ [AI UPSERT 실패 detail]", res.json())
            except Exception:
                pass
            return False

    except Exception as e:
        print(f"[AI Sync] 전송 중 에러: {e}")
        return False
