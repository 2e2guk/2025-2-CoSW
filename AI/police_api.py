import os
import time
import logging
from typing import Dict, List, Any, Optional
import requests
from urllib.parse import urlencode

log = logging.getLogger("police_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

BASE = os.environ.get("POLICE_API_BASE", "https://apis.data.go.kr/1320000/LosfundInfoInqireService").rstrip("/")
KEY  = os.environ.get("POLICE_API_KEY", "").strip()
assert KEY, "env POLICE_API_KEY must be set"

# 세션 + 재시도
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = requests.Session()
_retry = Retry(
    total=5,
    connect=3,
    read=3,
    backoff_factor=0.8,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "HEAD", "OPTIONS"]
)
_session.mount("https://", HTTPAdapter(max_retries=_retry))
_session.mount("http://",  HTTPAdapter(max_retries=_retry))

def _ymd(days_ago: int) -> str:
    return time.strftime("%Y%m%d", time.localtime(time.time() - 86400 * days_ago))

def _get_json(path: str, params: Dict[str, Any], timeout: float = 20.0) -> Dict[str, Any]:
    q = dict(params)
    q.setdefault("type", "json")
    q.setdefault("serviceKey", KEY)

    url = f"{BASE}/{path}"
    r = _session.get(url, params=q, timeout=timeout, headers={"Accept": "application/json"})
    r.raise_for_status()
    # 일부 케이스에선 빈 응답/HTML이 오기도 하므로 방어
    try:
        return r.json()
    except Exception:
        txt = (r.text or "").strip()
        if not txt:
            raise RuntimeError("empty response from police API")
        if txt.startswith("<"):
            raise RuntimeError("non-JSON (HTML/XML) response from police API")
        raise

def _extract_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    # 다양한 형태(body.items가 dict/None/[] 등) 방어
    body = (payload.get("response") or {}).get("body") or {}
    items = body.get("items")
    if not items:
        return []
    if isinstance(items, dict):
        arr = items.get("item")
        if not arr:
            return []
        if isinstance(arr, list):
            items = arr
        else:
            items = [arr]
    elif isinstance(items, list):
        pass
    else:
        return []

    out: List[Dict[str, Any]] = []
    for it in items:
        # 키 이름이 대/소문자 섞여오는 경우가 있어 중복 대응
        def g(*keys):
            for k in keys:
                if k in it:
                    return it[k]
            return None
        out.append({
            "atcId": g("atcId", "ATC_ID"),
            "fdPrdtNm": g("fdPrdtNm", "PRDT_NM"),
            "fdSbjt": g("fdSbjt", "FD_SBJT"),    # 제목
            "depPlace": g("depPlace", "CSTDY_PLACE"),
            "fdYmd": g("fdYmd", "FD_YMD"),
            "prdtClNm": g("prdtClNm", "PRDT_CL_NM"),
            "clrNm": g("clrNm", "CLR_NM"),
            "fdFilePathImg": g("fdFilePathImg", "FD_FILE_PATH_IMG"),
        })
    return out

def _fetch_pages_acc_to_period(start_ymd: str, end_ymd: str, target_n: int, rows: int) -> List[Dict]:
    items: List[Dict] = []
    page = 1
    while len(items) < target_n and page <= 5000:
        try:
            payload = _get_json(
                "getLosfundInfoAccToClAreaPd",
                {"numOfRows": rows, "pageNo": page, "START_YMD": start_ymd, "END_YMD": end_ymd},
                timeout=25.0
            )
            chunk = _extract_items(payload)
        except Exception as e:
            log.warning("[police_api] fetch getLosfundInfoAccToClAreaPd failed (attempt page %d): %s", page, e)
            # 다음 페이지 시도
            chunk = []

        if not chunk:
            # 아이템 없으면 중단
            break

        items.extend(chunk)
        page += 1
        # 과도한 속도 방지
        if page % 5 == 0:
            time.sleep(0.3)
    return items[:target_n]

def _fallback_by_name(start_ymd: str, end_ymd: str, target_n: int, rows: int) -> List[Dict]:
    # 대체 엔드포인트: 명칭/보관장소 기반. 명칭은 넓게.
    names = ["노트북", "휴대폰", "지갑", "키", "카드", "가방", "버즈", "에어팟"]
    items: List[Dict] = []
    for nm in names:
        page = 1
        while len(items) < target_n and page <= 2000:
            try:
                payload = _get_json(
                    "getLosfundInfoAccTpNmCstdyPlace",
                    {"numOfRows": rows, "pageNo": page, "PRDT_NM": nm, "START_YMD": start_ymd, "END_YMD": end_ymd},
                    timeout=25.0
                )
                chunk = _extract_items(payload)
            except Exception as e:
                log.warning("[police_api] fallback by name failed nm=%s page=%d: %s", nm, page, e)
                chunk = []
            if not chunk:
                break
            items.extend(chunk)
            page += 1
            if page % 5 == 0:
                time.sleep(0.3)
        if len(items) >= target_n:
            break
    return items[:target_n]

def fetch_items(mode: str = "limit", target_n: int = 1500, rows: int = 100, op: str = "op2") -> List[Dict]:
    """
    mode: 현재는 target_n 기준 페이징 수집
    op: 기간 프리셋. op1=최근 30일, op2=최근 60일, op3=최근 90일
    """
    if op == "op1":
        start_ymd = _ymd(30)
    elif op == "op3":
        start_ymd = _ymd(90)
    else:
        start_ymd = _ymd(60)
    end_ymd = _ymd(0)

    log.info("[police_api] fetch items %s~%s target_n=%d rows=%d", start_ymd, end_ymd, target_n, rows)

    items = _fetch_pages_acc_to_period(start_ymd, end_ymd, target_n, rows)

    if not items:
        # 1차가 비었으면 기간 확장 후 재시도(최대 120일)
        start_ymd2 = _ymd(120)
        log.warning("[police_api] primary fetch empty. retry with longer period %s~%s", start_ymd2, end_ymd)
        items = _fetch_pages_acc_to_period(start_ymd2, end_ymd, target_n, rows)

    if not items:
        # 그래도 비면 이름기반 대체 엔드포인트
        log.warning("[police_api] still empty. try fallback-by-name")
        items = _fallback_by_name(start_ymd, end_ymd, target_n, rows)

    log.info("[police_api] total fetched: %d", len(items))
    return items[:target_n]