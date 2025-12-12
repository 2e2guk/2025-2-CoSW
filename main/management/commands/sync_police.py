import os
import time
import requests
import xmltodict
import urllib3
from datetime import datetime, timedelta

from django.core.management.base import BaseCommand

from main.models import PoliceFoundItem
from main.utils import send_police_data_to_ai

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class Command(BaseCommand):
    help = "경찰청 데이터를 동기화하고 신규 데이터를 AI 서버로 전송합니다."

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=1,
            help="조회할 과거 기간(일 단위, 기본 1일)",
        )

    def handle(self, *args, **options):
        days_ago = options["days"]
        self.stdout.write(
            self.style.NOTICE(f"[Sync] 최근 {days_ago}일 치 데이터 동기화 시작...")
        )

        api_key = os.getenv("POLICE_API_KEY")
        if not api_key:
            self.stdout.write(self.style.ERROR("POLICE_API_KEY is missing"))
            return

        today = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y%m%d")
        base_url = "http://apis.data.go.kr/1320000/LosfundInfoInqireService/getLosfundInfoAccToClAreaPd"

        page_no = 1
        num_of_rows = 1000

        total_new_saved = 0
        total_updated = 0

        # AI 로 보낼 신규 raw item 들을 여기 모아두었다가, DB 저장 다 끝난 뒤 한 번에 전송
        ai_new_items = []

        while True:
            self.stdout.write(
                self.style.NOTICE(
                    f"  >> Page {page_no} 요청 중... ({start_date} ~ {today})"
                )
            )

            params = {
                "serviceKey": api_key,
                "START_YMD": start_date,
                "END_YMD": today,
                "numOfRows": str(num_of_rows),
                "pageNo": str(page_no),
            }

            try:
                response = requests.get(
                    base_url,
                    params=params,
                    verify=False,
                    timeout=100,
                )

                # -----------------------
                # 1) XML 파싱
                # -----------------------
                items_list = []
                try:
                    data_dict = xmltodict.parse(response.content)
                    body = data_dict.get("response", {}).get("body", {})
                    if not body:
                        # 더 이상 데이터 없음
                        break

                    items_wrapper = body.get("items")
                    if not items_wrapper:
                        self.stdout.write("      -> 해당 페이지에 데이터가 없습니다.")
                        break

                    items_list = items_wrapper.get("item")
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"      -> XML 파싱 실패: {e}"))
                    break

                if isinstance(items_list, dict):
                    items_list = [items_list]
                elif not items_list:
                    break

                # -----------------------
                # 2) 기존 데이터 조회
                #    (같은 atc_id + fd_sn 의 기존 레코드를 미리 가져와서 map 구성)
                # -----------------------
                keys = []
                for item in items_list:
                    atc_id = item.get("atcId")
                    fd_sn = item.get("fdSn")
                    if atc_id and fd_sn:
                        keys.append((atc_id, fd_sn))

                # atc_id 만으로 먼저 필터링한 후, (atc_id, fd_sn) 쌍으로 매핑
                atc_ids = list({k[0] for k in keys})
                existing_map = {}
                if atc_ids:
                    qs = PoliceFoundItem.objects.filter(atc_id__in=atc_ids)
                    for obj in qs:
                        existing_map[(obj.atc_id, obj.fd_sn)] = obj

                to_create = []
                to_update = []

                # -----------------------
                # 3) 각 아이템에 대해 new / update 분리
                # -----------------------
                for item in items_list:
                    atc_id = item.get("atcId")
                    fd_sn = item.get("fdSn")
                    if not atc_id or not fd_sn:
                        continue

                    # 날짜 파싱
                    fd_ymd = item.get("fdYmd")
                    found_date = None
                    if fd_ymd:
                        try:
                            fmt = "%Y-%m-%d" if "-" in fd_ymd else "%Y%m%d"
                            dt_obj = datetime.strptime(fd_ymd, fmt).date()
                            found_date = dt_obj
                            # AI 서버용으로는 YYYY-MM-DD 문자열로 맞춰줌
                            item["fdYmd"] = dt_obj.strftime("%Y-%m-%d")
                        except Exception:
                            pass

                    raw_img = item.get("fdFilePathImg", "")
                    final_img = raw_img[:490] if raw_img else None

                    defaults = {
                        "item_name": item.get("fdPrdtNm", "")[:190],
                        "item_desc": item.get("fdSbjt", ""),
                        "found_location": item.get("depPlace", "")[:190],
                        "found_date": found_date,
                        "police_image_url": final_img,
                        "raw_data": item,
                    }

                    key = (atc_id, fd_sn)

                    if key in existing_map:
                        # ✅ 기존 레코드: pk 있는 인스턴스를 가져와서 필드만 수정
                        obj = existing_map[key]
                        for field, value in defaults.items():
                            setattr(obj, field, value)
                        to_update.append(obj)
                    else:
                        # ✅ 신규 레코드: 아직 pk 없음, bulk_create 대상
                        obj = PoliceFoundItem(
                            atc_id=atc_id,
                            fd_sn=fd_sn,
                            **defaults,
                        )
                        to_create.append(obj)
                        # AI 서버에 보낼 raw dict도 같이 저장해둔다
                        ai_new_items.append(item)

                # -----------------------
                # 4) bulk_create / bulk_update 실행
                # -----------------------
                created_count = 0
                updated_count = 0

                if to_create:
                    PoliceFoundItem.objects.bulk_create(to_create)
                    created_count = len(to_create)

                if to_update:
                    # to_update 는 전부 existing_map 에서 온 인스턴스라 pk 존재
                    PoliceFoundItem.objects.bulk_update(
                        to_update,
                        [
                            "item_name",
                            "item_desc",
                            "found_location",
                            "found_date",
                            "police_image_url",
                            "raw_data",
                        ],
                    )
                    updated_count = len(to_update)

                total_new_saved += created_count
                total_updated += updated_count

                self.stdout.write(
                    f"      -> ➕ Bulk Insert {created_count}건, 🔁 Bulk Update {updated_count}건"
                )

                # 이 페이지의 아이템 수가 num_of_rows 보다 적으면 마지막 페이지
                if len(items_list) < num_of_rows:
                    break

                page_no += 1
                time.sleep(0.5)

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Error occurred in Page {page_no}: {e}")
                )
                break

        # -----------------------
        # 5) AI 서버로 신규 데이터 전송
        # -----------------------
        self.stdout.write(
            self.style.NOTICE(
                f"[Sync] DB 저장 완료. AI 서버 전송 시작 ({len(ai_new_items)}건)..."
            )
        )

        sent_ai = 0
        for item in ai_new_items:
            if send_police_data_to_ai(item):
                sent_ai += 1

        # -----------------------
        # 6) 최종 로그
        # -----------------------
        self.stdout.write(
            self.style.SUCCESS(
                f"Sync 완료! (신규 DB저장: {total_new_saved}, "
                f"DB 업데이트: {total_updated}, "
                f"AI 전송 성공: {sent_ai})"
            )
        )