import os
import requests
import xmltodict
import time
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from main.models import PoliceFoundItem
from main.utils import send_police_data_to_ai
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class Command(BaseCommand):
    help = "경찰청 데이터를 동기화하고 신규 데이터를 AI 서버로 전송합니다."

    def add_arguments(self, parser):
        parser.add_argument(
            "--days", type=int, default=1, help="조회할 과거 기간(일 단위)"
        )

    def handle(self, *args, **options):
        days_ago = options["days"]
        self.stdout.write(f"[Sync] 최근 {days_ago}일 치 데이터 동기화 시작...")

        api_key = os.getenv("POLICE_API_KEY")
        if not api_key:
            self.stdout.write(self.style.ERROR("POLICE_API_KEY is missing"))
            return

        today = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y%m%d")
        base_url = "http://apis.data.go.kr/1320000/LosfundInfoInqireService/getLosfundInfoAccToClAreaPd"

        page_no = 1
        num_of_rows = 1000
        total_saved = 0
        total_sent_ai = 0

        while True:
            self.stdout.write(
                f"   >> Page {page_no} 요청 중... ({start_date} ~ {today})"
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
                    base_url, params=params, verify=False, timeout=30
                )
                items_list = []
                try:
                    data_dict = xmltodict.parse(response.content)
                    body = data_dict.get("response", {}).get("body", {})
                    if not body:
                        break
                    items_wrapper = body.get("items")
                    if not items_wrapper:
                        self.stdout.write("      -> 해당 페이지에 데이터가 없습니다.")
                        break
                    items_list = items_wrapper.get("item")
                except:
                    pass

                if isinstance(items_list, dict):
                    items_list = [items_list]
                elif not items_list:
                    break

                current_batch_new = 0
                for item in items_list:
                    atc_id = item.get("atcId")
                    if not atc_id:
                        continue

                    fd_ymd = item.get("fdYmd")
                    found_date = None
                    if fd_ymd:
                        try:
                            fmt = "%Y-%m-%d" if "-" in fd_ymd else "%Y%m%d"
                            dt_obj = datetime.strptime(fd_ymd, fmt).date()
                            found_date = dt_obj
                            item["fdYmd"] = dt_obj.strftime("%Y-%m-%d")
                        except:
                            pass

                    raw_img = item.get("fdFilePathImg", "")
                    final_img = raw_img[:490] if raw_img else None

                    obj, created = PoliceFoundItem.objects.update_or_create(
                        atc_id=atc_id,
                        fd_sn=item.get("fdSn"),
                        defaults={
                            "item_name": item.get("fdPrdtNm", "")[:190],
                            "item_desc": item.get("fdSbjt", ""),
                            "found_location": item.get("depPlace", "")[:190],
                            "found_date": found_date,
                            "police_image_url": final_img,
                            "raw_data": item,
                        },
                    )

                    if created:
                        current_batch_new += 1
                        total_saved += 1
                        if send_police_data_to_ai(item):
                            total_sent_ai += 1

                self.stdout.write(
                    f"      -> 처리 완료. (신규 저장: {current_batch_new}건)"
                )
                if len(items_list) < num_of_rows:
                    break
                page_no += 1
                time.sleep(0.5)

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error: {e}"))
                break

        self.stdout.write(
            self.style.SUCCESS(
                f"Sync 완료! (신규 DB저장: {total_saved}, AI전송: {total_sent_ai})"
            )
        )
