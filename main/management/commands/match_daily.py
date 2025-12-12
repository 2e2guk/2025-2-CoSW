from django.core.management.base import BaseCommand
from django.core.mail import send_mail
from main.models import UserLostItem, NotificationLog
from main.utils import ai_infer


class Command(BaseCommand):
    help = "활성 분실물 요청 매칭 및 이메일 발송"

    def handle(self, *args, **options):
        self.stdout.write("[Cron] Daily Matching Started...")
        active_requests = UserLostItem.objects.filter(is_active=True)

        if not active_requests.exists():
            self.stdout.write("   -> 활성화된 요청이 없습니다.")
            return

        sent_count = 0

        for lost_req in active_requests:
            try:
                # AI 상위 3개 결과
                ai_results = ai_infer(lost_req.description, top_k=3)
                if not ai_results:
                    continue

                new_matches = []

                for idx, match in enumerate(ai_results):
                    # preview 안에서 상세 정보 꺼내기
                    preview = match.get("preview") or {}
                    match_id = str(match.get("atcId"))

                    # 이미 보낸 적 있는지 체크
                    if NotificationLog.objects.filter(
                        request=lost_req, atc_id=match_id, is_sent=True
                    ).exists():
                        continue

                    # 신규 매칭 리스트에 추가 (상세 정보 포함)
                    new_matches.append(
                        {
                            "id": match_id,
                            "name": preview.get("fdPrdtNm"),
                            "category": preview.get("prdtClNm"),
                            "place": preview.get("depPlace"),
                            "date": preview.get("fdYmd"),
                            "score": match.get("score"),
                            "rank": idx + 1,
                        }
                    )

                if new_matches:
                    # 이메일 본문 구성 (이름/카테고리/보관장소/날짜까지)
                    email_body = (
                        f"요청하신 '{lost_req.description}' 관련 유사 습득물:\n\n"
                    )
                    for item in new_matches:
                        email_body += (
                            f"[{item['rank']}위] {item['name']} "
                            f"(유사도: {item['score']:.2f})\n"
                        )
                        if item["category"]:
                            email_body += f"분류: {item['category']}\n"
                        if item["place"]:
                            email_body += f"보관장소: {item['place']}\n"
                        if item["date"]:
                            email_body += f"습득일자: {item['date']}\n"
                        email_body += f"관리번호: {item['id']}\n------------------\n"

                    email_body += "\nLost112에서 관리번호로 조회하세요."

                    # 실제 메일 발송
                    send_mail(
                        f"[Lost112] '{lost_req.description}' 유사 물품 알림",
                        email_body,
                        None,
                        [lost_req.user.email],
                        fail_silently=False,
                    )

                    # NotificationLog 기록
                    for item in new_matches:
                        NotificationLog.objects.create(
                            user=lost_req.user,
                            request=lost_req,
                            atc_id=item["id"],
                            rank=item["rank"],
                            similarity_score=item["score"],
                            is_sent=True,
                        )

                    self.stdout.write(f"   -> 📧 Sent to {lost_req.user.email}")
                    sent_count += 1

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error: {e}"))
                continue

        self.stdout.write(
            self.style.SUCCESS(f"Matching 완료! (총 {sent_count}명에게 발송됨)")
        )