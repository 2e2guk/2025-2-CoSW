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
            self.stdout.write("   -> 활성화된 요청이 없습니다.")
            return

        sent_count = 0
        for lost_req in active_requests:
            try:
                ai_results = ai_infer(lost_req.description, top_k=3)
                if not ai_results:
                    continue

                new_matches = []
                for idx, match in enumerate(ai_results):
                    match_id = str(match.get("atcId"))
                    if not NotificationLog.objects.filter(
                        request=lost_req, atc_id=match_id, is_sent=True
                    ).exists():
                        new_matches.append(
                            {
                                "id": match_id,
                                "name": match.get("fdPrdtNm"),
                                "score": match.get("score"),
                                "rank": idx + 1,
                            }
                        )

                if new_matches:
                    email_body = (
                        f"요청하신 '{lost_req.description}' 관련 유사 습득물:\n\n"
                    )
                    for item in new_matches:
                        email_body += f"[{item['rank']}위] {item['name']} (유사도: {item['score']:.2f})\n"
                        email_body += f"관리번호: {item['id']}\n------------------\n"
                    email_body += "\nLost112에서 관리번호로 조회하세요."

                    send_mail(
                        f"[Lost112] '{lost_req.description}' 유사 물품 알림",
                        email_body,
                        None,
                        [lost_req.user.email],
                        fail_silently=False,
                    )

                    for item in new_matches:
                        NotificationLog.objects.create(
                            user=lost_req.user,
                            request=lost_req,
                            atc_id=item["id"],
                            rank=item["rank"],
                            similarity_score=item["score"],
                            is_sent=True,
                        )
                    self.stdout.write(f"   -> 📧 Sent to {lost_req.user.email}")
                    sent_count += 1
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error: {e}"))
                continue

        self.stdout.write(
            self.style.SUCCESS(f"Matching 완료! (총 {sent_count}명에게 발송됨)")
        )
