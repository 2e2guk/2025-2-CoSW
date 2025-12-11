from rest_framework import serializers
from .models import *


class UserEmailSerializer(serializers.Serializer):
    email = serializers.EmailField()


# 분실물 검색
class LostItemCreateSerializer(serializers.Serializer):
    image = serializers.ImageField(required=False)
    description = serializers.CharField(required=False)

    def validate(self, data):
        if not data.get("image") and not data.get("description"):
            raise serializers.ValidationError("이미지 또는 설명 중 하나는 필요합니다.")
        return data
    

class UserLostItemListSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserLostItem
        fields = [
            "request_id",
            "description",
            "original_image_url",
            "is_active",
            "created_at",
        ]

    def get_original_image_url(self, obj):
        url = obj.original_image_url
        if not url:
            return None

        # 이미 http URL이면 그대로 반환
        if url.startswith("http://") or url.startswith("https://"):
            return url

        # request 객체 가져오기
        request = self.context.get("request")
        if not request:
            return url  # fallback (거의 발생 X)

        # 상대경로 → 절대경로 변환
        return request.build_absolute_uri(url)