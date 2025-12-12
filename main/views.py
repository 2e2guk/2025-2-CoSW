from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.parsers import MultiPartParser, FormParser
from .utils import *
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os

from .models import *
from .serializers import *
from .utils import *
from django.db import connection


class AuthEmailView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = UserEmailSerializer(data=request.data)

        if serializer.is_valid():
            email = serializer.validated_data["email"]

            # 1) 해당 이메일의 유저가 있는지 확인
            user, created = Users.objects.get_or_create(email=email)

            # 2) JWT 발급
            refresh = RefreshToken.for_user(user)

            return Response(
                {
                    "user_id": user.user_id,
                    "email": user.email,
                    "created": created,  # True면 이번 요청에서 새 회원가입됨
                    "access_token": str(refresh.access_token),
                },
                status=status.HTTP_200_OK,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



# 분실물 등록 + AI 유사도 검색 API
class LostItemCreateView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        serializer = LostItemCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        image = serializer.validated_data.get("image")
        description = serializer.validated_data.get("description")

        image_url = None
        if image:
            # MEDIA_ROOT 보장 및 안전한 파일 저장
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            fs = FileSystemStorage(location=settings.MEDIA_ROOT, base_url=settings.MEDIA_URL)
            filename = fs.get_available_name(image.name)
            saved_path = fs.save(filename, image)
            image_url = fs.url(saved_path)  # '/uploads/<filename>' 형태

        lost_item = UserLostItem.objects.create(
            user=request.user,
            description=description,
            original_image_url=image_url,
        )

        # --- AI 결과 가져오기 ---
        ai_results = ai_infer(description, top_k=3)

        # --- 프론트에서 필요한 형태로 변환 ---
        formatted = []
        for item in ai_results[:3]:
            preview = item.get("preview", {})

            formatted.append({
                "title": preview.get("fdPrdtNm"),
                "image": preview.get("fdFilePathImg"),
                "place": preview.get("depPlace"),
                "date": preview.get("fdYmd"),
                "detail_link": f"https://www.lost112.go.kr/find/findDetail.do?atcId={item.get('atcId')}",
                "score": item.get("score"),
            })

        return Response(
            {
                "registered_item": {
                    "request_id": lost_item.request_id,
                    "description": lost_item.description,
                    "original_image_url": lost_item.original_image_url,
                    "is_active": lost_item.is_active,
                    "created_at": lost_item.created_at,
                },
                "immediate_matches": formatted,
            },
            status=status.HTTP_201_CREATED
        )



# GET 사용자가 등록한 아이템 목록
class ItemsListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        items = UserLostItem.objects.filter(user=user).order_by("-created_at")
        serializer = UserLostItemListSerializer(items, many=True, context={"request": request})

        return Response({"items": serializer.data}, status=200)


# PATCH 분실물 알림 중지
class CeaseLostItemView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request, request_id):
        try:
            lost_item = UserLostItem.objects.get(request_id=request_id, user=request.user)
        except UserLostItem.DoesNotExist:
            return Response(
                {"error": "해당 요청이 존재하지 않거나 권한이 없습니다."},
                status=status.HTTP_404_NOT_FOUND
            )

        lost_item.is_active = False
        lost_item.save()

        return Response(
            {
                "request_id": request_id,
                "message": "알림이 중지되었습니다."
            },
            status=status.HTTP_200_OK
        )