# AI 런타임은 별도 가상환경에서 실행하므로 백엔드 이미지에는 포함하지 않음
FROM python:3.12-slim-bookworm

# 환경변수
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 백엔드 의존성만 설치 (AI 런타임 제외)
COPY services/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 소스 코드 복사
COPY . .

# 포트
EXPOSE 8000

# 실행 (docker-compose의 command가 없을 때 기본값)
CMD ["uvicorn", "services.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
