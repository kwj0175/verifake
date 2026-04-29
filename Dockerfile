FROM ubuntu:latest

# 1. 시스템 패키지 설치 및 pip 자체 업데이트
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    vim && \
    python3 -m pip install --upgrade pip

# 2. 라이브러리 미리 설치
# (requirements.txt가 변경되지 않으면 이 단계는 캐시되어 빌드 속도가 빨라집니다)
COPY services/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app