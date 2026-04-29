FROM ubuntu:latest

# 시스템 패키지 설치 및 pip 업데이트 (옵션 추가)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    vim && \
    python3 -m pip install --upgrade pip --break-system-packages

# 라이브러리 설치 시에도 동일한 옵션 추가
COPY services/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt --break-system-packages

WORKDIR /app