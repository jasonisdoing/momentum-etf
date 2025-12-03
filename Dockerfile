FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ✅ 메타 태그 추가 스크립트 복사 및 실행
COPY scripts/add_meta_tags.py .
RUN python add_meta_tags.py

# ✅ 프로젝트 소스 복사
COPY . .

# ✅ Streamlit 설정 파일 복사
COPY .streamlit /app/.streamlit
