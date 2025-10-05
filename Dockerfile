# 1. 베이스 이미지로 공식 Python 이미지를 사용합니다.
FROM python:3.12-slim

# 2. 환경 변수를 설정하여 파이썬 출력이 버퍼링되지 않도록 하고, .pyc 파일을 생성하지 않도록 합니다.
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
# 컨테이너의 기본 인코딩을 UTF-8로 설정하여 한글 깨짐을 방지합니다.
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV STREAMLIT_SERVER_ADDRESS 0.0.0.0
ENV STREAMLIT_SERVER_PORT 8501

# 3. debconf가 대화형 프롬프트를 표시하지 않도록 설정합니다.
ENV DEBIAN_FRONTEND=noninteractive

# 4. 시스템 패키지를 업데이트하고, 빌드에 필요한 도구를 설치합니다.
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# 5. 작업 디렉토리를 생성하고 설정합니다.
WORKDIR /app

# 6. 의존성 설치를 위해 requirements.txt 파일을 먼저 복사합니다.
#    이 파일이 변경되지 않으면 이 레이어는 캐시되어 빌드 속도가 향상됩니다.
COPY requirements.txt .

# 7. pip를 업그레이드하고 의존성을 설치합니다.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 8. 나머지 애플리케이션 소스 코드 전체를 작업 디렉토리로 복사합니다.
COPY . .

# 9. 컨테이너 시작 시 Streamlit 앱을 실행합니다.
EXPOSE 8501
CMD ["python", "run.py"]
