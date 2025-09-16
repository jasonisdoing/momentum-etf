# 1. 베이스 이미지로 Python 3.11 slim 버전을 사용합니다.
FROM python:3.11-slim

# 3. 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app

# 4. requirements.txt 파일을 먼저 복사하여 Docker 레이어 캐시를 활용합니다.
COPY requirements.txt .

# 5. pip를 최신 버전으로 업그레이드하고 의존성을 설치합니다.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. 나머지 프로젝트 소스 코드를 /app 디렉토리로 복사합니다.
COPY . .

# 7. Render가 웹 서비스에 사용할 포트를 지정합니다. Render는 이 포트로 트래픽을 라우팅합니다.
EXPOSE 10000

# 8. 웹 애플리케이션을 실행하는 명령입니다.
CMD ["streamlit", "run", "web_app.py", "--server.port=10000", "--server.address=0.0.0.0"]
