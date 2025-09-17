# 1. 베이스 이미지로 Python 3.12 slim 버전을 사용합니다. (requirements.txt 생성 기준)
FROM python:3.12-slim

# 2. 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app

# 3. 의존성 설치를 위해 requirements.txt 파일을 먼저 복사합니다.
#    이렇게 하면 소스 코드가 변경되어도 라이브러리를 다시 설치하지 않아 캐시를 활용할 수 있습니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 프로젝트의 모든 소스 코드를 컨테이너 안으로 복사합니다.
COPY . .

# 5. 컨테이너가 시작될 때 스케줄러를 실행합니다.
#    Cloud Run 인스턴스 기반 배포에서는 이 프로세스가 계속 실행됩니다.
CMD ["python", "scheduler.py"]
