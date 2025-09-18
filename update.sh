#!/bin/bash

# 스크립트 실행 중 오류가 발생하면 즉시 중단합니다.
set -e

# 1. Git 리포지토리에서 최신 코드를 가져옵니다.
echo ">>> 1. Pulling latest code from 'upgrade' branch..."
git pull origin upgrade

# 2. Docker Compose를 사용하여 컨테이너를 다시 빌드하고 재시작합니다.
# --build 옵션은 Dockerfile, requirements.txt, 소스 코드 등 변경사항을 이미지에 새로 반영합니다.
# -d 옵션은 컨테이너를 백그라운드에서 실행합니다.
echo ">>> 2. Rebuilding and restarting Docker containers..."
docker compose up -d --build

# 3. (선택사항) 사용하지 않는 오래된 도커 이미지를 정리하여 디스크 공간을 확보합니다.
echo ">>> 3. Pruning old Docker images..."
docker image prune -f

echo ">>> Deployment to Lightsail successful!"
