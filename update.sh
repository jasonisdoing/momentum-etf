#!/bin/bash

# 스크립트 실행 중 오류가 발생하면 즉시 중단
set -euo pipefail

# 실행 위치를 프로젝트 루트로 이동
cd /home/ubuntu/momentum-etf

# 1. Git 리포지토리에서 최신 코드 가져오기
echo ">>> 1. Pulling latest code from 'upgrade' branch..."
git fetch origin upgrade
git reset --hard origin/upgrade

# 2. Docker Compose로 컨테이너 빌드 및 재시작
echo ">>> 2. Rebuilding and restarting Docker containers..."
docker compose pull        # 새 이미지가 있으면 받아옴
docker compose up -d --build

# 3. 오래된 이미지/컨테이너/볼륨 정리 (선택사항)
echo ">>> 3. Pruning old Docker resources..."
docker image prune -f
docker container prune -f
docker volume prune -f

echo ">>> ✅ Deployment to Lightsail successful!"