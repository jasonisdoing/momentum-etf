# syntax=docker/dockerfile:1.7
FROM node:24-alpine

WORKDIR /app

COPY web/package.json web/package-lock.json* ./
# BuildKit cache mount 로 npm 다운로드 캐시 영구 보존.
# npm ci 는 package-lock.json 기반 reproducible 설치라 npm install 보다 안정/빠름.
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit --no-fund

COPY shared /shared
COPY web .

RUN --mount=type=cache,target=/app/.next/cache \
    npm run build

EXPOSE 3000

CMD ["npm", "run", "start", "--", "--hostname", "0.0.0.0", "--port", "3000"]
