# 도메인 · DNS

## 현재 설정

| 항목 | 값 |
|------|-----|
| 서비스 주소 | https://invest.jason.ai.kr |
| 서버 IP | 134.185.109.82 |
| 도메인 등록 | 가비아 (jason.ai.kr) |
| 네임서버 · DNS 관리 | Cloudflare |
| DNS 레코드 | A `invest` → 134.185.109.82, **DNS 전용(회색 구름)** |
| 인증서 | Let's Encrypt, acme-companion 자동 발급·갱신 |

등록기관(가비아)과 DNS 관리처(Cloudflare)가 다르다. **레코드는 Cloudflare 에서만 바꾼다** —
가비아 DNS 화면에 넣어도 반영되지 않는다.

Cloudflare 프록시(주황 구름)는 쓰지 않는다. 켜면 HTTP-01 인증서 검증이 리다이렉트로 실패하고,
SSL 모드가 Flexible 이면 무한 리다이렉트가 된다.

## 주소가 정해지는 곳

코드에 도메인을 적지 않는다. 서버 `.env` 의 **`APP_BASE_URL` 하나**가 단일 소스다.

| 대상 | 사용처 |
|------|--------|
| 슬랙 메시지 링크 | `utils/notification.py` 의 `app_link()` — 값이 없으면 링크 없이 라벨만 |
| 구글 OAuth 콜백 | `web/lib/auth.ts` 의 `getGoogleCallbackUrl()` → `{APP_BASE_URL}/api/auth/callback/google` |
| nginx-proxy 호스트 | `docker-compose.yml` 의 `VIRTUAL_HOST` · `LETSENCRYPT_HOST` (여기만 별도로 적는다) |
| vhost 커스텀 설정 | 서버 `~/apps/nginx-proxy/vhost.d/invest.jason.ai.kr` — **파일명이 곧 호스트명**. robots.txt 응답이 여기 있다 |

구글 OAuth 콘솔의 승인된 리디렉션 URI 에도 같은 주소가 등록돼 있어야 한다.

## 확인

```bash
dig +short invest.jason.ai.kr           # 서버 IP 가 나오면 DNS 전용 (104.x/172.67.x 면 CF 프록시)
curl -I https://invest.jason.ai.kr      # 미로그인 시 307 → /login
curl https://invest.jason.ai.kr/robots.txt   # vhost.d 설정이 살아 있으면 User-agent 응답
```

## 문제 대응

| 증상 | 원인 | 조치 |
|------|------|------|
| 호스트를 바꿨는데 인증서가 안 나옴 | acme-companion 의 `/app/letsencrypt_service_data` 가 옛 컨테이너 기준으로 굳음 (`signal_le_service` 로는 재생성 안 됨) | `docker restart nginx-proxy-acme` |
| 로그인만 `redirect_uri_mismatch` | `node_app` 이 옛 `APP_BASE_URL` 을 들고 있음 (배포는 이미지가 같으면 재생성하지 않음) | `docker compose up -d --no-deps --force-recreate node_app` |
| robots.txt 가 404 HTML | `vhost.d/{호스트명}` 파일이 옛 도메인 이름 그대로 | 파일명을 새 호스트로 바꾸고 `docker restart nginx-proxy` |
| 슬랙 링크에 주소가 없음 | `APP_BASE_URL` 미설정 | 서버 `.env` 에 설정 후 재생성 |
| 레코드를 넣었는데 반영 안 됨 | 가비아 DNS 화면에 등록함 | Cloudflare 에서 등록 |
