# 도메인 · DNS 정보

서버 자체 구성은 [server_infrastructure.md](server_infrastructure.md) 를 본다. 이 문서는
**도메인 등록·네임서버·DNS 레코드·인증서**만 다룬다.

## 현재 구성

| 항목 | 값 |
|------|-----|
| 서비스 도메인 | invest.jason.ai.kr |
| 옛 도메인 | etf.dojason.com (사용 중단 — `VIRTUAL_HOST` 에서 제거됨) |
| 도메인 등록기관(레지스트라) | 가비아 |
| 네임서버 | Cloudflare |
| DNS 레코드 관리 | Cloudflare 대시보드 |
| 서버 IP | 134.185.109.82 |
| 인증서 | Let's Encrypt (acme-companion 자동 발급·갱신) |

**등록기관과 DNS 관리처가 다르다.** 도메인 소유·갱신·결제는 가비아에서 하고,
가비아의 네임서버를 Cloudflare 로 지정해 두었기 때문에 **레코드 추가·수정은
가비아가 아니라 Cloudflare 에서 한다.** 가비아 DNS 관리 화면에 레코드를 넣어도
반영되지 않는다.

---

## DNS 레코드 등록 방법

Cloudflare → 해당 도메인 → DNS → 레코드 추가.

| 필드 | 값 |
|------|-----|
| 형식 | A |
| 이름 | invest (→ `invest.jason.ai.kr` 이 된다) |
| IPv4 주소 | 134.185.109.82 |
| 프록시 상태 | **DNS 전용 (회색 구름)** |
| TTL | 자동 |

### 프록시(주황 구름)를 켜지 않는 이유

이 시스템은 nginx-proxy + acme-companion 이 **HTTP-01 방식**으로 인증서를 받는다
(`http://도메인/.well-known/acme-challenge/...` 로 오는 검증 요청에 응답).
Cloudflare 프록시를 켜면 다음이 깨진다.

1. Cloudflare 의 **Always Use HTTPS** 가 켜져 있으면 검증 요청이 301 로 리다이렉트되어
   인증서 발급·갱신이 실패한다
2. Cloudflare **SSL 모드가 Flexible** 이면 CF 는 http 로 오리진에 붙고 오리진은 https 로
   되돌려 무한 리다이렉트가 된다
3. 운영 중인 `etf.dojason.com` 도 프록시 없이(오리진 IP 직접 노출) 동작해 왔다 —
   구성을 맞추는 편이 안전하다

프록시를 굳이 쓰려면 순서가 있다.

1. 회색 구름으로 두고 인증서 발급이 끝난 것을 확인한다
2. Cloudflare SSL/TLS 모드를 **Full (strict)** 로 바꾼다
3. 그 다음 프록시(주황 구름)를 켠다
4. 갱신 시점(60일 후)에 인증서가 정상 갱신되는지 확인한다 — 실패하면 되돌린다

---

## 도메인을 옮길 때 해야 하는 일

코드에는 도메인을 적지 않는다. **`APP_BASE_URL` 환경변수 하나가 단일 소스**이고,
슬랙 메시지 링크(`utils/notification.py` 의 `app_link`)와 구글 OAuth 콜백 주소
(`web/lib/auth.ts` 의 `getGoogleCallbackUrl`)가 이 값에서 파생된다.

1. **DNS** — Cloudflare 에 A 레코드 추가 (위 표대로, 회색 구름)
2. **`docker-compose.yml`** — `hybrid_proxy` 의 `VIRTUAL_HOST` · `LETSENCRYPT_HOST` 를 새 호스트로 교체.
   두 주소를 한동안 함께 열어야 하면 콤마로 나열한다 (`새주소,옛주소`)
3. **서버 `.env`** — `APP_BASE_URL=https://invest.jason.ai.kr`
4. **구글 OAuth 콘솔** — 승인된 리디렉션 URI 에
   `https://invest.jason.ai.kr/api/auth/callback/google` 추가.
   **이걸 빠뜨리면 로그인이 막힌다**
5. **재배포** — nginx-proxy 가 새 호스트를 인식하고 acme-companion 이 인증서를 새로 발급한다
6. 새 주소 정상 확인 후, 옛 도메인을 301 리다이렉트로 바꾸거나 `VIRTUAL_HOST` 에서 뺀다

### 확인 명령

```bash
dig +short invest.jason.ai.kr
```

서버 IP 가 그대로 나오면 DNS 전용이고, `104.x` / `172.67.x` 같은 주소가 나오면
Cloudflare 프록시가 켜진 것이다.

```bash
curl -I https://invest.jason.ai.kr
```

인증서 발급 상태는 서버에서 확인한다.

```bash
docker logs nginx-proxy-acme --tail 50
```

---

## 자주 겪는 문제

| 증상 | 원인 | 조치 |
|------|------|------|
| 인증서가 발급되지 않음 | Cloudflare 프록시 ON + Always Use HTTPS | 회색 구름으로 바꾸고 acme 로그 확인 |
| 무한 리다이렉트 | Cloudflare SSL 모드 Flexible | Full (strict) 로 변경 |
| 로그인만 실패 | 구글 OAuth 리디렉션 URI 미등록 | 콘솔에 새 도메인 콜백 추가 |
| 슬랙 링크가 옛 주소 | 서버 `.env` 의 `APP_BASE_URL` 미변경 | 값 수정 후 재배포 |
| 슬랙 링크에 주소가 없음 | `APP_BASE_URL` 미설정 (`app_link` 가 라벨만 남긴다) | 값 설정 |
| 레코드를 넣었는데 반영 안 됨 | 가비아 DNS 화면에 등록함 | 네임서버가 Cloudflare 이므로 Cloudflare 에서 등록 |
