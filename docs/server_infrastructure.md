# 서버 인프라 정보

## 접속 정보

| 항목 | 값 |
|------|-----|
| IP | ***.***.***.** |
| OS 사용자 | ubuntu |
| SSH 키 경로 | {SSH 키 파일 경로} |
| 도메인 | etf.dojason.com |

```bash
ssh -i {SSH 키 파일 경로} ubuntu@***.***.***.**
```

---

## Docker 컨테이너 구성

| 컨테이너명 | 이미지 | 역할 |
|------------|--------|------|
| momentum-etf-app-1 | momentum-etf-app | Node 웹 앱 (포트 80) |
| momentum-etf-fastapi_app-1 | momentum-etf-fastapi_app | 내부 FastAPI 백엔드 (포트 8000, 외부 미공개) |
| nginx-proxy | nginxproxy/nginx-proxy | 리버스 프록시 (80, 443) |
| nginx-proxy-acme | nginxproxy/acme-companion | SSL 인증서 자동 갱신 |

---

## nginx-proxy 구성

### vhost.d 경로

nginx-proxy는 도메인별 커스텀 nginx 설정을 `/etc/nginx/vhost.d/` 에서 읽는다.
호스트 경로(bind mount)는 다음과 같다:

```
/home/ubuntu/apps/nginx-proxy/vhost.d/
```

### 파일 종류

| 파일명 | 적용 위치 | 용도 |
|--------|-----------|------|
| `{domain}` | server 블록 내부 (location 블록 밖) | 새 location 블록 추가 등 서버 레벨 설정 |
| `{domain}_location` | `location /` 블록 내부 | 기본 proxy location 내 추가 설정 |
| `{domain}_location_override` | `location /` 블록 전체 대체 | proxy location 완전 교체 |

### 설정 변경 절차

1. `/home/ubuntu/apps/nginx-proxy/vhost.d/{파일}` 생성 또는 수정
2. nginx-proxy 재시작으로 템플릿 재렌더링 (단순 reload로는 적용 안 됨)

```bash
docker restart nginx-proxy
```

### 예시: robots.txt 응답 추가

파일: `/home/ubuntu/apps/nginx-proxy/vhost.d/etf.dojason.com`

```nginx
location = /robots.txt {
    return 200 'User-agent: *\nAllow: /\n';
    add_header Content-Type text/plain;
}
```

---

## 앱 배포 구조

- Node 웹 앱이 컨테이너 내부 포트 80으로 실행
- FastAPI 내부 API가 컨테이너 내부 포트 8000으로 실행
- Node 앱이 내부 네트워크에서 FastAPI를 호출
- nginx-proxy가 `etf.dojason.com` 요청을 momentum-etf-app-1 컨테이너로 프록시
- SSL은 acme-companion이 Let's Encrypt 인증서로 자동 처리

---

## 배치 운영

### VM cron 제거

1 OCPU ARM VM 에서 배치 실행 시 CPU 100% 폭주로 시스템 다운이 반복돼,
momentum-etf 의 VM cron 항목은 모두 제거되었다 (`infra/cron/install.sh --uninstall`).

> **leverage-switching 앱 통합(2026-06)**: 같은 VM 에서 별도 호스트 cron 으로 돌던
> `leverage-switching`(레버리지 스위칭) 앱은 폐기되었고, 전략이 momentum-etf 의
> `leverage/` 패키지로 이전되었다. 추천 배치는 momentum-etf 배치 체계의 `leverage_switch`
> 잡(`scripts/leverage_recommend_switch.py`)으로 편입되어 동일한 crontab·스케줄러·큐로 실행된다.
> leverage-switching VM cron 제거는
> `bash ~/apps/leverage-switching/infra/cron/install.sh --uninstall`.

### 로컬 스케줄러로 전환

모든 momentum-etf 자동 배치는 로컬(Mac) 의 `infra/server_scheduler.py` 가 실행한다.
이 프로세스는 `infra/cron/crontab` 파일을 단일 진실 소스로 파싱하여
APScheduler 에 등록한다.

- 락 메커니즘: MongoDB `batch_locks` 컬렉션 (`_id=<job_name>` unique)
  → 로컬 자동 실행과 `/system` 화면 수동 실행이 동일한 락을 거치므로 중복 방지됨
- 락 소유자 식별: `APP_TYPE` 환경변수 (`Local` vs 미설정 시 `PROD`)
- 노트북이 꺼져 있던 시간의 미실행 분은 따라잡지 않는다 (misfire_grace_time=None)

### 로컬 전용 잡 (워커 친화도)

큐는 서버·로컬 워커가 공유하지만, **무거운 계산 + 결과가 로컬 파일에 남는** 잡은 로컬 워커만
픽하게 한다 (`utils/batch_queue.py` 의 `LOCAL_ONLY_JOBS = {"leverage_tune", "momentum_backtest"}`).

- `enqueue` 가 잡 doc 에 `local_only` 플래그를 자동 기록(잡 이름 기준) → 모든 트리거 경로
  (레버리지·모멘텀 화면 버튼 / `/batch` 클릭 / 스케줄러)에 일관 적용.
- `claim_next_pending` 은 워커가 `APP_TYPE != "Local"` 이면 `local_only: {$ne: True}` 로 필터 →
  **서버 워커는 튜닝/백테스트를 claim 하지 않는다.** 로컬 워커는 전부 claim.
- 이유: ① 결과 파일(`leverage/zresults/`, `backtest/results/`)이 실행한 머신 FS 에만 남아
  로컬 UI 에서만 보임(공유 스토리지 아님), ② 약한 서버 VM 에서 멀티프로세스 돌리면 CPU 폭주.
- 주의: **로컬 워커가 꺼져 있으면 이 두 잡은 pending 으로 대기**(서버가 안 가져감). `/batch` 에는
  "수동 실행 · 로컬 전용" 으로 표시된다.

### 스케줄러·배치 작성 시 주의

`infra/server_scheduler.py` 가 `infra/cron/crontab` 을 파싱할 때의 비자명한 동작:

- **무인자 스크립트만 실행**: `python <script.py>` 형태만 파싱하며 `-m`/추가 인자는 인식하지 못한다. 인자가 필요한 진입점은 무인자 **래퍼 스크립트**로 감싼다(예: `scripts/leverage_recommend_switch.py`, `scripts/leverage_tune_switch.py`).
- **주석 cron 라인도 활성으로 파싱**: 주석(`#`) 처리된 잡 라인도 등록될 수 있으므로, 잡 비활성화는 주석이 아니라 **라인 삭제**로 한다.
- 배치 코드/스크립트는 Docker 이미지에 포함되므로 변경 시 **재배포** 필요. `crontab`/`run_batch` 는 `./infra/cron` 마운트로 즉시 반영.
- leverage 전략 데이터는 `ticker_type="etf"`(MongoDB 캐시 키)로 조회하며 대상은 모두 한국 ETF.

### VM 의 역할 (현재)

- Docker 컨테이너 (웹 + FastAPI + MongoDB + nginx-proxy) 만 가동
- 자동 배치 없음. 수동 실행이 필요하면 `/system` 화면의 버튼으로 트리거

### 가격 캐시 — KOR 풀 동적 sleep (KRX rate-limit 회피)

**증상**: 서버(OCI 춘천) 에서 가격 캐시 KOR 풀 처리 중 30~33 종목 즈음 KRX 응답이
멈추면서 작업이 hang 으로 빠진다. 로컬(한국 ISP)에서는 발생하지 않는다.

**원인**: KRX 가 단위 시간당 호출 빈도로 IP 차단을 거는 것으로 보인다. 서버는
응답이 너무 빨라 (종목당 ~0.1s) 분당 600회 호출이 발생하면서 차단된다.

**대응 (구현)**: `scripts/stock_price_cache_updater.py` 의 KOR 풀 직렬 루프에
**종목당 동적 sleep** 을 적용한다. 종목 처리 elapsed 가 목표 간격 미만이면
부족분만큼 채워서 호출 빈도를 일정 수준 이하로 유지한다.

- 목표 간격: `KOR_FETCH_TARGET_MS` 환경변수 (기본 **300ms**)
- 적용 대상: `country_code == "kor"` 풀만 (US/AUS 는 yfinance 일괄 prefetch 라 무영향)
- 로그 표시 예: `소요 0.1s + 0.2s 대기(속도조절)`
- 로컬은 종목당 자연 소요(0.2~0.4s) 가 이미 충분히 느려 sleep 0 → 영향 없음
- 서버는 0.1s × ~60종목 부족분 = 약 +12s 보충 → 풀 전체 +1~2분, 30분 timeout 한참 여유

**튜닝**: 위 기본값으로도 차단되면 환경변수를 늘린다 (예: `500` → 분당 ~100회).
컨테이너에 환경변수가 없으면 코드 기본값(300ms) 이 적용된다.
