# 서버 인프라 · 배치 · 도메인

## 서버

| 항목 | 값 |
| --- | --- |
| 호스트 | OCI 춘천, 1 OCPU ARM, Ubuntu, 사용자 `ubuntu` |
| 접속 | `ssh -i {SSH 키} ubuntu@{IP}` |
| 앱 경로 | `/home/ubuntu/apps/momentum-etf` |
| 컨테이너 | `momentum-etf-app-1`(Node, 80) · `momentum-etf-fastapi_app-1`(FastAPI, 8000, 내부) · MongoDB · `nginx-proxy` · `nginx-proxy-acme` |
| 서버 `data/` | 읽기 전용 마운트 — 서버에서 갱신돼야 하는 데이터는 파일이 아니라 DB(`index_constituents` 등) |

VM 의 역할은 컨테이너 가동뿐이다. 자동 배치는 돌지 않는다.

## 배치

- **모든 자동 배치는 로컬(Mac)** 의 `infra/server_scheduler.py` 가 `infra/cron/crontab` 을 파싱해 APScheduler 로 실행한다. VM cron 은 CPU 100% 다운이 반복돼 제거했다(`infra/cron/install.sh --uninstall`).
- `crontab` 이 배치 정의의 단일 소스. 잡 이름 = action 키. 비활성화는 주석이 아니라 **라인 삭제**(과거 파서가 주석 라인을 등록한 적 있음). 스크립트 뒤 인자는 전달되지만 `-m` 은 안 된다.
- 락: Mongo `batch_locks`(`_id=<job>`). 로컬 자동 실행과 `/batch` 수동 실행이 같은 락을 쓴다. 소유자는 `APP_TYPE`(`Local` / 미설정=PROD). 꺼져 있던 시간의 누락분은 따라잡지 않는다.
- 큐는 서버·로컬이 공유. 로컬에만 결과가 남는 잡은 `utils/batch_queue.LOCAL_ONLY_JOBS` 에 등록하면 서버 워커가 claim 하지 않는다.
- 배치 코드는 Docker 이미지에 포함 → 변경 시 재배포. `crontab`/`run_batch` 는 마운트라 즉시 반영.
- 로그: `logs/cron/<job>.log`. 실패 시에만 래퍼 슬랙 알림.
- 배치 추가·삭제 시 함께 고칠 7곳은 [developer_guide.md](developer_guide.md) §4.

### 배치 목록 (crontab 기준)

| 잡 | 스크립트 | 비고 |
| --- | --- | --- |
| `cache_refresh` / `cache_refresh_full` | `stock_price_cache_updater.py` | 매시 증분, 하루 1회 `--full`(수정주가 재정렬) |
| `reference_meta_updater` | `stock_reference_meta_updater.py` | 배치 B: 식별·상세 메타, ETF 구성종목·배당 |
| `price_metrics_updater` | `stock_price_metrics_updater.py` | 배치 A: 거래량·기간수익률 등 가격 파생 |
| `data_aggregate` | `collect_data.py` | 일별 원장 → 주/월 재집계 |
| `asset_summary` | `slack_asset_summary.py` | 자산 요약 슬랙 |
| `market_hours_analysis` | `analyze_market_hours.py` | |
| `us_market_stocks` / `aus_market_stocks` | `update_us_market_stocks.py` / `update_aus_market_stocks.py` | `index_constituents` 갱신. 구성종목 수가 범위 밖이면 저장 안 하고 실패 |
| `us_market_etfs` | `update_us_market_etfs.py` | 미국 ETF 마켓 목록(`etf_market_master`) — KIS 미국 마스터 + yfinance |
| `market_breadth` | `collect_market_breadth.py` | 한국 마감 후·미국 마감 후 |
| `db_backup` | `backup_mongo_full.py` | 로컬 폴더라 LOCAL 전용 |
| `holdings_alarm`, `strategy_mix_notify`, `live_24h_slack`, `leverage_ma_cross`, `broker_balance_sync` 등 | 동명 스크립트 | 알림·동기화 |

## 배포

- Node 앱이 내부 네트워크로 FastAPI 호출. nginx-proxy 가 도메인 요청을 Node 컨테이너로 프록시, acme-companion 이 Let's Encrypt 인증서 자동 갱신.
- GitHub Actions(`.github/workflows/`)로 배포. 배포 실패 시 앱이 죽어야 한다 — rollback·build-first 로 가리지 않는다.
- 폐기된 기능은 배포 때 서버 DB 문서·crontab 라인도 같이 지운다.

### nginx-proxy 커스텀 설정

호스트 `/home/ubuntu/apps/nginx-proxy/vhost.d/` — `{domain}`(server 블록), `{domain}_location`, `{domain}_location_override`. **파일명이 곧 호스트명**. 수정 후 `docker restart nginx-proxy`(reload 로는 템플릿이 재렌더링되지 않음).

## 도메인 · DNS · 인증서

| 항목 | 값 |
| --- | --- |
| 서비스 주소 | https://invest.jason.ai.kr |
| 등록기관 | 가비아 (`jason.ai.kr`) |
| DNS 관리 | Cloudflare — **레코드는 여기서만**(가비아 DNS 화면은 반영 안 됨) |
| 레코드 | A `invest` → 서버 IP, **DNS 전용(회색 구름)** |
| 인증서 | Let's Encrypt, acme-companion |

Cloudflare 프록시(주황 구름)는 쓰지 않는다 — HTTP-01 검증이 리다이렉트로 실패하고 Flexible SSL 이면 무한 리다이렉트.

코드에 도메인을 적지 않는다. 서버 `.env` 의 **`APP_BASE_URL` 하나**가 단일 소스(슬랙 링크 `utils/notification.app_link`, 구글 OAuth 콜백 `web/lib/auth.ts`). 예외는 `docker-compose.yml` 의 `VIRTUAL_HOST`·`LETSENCRYPT_HOST` 와 `vhost.d/` 파일명. 구글 OAuth 콘솔의 리디렉션 URI 도 같은 주소.

## 환경변수 (`.env`)

| 키 | 용도 |
| --- | --- |
| `APP_BASE_URL` | 서비스 주소 단일 소스 |
| `APP_TYPE` | `Local` 이면 로컬 워커·락 소유자 |
| `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID` | 알림 |
| `DART_API_KEY` | OpenDART 재무 조회 |
| `KRX_DATA_ID`, `KRX_DATA_PW` | KRX 로그인 헬퍼(보관용, 자동 수집은 약관 금지) |
| `TUNING_WORKERS` | 튜닝 병렬 수 덮어쓰기(기본 코어 수) |
| MongoDB 접속 | `utils/db_manager.py` 참조 |

## 겪은 문제

| 증상 | 원인 | 조치 |
| --- | --- | --- |
| 서버에서 KOR 가격 캐시가 30종목쯤에서 hang | KRX 호출 빈도 IP 차단(서버는 응답이 빨라 분당 600회) | `stock_price_cache_updater.py` 의 `KOR_FETCH_TARGET_SECONDS` 로 종목당 최소 간격. 차단되면 값을 늘린다 |
| VM 에서 배치 돌리면 시스템 다운 | 1 OCPU ARM CPU 100% | VM cron 제거, 로컬 스케줄러 |
| 호스트 변경 후 인증서 미발급 | acme-companion 이 옛 컨테이너 데이터 보유 | `/app/letsencrypt_service_data` 정리 후 재시작 |
| 로그인만 `redirect_uri_mismatch` | `node_app` 이 옛 `APP_BASE_URL` | 컨테이너 재생성 |
| robots.txt 가 404 HTML | `vhost.d/` 파일명이 옛 도메인 | 파일명을 새 호스트로 |
| DNS 레코드가 반영 안 됨 | 가비아에 등록함 | Cloudflare 에 등록 |
| 배포 실패 `DISK_USE: 100%` | compose 가 `:latest` 를 참조해 배포마다 직전 이미지가 dangling 으로 남는데 정리하는 곳이 없었다(2026-04 에 넣은 `prune -af` 를 2026-05 에 부하 때문에 제거). 3개월간 1,140개 32.6GB 누적 | 배포 끝에 `docker image prune -f` 복원(`-a` 없이 — 태그 이미지까지 지우면 다음 배포에 다시 받아 1 OCPU 에 부하). 배포 **전** 디스크 5GB 미만이면 서버를 건드리기 전에 실패 |
| 디스크를 비웠는데도 배포가 계속 실패 (fastapi 헬스체크 30초 초과) | 디스크 100% 때 mongodb 컨테이너가 재시작하며 네트워크 엔드포인트를 못 붙였다. `inspect` 에 네트워크 이름은 있는데 **IP 가 빈 값**이라 `mongodb` DNS 해석 실패 | `docker compose up -d --force-recreate mongodb` 로 컨테이너 재생성(데이터는 볼륨이라 보존). `docker network inspect momentum-etf_default` 의 Containers 목록에 mongodb 가 있는지로 판별 |
