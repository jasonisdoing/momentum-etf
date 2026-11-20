# 서버 인프라 · 배치 · 도메인

## 서버

| 항목 | 값 |
| --- | --- |
| 호스트 | Mac mini (macOS, 24시간 상시 운영 — 자동 잠자기 끄기 · 정전 후 자동 시동) |
| 앱 경로 | `/Users/jason/DEV/momentum-etf` |
| `launchd` 서비스 | `com.momentum-etf.web`(Next.js `:3000`) · `com.momentum-etf.fastapi`(FastAPI `:8000`) · `com.momentum-etf.scheduler`(`server_scheduler.py`) · `cloudflared` |
| MongoDB | 맥미니 로컬(`:27017`, Homebrew) — 맥북 프로는 LAN IP(`192.168.68.55:27017`)로 직접 연결 |
| 로컬 개발 | `run_local_dev.py`(`APP_TYPE=Local`, cron 비활성) |

호스트의 시스템 cron은 사용하지 않는다.

## 배치

- 자동 배치는 맥미니의 `com.momentum-etf.scheduler`(`infra/cron/server_scheduler.py`)가 `infra/cron/crontab`을 읽어 큐(`batch_queue`)에 등록하고 처리한다.
- 로컬 워커(`APP_TYPE=Local`)는 `config.LOCAL_QUEUE_WORKER_ENABLED`에 따라 동작한다: `False`(기본)면 `LOCAL_ONLY_JOBS`(`db_backup`)만 claim하고 나머지는 맥미니 서버만 처리하며, `True`면 일반 배치도 서버·로컬이 함께 처리한다.
- `crontab` 이 배치 정의의 단일 소스. 잡 이름 = action 키. 비활성화는 주석이 아니라 **라인 삭제**(과거 파서가 주석 라인을 등록한 적 있음). 스크립트 뒤 인자는 전달되지만 `-m` 은 안 된다.
- 락: Mongo `batch_locks`(`_id=<job>`). 자동 실행과 `/batch` 수동 실행이 같은 락을 쓴다. 소유자는 `APP_TYPE`(`Local` / 미설정=PROD). 꺼져 있던 시간의 누락분은 따라잡지 않는다.
- 큐(`batch_queue`)는 대기·실행 중 항목을 24시간 TTL로, 끝난 항목은 30일(`RUN_HISTORY_DAYS`) 보관한다. 이 이력 전체(최근 30일)의 평균이 `/batch`의 대기·예상시간의 원천이다.
- 큐는 서버·로컬이 공유. 로컬에만 결과가 남는 잡은 `utils/batch_queue.LOCAL_ONLY_JOBS` 에 등록하면 서버 워커가 claim 하지 않는다.
- 스케줄러는 시작할 때 `infra/cron/crontab`을 읽으므로 시간표·코드 변경 후 재시작(또는 배포)해야 반영된다.
- 미장까지 다루는 배치(잔고 동기화·사고팔기 알림·슬랙 주문 결과 알림 등)는 월~토 24시간 돈다 — 종목마다 그 시장의 거래일·거래 시간을 배치 안에서 판정하고, 아니면 바로 끝난다.
- 로그: `logs/cron/<job>.log`. 실패 시에만 래퍼 슬랙 알림. 환율 실패 예외에는 통화·Yahoo 심볼·시도 횟수·원본 오류가 포함된다.
- 배치 추가·삭제 시 함께 고칠 7곳은 [developer_guide.md](developer_guide.md) §4.
- 미국 가격 캐시는 야후 수집 뒤 네이버로 빈 OHLC를 복구하고, 토스로 최근 완료 거래일의 빈 거래량을 복구한 다음 `stock_meta` 가격 파생값을 계산한다. 복구 실패 항목은 가격 캐시 점검 알림에 포함한다.

### 배치 목록 (crontab 기준)

| 잡 | 스크립트 | 비고 |
| --- | --- | --- |
| `cache_refresh_kor` / `cache_refresh_us` / `cache_refresh_au` / `cache_refresh_full` | `stock_price_cache_updater.py` | 매시 증분은 시장별(`--country`)로 나눠 시각을 엇갈려 돈다(한 번에 돌면 data 레인을 오래 잡아 짧은 배치가 길게 기다렸다). 하루 1회 `--full`(수정주가 재정렬)은 전체 |
| `reference_meta_updater` | `stock_reference_meta_updater.py` | 배치 B: 식별·상세 메타, ETF 구성종목·배당 |
| `price_metrics_updater` | `stock_price_metrics_updater.py` | 배치 A: 거래량·기간수익률 등 가격 파생 |
| `data_aggregate` | `collect_data.py` | 일별 원장 → 주/월 재집계 |
| `asset_summary` | `slack_asset_summary.py` | 전용 슬랙 채널의 한국시간 날짜별 자산 요약 본문과 스레드 갱신. 메시지 ID는 `system_config.asset_summary_slack`에 저장 |
| `market_hours_analysis` | `analyze_market_hours.py` | |
| `us_market_stocks` / `aus_market_stocks` | `update_us_market_stocks.py` / `update_aus_market_stocks.py` | `index_constituents` 갱신. 구성종목 수가 범위 밖이면 저장 안 하고 실패. 미국 배치는 S&P500(SPY 공식 구성종목)·나스닥100(나스닥 공식 API)을 적재하고, 두 명단 중 종목풀 미등록 종목을 슬랙으로 알림 |
| `kor_market_stocks` | `update_kor_market_stocks.py` | 한국 지수 명단 갱신 후 시장 화면 「통합」 명단(코스피·코스닥 시총 상위) 중 종목풀 미등록 종목을 슬랙으로 알림 |
| `us_market_etfs` | `update_us_market_etfs.py` | 미국 ETF 마켓 목록(`etf_market_master`) — KIS 미국 마스터 + yfinance |
| `market_breadth` | `collect_market_breadth.py` | 한국 마감 후·미국 마감 후 |
| `db_backup` | `backup_mongo_full.py` | 로컬 폴더라 LOCAL 전용 |
| `holdings_alarm`, `strategy_mix_notify`, `strategy_trade_notify`, `slack_mock_order_notify`, `live_24h_slack`, `leverage_ma_cross`, `broker_balance_sync` 등 | 동명 스크립트 | 알림·동기화 |

- `strategy_trade_notify`는 종목의 마지막 거래 구간이 끝나면 주문·체결에 사용되지 않은 자동 슬랙 신호 스레드를 정리한다. 수동 신호는 세션 정리에서 제외한다. 시간은 `config.INSTRUMENT_TRADING_WINDOWS`를 사용한다.

## 배포

- Node 앱(`:3000`)이 로컬 FastAPI(`:8000`)를 호출한다.
- GitHub Actions([deploy.yml](../.github/workflows/deploy.yml), `upgrade` 브랜치 push · `workflow_dispatch`)가 맥미니 Self-Hosted Runner(`runs-on: self-hosted`)에서 직접 실행한다: `git reset --hard origin/upgrade` → `requirements.txt`·`package*.json` 변경 시 의존성 설치 → `npm --prefix web run build` → `launchctl kickstart -k` 로 `fastapi` → `web` → `scheduler` 순차 재시작 및 헬스체크.
- scheduler 는 종료 신호(배포·재시작)를 받으면 새 배치를 더 가져가지 않고, data 레인 배치는 바로 중단해 대기열 맨 앞으로 되돌려 새 scheduler 가 먼저 다시 실행하며, light 레인(알림·주문, 수 초~1분)은 중간에 끊으면 정합성이 깨질 수 있어 끝날 때까지 기다린다(`infra/server_scheduler.py`). 맥미니 plist(`~/Library/LaunchAgents/com.momentum-etf.scheduler.plist`)의 `ExitTimeOut` 은 macOS 상한인 60초로 적용된다(90 이상을 넣어도 60) — 이보다 긴 light 배치(드묾)는 강제 종료되고 stale 정리가 알린다. 배포 플래그(`batch_locks.__deploy__`)는 화면 「배포 중」 표시용이다.
- 배포 중 빌드·헬스체크가 실패하면 직전 커밋(`PREV_SHA`)으로 자동 롤백·재시작해 앱과 배치를 복구하고 슬랙 실패 알림을 보낸다.
- 폐기된 기능은 배포 때 서버 DB 문서·crontab 라인도 같이 지운다.

## 도메인 · DNS · 인증서

| 항목 | 값 |
| --- | --- |
| 서비스 주소 | https://invest.jason.ai.kr |
| 등록기관 | 가비아 (`jason.ai.kr`) |
| DNS 관리 | Cloudflare — **레코드는 여기서만**(가비아 DNS 화면은 반영 안 됨) |
| 레코드 · 터널 | CNAME `invest` → Cloudflare Tunnel(`cloudflared`) → 맥미니 `http://127.0.0.1:3000` |
| 인증서 | Cloudflare Edge SSL/TLS |

코드에 도메인을 적지 않는다. 서버 `.env` 의 **`APP_BASE_URL` 하나**가 단일 소스(슬랙 링크 `utils/notification.app_link`, 구글 OAuth 콜백 `web/lib/auth.ts`). 구글 OAuth 콘솔의 리디렉션 URI 도 같은 주소.

## 환경변수 (`.env`)

| 키 | 용도 |
| --- | --- |
| `APP_BASE_URL` | 서비스 주소 단일 소스 |
| `APP_TYPE` | `Local` 이면 로컬 워커·락 소유자 |
| `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID` | 알림 |
| `SLACK_SIGNING_SECRET` | 슬랙 인터랙션 웹훅(`/api/slack/interactions`) 서명 검증. 슬랙 앱의 Interactivity Request URL 이 `{APP_BASE_URL}/api/slack/interactions` |
| `DART_API_KEY` | OpenDART 재무 조회 |
| `NAMU_PLUG_API_KEY`, `NAMU_PLUG_API_SECRET` | 나무증권 API — 키 하나로 여러 계좌(연동에 계좌번호 저장) |
| `{계좌ID}_API_KEY`, `{계좌ID}_API_SECRET` | 메리츠증권 API — 키가 증권사 계좌 하나에 묶여 우리 계좌 ID 를 이름으로 쓴다(예: `m_1_API_KEY`, 대소문자 그대로) |
| `MERITZ_MAC_ADDRESS` | 메리츠 요청 필수 헤더 `mac_address`(대문자 16진수 12자리). 프로세스가 읽는 MAC 은 실행마다 달라 고정값으로 둔다 |
| `KRX_DATA_ID`, `KRX_DATA_PW` | KRX 로그인 헬퍼(보관용, 자동 수집은 약관 금지) |
| MongoDB 접속 | `utils/db_manager.py` 참조 |

## 겪은 문제

| 증상 | 원인 | 조치 |
| --- | --- | --- |
| KOR 가격 캐시가 30종목쯤에서 hang | KRX 호출 빈도 IP 차단(분당 수백 회 요청) | `stock_price_cache_updater.py` 의 `KOR_FETCH_TARGET_SECONDS` 로 종목당 최소 간격 유지. 차단되면 값을 늘린다 |
| 로그인만 `redirect_uri_mismatch` | 웹 프로세스가 옛 `APP_BASE_URL` 을 물고 있음 | `.env` 수정 후 `com.momentum-etf.web` 재시작 |
| DNS 레코드가 반영 안 됨 | 가비아에 등록함 | Cloudflare 에 등록 |
