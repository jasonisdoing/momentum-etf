# 개발자 가이드

유지보수를 위한 문서다. 무엇이 있고 어디가 단일 소스인지, 코드만 봐서는 알 수 없는 결정만 적는다.
화면 사용법·설정값·컬럼은 적지 않는다 — 코드가 단일 소스다.

## 1. 구조

| 층 | 위치 | 역할 |
| --- | --- | --- |
| 웹 | `web/app/<화면>/`, `web/app/api/` | Next.js 화면과 API 프록시. 로직 없는 프록시는 `web/lib/fastapi-proxy.ts` 의 `createFastApiProxy` 로 쓴다 |
| 백엔드 | `fastapi_app/routes/` | 내부 FastAPI(`/internal/...`). 웹만 호출한다 |
| 도메인 서비스 | `utils/<도메인>_service.py` | 화면·API·스크립트가 공용으로 쓰는 계산·오케스트레이션 |
| 저장/IO | `utils/<도메인>_store.py` · `_io.py` | DB·파일 읽기/쓰기 |
| 외부 연동 | `services/` | 외부 API 1소스 1파일, 자체 TTL 캐시 포함. 화면·유틸은 외부를 직접 부르지 않는다 |
| 순수 계산 | `core/strategy/` | 네트워크·DB 없는 지표 계산 |
| 배치 | `scripts/` + `infra/cron/crontab` + `infra/server_scheduler.py` | 운영은 [server_infrastructure.md](server_infrastructure.md) |
| 정적 데이터 | `data/` | 거래일 캘린더 등 연 단위로만 바뀌는 것. 서버에선 읽기 전용 마운트 |
| 레버리지 | `leverage/` | 폐기된 `leverage-switching` 앱에서 이전. 자체 엔진, 시세·슬랙은 `utils` 재사용 |

기존 파일은 옮기지 않는다 — 신규 코드부터 위 기준으로 수렴시킨다.

### 로컬 실행

```bash
python run_local_dev.py            # FastAPI(8000) + Next dev(3000)
python infra/server_scheduler.py   # 배치 스케줄러 (crontab 파싱 → APScheduler)
```

## 2. 화면과 모듈

| 화면 | 역할 | 서비스 |
| --- | --- | --- |
| `/pools-rank` | 종목풀 순위(추세%)와 종목 관리 | `utils/rankings.py`, `core/strategy/metrics.py` |
| `/pools-settings` | 종목풀 설정 | `utils/pool_settings_store.py` |
| `/pools-backtest` | 종목풀 백테스트 | `utils/pool_signal_backtest_service.py` |
| `/strategy-momentum` | 모멘텀 전략(주간) 선정·백테스트·튜닝 | `utils/momentum_service.py`, `momentum_backtest.py`, `momentum_tuning.py` |
| `/strategy-new-high` | 신고가 전략 선정·백테스트·튜닝 | `utils/new_high_service.py`, `new_high_backtest.py`, `new_high_tuning.py` |
| `/strategy-mix` | 두 전략 합성 — 목표 비중·오늘의 액션·백테스트(열람 전용) | `utils/strategy_mix_service.py` |
| `/leverage-settings` | 레버리지 이동평균 크로스 설정·튜닝 | `leverage/` |
| `/account-settings` | 계좌 메타·증권사 연동·합성 슬리브별 종목풀·보유종목 알림 On/Off | `utils/account_settings_store.py`, `utils/holdings_alarm_service.py` |
| `/assets`, `/asset-helper`, `/holdings*` | 자산·보유·목표 비중 | `utils/asset_helper_service.py`, `portfolio_master` |
| `/daily`, `/weekly`, `/monthly`, `/yearly`, `/dashboard`, `/snapshots` | 일별 원장과 집계 | `utils/daily_fund_service.py` 등 |
| `/market-trend`, `/market`, `/live-24h` | 시장 지수 추세·지표 | `services/toss_market_service.py` 등 |
| `/kor-market-stock`, `/us-market-stock`, `/aus-market-stock`, `/kor-market-etf` | 시장별 종목 탐색·종목풀 추가 | `index_constituents`, KIS 마스터 |
| `/ticker`, `/compare` | 개별 종목 상세·비교 | `services/etf_holdings_service.py` |
| `/batch`, `/system` | 배치 수동 실행·상태 | `utils/system_service.py`, `utils/batch_queue.py` |
| `/data-tables` | DB 컬렉션 카탈로그 — 분류·크기·고아 데이터 | `utils/data_table_catalog.py` |
| `/m` | 폰 전용. 모바일 전용 API 는 만들지 않는다(`web/app/m/mobile-data.ts` 가 기존 API 합성) | |

전략 공용 모듈:
- 전략 시작일은 모멘텀의 `pool_settings.MOMENTUM_START_DATE`, 신고가·포트폴리오의 풀별 `start_date`에 저장한다. 검증은 `utils/strategy_settings.py`, 개별 운용 현황과 합성은 같은 저장일로 계산한다.
- `utils/strategy_settings.py` — 선택지 밖 저장값을 첫 선택지로 보정하고 내역을 돌려준다(화면만; 배치·백테스트는 엄격 검증).
- `utils/strategy_tuning.py` — 선택지 전 조합 백테스트의 지표·축별 평균·병렬 실행. 부모가 데이터를 프리로드해 spawn 워커에 넘기며, 서버 전체에 튜닝 하나만 허용한다. 새 실행·중단 API는 기존 프로세스 풀을 직접 종료한다.

## 3. DB 단일 소스

| 컬렉션 / 문서 | 내용 |
| --- | --- |
| `pool_settings` | 종목풀 정의(국가·통화·벤치마크·풀 성격 stock/etf·보유 종목 수·순위용 이평). 화면 `/pools-settings` |
| `account_settings` | 계좌 정의·합성 배분. 추가/삭제는 DB 직접 |
| `stock_meta` | 종목 관리 원본(버킷·종목명). 삭제는 즉시 하드 딜리트 |
| `stock_cache_meta` | 저빈도 메타(`meta_cache`)·ETF 구성종목(`holdings_cache`) |
| 가격 캐시 | `utils/cache_utils.py` Parquet → Mongo. 요청한 풀만 읽고 다른 풀로 fallback 하지 않는다. **소유자 캐시는 `cache_<소유자>_stocks`, 소유자 없는 참조 시세(환율·레버리지 지수)는 `reference_*`** — 수명이 정반대라 이름 형식을 나눠 둔다(§6) |
| `index_constituents` | SP500/NDX100/ASX200/KOSPI200/KOSDAQ150 구성종목. 파일이 아닌 DB 인 이유: 서버 `data/` 가 읽기 전용. 한국은 공식 API 가 없어 **추종 ETF 보유종목**을 명단으로 쓴다(`KOR_INDEX_SOURCES`) |
| `system_config.momentum_settings` / `new_high_settings` | 전략 설정, 풀별 `settings_by_pool` |
| `leverage_config` / `leverage_state` | 레버리지 전략 설정·상태 |
| `portfolio_master` | 계좌 보유·목표 비중·자산 헬퍼 설정(단일 컬렉션) |
| `daily_fund_data` → `weekly_fund_data` / `monthly_fund_data` | 일별 원장이 기준, 주/월은 재집계 |
| `batch_locks`, `batch_queue` | 배치 락·큐 |
| `daily_snapshots` | 일자별 계좌 자산 스냅샷. **계좌를 지워도 남긴다** — 지우면 지난 수익률 그래프가 바뀐다 |

컬렉션이 무엇에 속하는지는 **`utils/data_table_catalog.py` 가 단일 소스**다 (§4 참조).

거래일 캘린더만 파일(`data/country/<국가>/market_calendars.json`)이다. 없거나 범위 밖이면 에러.

## 4. 지켜야 할 규칙

- **암묵적 기본값 금지.** 설정이 없으면 에러. 화면은 폼 대신 실패 사유를 보여준다.
- **호주 티커는 `ASX:` 접두사.** 미국과 영문 티커가 겹친다(`TECH`, `HACK`). DB·API·화면까지 붙인 채로 유통하고, 외부 호출 직전에만 `utils/asx_ticker.py` 로 벗긴다. 구성종목의 상장 국가는 수집 소스 신호로만 판별(추정 금지).
- **캐시 TTL 은 `config.py` 상수 4개**(`CACHE_TTL_LIVE/COMPUTE/SLOW/META`)와 `utils/ttl_cache.TtlCache` 만 쓴다.
- **실시간 가격 외에는 종목 캐시에서 읽는다.** 화면 진입 시 외부 원천을 다시 부르지 않는다.
- **최신 거래일 기준 날짜는 모든 시장 공통 한국 날짜.**
- **제외 종목(`exclude_from_ranking`)** 은 비교 기준일 뿐 — 모든 선정·백테스트 유니버스에서 제외.
- **배치 추가·삭제는 7곳을 함께 고친다**: `utils/system_service.py` 의 `SystemAction`·`SCHEDULE_ROWS`·`_SCRIPT_BY_ACTION`, `web/app/api/system/route.ts` 의 `allowed`, `web/lib/system-store.ts`, `web/app/batch/SystemManager.tsx`, `infra/cron/crontab`. 한 곳이 빠지면 `/batch` 에서 400.
- **새 컬렉션은 `utils/data_table_catalog.py` 에 등록한다.** 종목풀·계좌 삭제(`purge_owner`)와 고아 점검(`scan_orphans`)이 이 카탈로그 하나만 본다. 등록하지 않으면 `/data-tables` 에 **미분류**로 뜨고, 소유자를 지울 때 함께 정리되지 않는다.
- **종목풀·계좌 삭제는 `purge_owner()` 를 쓴다.** 지울 자리를 삭제 함수가 직접 들고 있으면 컬렉션이 늘 때마다 한쪽만 갱신돼 찌꺼기가 남는다(§6).
- **폐기는 코드·DB·crontab·문서까지 전부 제거.**
- 실시간 값을 주는 Next API 는 응답에 `Cache-Control: no-store`.

### 자산 수익률 정의

- 기간 수익률 = `period_profit / 직전 기간 총자산`. 분자는 누적손익 차분이라 입출금이 제거된다.
- 누적 수익률 = `cumulative_profit / total_principal` (ROI).
- 합계 행은 `/daily` 입력 입출금으로 정확하고, 계좌별 행은 스냅샷 원금 차이로 추정한다 — 인출일에 원금 수정을 같은 날 반영하지 않으면 왜곡된다.
- 프론트는 백엔드 값을 그대로 쓴다(자체 계산 금지).

## 5. 화면 UI 표준

`/pools-rank` 레이아웃이 기준. 메뉴 헤더(메뉴명 / 요약) → 메인 헤더(`appMainHeader`: 주 제어 / 특별 버튼) → CRUD 액션 헤더(버튼 2개 이상일 때만 별도 줄) → 테이블(AG Grid, `AppAgGrid`).
컨트롤 높이 30px 은 `globals.css` 공통 클래스에서만 정한다. 공용 셀 표기는 `web/lib/grid-cells.tsx`. 업종·섹터 노출은 풀 성격(`pool_kind`)이 기준.

## 6. 겪은 문제

| 증상 | 원인 | 조치 |
| --- | --- | --- |
| 튜닝 중 Mongo 타임아웃, 다른 화면까지 실패 | 워커 여러 개가 동시에 DB 읽음 | 부모 프리로드 + 워커 캐시 시딩(`strategy_tuning.seed_worker_caches`) |
| 중단·새 실행 뒤 튜닝 워커 누적 | 요청 스트림이 첫 결과를 기다리는 동안 연결 종료를 처리하지 못함 | 서버 전역 실행 핸들 + 프로세스 풀 강제 종료 |
| 호주 종목이 미국 변동률로 표시 | 티커만으로 시장 구분 불가 | `ASX:` 접두사 규칙 |
| NASDAQ100 갱신이 한 달간 조용히 실패 | 위키 문서 이동 | 구성종목 수 범위 검증, 벗어나면 실패 처리·슬랙 |
| `/batch` 버튼 400 | 배치 목록 7곳 중 누락 | 위 체크리스트 |
| 삭제한 종목풀 데이터가 수백 건 남음 | 지울 자리가 `delete_pool`/`delete_account` 안에 흩어져 컬렉션이 늘 때 누락 | `utils/data_table_catalog.py` 로 소유 관계를 선언하고 삭제·점검·화면이 공유 |
| 고아 점검이 환율 캐시를 "주인 없는 데이터" 로 잡음 | OHLCV 저장 코드를 재사용하려고 환율·레버리지도 `cache_<토큰>_stocks` 에 넣어, 소유자 캐시와 이름 형식이 같았다 | 참조 시세를 `reference_*` 로 분리(`cache_utils._REFERENCE_COLLECTIONS`). 예외 목록으로 막지 않고 형식을 갈랐다 |
| `/batch` 에 뜨는데 실행 버튼이 400 (`kor_dividend_stocks`) | `SCHEDULE_ROWS` 에만 등록되고 `SystemAction`·`_SCRIPT_BY_ACTION`·crontab 누락 | 위 7곳 체크리스트대로 채움 |
