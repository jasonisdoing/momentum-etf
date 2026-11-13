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

미국 ETF 구성종목은 `utils/stock_meta_updater.py`의 공통 수집 경로를 거쳐 저장한다. 공식 연동은 ETF 이름으로 정한 발행사(`utils/data_source_catalog.py`의 발행사 규칙)의 것만 호출한다 — 한 운용사 사이트가 막혀도 다른 발행사 ETF 수집은 영향을 받지 않는다. Cloudflare로 막힌 사이트는 `utils/http_session.py`의 `browser_get`이 브라우저 지문을 바꿔 가며 받는다. 공식 연동은 `services/invesco_us_service.py`·`services/ishares_us_service.py`·`services/ishares_au_service.py`·`services/spdr_us_service.py`·`services/vaneck_us_service.py`·`services/vistashares_us_service.py`·`services/ark_us_service.py`·`services/firsttrust_us_service.py`·`services/fidelity_us_service.py`·`services/tortoise_us_service.py`·`services/globalx_us_service.py`·`services/globalx_au_service.py`·`services/vaneck_au_service.py`·`services/victoryshares_us_service.py`·`services/vanguard_us_service.py`·`services/direxion_us_service.py`·`services/roundhill_us_service.py`·`services/alps_us_service.py`·`services/marketdesk_us_service.py`·`services/wisdomtree_us_service.py`·`services/schwab_us_service.py`·`services/proshares_us_service.py`·`services/amplify_us_service.py`, 호주 SPDR 전체 구성종목은 `services/spdr_au_service.py`, SPDR 공시 검증은 `services/spdr_holdings_service.py`, 운용사 공통 XLSX 판독은 `services/xlsx_holdings_service.py`가 담당한다. Global X 공식 페이지 JSON 판독은 `services/globalx_page_service.py`에서 공유한다. iShares 미국·호주의 상품 명단·CSV 검증·정규화는 `services/ishares_holdings_service.py`, Filepoint CSV 검증·정규화는 `services/filepoint_holdings_service.py`, 외부 소스 목록은 `utils/data_source_catalog.py`가 담당한다. 실물 보유 상품은 `services/us_physical_etf_service.py`에서 확인한 공식 공시를 `holdings_cache.disclosure`에 저장하며 주식 구성종목 수집과 구분한다. 구성종목의 상장 시장을 식별할 수 없거나 시세 조회 대상이 아닌 자산은 목록에 유지하되 가격 조회에서 제외한다. 미국 상장 명단은 `utils/us_etf_market_service.py`의 KIS 공통 명단을 사용한다.

### 로컬 실행

```bash
python run_local_dev.py            # FastAPI(8000) + Next dev(3000)
python infra/server_scheduler.py   # 배치 스케줄러 (crontab 파싱 → APScheduler)
```

## 2. 화면과 모듈

| 화면 | 역할 | 서비스 |
| --- | --- | --- |
| `/pools-rank` | 종목풀 순위(추세%)와 종목 관리 | `utils/rankings.py`, `core/strategy/metrics.py` |
| `/pools-risk-return` | 종목풀 종목별 CAGR·MDD 산점도 | `utils/pool_risk_return_service.py` |
| `/pools-settings` | 종목풀 설정 | `utils/pool_settings_store.py` |
| `/pools-backtest` | 종목풀 백테스트 | `utils/pool_signal_backtest_service.py` |
| `/strategy-momentum` | 모멘텀 전략(주간) 선정·백테스트·튜닝 | `utils/momentum_service.py`, `momentum_backtest.py`, `momentum_tuning.py` |
| `/strategy-new-high` | 신고가 전략 선정·백테스트·튜닝 | `utils/new_high_service.py`, `new_high_backtest.py`, `new_high_tuning.py` |
| `/strategy-mix` | 계좌 평가액 기준 합성 운용 — 회수·채우기(합성 백테스트 없음, 백테스트는 각 전략 화면에만 있다) | `utils/strategy_mix_service.py`(계좌·슬리브 설정), `utils/strategy_mix_positions.py`(목표·액션), `utils/mix_sleeve.py`(전략별 현재 상태 어댑터) |
| `/strategy-trade` | 계좌 설정에서 사고팔기(`trade_enabled`)를 켠 운영 계좌(켤 때 증권사 API 연동·한국·미국 계좌인지 검증, 끈 계좌의 전략은 `load_settings`에서 빠져 화면·알림·주문·원장 배치 어디에도 안 돈다)에 그 국가의 개별주·ETF 종목풀 종목을 직접 등록하는 분할 매매 전략(백테스트 없음). 국가별 규칙(금액 표기·호가·주문일자·거래소·마감 경고)은 `utils/strategy_trade_market.py`·`core/strategy/trade/rules.py`가 단일 소스이고, 계좌의 일 주문 한도 `api_order_max_amount`는 계좌 통화 기준이다. 설정은 `system_config.strategy_trade_settings`(계좌별 슬랙 스위치는 `accounts.{계좌}`, 전략과 가격 출처는 `strategies`). 합성과 별개로 계좌 단위로 운영한다. 다음 회차를 계좌 통화 예수금(잔고 동기화 값)으로 살 수 없으면 한도 소진으로 매수 신호·주문을 막는다. 운영 주문은 전략의 운용 계좌로만 나간다. 보유 회차와 회차별 체결가는 평균단가 역산이 아니라 증권사 일별 체결 조회를 재생한 회차 원장(`strategy_trade_ledger`, 슬랙 신호 주문·수동 주문 구분 없이 반영 — 마지막으로 확정한 전 거래일 마감 회차(기준점, `strategy_trade_ledger_anchors`)에서 이어 재생하고, 기준점이 없거나 잔고와 맞지 않으면 최근 30일을 재생하되 그 전 보유는 체결가를 모르는 기초분으로 두어 모두 팔렸을 때만 확정한다. 지난 날짜 조회 결과는 `strategy_trade_day_fills`, 계산한 원장은 `strategy_trade_ledgers` 컬렉션에 저장 — `strategy_trade_notify` 배치가 슬랙 스위치와 무관하게 먼저 새로 계산하고 화면은 저장본만 읽으며, 주문 직전 확인만 그 순간 다시 계산)이 정하며, 체결 이력이 증권사 잔고와 맞지 않는 전략은 평균단가로 대신하지 않고 오류로 표시한다. 증권사 앱에 같은 종목·구분의 미체결 수동 주문이 있으면 슬랙 주문을 막는다(`strategy_trade_orders._require_no_manual_order`). 종목 표시 이름은 한국은 종목명, 미국은 티커다(`strategy_trade_service.trade_display_name`). 신호는 같은 회차·구분을 그날 한 번만 보내고(재알림 없음) 그 메시지(`sent_log.signal_ts`)의 스레드에 회차별 주문 메시지(슬랙 버튼, 주문이 이미 있는 회차는 제외)로 이어지며(스레드에서 첫 주문이 접수되면 신호 메시지에 모래시계 반응을 달고 주문 메시지가 모두 체결되면 ✅로 바꾼다 — 슬랙 앱에 `reactions:write` 권한 필요, 스레드 기록은 `system_config.strategy_trade_settings.signal_threads`), 사고팔기가 보내는 모든 슬랙 메시지의 ts 는 `strategy_trade_slack_messages` 컬렉션에 남고(`utils/strategy_trade_threads.send_thread_message`), 신호 회수(가격 복귀·설정 변경·사고팔기에서 제외·장 마감 미사용·화면 슬랙 뱃지)는 모두 `strategy_trade_threads.withdraw_signal` 하나를 거쳐 주문이 없는 신호만 본문을 표준 한 줄로 바꾸고 답글을 지운다 주문 직전 서버가 그 회차의 보유 상태(가격은 확인 모달이 신호 대비 %로 보여 주고 사람이 승인)와 같은 회차·구분 중복(진행 중인 주문)을 막는다(`strategy_trade_orders.place_order`) | `utils/strategy_trade_service.py`, `strategy_trade_notify.py`, `strategy_trade_threads.py`, `strategy_trade_ledger.py`, `core/strategy/trade/` |
| `/leverage-settings` | 레버리지 이동평균 크로스 설정·튜닝 | `leverage/` |
| `/account-settings` | 계좌 메타(숨김 `hidden` 계좌는 `load_live_account_docs`에서 빠져 자산·선택·집계·배치 어디에도 안 나오며, 현금·보유·원금이 모두 0일 때만 켠다)·증권사 연동·합성 슬리브별 종목풀·보유종목 알림 On/Off. 증권사 연동 커넥터는 `broker_api_service.PROVIDERS` 가 단일 소스다 — 나무증권은 키 하나로 여러 계좌를 보므로 연동에 증권사 계좌번호(`account_no`)를 저장하고, 메리츠증권은 키가 계좌에 묶여(`per_account_keys`, 키 이름 `{계좌ID}_API_KEY`) API 가 계좌번호를 주지 않아 사용자가 입력한 표시용 마스킹 번호(`account_mask`)만 저장한다. 메리츠는 잔고 동기화(국내·미국 계좌 — 계좌 국가와 다른 잔고, 계좌 현금 통화에 없는 예수금, 소수점 잔고는 오류로 막는다)와 국내주식 사고팔기 주문(호가·KRX 지정가·취소·주문내역, 증권사 경고 응답은 자동 재전송하지 않는다)이 붙어 있고 모의투자 서버가 없어 운영 계좌만 연동된다(`services/meritz_api_service.py`, 토큰은 `system_config` 의 `meritz_token:{계좌ID}` 로 공유). 계좌에는 구분(`env`: 운영 `live`·모의 `mock`, 만든 뒤 변경 불가)이 있고 **모의 계좌는 이 화면에만 나온다** — 계좌 목록을 쓰는 곳은 `load_live_account_docs`·`settings_loader`(운영만)를 거치며, 운영 계좌의 API 주문은 계좌 스위치·정규장·하루 건수/금액 한도를 통과해야 나간다(`strategy_trade_orders._require_order_allowed`, 모의 계좌는 적용 안 함). 모의 계좌에는 합성·사고팔기·알림을 켤 수 없고, 모의 계좌는 증권사 `03` 계좌만·운영 계좌는 그 밖만 연동된다. 사고팔기 주문 경로(잔고·호가·주문·취소·주문내역)는 계좌 ID 로 증권사를 고르는 `services/broker_trade_service.py` 하나를 거치고 주문 기록·회차 원장도 계좌 ID 로 묶는다. 계좌 행의 「API 주문」 모달은 연동된 계좌로 사고팔기 상대 1호가 지정가 주문·취소·상태 확인을 한다(모달의 직접 주문은 매도만, 매수는 슬랙 주문 메시지의 확인 모달로만) — **운영 계좌는 실제 주문이 나가며**(빨강 테두리·배지), 모의 계좌는 모의투자 서버로만 간다(초록). 나무증권 서버는 요청이 아니라 계좌 구분(`account_env`: 계좌 설정에 연동돼 있고 구분이 증권사 계좌구분 03 과 맞아야 통과)으로 정하며, SDK 는 서버를 전역 환경변수로 읽어 조회와 섞이므로 쓰지 않고 **직접 요청**한다, 주문 기록은 `strategy_trade_orders` 컬렉션. 슬랙 버튼→모달→웹훅(`/api/slack/interactions`, 로그인 제외·서명 검증은 FastAPI) 왕복도 같은 모의 주문 경로를 타며, 모든 액션(주문·취소)은 확인 모달을 거치고 메시지 버튼이 상태에 따라 바뀐다. 접수 뒤 체결·취소·거부·만료(미체결은 당일 주문이라 접수일이 지났거나 정규장이 끝났으면 만료로 본다)는 `slack_mock_order_notify` 배치(1분)가 스레드로 알리고 메시지 버튼을 걷어내며, 마감(15:30) 직전(15:20~)까지 미체결인 주문은 한 번 경고하고 취소 버튼을 다시 띄운다 | `utils/account_settings_store.py`, `utils/holdings_alarm_service.py`, `services/broker_api_service.py`, `services/meritz_api_service.py`, `services/broker_order_service.py`, `services/broker_trade_service.py`, `utils/strategy_trade_orders.py`, `utils/slack_mock_orders.py` |
| `/assets`, `/holdings*` | 자산·보유 | `utils/holdings_detail_service.py`, `portfolio_master` |
| `/daily`, `/weekly`, `/monthly`, `/yearly`, `/dashboard`, `/snapshots` | 일별 원장과 집계 | `utils/daily_fund_service.py` 등 |
| `/market-trend`, `/market-calendar`, `/market`, `/live-24h` | 시장 지수 추세·캘린더·지표 | `utils/market_trend_service.py`, `utils/market_calendar_service.py`, `services/toss_market_service.py` 등 |
| `/kor-market-stock`, `/us-market-stock`, `/aus-market-stock`, `/kor-market-etf` | 시장별 종목 탐색·종목풀 추가 | `index_constituents`, KIS 마스터 |
| `/ticker`, `/compare` | 개별 종목 상세·비교 | `services/etf_holdings_service.py` |
| `/batch`, `/system` | 배치 수동 실행·상태 | `utils/system_service.py`, `utils/batch_queue.py` |
| `/data-tables` | DB 컬렉션 카탈로그 — 분류·크기·고아 데이터 | `utils/data_table_catalog.py` |
| `/m` | 폰 전용. 모바일 전용 API 는 만들지 않는다(`web/app/m/mobile-data.ts` 가 기존 API 합성) | |

사고팔기의 구버전 화면·배치가 공유 DB를 더 이상 읽지 않으면 `strategy_trade_settings.accounts.*.pools`를 일괄 제거한다. `strategies.*.pool`은 가격 출처와 기존 주문 연결에 사용한다.
`strategy_trade_settings.strategies.*.sort_order`는 계좌별 사고팔기 목록의 표시 순서다.
`strategy_trade_settings.accounts.*.groups`는 사고팔기 표의 계좌별 표시 그룹이며 자산 표의 그룹과 별도로 저장한다.

전략 공용 모듈:
- 전략 시작일은 모멘텀의 `pool_settings.MOMENTUM_START_DATE`, 신고가·포트폴리오의 풀별 `start_date`에 저장한다. 검증은 `utils/strategy_settings.py`, 개별 운용 현황과 합성은 같은 저장일로 계산한다.
- `utils/strategy_settings.py` — 선택지 밖 저장값을 첫 선택지로 보정하고 내역을 돌려준다(화면만; 배치·백테스트는 엄격 검증).
- `utils/strategy_tuning.py` — 선택지 전 조합 백테스트의 지표·축별 평균·병렬 실행. 부모가 데이터를 프리로드해 spawn 워커에 넘기며, 서버 전체에 튜닝 하나만 허용한다. 새 실행·중단 API는 기존 프로세스 풀을 직접 종료한다.

## 3. DB 단일 소스

| 컬렉션 / 문서 | 내용 |
| --- | --- |
| `fx_quote_cache` | `services/price_service.py`의 통화별 환율 공유 캐시. 서버·배치가 유효기간 내 Yahoo 조회 결과를 재사용 |
| `pool_settings` | 종목풀 정의(국가·통화·벤치마크·풀 성격 stock/etf·보유 종목 수·이평 종류와 기간)와 모멘텀 설정(`MOMENTUM_START_DATE`·`ADR_FLOOR`·`ENTRY_VOL_MULT`·`INDUSTRY_CAP_PCT` — 상한은 개별주 풀만). 화면 `/pools-settings` |
| `account_settings` | 계좌 정의(운영/모의 구분 `env` 포함)·합성 배분 |
| `stock_meta` | 종목 관리 원본(버킷·종목명). 삭제는 즉시 하드 딜리트 |
| `stock_cache_meta` | 저빈도 메타(`meta_cache`)·ETF 구성종목(`holdings_cache`). 구성종목 갱신은 저장된 계산 캐시를 제거하며, 상세·비교 화면은 `services/stock_cache_service.py`의 원본 식별값으로 메모리 캐시도 검증한다 |
| `pool_strategy_backtest` | 종목풀별 저장된 전략 백테스트 결과. 풀 삭제 시 함께 제거 |
| `cache_refresh_status` | 종목풀별 가격 캐시 갱신 완료 시각. 풀 삭제 시 함께 제거 |
| 가격 캐시 | `utils/cache_utils.py` Parquet → Mongo. 요청한 풀만 읽고 다른 풀로 fallback 하지 않는다. **소유자 캐시는 `cache_<소유자>_stocks`, 소유자 없는 참조 시세(환율·레버리지 지수)는 `reference_*`** — 수명이 정반대라 이름 형식을 나눠 둔다(§6) |
| `index_constituents` | SP500/NDX100/ASX200/KOSPI200/KOSDAQ150 구성종목. 파일이 아닌 DB 인 이유: 서버 `data/` 가 읽기 전용. 한국은 공식 API 가 없어 **추종 ETF 보유종목**을 명단으로 쓴다(`KOR_INDEX_SOURCES`) |
| `system_config.momentum_settings` / `new_high_settings` | 전략 설정, 풀별 `settings_by_pool` |
| `leverage_config` / `leverage_state` | 레버리지 전략 설정·상태 |
| `portfolio_master` | 계좌 보유·메모·표시용 그룹 구분선(단일 컬렉션) |
| `daily_fund_data` → `weekly_fund_data` / `monthly_fund_data` | 일별 원장이 기준, 주/월은 재집계 |
| `batch_locks`, `batch_queue` | 배치 락·큐 |
| `daily_snapshots` | 일자별 계좌 자산 스냅샷. **계좌를 지워도 남긴다** — 지우면 지난 수익률 그래프가 바뀐다 |

컬렉션이 무엇에 속하는지는 **`utils/data_table_catalog.py` 가 단일 소스**다 (§4 참조).

거래일 캘린더만 파일(`data/country/<국가>/market_calendars.json`)이다. 없거나 범위 밖이면 에러.

## 4. 지켜야 할 규칙

- **암묵적 기본값 금지.** 설정이 없으면 에러. 화면은 폼 대신 실패 사유를 보여준다.
- **호주 티커는 `ASX:` 접두사.** 미국과 영문 티커가 겹친다(`TECH`, `HACK`). DB·API·화면까지 붙인 채로 유통하고, 외부 호출 직전에만 `utils/asx_ticker.py` 로 벗긴다. 구성종목의 상장 국가는 수집 소스 신호로만 판별(추정 금지).
- **캐시 TTL 은 `config.py` 상수 3개**(`CACHE_TTL_LIVE/COMPUTE/SLOW`)와 `utils/ttl_cache.TtlCache` 만 쓴다.
- **종목 가격 경로는 수집 어댑터(`utils/realtime_quotes.py`) → 공통 시세 서비스(`services/price_service.py`) → 봉 반영(`utils/effective_prices.py`)이다.** 화면·전략·알림은 공통 시세·확정 종가 결과만 사용한다. 거래대금·NAV 수집은 가격 판정과 별개다.
- **최신 거래일 기준 날짜(화면 `as_of`)는 모든 시장 공통 한국 날짜.** 단 **실시간을 붙일 봉의 날짜**는 그 시장 현지 날짜다(`utils/effective_prices.effective_bar_date`) — 한국 날짜로 미국 봉을 세면 하루 어긋난다.
- **제외 종목(`exclude_from_trading`)** 은 **거래 제외**다 — 순위 번호는 받고(순위·모멘텀 화면 번호가 같아야 한다), 모든 선정·백테스트 매매에서는 뺀다. 모멘텀은 `compute_signals(excluded=…)` 로 진입만 막는다.
- **배치 추가·삭제는 7곳을 함께 고친다**: `utils/system_service.py` 의 `SystemAction`·`SCHEDULE_ROWS`·`_SCRIPT_BY_ACTION`, `web/app/api/system/route.ts` 의 `allowed`, `web/lib/system-store.ts`, `web/app/batch/SystemManager.tsx`, `infra/cron/crontab`. 한 곳이 빠지면 `/batch` 에서 400.
- **새 컬렉션은 `utils/data_table_catalog.py` 에 등록한다.** 종목풀·계좌 삭제(`purge_owner`)와 고아 점검(`scan_orphans`)이 이 카탈로그 하나만 본다. 등록하지 않으면 `/data-tables` 에 **미분류**로 뜨고, 소유자를 지울 때 함께 정리되지 않는다.
- 사고팔기 체결 조회 캐시와 회차 원장·기준점은 `account_id`로 계좌 삭제에 연동한다. 증권사 주문 이력은 계좌 삭제 후에도 보존한다.
- **종목풀·계좌 삭제는 `purge_owner()` 를 쓴다.** 지울 자리를 삭제 함수가 직접 들고 있으면 컬렉션이 늘 때마다 한쪽만 갱신돼 찌꺼기가 남는다(§6).
- **소유자 삭제와 가격 캐시·저장 백테스트 쓰기는 같은 소유자 잠금으로 직렬화한다.** 정리 실패 시 소유자 문서를 지우지 않는다.
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
| 주말 나스닥선물 변동률이 -1.01%→+0.57%→+1.60%로 변경 | 시간봉 마감 추정과 연속 선물 일봉 비교 모두 quote 기준가와 불일치. 야후 금요일 일봉도 사후 변경됨 | `price_service`는 NQ=F의 단일 history metadata 응답에서 현재가·전일 기준가·제공 등락률을 함께 읽는다. 필수값 누락 시 오류, 일봉·시간봉 대체 폐기. 환율 경로 유지 |
| 튜닝 중 Mongo 타임아웃, 다른 화면까지 실패 | 워커 여러 개가 동시에 DB 읽음 | 부모 프리로드 + 워커 캐시 시딩(`strategy_tuning.seed_worker_caches`) |
| 중단·새 실행 뒤 튜닝 워커 누적 | 요청 스트림이 첫 결과를 기다리는 동안 연결 종료를 처리하지 못함 | 서버 전역 실행 핸들 + 프로세스 풀 강제 종료 |
| 호주 종목이 미국 변동률로 표시 | 티커만으로 시장 구분 불가 | `ASX:` 접두사 규칙 |
| NASDAQ100 갱신이 한 달간 조용히 실패 | 위키 문서 이동 | 구성종목 수 범위 검증, 벗어나면 실패 처리·슬랙 |
| `/batch` 버튼 400 | 배치 목록 7곳 중 누락 | 위 체크리스트 |
| 삭제한 종목풀 데이터가 수백 건 남음 | 지울 자리가 `delete_pool`/`delete_account` 안에 흩어져 컬렉션이 늘 때 누락 | `utils/data_table_catalog.py` 로 소유 관계를 선언하고 삭제·점검·화면이 공유 |
| 고아 점검이 환율 캐시를 "주인 없는 데이터" 로 잡음 | OHLCV 저장 코드를 재사용하려고 환율·레버리지도 `cache_<토큰>_stocks` 에 넣어, 소유자 캐시와 이름 형식이 같았다 | 참조 시세를 `reference_*` 로 분리(`cache_utils._REFERENCE_COLLECTIONS`). 예외 목록으로 막지 않고 형식을 갈랐다 |
| `/batch` 에 뜨는데 실행 버튼이 400 (`kor_dividend_stocks`) | `SCHEDULE_ROWS` 에만 등록되고 `SystemAction`·`_SCRIPT_BY_ACTION`·crontab 누락 | 위 7곳 체크리스트대로 채움 |

합성 액션 사유는 `core/strategy/mix/actions.py`에서 조립하며 화면·슬랙이 같은 문구를 읽는다. 포트폴리오 `current_positions`의 거래 내역 중 기준일 거래를 `SlotState.engine_trades`로 전달하며 목표 계산에는 사용하지 않는다.

`utils/share_allocation.py`에는 슬롯 엔진의 백테스트 시작 자본(`backtest_initial_capital`)만 남아 있다. 목표 금액을 정수 주수로 나누던 배분은 합성이 고정 기준금액 회수·채우기로 바뀌며 폐기했다.

2026-09-09 전환: 기존 설정값은 새 정책에 전용하지 않고 사용자 결정으로 전 계좌 0 초기화 후 옛 필드를 제거했다. 전환 전 로컬 백업은 `backups/mix-allowance-20260909T090452Z.json`이며 저장소 추적 대상이 아니다.

합성 운용 응답의 `currency`·`krw_rate`는 계좌 국가 통화와 평가에 사용한 환율이다. 화면의 총액·평가액·목표액·배분액은 원화 계산값을 해당 환율로 나누어 표시하고, 종목 가격은 원래 현지 통화 값을 표시한다.

모멘텀·신고가 `current_positions`의 `target_holdings`·`target_entries`는 실시간 표시를 적용하기 전 확정 엔진 상태다. 합성 목표는 이 상태만 읽으며, 목표 밖 계좌 종목의 평가도 엔진 기준일까지의 종가로 제한한다.

합성 계좌 설정은 `mix_harvest_pct`·`mix_refill_pct`를 함께 저장한다. 운용 기준금액은 저장하지 않고 계좌 평가액에서 계산한다.
`mix/capital_policy.py`가 화면·슬랙의 단일 판정 경로다. 미설정 계좌는 주문을 생성하지 않는다.
