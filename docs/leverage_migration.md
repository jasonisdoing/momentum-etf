# leverage(레버리지 스위칭) 이전 현황

별도 앱 `leverage-switching` 의 전략을 momentum-etf 로 흡수한 작업의 현황·구조·남은 작업 정리.
(leverage-switching 은 폐기 예정 — README 에 RETIRED 표기, deploy 비활성화, VM cron uninstall 안내 완료.)

## 1. 목표
- 레버리지 스위칭(switch) 전략을 momentum-etf 에 **서브메뉴/배치**로 흡수.
- 데이터·Slack·스케줄러는 momentum-etf 인프라 재사용. UI 와 설정 DB화는 이후 단계.
- 최종적으로 leverage-switching 앱 폐기.
- 무한매수법(buy)은 운영하지 않기로 결정 → 코드·설정·문서에서 모두 제거함.

## 2. 완료된 작업

### 2.1 전략 코어 패키지 — `leverage/`
- `leverage/engine/` — 엔진(backtest/runner.py, signals.py, settings.py, tune/runner.py). 원본 `logic/` 를 그대로 이식.
- `leverage/data_adapter.py` — **핵심 글루**. 원본 `data.py`(pykrx/yfinance 직접) 대신 momentum-etf `utils.data_loader.fetch_ohlcv(ticker_type="etf")` 로 시/종가 조회. `current_trading_day()`(=`get_latest_trading_day`), `realtime_price()`(=`fetch_naver_realtime_price`) 제공.
- `leverage/constants.py` — `INITIAL_CAPITAL_KRW`, `MARKET_SCHEDULES`, 경로 상수(`CONFIG_DIR/ZRESULTS_DIR/STATE_DIR`).
- `leverage/notify.py` — Slack 전송을 momentum-etf `utils.notification.send_slack_message_v2` 로 위임(블록 구성 로직은 유지).
- `leverage/report.py` — 표/금액 포맷(원본 utils/report.py).
- `leverage/config/switch.json` — 전략 설정(파일 기반, 추후 DB화).
- `leverage/{backtest,recommend,tune}.py` — CLI 엔트리. 실행: `python -m leverage.backtest switch` 등.

### 2.2 검증
- `python -m leverage.backtest switch` → 원본과 **CAGR/MDD 1원 단위 동일**(이식 무결성 확인).
- `python -m leverage.recommend switch [--slack]` → 정상. Slack 은 momentum-etf 채널로 전송 확인.

### 2.3 동작 규칙(장중)
- 신호/포지션/평단/보유일 = **마지막으로 닫힌 거래일 종가**로 확정(`drop_today` + `current_trading_day`).
- 현재가·일간·누적·드로다운·회복필요·설명 = **오늘 실시간**으로 매시간 갱신.
- `fetch_ohlcv` 가 이미 실시간가를 반영하므로 별도 실시간 호출 불필요.

### 2.4 배치/스케줄 통합
- `infra/cron/crontab` — `leverage_switch` 잡 추가(평일 09:05~16:05 매시 :05).
- `scripts/leverage_recommend_switch.py` — 무인자 래퍼(스케줄러/run_batch 호환). 루트를 sys.path 에 추가 후 `leverage.recommend` 호출.
- `/batch` 수동 트리거 등록 — `utils/system_service.py`(SystemAction Literal, SCHEDULE_ROWS, _SCRIPT_BY_ACTION) + 프론트(`web/app/api/system/route.ts` allowed Set, `web/lib/system-store.ts` SystemAction, `web/app/batch/SystemManager.tsx` SystemJobKey).
- `infra/cron/run_batch.py` — `SUCCESS_NOTIFICATION_DISABLED_JOBS` 에 `leverage_switch` 추가(성공 알림 끔, 실패 알림 유지).

## 3. 실행/테스트 방법
- CLI: `python -m leverage.backtest switch` / `python -m leverage.recommend switch --slack` / `python -m leverage.tune switch`
- `/batch` 페이지: "레버리지 스위칭 추천" 클릭 → 큐 → worker(`scripts/leverage_recommend_switch.py`) → Slack
- 자동: 서버 scheduler 가 crontab 의 `leverage_switch` 를 enqueue → worker 실행

## 4. 남은 작업
1. **UI 서브메뉴** — recommend 결과(dict)를 보여줄 FastAPI route(`fastapi_app/routes/leverage.py`) + service(`utils/leverage_service.py`) + Next 페이지(`web/app/leverage-switching/`) + `AppShell.tsx` 메뉴. (momentum-etf 패턴: market-trend 라우트/서비스/페이지 복제)
2. **설정 파일 → DB/UI 전환** — `leverage/config/*.json` → momentum-etf 설정 시스템(settings_loader/DB)으로. UI 에서 파라미터 편집.
3. **`tune.py` 검증** — momentum-etf 데이터로 `python -m leverage.tune switch` 1회 확인(현재 backtest/recommend 만 검증됨).
4. **leverage-switching 앱 폐기 마무리** — VM 에서 `install.sh --uninstall`, repo 아카이브.

(완료: 문서 갱신 — `docs/server_infrastructure.md` 에 leverage 배치 반영됨.
 결정/완료: 무한매수법(buy)은 운영 안 함 → 코드·설정·문서에서 제거.)

## 5. 주의/참고
- 스케줄러는 `python <script.py>` 무인자 형식만 파싱(`-m`/추가 인자 불가) → 래퍼 스크립트 사용.
- 스케줄러는 **주석 처리된 cron 라인도 활성으로 파싱** → 비활성화는 라인 제거로.
- leverage 코드/스크립트는 Docker 이미지에 포함되므로 변경 시 재배포 필요. crontab/run_batch 는 `./infra/cron` 마운트.
- 데이터는 `ticker_type="etf"`(MongoDB 캐시 키). 대상 종목은 모두 한국 ETF.
