# momentum-etf
ETF 추세추종 전략 기반의 트레이딩 시뮬레이션 및 분석 도구

간단한 운용, 시뮬레이션, 일일 액션 산출을 위한 스크립트 모음입니다. 모든 결과는 투자 참고용이며, 실제 투자 결정에 대한 책임은 투자자 본인에게 있습니다.

구성 개요 / 폴더 구조
---------------------

- `logic/`: 매매 전략(로직) 정의.
- `utils/`: 공통 유틸리티 모듈
  - `data_loader.py`: 데이터 로딩 및 API 호출.
  - `indicators.py`: 보조지표 계산 (SuperTrend 등).
  - `report.py`: 리포트, 로그 포맷팅 및 테이블 렌더링.
- `scripts/`: 각종 유틸리티 및 분석 스크립트 모음.
  - `update_price_cache.py`: 국가별 종목 OHLCV 데이터를 캐시에 선다운로드/증분 갱신.
- `run.py`: 메인 실행 진입점.
- `test.py`: 과거 구간 백테스트 실행 및 `logs/test.log` 생성.
- `web_app.py`: Streamlit 기반 웹 UI. 오늘의 현황을 시각적으로 보여줍니다.
- `settings.py`: 모든 전략에 공통으로 적용되는 전역 설정.
설치 및 준비
------------

1) Python 가상환경 구성(권장)
2) 데이터베이스 준비
- **MongoDB**: 연결 정보(`MONGO_DB_CONNECTION_STRING`)를 `settings.py` 또는 환경 변수에 설정해야 합니다. 모든 거래 내역, 종목 마스터, 평가금액 등은 DB에 저장됩니다.
- **종목 마스터**: 웹 앱의 `마스터정보` 탭을 통해 투자 유니버스에 포함할 종목을 관리합니다.

주요 사용법
-----------

1) 웹앱으로 현황 확인

웹 브라우저를 통해 오늘의 현황을 시각적으로 확인하고, 거래 내역, 종목 등 데이터를 관리합니다.

    python run.py

2) 오늘/다음 거래일 현황 조회 (CLI)

과거 시뮬레이션 없이 “현재 보유 + 오늘 신호”를 바탕으로 다음 거래일에 대한 매매 신호를 제안합니다.

    python cli.py <국가코드> --signal

예: `python cli.py coin --signal`

3) 백테스트 로그 생성 (CLI)

과거 구간에 대해 백테스트를 실행합니다.

    python cli.py <국가코드> --test --account <계좌코드>

예: `python cli.py coin --test --account b1`

4) 파라미터 튜닝 (CLI)

`cli.py`를 통해 파라미터 튜닝을 실행하여 각 전략의 최적 파라미터를 찾습니다.

    python cli.py <국가코드> --tune

- 주의: 매우 많은 조합을 테스트하므로 실행에 오랜 시간이 걸릴 수 있습니다.
- 스크립트 상단에서 테스트할 파라미터 범위를 조절할 수 있습니다.
- 최종적으로 최고 CAGR, 최저 MDD, 최고 Calmar Ratio(위험 조정 수익률) 등을 기록한 파라미터와 성과를 각각 출력합니다.

5) ETF 섹터 분류 (AI 사용)

`scripts/categorize_etf.py` 스크립트를 실행하여 `data/<국가코드>/etf_raw.txt` 파일의 ETF들을 AI를 이용해 섹터별로 자동 분류하고 `data/<국가코드>/etf_categorized.csv` 파일에 저장합니다.

    # momentum-etf
ETF 추세추종 전략 기반의 트레이딩 시뮬레이션 및 분석 도구

간단한 운용, 시뮬레이션, 일일 액션 산출을 위한 스크립트 모음입니다. 모든 결과는 투자 참고용이며, 실제 투자 결정에 대한 책임은 투자자 본인에게 있습니다.

구성 개요 / 폴더 구조
---------------------

- `logic/`: 매매 전략(로직) 정의.
- `utils/`: 공통 유틸리티 모듈
  - `data_loader.py`: 데이터 로딩 및 API 호출.
  - `indicators.py`: 보조지표 계산 (SuperTrend 등).
  - `report.py`: 리포트, 로그 포맷팅 및 테이블 렌더링.
- `scripts/`: 각종 유틸리티 및 분석 스크립트 모음.
  - `update_price_cache.py`: 국가별 종목 OHLCV 데이터를 캐시에 선다운로드/증분 갱신.
- `run.py`: 메인 실행 진입점.
- `test.py`: 과거 구간 백테스트 실행 및 `logs/test.log` 생성.
- `web_app.py`: Streamlit 기반 웹 UI. 오늘의 현황을 시각적으로 보여줍니다.
- `settings.py`: 모든 전략에 공통으로 적용되는 전역 설정.
설치 및 준비
------------

1) Python 가상환경 구성(권장)
2) 데이터베이스 준비
- **MongoDB**: 연결 정보(`MONGO_DB_CONNECTION_STRING`)를 `settings.py` 또는 환경 변수에 설정해야 합니다. 모든 거래 내역, 종목 마스터, 평가금액 등은 DB에 저장됩니다.
- **종목 마스터**: 웹 앱의 `마스터정보` 탭을 통해 투자 유니버스에 포함할 종목을 관리합니다.

주요 사용법
-----------

1) 웹앱으로 현황 확인

웹 브라우저를 통해 오늘의 현황을 시각적으로 확인하고, 거래 내역, 종목 등 데이터를 관리합니다.

    python run.py

2) 오늘/다음 거래일 현황 조회 (CLI)

과거 시뮬레이션 없이 “현재 보유 + 오늘 신호”를 바탕으로 다음 거래일에 대한 매매 신호를 제안합니다.

    python cli.py <국가코드> --signal

예: `python cli.py coin --signal`

3) 백테스트 로그 생성 (CLI)

과거 구간에 대해 백테스트를 실행합니다.

    python cli.py <국가코드> --test --account <계좌코드>

예: `python cli.py coin --test --account b1`

4) 파라미터 튜닝 (CLI)

`cli.py`를 통해 파라미터 튜닝을 실행하여 각 전략의 최적 파라미터를 찾습니다.

    python cli.py <국가코드> --tune

- 주의: 매우 많은 조합를 테스트하므로 실행에 오랜 시간이 걸릴 수 있습니다.
- 스크립트 상단에서 테스트할 파라미터 범위를 조절할 수 있습니다.
- 최종적으로 최고 CAGR, 최저 MDD, 최고 Calmar Ratio(위험 조정 수익률) 등을 기록한 파라미터와 성과를 각각 출력합니다.

5) ETF 섹터 분류 (AI 사용)

`scripts/categorize_etf.py` 스크립트를 실행하여 `data/<국가코드>/etf_raw.txt` 파일의 ETF들을 AI를 이용해 섹터별로 자동 분류하고 `data/<국가코드>/etf_categorized.csv` 파일에 저장합니다.

    python scripts/categorize_etf.py <국가코드>

- **사전 준비**: `pip install google-generativeai python-dotenv` 라이브러리 설치가 필요합니다.
- **API 키 설정**: 프로젝트 루트 디렉터리에 `.env` 파일을 만들고 API 키를 저장합니다.
  - `GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"` (Google AI Studio에서 발급)
- 실행 시 `etf_categorized.csv` 파일에 `category` 컬럼이 추가/업데이트됩니다.

6) 스케줄러로 자동 실행 (APScheduler)

장 마감 이후 자동으로 현황을 계산하고(교체매매 신호 포함) 슬랙(Slack)으로 알림을 보낼 수 있습니다.

1. 의존성 설치: `pip install -r requirements.txt`
2. (선택) 환경 변수로 스케줄/타임존 설정
   - `SCHEDULE_ENABLE_KOR|AUS|COIN` = `1`/`0` (기본 1)
   - `SCHEDULE_KOR_CRON` = `"10 18 * * 1-5"` (서울 18:10 평일)
   - `SCHEDULE_AUS_CRON` = `"10 18 * * 1-5"` (시드니 18:10 평일)
   - `SCHEDULE_COIN_CRON` = `"5 0 * * *"` (매일 00:05)
   - `SCHEDULE_KOR_TZ` = `Asia/Seoul`, `SCHEDULE_AUS_TZ` = `Asia/Seoul`, `SCHEDULE_COIN_TZ` = `Asia/Seoul`
   - `RUN_IMMEDIATELY_ON_START` = `1` 이면 시작 시 즉시 한 번 실행
   - `SCHEDULE_ENABLE_CACHE` = `1`/`0` (기본 1)
   - `SCHEDULE_CACHE_CRON` = `"30 3 * * *"` (서울 03:30)
   - `SCHEDULE_CACHE_TZ` = `Asia/Seoul`
   - `CACHE_START_DATE` = `2020-01-01` (캐시 초기화 시작일 기본값)
   - `CACHE_COUNTRIES` = `kor,aus,coin`
3. 실행: `python s.py`

가격 캐시만 따로 갱신하려면:

```
python scripts/update_price_cache.py --country all --start 2020-01-01
```

슬랙(Slack) 알림은 `.env` 파일에 `KOR_SLACK_WEBHOOK` 등 국가별 웹훅 URL을 설정하면 자동으로 동작합니다.

5) (선택) 급등주 찾기

pykrx 라이브러리를 사용하여 한국 시장의 급등 ETF를 찾아봅니다.

    python scripts/find.py --type etf --min-change 3.0


전략/로직 요약
-------------


  - 가격이 지정된 기간의 이동평균선 위에 있을 때.
- **매도 신호**:
  - **추세이탈**: 가격이 이동평균선 아래로 내려갈 때.

### 공통 리스크 관리 규칙 (백테스트)
  - **가격기반손절(CUT)**: 보유수익률 ≤ `HOLDING_STOP_LOSS_PCT`
  - **쿨다운**: 매수/매도 후 `COOLDOWN_DAYS` 동안 반대 방향 거래 금지

설정 체계
-------------------

- 공통 설정(웹앱 ‘설정’ 탭 → 공통 설정)
  - 모든 국가가 동일하게 사용하는 전역 파라미터를 DB에 저장하여 관리합니다.
  - MARKET_REGIME_FILTER_ENABLED / MARKET_REGIME_FILTER_TICKER / MARKET_REGIME_FILTER_MA_PERIOD
  - HOLDING_STOP_LOSS_PCT / COOLDOWN_DAYS
  - 주의: HOLDING_STOP_LOSS_PCT는 양수로 입력해도 자동으로 음수로 저장/해석됩니다. (예: 10 → -10)
- 국가별 전략 파라미터(웹앱 각 국가 탭 → 설정):
  - portfolio_topn, ma_period,
  - replace_weaker_stock, replace_threshold
  - 각 국가별로 DB에 저장되어 해당 국가 현황/백테스트에 반영됩니다.

주의/제약
--------

- `signals.py`는 `pandas_market_calendars`를 이용하여 거래일 판정을 수행합니다. 캘린더 업데이트 지연 시 휴장 판단이 달라질 수 있습니다.
- `signals.py` 경량 모드는 “내일 할 일” 계산에 특화되어 있습니다.
- `signals.py`는 장중 실행 시, 네이버 금융 웹 스크레이핑을 통해 실시간 가격을 가져오려고 시도합니다. 이 기능은 네이버 웹사이트 구조 변경 시 동작하지 않을 수 있으며, 비공식적인 방법이므로 안정성을 보장하지 않습니다.
- 모든 결과는 참고용이며, 실거래 적용 전 리스크/수수료/세금/체결 슬리피지 등을 반드시 반영해 재검증하세요.

## 개발 참고사항 (Developer Notes)

### 데이터 조회 로직 (Data Fetching Logic)

`signals.py`에서 여러 종목의 시세 데이터를 조회할 때, `yfinance` API의 안정성 문제로 인해 반드시 **순차적으로 처리**해야 합니다. 병렬로 처리할 경우 API 요청이 실패하거나 데이터가 오염될 수 있습니다. 이 문제는 과거에 발생하여 의도적으로 순차 처리로 변경된 사항이므로, 향후 코드 수정 시 이 부분을 반드시 고려해야 합니다.

- **사전 준비**: `pip install google-generativeai python-dotenv` 라이브러리 설치가 필요합니다.
- **API 키 설정**: 프로젝트 루트 디렉터리에 `.env` 파일을 만들고 API 키를 저장합니다.
  - `GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"` (Google AI Studio에서 발급)
- 실행 시 `etf_categorized.csv` 파일에 `category` 컬럼이 추가/업데이트됩니다.

6) 스케줄러로 자동 실행 (APScheduler)

장 마감 이후 자동으로 현황을 계산하고(교체매매 신호 포함) 슬랙(Slack)으로 알림을 보낼 수 있습니다.

1. 의존성 설치: `pip install -r requirements.txt`
2. (선택) 환경 변수로 스케줄/타임존 설정
   - `SCHEDULE_ENABLE_KOR|AUS|COIN` = `1`/`0` (기본 1)
   - `SCHEDULE_KOR_CRON` = `"10 18 * * 1-5"` (서울 18:10 평일)
   - `SCHEDULE_AUS_CRON` = `"10 18 * * 1-5"` (시드니 18:10 평일)
   - `SCHEDULE_COIN_CRON` = `"5 0 * * *"` (매일 00:05)
   - `SCHEDULE_KOR_TZ` = `Asia/Seoul`, `SCHEDULE_AUS_TZ` = `Australia/Sydney`, `SCHEDULE_COIN_TZ` = `Asia/Seoul`
   - `RUN_IMMEDIATELY_ON_START` = `1` 이면 시작 시 즉시 한 번 실행
   - `SCHEDULE_ENABLE_CACHE` = `1`/`0` (기본 1)
   - `SCHEDULE_CACHE_CRON` = `"30 3 * * *"` (서울 03:30)
   - `SCHEDULE_CACHE_TZ` = `Asia/Seoul`
   - `CACHE_START_DATE` = `2020-01-01` (캐시 초기화 시작일 기본값)
   - `CACHE_COUNTRIES` = `kor,aus,coin`
3. 실행: `python s.py`

가격 캐시만 따로 갱신하려면:

```
python scripts/update_price_cache.py --country all --start 2020-01-01
```

슬랙(Slack) 알림은 `.env` 파일에 `KOR_SLACK_WEBHOOK` 등 국가별 웹훅 URL을 설정하면 자동으로 동작합니다.

5) (선택) 급등주 찾기

pykrx 라이브러리를 사용하여 한국 시장의 급등 ETF를 찾아봅니다.

    python scripts/find.py --type etf --min-change 3.0


전략/로직 요약
-------------


  - 가격이 지정된 기간의 이동평균선 위에 있을 때.
- **매도 신호**:
  - **추세이탈**: 가격이 이동평균선 아래로 내려갈 때.

### 공통 리스크 관리 규칙 (백테스트)
  - **가격기반손절(CUT)**: 보유수익률 ≤ `HOLDING_STOP_LOSS_PCT`
  - **쿨다운**: 매수/매도 후 `COOLDOWN_DAYS` 동안 반대 방향 거래 금지

설정 체계
-------------------

- 공통 설정(웹앱 ‘설정’ 탭 → 공통 설정)
  - 모든 국가가 동일하게 사용하는 전역 파라미터를 DB에 저장하여 관리합니다.
  - MARKET_REGIME_FILTER_ENABLED / MARKET_REGIME_FILTER_TICKER / MARKET_REGIME_FILTER_MA_PERIOD
  - HOLDING_STOP_LOSS_PCT / COOLDOWN_DAYS
  - 주의: HOLDING_STOP_LOSS_PCT는 양수로 입력해도 자동으로 음수로 저장/해석됩니다. (예: 10 → -10)
- 국가별 전략 파라미터(웹앱 각 국가 탭 → 설정):
  - portfolio_topn, ma_period,
  - replace_weaker_stock, replace_threshold
  - 각 국가별로 DB에 저장되어 해당 국가 현황/백테스트에 반영됩니다.

주의/제약
--------

- tactics.py는 pandas_market_calendars를 이용하여 거래일 판정을 수행합니다. 캘린더 업데이트 지연 시 휴장 판단이 달라질 수 있습니다.
- tactics.py 경량 모드는 “내일 할 일” 계산에 특화되어 있습니다.
- tactics.py는 장중 실행 시, 네이버 금융 웹 스크레이핑을 통해 실시간 가격을 가져오려고 시도합니다. 이 기능은 네이버 웹사이트 구조 변경 시 동작하지 않을 수 있으며, 비공식적인 방법이므로 안정성을 보장하지 않습니다.
- 모든 결과는 참고용이며, 실거래 적용 전 리스크/수수료/세금/체결 슬리피지 등을 반드시 반영해 재검증하세요.
