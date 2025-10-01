# momentum-etf

ETF 추세추종 전략 기반의 트레이딩 시뮬레이션 및 분석 도구

간단한 운용, 시뮬레이션, 일일 액션 산출을 위한 스크립트 모음입니다. 모든 결과는 투자 참고용이며, 실제 투자 결정에 대한 책임은 투자자 본인에게 있습니다.

## 구성 개요 / 폴더 구조

- `logic/`: 매매 전략(로직) 정의 및 추천 파이프라인
  - `strategies/momentum/`: 이동평균 기반 모멘텀 전략
    - `backtest.py`: 백테스트 실행 엔진
    - `recommends.py`: 전략별 추천 생성 로직(결정 테이블)
    - `shared.py`: 공통 유틸리티
  - `recommends/`: 추천 파이프라인과 유틸 (루트 recommends 이관)
    - `pipeline.py`: 추천 생성 파이프라인 진입점(`main`, `generate_signal_report`)
    - `history.py`: 보유일/쿨다운 계산 유틸
    - `schedule.py`: 개장여부/다음 거래일/스케줄 타깃 날짜 계산
    - `logger.py`: 추천 전용 파일 로거
    - `market.py`: 시장 레짐 상태 문자열 생성(웹 UI 헤더용)
- `utils/`: 공통 유틸리티 모듈
  - `data_loader.py`: 데이터 로딩 및 API 호출
  - `indicators.py`: 기술적 지표 계산 (이동평균, SuperTrend, ATR 등)
  - `report.py`: 리포트, 로그 포맷팅 및 테이블 렌더링
  - `db_manager.py`: 데이터베이스 관리
  - `account_registry.py`: 계좌 관리
- `scripts/`: 각종 유틸리티 및 분석 스크립트 모음
  - `update_price_cache.py`: 국가별 종목 OHLCV 데이터를 캐시에 선다운로드/증분 갱신
  - `categorize_etf.py`: AI를 이용한 ETF 섹터 자동 분류
- `pages/`: Streamlit 웹앱 페이지들
- `data/`: 데이터 저장소
  - `kor/`, `aus/`, `coin/`: 국가별 데이터
- `run.py`: 메인 실행 진입점
- `test.py`: 과거 구간 백테스트 실행 (로직은 `logic/backtest/runner.py`로 위임)
- `tune.py`: 파라미터 튜닝 (별도 프로세스 실행, 로직은 `logic/tune/runner.py`에 위치)
- `settings.py`: 모든 전략에 공통으로 적용되는 전역 설정

## 문서

- [추천 규칙 명세](docs/recommend-rules.md)
- [개발 규칙(개발자 가이드)](docs/development-rules.md)

## 설치 및 준비

### 1) Python 가상환경 구성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2) 데이터베이스 준비
- **MongoDB**: 연결 정보(`MONGO_DB_CONNECTION_STRING`)를 환경 변수에 설정

### 3) 환경 변수 설정 (선택사항)
`.env` 파일을 생성하여 다음 변수들을 설정할 수 있습니다:
```env
MONGO_DB_CONNECTION_STRING=mongodb://localhost:27017/momentum_etf
GOOGLE_API_KEY=your_google_api_key
KOR_SLACK_WEBHOOK=your_slack_webhook_url
AUS_SLACK_WEBHOOK=your_slack_webhook_url
```

### 4) 서버 시간대 설정 (필수)
배포 서버의 시스템 시간이 KST(Asia/Seoul)와 동기화되어 있어야 추천 기준일과 로그 파일이 올바르게 생성됩니다.

- **시간대 지정**
  ```bash
  sudo timedatectl set-timezone Asia/Seoul
  ```
- **NTP 동기화 활성화**
  ```bash
  sudo timedatectl set-ntp true
  ```
- **설정 확인**
  ```bash
  timedatectl status
  ```

명령 실행 후에는 관련 서비스(APScheduler 등)를 재시작하여 새 시간 설정이 반영되도록 하세요.

## 주요 사용법

### 1) 웹앱으로 현황 확인
웹 브라우저를 통해 오늘의 현황을 시각적으로 확인하고, 거래 내역, 종목 등 데이터를 관리합니다.

```bash
python run.py
```

### 2) 실시간 추천 조회 (CLI)
과거 시뮬레이션 없이 "현재 보유 + 오늘 추천"를 바탕으로 다음 거래일에 대한 매매 추천를 제안합니다.


### 3) 백테스트 실행 (CLI)
과거 구간에 대해 백테스트를 실행합니다.

```bash
python cli.py <계좌코드> --test
```

예시:
```bash
python cli.py b1 --test
python cli.py m1 --test
```

### 4) 파라미터 튜닝 (CLI)
`cli.py`를 통해 파라미터 튜닝을 실행하여 각 전략의 최적 파라미터를 찾습니다.

```bash
python cli.py <계좌코드> --tune
```

**주의사항:**
- 매우 많은 조합을 테스트하므로 실행에 오랜 시간이 걸릴 수 있습니다
- 스크립트 상단에서 테스트할 파라미터 범위를 조절할 수 있습니다
- 최종적으로 최고 CAGR, 최저 MDD, 최고 Calmar Ratio(위험 조정 수익률) 등을 기록한 파라미터와 성과를 각각 출력합니다

### 결과 파일 및 로그 경로

- **추천 결과(요약/상세) 저장**
  - DB 저장: `utils.db_manager.save_signal_report_to_db()`로 저장되어 웹앱에서 조회됩니다
  - 파일 저장(상세 로그): `results/signal_{account}_{YYYY-MM-DD}.log`
- **추천 전용 파일 로그**
  - 경로: `logs/YYYY-MM-DD.log` (`logic/signals/logger.py`)
  - 내용: 추천 생성 과정의 디테일/디버그 로그
- **백테스트 로그**
  - 경로: `logs/test_{country}_{account}.log`
  - 트리거: `cli.py --test` 실행 시 `test.py`에서 파일 로깅
- **튜닝 로그**
  - 경로: `logs/tune_{country}_{account}.log`
  - 트리거: `cli.py --tune` (별도 프로세스 실행)

### 5) ETF 섹터 분류 (AI 사용)
`scripts/categorize_etf.py` 스크립트를 실행하여 `data/<국가코드>/etf_raw.txt` 파일의 ETF들을 AI를 이용해 섹터별로 자동 분류하고 `data/<국가코드>/etf_categorized.csv` 파일에 저장합니다.

```bash
python scripts/categorize_etf.py <국가코드>
```

**사전 준비:**
- `pip install google-generativeai python-dotenv` 라이브러리 설치 필요
- Google AI Studio에서 API 키 발급 후 `.env` 파일에 `GOOGLE_API_KEY` 설정

### 6) 스케줄러로 자동 실행 (APScheduler)
장 마감 이후 자동으로 현황을 계산하고(교체매매 추천 포함) 슬랙(Slack)으로 알림을 보낼 수 있습니다.

1. 의존성 설치: `pip install -r requirements.txt`
2. (선택) 환경 변수로 스케줄/타임존 설정:
   - `SCHEDULE_ENABLE_KOR|AUS` = `1`/`0` (기본 1)
   - `SCHEDULE_KOR_CRON` = `"10 18 * * 1-5"` (서울 18:10 평일)
   - `SCHEDULE_AUS_CRON` = `"10 18 * * 1-5"` (시드니 18:10 평일)
   - `SCHEDULE_KOR_TZ` = `Asia/Seoul`, `SCHEDULE_AUS_TZ` = `Australia/Sydney`
   - `RUN_IMMEDIATELY_ON_START` = `1` 이면 시작 시 즉시 한 번 실행
   - `SCHEDULE_ENABLE_CACHE` = `1`/`0` (기본 1)
   - `SCHEDULE_CACHE_CRON` = `"30 3 * * *"` (서울 03:30)
   - `SCHEDULE_CACHE_TZ` = `Asia/Seoul`
   - `CACHE_START_DATE` = `2020-01-01` (캐시 초기화 시작일 기본값)
   - `CACHE_COUNTRIES` = `kor,aus,coin`
3. 실행: `python aps.py`

가격 캐시만 따로 갱신하려면:
```bash
python scripts/update_price_cache.py --country all --start 2020-01-01
```

### 7) 급등주 찾기 (선택사항)
pykrx 라이브러리를 사용하여 한국 시장의 급등 ETF를 찾아봅니다.

```bash
python scripts/find.py --type etf --min-change 3.0
```

## 전략/로직 요약

### 매매 추천
- **매수 추천**: 가격이 지정된 기간의 이동평균선 위에 있을 때
- **매도 추천**:
  - **추세이탈**: 가격이 이동평균선 아래로 내려갈 때
  - **손절**: 보유 수익률이 손절 기준을 하회할 때

### 공통 리스크 관리 규칙 (백테스트)
- **가격기반손절(CUT)**: 보유수익률 ≤ `HOLDING_STOP_LOSS_PCT`
- **쿨다운**: 매수/매도 후 `COOLDOWN_DAYS` 동안 반대 방향 거래 금지

## 설정 체계

### 공통 설정
모든 국가가 동일하게 사용하는 전역 파라미터를 파일에서 관리합니다:
- `MARKET_REGIME_FILTER_ENABLED`: 시장 레짐 필터 사용 여부
- `MARKET_REGIME_FILTER_TICKER`: 레짐 필터 지수 티커
- `MARKET_REGIME_FILTER_MA_PERIOD`: 레짐 필터 이동평균 기간
- `HOLDING_STOP_LOSS_PCT`: 보유 손절 비율
- `COOLDOWN_DAYS`: 거래 쿨다운 기간

**주의**: `HOLDING_STOP_LOSS_PCT`는 양수로 입력해도 자동으로 음수로 저장/해석됩니다. (예: 10 → -10)

### 국가별 전략 파라미터
- `portfolio_topn`: 포트폴리오 최대 보유 종목 수
- `ma_period`: 이동평균 기간
- `replace_weaker_stock`: 약한 종목 교체 여부
- `replace_threshold`: 종목 교체 임계값

각 국가별로 DB에 저장되어 해당 국가 현황/백테스트에 반영됩니다.

## 코드 구조 개선사항

### 최근 리팩토링(2025-09)
1. **signals 분리/이관**: 루트 `signals.py` 삭제. 추천 파이프라인/스케줄/로거/히스토리/마켓 상태를 `logic/signals/`로 모듈화
   - 파이프라인: `logic/signals/pipeline.py` (`main`, `generate_signal_report`)
   - 스케줄: `logic/signals/schedule.py` (거래일/개장여부/스케줄 날짜)
   - 로거: `logic/signals/logger.py` (파일 로깅)
   - 히스토리: `logic/signals/history.py` (보유일/쿨다운)
   - 마켓 상태: `logic/signals/market.py` (웹 UI 헤더용 문자열)
2. **포맷팅/정밀도 일원화**: 금액/퍼센트/표 렌더링은 `utils.report`, 요약 문구는 `utils.notification` 사용으로 통일
3. **벤치마크/스케줄/로거 분리**: 레이어 간 의존성 정리로 테스트/유지보수 용이성 향상
4. **백테스트/튜닝 모듈화**:
   - 백테스트 러너: `logic/backtest/runner.py` (현재 `test.py`를 위임)
   - 튜닝 러너: `logic/tune/runner.py` (A안: `tune.py`를 별도 프로세스로 실행, 로직은 runner에 위치)

### 최근 개선된 기능들
1. **중복 코드 제거**: 이동평균 계산 로직을 `utils/indicators.py`의 공통 함수로 통합
2. **함수명 개선**: 더 직관적이고 이해하기 쉬운 함수명과 변수명 사용
3. **주석 한글화**: 모든 주석을 한글로 번역하고 상세한 설명 추가
4. **코드 최적화**: 중복된 데이터 처리 로직을 공통 함수로 분리하여 성능 향상
5. **타입 힌트 개선**: 더 명확한 타입 힌트 추가

### 주요 공통 함수들
- `calculate_moving_average_signals()`: 이동평균 기반 추천 계산
- `calculate_ma_score()`: 이동평균 대비 수익률 점수 계산

## 주의/제약사항

- 거래일 판정/개장여부는 `logic/signals/schedule.py`에서 처리합니다. 캘린더/공휴일 변동 시 판단이 달라질 수 있습니다.
- 장중 실시간 가격 조회(네이버/거래소 API 등)는 외부 사이트 변경 시 실패할 수 있습니다. 비공식 소스는 안정성을 보장하지 않습니다.
- 모든 결과는 참고용이며, 실거래 적용 전 리스크/수수료/세금/체결 슬리피지 등을 반드시 반영해 재검증하세요.

## 개발 참고사항 (Developer Notes)

### 데이터 조회 로직 (Data Fetching Logic)
`logic/signals/pipeline.py`에서 여러 종목의 시세 데이터를 조회할 때, 외부 API(yfinance 등)의 특성상 반드시 **순차 처리**해야 할 구간이 있습니다. 무분별한 병렬화는 요청 실패나 데이터 오염을 유발할 수 있으니 주의하세요.

### 코드 스타일 가이드
- 함수명과 변수명은 명확하고 직관적으로 작성
- 모든 함수에는 상세한 docstring 추가
- 주석은 한글로 작성하여 가독성 향상
- 타입 힌트를 적극 활용하여 코드 안정성 확보
