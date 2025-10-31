# momentum-etf

ETF 추세추종 전략 기반의 트레이딩 시뮬레이션 및 분석 도구

간단한 운용, 시뮬레이션, 일일 액션 산출을 위한 스크립트 모음입니다. 모든 결과는 투자 참고용이며, 실제 투자 결정에 대한 책임은 투자자 본인에게 있습니다.

## 구성 개요 / 폴더 구조

- `logic/`: 매매 전략(로직) 정의 및 추천/백테스트 파이프라인
  - `strategies/maps/`: 이동평균 기반 모멘텀 전략 구현체
    - `backtest.py`: 백테스트 실행 엔진
    - `recommend.py`: 전략별 추천 생성 로직 (계정 단위 추천)
    - `shared.py`: 공통 유틸리티
  - `recommend/`: 추천 파이프라인과 유틸리티
    - `pipeline.py`: 추천 생성 파이프라인 진입점 (`generate_account_recommendation_report`)
    - `history.py`: 보유일/쿨다운 계산 유틸
    - `schedule.py`: 개장 여부/다음 거래일/스케줄 타깃 날짜 계산
    - `logger.py`: 추천 전용 파일 로거
    - `market.py`: 시장 레짐 상태 조회 (대시보드 참고용)
  - `common/`: 추천과 백테스트에서 공통으로 사용하는 로직
    - `portfolio.py`: 보유 종목 상태 계산 공통 함수 (`count_current_holdings`, `get_hold_states` 등)
    - `signals.py`: 매수 신호 판단 공통 함수
    - `filtering.py`: 카테고리 중복 필터링 공통 함수
- `utils/`: 공통 유틸리티 모듈
  - `data_loader.py`: 데이터 로딩 및 API 호출
  - `indicators.py`: 기술적 지표 계산 (이동평균, SuperTrend, ATR 등)
  - `report.py`: 리포트, 로그 포맷팅 및 테이블 렌더링
  - `db_manager.py`: 데이터베이스 관리
  - `account_registry.py`: 계정/국가 설정 로더 및 공통 설정 헬퍼
  - `country_registry.py`: 구 코드 호환을 위한 래퍼
- `scripts/`: 각종 유틸리티 및 분석 스크립트 모음
  - `update_price_cache.py`: 국가별 종목 OHLCV 데이터를 캐시에 선다운로드/증분 갱신
  - `find.py`: 급등 ETF 검색 도구
- `app_pages/`: Streamlit 웹앱 페이지들
  - `account_page.py`: 계정별 추천/현황 페이지
  - `trade.py`: 관리자용 거래 관리 페이지 (로그인 필요)
  - `migration.py`: 계정 ID/거래 데이터 마이그레이션 페이지 (로그인 필요)
- `data/`: 데이터 저장소
  - `settings/`: 설정 파일
    - `account/*.json`: 계정별 전략 설정
    - `common.py`: 공통 설정
    - `schedule_config.json`: APScheduler 설정
  - `stocks/`: 국가별 종목 리스트 (aus.json, kor.json, us.json)
  - `results/`: 백테스트/튜닝 결과 로그
- `app.py`: Streamlit 웹앱 메인 진입점
- `run.py`: 웹앱 실행 스크립트 (app.py 래퍼)
- `recommend.py`: CLI 추천 생성 스크립트
- `backtest.py`: CLI 백테스트 실행 스크립트
- `tune.py`: CLI 파라미터 튜닝 스크립트
- `aps.py`: APScheduler 자동 실행 스크립트

## 문서

- [추천 규칙 명세](docs/recommend-rules.md)
- [개발 규칙(개발자 가이드)](docs/development-rules.md)

## 설치 및 준비

### 1) Python 가상환경 구성 (권장)

#### (A) pyenv + pyenv-virtualenv 사용

1. pyenv 설치 (macOS/Homebrew 예시)

   ```bash
   brew install pyenv pyenv-virtualenv
   ```

2. 쉘 초기화 스크립트에 pyenv 설정 추가 (예: `~/.zshrc`)

   ```bash
   eval "$(pyenv init -)"
   eval "$(pyenv virtualenv-init -)"
   ```

3. 이 저장소에서 사용할 Python 버전을 설치하고 가상환경 생성

   ```bash
   pyenv install 3.12.11  # 원하는 버전으로 변경 가능
   pyenv virtualenv 3.12.11 momentum-etf
   ```

4. 프로젝트 디렉터리에서 로컬 가상환경 지정 후 의존성 설치

   ```bash
   cd momentum-etf
   pyenv local momentum-etf
   pip install -r requirements.txt
   ```

#### (B) 표준 venv 사용

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

- `/` : 대시보드(빈 페이지)
- `/<account_id>` : 계정별 추천 페이지 (로그인 불필요)
- `/admin` : 거래 관리 페이지 (로그인 필요)
- `/migration` : 계정 ID 마이그레이션 도구 (로그인 필요)

```bash
python run.py
```

### 2) 추천 조회 (CLI)

과거 시뮬레이션 없이 "현재 보유 + 오늘 추천"를 바탕으로 다음 거래일에 대한 매매 추천를 제안합니다.
추천은 전날 종가 기준으로 계산되어 백테스트와 동일한 로직을 사용합니다.

```bash
python recommend.py <account_id> [--date YYYY-MM-DD] [--output 경로]
```

### 3) 백테스트 실행 (CLI)

과거 구간에 대해 백테스트를 실행합니다.

```bash
python backtest.py <account_id> [--output 경로]
```

### 4) 파라미터 튜닝 (CLI)

`tune.py`를 통해 파라미터 튜닝을 실행하여 각 전략의 최적 파라미터를 찾습니다.

```bash
python tune.py <account_id> [--output 경로]
```

**주의사항:**

- 매우 많은 조합을 테스트하므로 실행에 오랜 시간이 걸릴 수 있습니다
- 스크립트 상단에서 테스트할 파라미터 범위를 조절할 수 있습니다
- 최종적으로 최고 CAGR, 최저 MDD, 최고 Calmar Ratio(위험 조정 수익률) 등을 기록한 파라미터와 성과를 각각 출력합니다

### 결과 파일 및 로그 경로

모든 결과 로그는 계정별 폴더로 구조화되어 저장됩니다: `data/results/<account_id>/`

- **추천 결과(요약/상세) 저장**
  - DB 저장: `utils.db_manager.save_signal_report_to_db()`로 저장되어 웹앱에서 조회됩니다
  - 파일 저장(상세 로그): `data/results/<account_id>/recommend_{YYYY-MM-DD}.log`
- **추천 전용 파일 로그**
  - 경로: `logs/YYYY-MM-DD.log` (`logic/recommend/logger.py`)
  - 내용: 추천 생성 과정의 디테일/디버그 로그
- **백테스트 로그**
  - 경로: `data/results/<account_id>/backtest_{YYYY-MM-DD}.log`
  - 트리거: `python backtest.py <account_id>` 실행 시 자동 생성
- **튜닝 로그**
  - 경로: `data/results/<account_id>/tune_{YYYY-MM-DD}.log`
  - 트리거: `python tune.py <account_id>` 실행 시 자동 생성

### 5) 스케줄러로 자동 실행 (APScheduler)

장 마감 이후 자동으로 현황을 계산하고(교체매매 추천 포함) 슬랙(Slack)으로 알림을 보낼 수 있습니다.

1. 의존성 설치: `pip install -r requirements.txt`
2. 실행: `python aps.py`

가격 캐시만 따로 갱신하려면:

```bash
python scripts/update_price_cache.py --country all --start 2020-01-01
```

### 6) 상승중인 ETF 찾기 (선택사항)

pykrx 라이브러리를 사용하여 한국 시장의 급등 ETF를 찾아봅니다.

```bash
python scripts/find.py --type etf --min-change 3.0
```

## 시스템 아키텍처

### 핵심 개념

이 시스템은 **ETF 모멘텀 추세 추종 전략**을 기반으로 한 반자동 포트폴리오 운용 시스템입니다.
이동평균선 대비 가격 위치, 최근 수익률, 연속 상승 일수 등을 종합해 ETF별 점수를 계산하고,
상위 종목을 자동으로 추천하며 시장 위험을 감지하여 리스크를 관리합니다.

### 주요 컴포넌트

#### 1. 추세 분석 엔진 (MAPS Strategy)

- **위치**: `strategies/maps/`
- **기능**: 이동평균 기반 모멘텀 점수 계산
- **입력**: OHLCV 데이터, MA 기간, 포트폴리오 크기
- **출력**: ETF별 추천 점수 및 포지션 상태

#### 2. RSI 과매수 감지

- **위치**: `strategies/rsi/`
- **기능**: RSI 지표로 과열 종목 감지 및 매도 신호 생성
- **임계값**: 계정별 설정 가능 (일반적으로 5~30)

#### 3. 시장 레짐 모니터링

- **위치**: `logic/recommend/market.py`
- **기능**: 주요 지수(S&P 500, NASDAQ 등)의 이동평균 대비 위치 추적
- **동작**:
  - 시장 위험 시: 신규 매수 차단 또는 투자 비중 축소
  - 시장 안정 시: 정상 운영

#### 4. 포지션 관리 시스템

ETF별로 다음 상태를 추적하고 관리합니다:

| 상태 | 설명 | 트리거 조건 |
|------|------|------------|
| `WAIT` | 대기 | 조건 미충족, 제약 발생 |
| `BUY` | 신규 매수 | 상승세 뚜렷, 리스크 낮음, 슬롯 여유 |
| `HOLD` | 보유 유지 | 추세 유지 중 |
| `SELL_TREND` | 추세 이탈 매도 | 가격이 이동평균선 아래로 하락 |
| `SELL_RSI` | RSI 과매수 매도 | RSI 임계값 이하 |
| `CUT_STOPLOSS` | 손절 | 손실률이 설정 한도 초과 |
| `BUY_REPLACE` | 교체 매수 | 기존 종목보다 점수가 임계값 이상 높음 |
| `SELL_REPLACE` | 교체 매도 | 더 나은 후보로 교체 |

#### 5. 리스크 관리 레이어

다층 방어 체계로 리스크를 제어합니다:

- **카테고리 중복 방지**: 동일 섹터 ETF 중복 편입 차단
- **쿨다운 메커니즘**: 최근 거래 후 일정 기간 재거래 제한
- **RSI 과매수 관리**: 과열 종목 자동 감지 및 매도
- **데이터 검증**: 가격 데이터 누락 시 거래 중단
- **현금 관리**: 투자 가능 현금 범위 내에서만 매수

**참고**: 시장 레짐 정보는 대시보드에서 참고용으로만 표시되며, 실제 매매에는 영향을 주지 않습니다.

### 데이터 플로우

```
1. 데이터 수집 (yfinance, cache)
   ↓
2. 기술적 지표 계산 (MA, RSI, 수익률)
   ↓
3. ETF별 점수 계산 (MAPS + RSI)
   ↓
4. 포지션 상태 결정 (BUY/HOLD/SELL/REPLACE)
   ↓
5. 리스크 필터 적용 (카테고리, 쿨다운, 현금)
   ↓
6. 최종 추천 생성 및 저장 (DB + 파일)
```

### 수익률 측정 방식

시스템은 세 가지 방식으로 성과를 측정합니다:

#### (1) 백테스트

- **목적**: 전략 검증 및 파라미터 최적화
- **가격**: 다음날 시초가
- **슬리피지**: 한국 0.5%, 호주 1%, 미국 0.3%
- **특징**: 실제보다 불리한 조건으로 보수적 추정

#### (2) 가상 거래 수익률 (Momentum ETF)

- **목적**: 실제 투자 성과 측정
- **가격**: 거래일 종가
- **슬리피지**: 없음 (벤치마크와 동일 조건)
- **리밸런싱**: 동적 균등 분배

#### (3) 벤치마크 (Buy & Hold)

- **목적**: 단순 보유 전략 대비 성과 비교
- **가격**: 시작일 종가 → 최신 종가
- **슬리피지**: 없음
- **리밸런싱**: 없음

### 전략 파라미터

주요 튜닝 가능 파라미터:

| 파라미터 | 설명 | 범위 | 위치 |
|---------|------|------|------|
| `MA_PERIOD` | 이동평균 기간 | 10~100일 | `data/settings/account/*.json` |
| `PORTFOLIO_TOPN` | 포트폴리오 목표 종목 수 | 3~10개 | 계정 설정 |
| `REPLACE_SCORE_THRESHOLD` | 교체 점수 임계값 | 0~3점 | 계정 설정 |
| `OVERBOUGHT_SELL_THRESHOLD` | RSI 과매수 임계값 | 5~30점 | 계정 설정 |
| `COOLDOWN_DAYS` | 쿨다운 기간 | 0~5일 | `config.py` |
| `MARKET_REGIME_MA` | 시장 레짐 MA 기간 (참고용) | 10~100일 | 공통 설정 |

파라미터 최적화는 `tune.py`를 통해 수행합니다.

## 설정 체계

### 공통 설정

모든 국가가 동일하게 사용하는 전역 파라미터를 `config.py`에서 관리합니다:

- `COOLDOWN_DAYS`: 거래 쿨다운 기간
- `MARKET_REGIME_FILTER_TICKER_MAIN`: 전략에 적용되는 주요 레짐 필터 지수 티커
- `MARKET_REGIME_FILTER_TICKERS_AUX`: 대시보드에 참고용으로 노출할 보조 지수 리스트
- `MARKET_REGIME_FILTER_MA_PERIOD`: 시장 레짐 필터 이동평균 기간
- `MARKET_REGIME_FILTER_COUNTRY`: 레짐 필터 데이터 조회에 사용할 시장 코드(`kor`, `us` 등)
- `MARKET_SCHEDULES`: 국가별 시장 거래 시간표 (한국: 9:00-14:00, 호주: 8:00-15:00, 간격: 60분)

### 계정별 전략 파라미터

각 계정의 설정은 `data/settings/account/{account_id}.json`에 저장됩니다:

**전략 설정 (`strategy`):**

- `tuning`: 튜닝으로 찾은 최적 파라미터
  - `MA_PERIOD`: 이동평균 기간
  - `MA_TYPE`: 이동평균 타입 (SMA, EMA, HMA 등)
  - `PORTFOLIO_TOPN`: 포트폴리오 목표 종목 수
  - `REPLACE_SCORE_THRESHOLD`: 교체 점수 임계값
  - `OVERBOUGHT_SELL_THRESHOLD`: RSI 과매수 임계값
  - `COOLDOWN_DAYS`: 쿨다운 기간
  - `CORE_HOLDINGS`: 핵심 보유 종목 리스트

**표시 설정:**

- `name`: 계정 표시 이름
- `country_code`: 국가 코드 (kor, aus, us)
- `icon`: 아이콘 (이모지)
- `initial_cash`: 초기 자본금
- `slack_channel`: 슬랙 알림 채널 ID

## 코드 구조 개선사항

### 최근 리팩토링(2025-10)

1. **계정 중심 구조로 전환**: `data/settings/account/*.json` 기반으로 추천/백테스트가 동작하도록 전면 수정
2. **Streamlit 페이지 정비**: 거래 관리(`trade.py`)와 계정 마이그레이션(`migration.py`)을 분리하고 로그인 후 접근하도록 구성
3. **추천 결과 저장 방식 개선**: 계정 ID와 국가 코드 두 경로에 결과를 저장해 UI와 스케줄러가 일관된 데이터를 참조하도록 변경
   - 히스토리: `logic/recommend/history.py` (보유일/쿨다운)
   - 마켓 상태: `logic/recommend/market.py` (웹 UI 헤더용 문자열)
2. **포맷팅/정밀도 일원화**: 금액/퍼센트/표 렌더링은 `utils.report`, 요약 문구는 `utils.notification` 사용으로 통일
3. **벤치마크/스케줄/로거 분리**: 레이어 간 의존성 정리로 테스트/유지보수 용이성 향상
4. **백테스트/튜닝 모듈화**:
   - 백테스트 러너: `logic/backtest/account_runner.py`, `logic/backtest/portfolio_runner.py`
   - 튜닝 러너: `logic/tune/runner.py`
5. **수익률 계산 개선 (2025-10-19)**:
   - 백테스트: 다음날 시초가 + 슬리피지 (보수적)
   - 실제 거래: 당일 종가, 슬리피지 없음, 동적 리밸런싱
   - 벤치마크: 시작일 종가 → 최신 종가 (Buy & Hold)
6. **시스템 문서화**:
   - `SYSTEM_SUMMARY.md`: 투자자용 시스템 요약
   - `SYSTEM_DETAILS.md`: 투자자용 상세 매뉴얼
   - 웹앱에서 마크다운 파일 기반으로 설명 표시

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

- 거래일 판정/개장여부는 `logic/recommend/schedule.py`에서 처리합니다. 캘린더/공휴일 변동 시 판단이 달라질 수 있습니다.
- 모든 결과는 참고용이며, 실거래 적용 전 리스크/수수료/세금/체결 슬리피지 등을 반드시 반영해 재검증하세요.
- 백테스트는 보수적 추정(시초가 + 슬리피지)이므로 실제 성과와 차이가 있을 수 있습니다.

## 개발 참고사항 (Developer Notes)

### 데이터 조회 로직 (Data Fetching Logic)

`logic/recommend/pipeline.py`에서 여러 종목의 시세 데이터를 조회할 때, 외부 API(yfinance 등)의 특성상 반드시 **순차 처리**해야 할 구간이 있습니다. 무분별한 병렬화는 요청 실패나 데이터 오염을 유발할 수 있으니 주의하세요.

### 코드 스타일 가이드

- 함수명과 변수명은 명확하고 직관적으로 작성
- 모든 함수에는 상세한 docstring 추가
- 주석은 한글로 작성하여 가독성 향상
- 타입 힌트를 적극 활용하여 코드 안정성 확보


### 데이터 사용 시점

- **추천, 백테스트, 튜닝 모두 동일한 로직 사용**: 최근 마감된 거래일의 종가까지만 사용합니다.
- **의사결정 일관성**: 추천과 백테스트가 동일한 데이터와 로직을 사용하여 재현성을 보장합니다.
- **정보성 실시간 데이터**: 화면 표시용으로 네이버/yfinance API를 통해 현재가, NAV, 괴리율을 조회하지만, 이는 매매 의사결정에 영향을 주지 않습니다.
