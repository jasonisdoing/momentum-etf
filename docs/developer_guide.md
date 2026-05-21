# 개발자 가이드 (Developer Guide)

이 문서는 Momentum ETF 순위 시스템의 아키텍처, 데이터 흐름, 그리고 개발 시 반드시 지켜야 할 정합성 원칙을 설명합니다.

## 0. 로컬 실행

개발과 자동 배치는 **두 개의 터미널** 로 나누어 실행합니다.

```bash
# 터미널 1 — 웹 (FastAPI + Next dev)
python run_local_dev.py

# 터미널 2 — 배치 스케줄러 (APScheduler)
python run_local_scheduler.py
```

- `run_local_dev.py` 는 FastAPI(`uvicorn`, 포트 8000) 와 Next dev(`npm run dev`, 포트 3000) 를 함께 띄웁니다.
- `run_local_scheduler.py` 는 `infra/cron/crontab` 을 파싱해 APScheduler 로 자동 배치를 돌립니다.
  - `APP_TYPE=Local` 로 자동 설정되어 `batch_locks` 의 owner 가 구분됩니다.
  - 노트북이 꺼져 있었던 시간의 누락 분은 따라잡지 않습니다 (다음 예약 시각부터 동작).
  - Ctrl+C 로 깔끔히 종료됩니다.
- VM 의 cron 은 제거되어 있으므로, **자동 배치를 돌리려면 터미널 2 가 켜져 있어야** 합니다.
- 수동 1회 실행은 `/system` 화면의 버튼으로도 가능하며, 동일한 `batch_locks` 락을 사용하므로 자동 실행과 충돌하지 않습니다.

## 1. 시스템 아키텍처

### 모듈 구조
*   `backtest/`: 백테스트 전용 실행기, 스윕 설정, 결과 로그 생성 엔진
*   `core/strategy/`: 지표/점수/비중 계산 공용 전략 유틸
*   `services/`: **외부 API/데이터 연동 통합 계층**
    *   `price_service.py`: 실시간 가격/환율 오케스트레이션 및 TTL 캐시
    *   `reference_data_service.py`: KIS ETF 마스터, 종목 메타데이터, 상장일 조회
    *   `etf_holdings_service.py`: 한국 ETF 구성종목 비중을 네이버 `ETFComponent` API로 조회해 메타 캐시에 저장할 형태로 정규화합니다. 국내 구성종목은 6자리 종목코드, 해외 구성종목은 `componentReutersCode`에서 추출한 심볼을 표시용 `ticker`로 사용하고, 원본 ISIN은 `raw_code`에 저장합니다. 해외 구성종목 가격 조회는 응답 시점에 Yahoo를 사용하고 서비스 메모리 TTL 캐시를 적용합니다.
    *   `vkospi_service.py`: VKOSPI 등 외부 시장 지표 연동 및 메모리 캐시
    *   `fear_greed_service.py`: CNN 공포탐욕지수 연동 및 메모리 캐시
    *   **원칙**: 새로운 시장 지표, 가격 정보, 외부 데이터 크롤링 등은 혼동을 막기 위해 모두 이 폴더에서 각각의 서비스로 관리하고, 자체 캐시 시스템(TTL 등)을 구축합니다.
*   `utils/rankings.py`: 순위 테이블 계산 전용 유틸
*   `scripts/`: 데이터 수집, 캐시 갱신, 일별 원장 시드/집계 생성 등 유틸리티 스크립트
*   `utils/`:
    *   `cache_utils.py`: **Parquet 기반 캐시 I/O** 및 직렬화 관리
    *   `data_loader.py`: OHLCV 수집/보완 및 원천 fetch 함수
    *   `ai_summary.py`: AI용 요약 데이터 생성 공용 유틸
    *   `daily_fund_service.py`: `daily_fund_data` 일별 원장 조회/수정/주별 시드 이관
    *   `weekly_service.py`: `daily_fund_data` 기준 주별 재집계 및 `weekly_fund_data` 조회/비고 수정
    *   `monthly_service.py`: `daily_fund_data` 기준 월별 재집계 및 `monthly_fund_data` 조회/비고 수정
*   `.github/workflows/`: GitHub Actions를 이용한 일일 배포 및 자동화 정의
*   `accounts.json`: 계좌 메타데이터 단일 설정 파일. 각 계좌의 `ticker_types`는 해당 계좌가 보유할 수 있는 종목풀 목록이며, 보유종목이 종목풀에서 제거된 뒤에도 가격/메타 캐시 갱신 대상의 ticker_type을 결정하는 기준입니다.

### 데이터 파이프라인 및 캐싱
1.  **데이터 수집**: `pykrx`, `yfinance` 등을 통해 원천 데이터 수집.
2.  **Parquet 캐싱**: 수집된 데이터는 `utils/cache_utils.py`를 통해 **Apache Parquet** 포맷으로 MongoDB에 저장됩니다. (기존 Pickle 방식의 버전 충돌 문제를 해결)
3.  **서비스 오케스트레이션**:
    *   `services/price_service.py`가 실시간 가격/환율과 TTL 캐시를 관리합니다.
    *   `services/reference_data_service.py`가 KIS ETF 목록과 메타데이터 조회를 관리합니다.
    *   `services/etf_holdings_service.py`가 한국 ETF 구성종목 비중을 네이버 `ETFComponent` API로 수집하고, 응답 시점에는 메타 캐시에 저장된 구성종목에 해외 가격만 Yahoo TTL 캐시로 보조합니다.
4.  **지표 계산**: `core/strategy/metrics.py`가 이동평균과 점수를 계산.
5.  **순위 생성**: `utils/rankings.py`가 종목별 점수, 규칙별 추세, RSI, 기간 수익률을 합쳐 화면용 DataFrame 생성.

### 일별 원장 원칙

*   `daily_fund_data`는 앞으로 일/주/월 집계의 기준이 될 일별 원장 컬렉션입니다.
*   현재 초기 단계에서는 기존 `weekly_fund_data`의 종료일 row를 `daily_fund_data.date`로 시드 이관해 sparse 일별 원장으로 사용합니다.
*   이 시드 데이터는 과거 일별 복원값이 아니라, **주별 종료일 스냅샷을 일별 원장에 옮긴 값**으로 취급합니다.
*   초기 시드는 명시 스크립트 `./.venv/bin/python scripts/seed_daily_fund_data.py`로만 생성합니다. 런타임에서 자동 시드를 만들지 않습니다.
*   통합 데이터 집계는 `./.venv/bin/python scripts/collect_data.py`가 담당하며, 미래 `daily_fund_data` row를 먼저 정리한 뒤 오늘 row를 upsert 하고 이어서 `daily_fund_data`에서 주별/월별 마지막 영업일 snapshot을 읽어 `weekly_fund_data`, `monthly_fund_data`를 다시 생성합니다.
*   주별/월별 수동 수정은 `memo`만 허용하며, 금액 관련 필드는 모두 일별 원장에서 유도됩니다.

### 종목 캐시 용어

이 프로젝트에서 **종목 캐시**는 다음 두 가지를 합친 상위 개념으로 사용합니다.

1.  **가격 캐시**
    *   OHLCV, 종가 시계열, 실시간 스냅샷
    *   `utils/cache_utils.py`, `utils/data_loader.py`, `services/price_service.py`
    *   `scripts/stock_price_cache_updater.py`는 종목풀 인자를 받지 않고 항상 전체 종목풀의 가격 캐시를 갱신합니다.
2.  **메타 캐시**
    *   상장일, 배당률, 보수, 순자산총액/시가총액, 업종, ETF 구성종목 같은 저빈도 정보
    *   Mongo `stock_cache_meta` 컬렉션
    *   `utils/stock_cache_meta_io.py`, `services/stock_cache_service.py`
    *   한국 ETF 저빈도 메타와 구성종목은 `scripts/stock_meta_cache_updater.py`가 네이버 `ETFBase`, `ETFDividend`, `ETFComponent`를 조회해 `stock_cache_meta.meta_cache`, `stock_cache_meta.holdings_cache`로 저장합니다.
    *   미국 개별주는 네이버 `foreign/market/stock/global`에서 업종, 배당률, 시가총액을 조회해 `stock_meta.etf_category`와 `stock_cache_meta.meta_cache`에 저장합니다. 미국 개별주에는 보수 개념을 적용하지 않습니다.
    *   종목풀에 등록되지 않았더라도 포트폴리오 마스터에서 현재 보유 중인 티커는 계좌 `ticker_types` 기준으로 종목 메타/가격 캐시 갱신 대상에 포함됩니다.

`stock_meta` 컬렉션은 종목 관리 원본(버킷, 종목명 등)으로 유지하고, 저빈도 메타 캐시는 `stock_cache_meta`로 분리하는 것을 기본 방향으로 삼습니다. 종목 삭제는 별도 휴지통 없이 즉시 하드 딜리트를 기본으로 합니다.

국가별 거래일 캘린더는 DB가 아니라 파일 캐시로 관리합니다. 런타임은 `zcountry/{country}/market_calendars.json`만 읽고, 파일이 없거나 범위를 벗어나면 즉시 에러를 발생시킵니다.

### 서비스 사용 원칙

1.  실시간 가격/환율 조회는 `services/price_service.py`를 먼저 사용합니다.
2.  KIS ETF 마스터, 종목 메타데이터, 상장일 조회는 `services/reference_data_service.py`를 먼저 사용합니다.
3.  **외부 연동 일원화**: 시장 지표나 새로운 데이터를 외부에서 파싱/API 통신해야 한다면 `services/` 하위에 위치시키며, 무분별한 요청을 방지하기 위한 캐싱 로직을 포함해야 합니다.
4.  화면 계층과 일반 유틸은 외부 데이터 소스를 직접 호출하지 않고, 가능하면 `services/` 계층을 통해 접근합니다.
5.  `utils/data_loader.py`는 원천 fetch 함수와 OHLCV 보완 로직을 포함하지만, 신규 호출부를 작성할 때는 직접 진입점으로 우선 사용하지 않습니다.
6.  실시간 가격데이터를 제외한 값은 우선 `종목 캐시`에 저장하고 읽습니다. 화면 진입 시 외부 원천을 다시 호출하지 않고, 캐시된 메타/구성종목을 한꺼번에 읽어 응답 속도를 유지하는 것을 기본 원칙으로 삼습니다.
7.  한국 ETF 구성종목 비중은 `stock_cache_meta.holdings_cache`를 우선 사용합니다. `/ticker`는 구성종목 목록/비중을 실시간 조회하지 않습니다.
8.  배당률, 보수, 순자산총액/시가총액, 상장일 같은 저빈도 메타는 `stock_cache_meta.meta_cache`를 우선 사용합니다.
9.  앞으로 새로운 저빈도 항목(예: ETF 메타, 구성종목 속성, 기초지수 관련 부가정보)을 발견하면, 실시간 가격데이터가 아닌 이상 먼저 `stock_cache_meta`에 저장하는 방향을 우선 원칙으로 삼습니다.
10. 실시간 또는 준실시간 값을 내려주는 Next API 라우트는 요청 fetch뿐 아니라 응답 헤더에도 `Cache-Control: no-store`를 명시해 브라우저/중간 계층 캐시를 차단합니다.
11. 가격 캐시 조회는 요청한 `ticker_type` 또는 국가 캐시만 엄격하게 조회합니다. 다른 종목풀 캐시로 자동 fallback 하지 않습니다. 계좌 보유 화면처럼 여러 종목풀이 섞인 경우에만 호출부에서 전체 종목풀 조회를 명시적으로 선택합니다.

### 자산 수익률 계산 정책 (단일 출처)

자산 화면(자산 관리 `/assets`, 일별 `/daily`, 주별 `/weekly`, 월별 `/monthly`, 연별 `/yearly`, 대시보드 `/dashboard`)의 모든 수익률 지표는 아래 규칙을 따릅니다. 분모/분자 정의를 바꾸려면 이 절을 먼저 수정하고 코드를 동기화합니다.

- **기간 수익률 (일/주/월/년) — 입출금 제거 1기간 수익률**
  - 공식: `period_return_pct = period_profit / previous_total_assets × 100`
  - 분자(`period_profit`)는 `cumulative_profit`의 차분으로 계산되며, `total_principal` 누적이 입출금을 흡수해 입출금 영향이 자동 제거됩니다.
  - 분모는 직전 기간 종료 시점의 총자산으로 고정해, 입금/출금 자체가 해당 기간 수익률 기준금액을 흔들지 않게 합니다.
- **누적 수익률 — ROI(Return on Investment)**
  - 공식: `cumulative_return_pct = cumulative_profit / total_principal × 100`
  - `cumulative_profit = total_assets - total_principal - total_expense_누적`.
  - 기간 수익률을 복리 누적한 값이 아니라 투입 원금 대비 총 수익을 보는 단순 비율입니다.

화면별 매핑:

| 화면 | 일(%) | 주(%) | 월(%) | 년(%) | 누적(%) |
|------|-------|-------|-------|-------|---------|
| /daily | 입출금 제거 1일 | — | — | — | ROI |
| /weekly | — | 입출금 제거 1주 | — | — | ROI |
| /monthly | — | — | 입출금 제거 1월 | — | ROI |
| /yearly | — | — | — | 입출금 제거 1년 | ROI |
| /assets | 입출금 제거 1일 | 입출금 제거 1주 | — | — | ROI |
| /dashboard | 입출금 제거 1일 | 입출금 제거 1주 | 입출금 제거 1월 | 입출금 제거 1년 | ROI |

같은 일자에서는 모든 화면의 일(%) 값이 동일합니다.

#### 합계 행 vs 계좌별 행의 현금흐름 처리 차이 (`/assets`, `/dashboard`)

- **합계 행 (정확)**: `daily_fund_data` / `weekly_fund_data` 최신 doc 의 `daily_profit` / `weekly_profit` 을 그대로 사용합니다. 사용자가 `/daily` 화면에서 입력한 `deposit_withdrawal`, `withdrawal_personal`, `withdrawal_mom`, `nh_principal_interest` 가 분자에서 직접 차감되어 **시장 변동분만** 손익으로 잡힙니다.
- **계좌별 행 (추정)**: 계좌별 입출금 명시 데이터가 없어, `daily_snapshots` 에서 `오늘 total_principal − 어제 total_principal` 차이를 입출금으로 **추정** 합니다. 추정 한계:
  - 사용자가 인출 전에 `portfolio_master.total_principal` 을 미리 수정해버리면, 어제·오늘 스냅샷의 원금이 같아 입출금 추정이 0 으로 잡히고, 실제 인출액이 계좌별 손익에 통째로 손실로 표시될 수 있습니다.
  - 반대로 원금 단순 정정(입출금 없음)을 한 경우에도 그 차이가 입출금으로 잡혀 손익을 왜곡할 수 있습니다.
- 따라서 **계좌별 일(%) / 주(%) 는 참고용**이며, 정확한 일/주 손익은 **합계 행** 또는 `/daily` / `/weekly` 화면을 기준으로 합니다.
- 인출/입금이 발생했을 때 계좌별 행도 정확하게 보고 싶다면, 인출 발생일에는 **포트폴리오 원금 수정을 그날 안에 함께** 반영하는 운영 규칙이 필요합니다 (당일 원금 차이 = 당일 입출금).

구현 위치:

- 백엔드(Python): `utils/daily_fund_service.py`, `utils/weekly_service.py`, `utils/monthly_service.py`, `utils/yearly_service.py`, `utils/dashboard_service.py` 의 `calculate_period_return_pct` / `_apply_running_total_principal` / `_calculate_weekly_docs` / `load_dashboard_data`.
- 프론트엔드: `/assets`(`web/app/assets/AssetsManager.tsx`)는 백엔드의 `daily_return_pct`/`weekly_return_pct` 값을 그대로 사용합니다(자체 계산 금지).
- `/ticker`의 "포트폴리오 변동(%)"은 별도 지표(ETF 구성종목 가중평균)이며 본 정책과 무관합니다.

데이터 무결성:

- `total_principal`은 입출금 발생 시 즉시 반영되어야 합니다. 누락 시 모든 기간 수익률이 왜곡됩니다.
- 정책 변경 시 raw 데이터(`total_assets`, `total_principal`, `deposit_withdrawal`, `total_expense`)는 그대로 유지되고 파생 필드만 계산식이 바뀌므로, 정책 변경 자체에는 재집계가 필요하지 않습니다. 화면 새로고침 시점부터 적용됩니다.
- 과거 일별 입출금 값을 수정한 경우에는 주/월/년 raw 집계(`deposit_withdrawal`, `total_assets` 등)를 다시 만들기 위해 관련 집계 버튼을 눌러야 합니다.

## 2. 순위 화면 정합성 원칙

> **Critical**: 이 시스템은 **순위 화면이 단일 진실 원천(single source of truth)** 입니다. 화면에서 보이는 값은 계좌 종목 목록, 가격 캐시, 실제 보유 데이터로 직접 계산되어야 합니다.

### 핵심 구조
| 파일/경로 | 역할 |
|-----------------|------|
| `web/app/*` | Next.js 기반 사용자 화면과 API 라우트 |
| `utils/rankings.py` | 순위 계산과 정렬 |
| `core/strategy/metrics.py` | 이동평균 점수 계산 |
| `services/price_service.py` | 실시간 가격/환율 조회의 공식 진입점 |
| `services/reference_data_service.py` | ETF 마스터/메타데이터/상장일 조회의 공식 진입점 |
| `utils/account_notes.py` | 계좌 메모 저장/조회 |

### 핵심 일관성 체크리스트

1.  **입력 단순화**: 종목풀 설정의 `MA_TYPE`, `MA_MONTHS`를 사용하고, 순위 화면에서도 같은 단일 MA 기준만 변경할 수 있다.
2.  **정렬 기준 고정**: `점수`가 있는 종목을 `점수` 내림차순으로 정렬하고, 계산 불가 종목은 맨 아래로 보낸다.
3.  **데이터 기준**:
    *   모든 의사결정은 **판단 시점의 전일 종가 데이터**를 기준으로 함
    *   "오늘"의 순위는 "어제까지의 마감 데이터"를 보고 계산된 것임
    *   최신 거래일 판단의 기준 날짜는 모든 시장 공통으로 **한국 날짜**를 사용함
4.  **엄격한 설정 원칙 (Rule 7)**:
    *   코드에 암묵적인 기본값(fallback)을 사용하지 않습니다.
    *   필수 순위 파라미터가 누락된 경우 명확한 `ValueError`를 발생시킵니다.

## 3. 전략 설정 규칙

종목풀 설정 포맷(`pools.json`):

```json
{
  "all": {
    "TOP_N_HOLD": 3,
    "HOLDING_BONUS_SCORE": 10,
    "MA_TYPE": "ALMA",
    "MA_MONTHS": 5,
    "RSI_LIMIT": 100,
    "include": ["kor_kr", "kor_us", "kor"]
  },
  "pools": [
    {
      "order": 1,
      "ticker_type": "kor_kr",
      "icon": "🇰🇷",
      "name": "국내상장 국내",
      "country_code": "kor",
      "currency": "KRW",
      "MA_TYPE": "SMA",
      "MA_MONTHS": 10
    }
  ]
}
```

종목풀 설정의 `country_code`는 현재 `kor`, `au`, `us`를 허용합니다.

검증 원칙(현재 운영):

* 전체 종목풀: `all.TOP_N_HOLD`, `all.HOLDING_BONUS_SCORE`, `all.MA_TYPE`, `all.MA_MONTHS`, `all.RSI_LIMIT`, `all.include` 필수
* 개별 종목풀: `MA_TYPE`, `MA_MONTHS` 필수
* 필수값 누락 시 fallback 없이 명시적 에러

## 4. 테스트 및 검증

코드를 수정할 때는 다음 절차를 따르세요.

1.  **로직 수정**: `utils/rankings.py`, `core/strategy/metrics.py`, `web/app/*`, `web/lib/*`를 우선 확인
    *   가격/환율 문제면 `services/price_service.py`를 함께 확인
    *   KIS ETF 목록/메타데이터/상장일 문제면 `services/reference_data_service.py`를 함께 확인
2.  **검증**:
    *   순위 화면에서 종목풀 변경 또는 `MA` 변경 시 컬럼과 점수가 즉시 갱신되는지 확인
    *   실제 보유 종목이 녹색 행으로 표시되는지 확인
3.  **확인**:
    *   `점수`, `추세` 컬럼이 `현재가` 뒤에 표시되는지 확인

## 5. 순위 화면의 정의

**"순위(Rank)"**는 종목풀의 현재 종목 유니버스에서 `MA_TYPE`, `MA_MONTHS` 기준 점수를 계산한 결과입니다.

### 핵심 원칙
1.  **화면 기준 계산**: 순위는 별도 저장 결과를 읽지 않고, 가격 캐시와 계좌 종목으로 즉시 계산합니다.
2.  **실보유 구분**: 실제 보유 종목만 행 색상으로 표시합니다.
3.  **정렬 규칙**: `점수` 내림차순, `점수` 계산 불가 종목은 맨 아래입니다.
4.  **계좌 종목 직접 관리**: 계좌가 자신의 종목 유니버스를 직접 보유하며, 별도 종목풀 fallback은 사용하지 않습니다.
5.  **고정 종목 표시**: `exclude_from_ranking=true`인 고정 종목은 순위 번호 없이 현재 위치만 보여줍니다.

## 6. 화면 UI 표준

AG Grid 기반 주요 화면은 현재 `/pools`에서 정리한 레이아웃을 공통 기준으로 사용합니다.

### 공통 레이아웃 순서
1.  **메뉴 헤더**
    *   왼쪽: 메뉴명
    *   오른쪽: 총 개수, 선택 개수, 활성 주차 같은 정보
2.  **메인 헤더**
    *   왼쪽: 계좌 셀렉터, 보기 전환 토글, 검색/필터 같은 주 제어
    *   오른쪽: CRUD가 아닌 특별한 버튼, 예를 들어서 금액 가리기
3.  **보조 액션 헤더**
    *   버튼이 있을 때만 사용
    *   버튼은 항상 오른쪽 정렬
    *   `추가`, `저장`, `삭제` 같은 CRUD 액션만 둔다
4.  **테이블**
    *   카드 내부 스크롤을 사용

### 공통 스타일 규칙

1.  헤더 높이, 버튼 높이, 입력창 높이는 `web/app/globals.css`의 공통 클래스 기준으로 맞춘다.
2.  새 화면을 만들 때 개별 인라인 스타일보다 공통 클래스(`appMainHeader`, `appActionHeader`, `appHeaderMetrics`, `appSegmentedToggle`)를 우선 사용한다.
3.  토글 버튼은 글자 길이에 맞는 폭을 사용하고, 불필요한 고정 최소폭을 두지 않는다.
4.  컬럼 헤더 텍스트는 가운데 정렬, 데이터 셀은 텍스트 왼쪽 정렬 / 숫자 오른쪽 정렬을 유지한다.
5.  수정 가능한 셀 강조, 수정된 셀 강조, 선택/삭제 버튼 배치는 화면마다 임의로 다르게 만들지 않는다.
6.  종목 관리 삭제는 즉시 하드 딜리트로 처리하고, 삭제 사유/삭제일자/휴지통 개념을 새로 만들지 않는다.

### 화면별 예외 원칙

1.  `/market`처럼 검색 입력이 여러 개인 화면은 메인 헤더 한 줄을 유지하는 전용 예외 클래스를 둘 수 있다.
2.  예외를 추가하더라도 메인 헤더, 보조 액션 헤더, 테이블의 3단 구조 자체는 최대한 유지한다.
3.  새 예외 스타일을 만들면 먼저 공통 클래스 확장으로 해결 가능한지 검토하고, 불가할 때만 화면 전용 클래스를 추가한다.
