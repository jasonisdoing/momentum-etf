# 개발자 가이드 (Developer Guide)

이 문서는 Momentum ETF 순위 시스템의 아키텍처, 데이터 흐름, 그리고 개발 시 반드시 지켜야 할 정합성 원칙을 설명합니다.

## 1. 시스템 아키텍처

### 모듈 구조
*   `core/strategy/`: 지표/점수/비중 계산 공용 전략 유틸
*   `services/`: **외부 API/데이터 연동 통합 계층**
    *   `price_service.py`: 실시간 가격/환율 오케스트레이션 및 TTL 캐시
    *   `reference_data_service.py`: KIS ETF 마스터, 종목 메타데이터, 상장일 조회
    *   `vkospi_service.py`: VKOSPI 등 외부 시장 지표 연동 및 메모리 캐시
    *   `fear_greed_service.py`: CNN 공포탐욕지수 연동 및 메모리 캐시
    *   **원칙**: 새로운 시장 지표, 가격 정보, 외부 데이터 크롤링 등은 혼동을 막기 위해 모두 이 폴더에서 각각의 서비스로 관리하고, 자체 캐시 시스템(TTL 등)을 구축합니다.
*   `utils/rankings.py`: 순위 테이블 계산 전용 유틸
*   `scripts/`: 데이터 수집, 캐시 갱신 등 유틸리티 스크립트
*   `utils/`:
    *   `cache_utils.py`: **Parquet 기반 캐시 I/O** 및 직렬화 관리
    *   `data_loader.py`: OHLCV 수집/보완 및 원천 fetch 함수
    *   `ai_summary.py`: AI용 요약 데이터 생성 공용 유틸
*   `.github/workflows/`: GitHub Actions를 이용한 일일 배포 및 자동화 정의

### 데이터 파이프라인 및 캐싱
1.  **데이터 수집**: `pykrx`, `yfinance` 등을 통해 원천 데이터 수집.
2.  **Parquet 캐싱**: 수집된 데이터는 `utils/cache_utils.py`를 통해 **Apache Parquet** 포맷으로 MongoDB에 저장됩니다. (기존 Pickle 방식의 버전 충돌 문제를 해결)
3.  **서비스 오케스트레이션**:
    *   `services/price_service.py`가 실시간 가격/환율과 TTL 캐시를 관리합니다.
    *   `services/reference_data_service.py`가 KIS ETF 목록과 메타데이터 조회를 관리합니다.
4.  **지표 계산**: `core/strategy/metrics.py`가 이동평균과 점수를 계산.
5.  **순위 생성**: `utils/rankings.py`가 종목별 통합점수, 추세점수, 고점점수, RSI, 기간 수익률을 합쳐 화면용 DataFrame 생성.

### 서비스 사용 원칙

1.  실시간 가격/환율 조회는 `services/price_service.py`를 먼저 사용합니다.
2.  KIS ETF 마스터, 종목 메타데이터, 상장일 조회는 `services/reference_data_service.py`를 먼저 사용합니다.
3.  **외부 연동 일원화**: 시장 지표나 새로운 데이터를 외부에서 파싱/API 통신해야 한다면 `services/` 하위에 위치시키며, 무분별한 요청을 방지하기 위한 캐싱 로직을 포함해야 합니다.
4.  화면 계층과 일반 유틸은 외부 데이터 소스를 직접 호출하지 않고, 가능하면 `services/` 계층을 통해 접근합니다.
5.  `utils/data_loader.py`는 원천 fetch 함수와 OHLCV 보완 로직을 포함하지만, 신규 호출부를 작성할 때는 직접 진입점으로 우선 사용하지 않습니다.

## 2. 순위 화면 정합성 원칙

> **Critical**: 이 시스템은 **순위 화면이 단일 진실 원천(single source of truth)** 입니다. 화면에서 보이는 값은 계좌 종목 목록, 가격 캐시, 실제 보유 데이터로 직접 계산되어야 합니다.

### 핵심 구조
| 파일/경로 | 역할 |
|-----------------|------|
| `web/app/*` | Next.js 기반 사용자 화면과 API 라우트 |
| `utils/rankings.py` | 순위 계산과 정렬 |
| `core/strategy/metrics.py` | 이동평균 점수 및 지속일 계산 |
| `services/price_service.py` | 실시간 가격/환율 조회의 공식 진입점 |
| `services/reference_data_service.py` | ETF 마스터/메타데이터/상장일 조회의 공식 진입점 |
| `utils/account_notes.py` | 계좌 메모 저장/조회 |

### 핵심 일관성 체크리스트

1.  **입력 단순화**: 화면에서 바꾸는 파라미터는 `MA_TYPE`, `MA_MONTHS` 두 개뿐이다.
2.  **정렬 기준 고정**: 통합점수가 있는 종목을 통합점수 내림차순으로 정렬하고, 계산 불가 종목은 맨 아래로 보낸다.
3.  **데이터 기준**:
    *   모든 의사결정은 **판단 시점의 전일 종가 데이터**를 기준으로 함
    *   "오늘"의 순위는 "어제까지의 마감 데이터"를 보고 계산된 것임
    *   최신 거래일 판단의 기준 날짜는 모든 시장 공통으로 **한국 날짜**를 사용함
4.  **엄격한 설정 원칙 (Rule 7)**:
    *   코드에 암묵적인 기본값(fallback)을 사용하지 않습니다.
    *   필수 순위 파라미터가 누락된 경우 명확한 `ValueError`를 발생시킵니다.

## 3. 전략 설정 규칙

계좌 설정 포맷(`zaccounts/<order>_<account>/config.json`):

```json
{
  "icon": "🇰🇷",
  "name": "국내 계좌",
  "country_code": "kor",
  "currency": "KRW",
  "MA_TYPE": "ALMA",
  "MA_MONTHS": 6
}
```

계좌 설정의 `country_code`는 현재 `kor` 또는 `au`만 허용합니다.

검증 원칙(현재 운영):

* 계좌: `MA_TYPE`, `MA_MONTHS` 필수
* 필수값 누락 시 fallback 없이 명시적 에러

## 4. 테스트 및 검증

코드를 수정할 때는 다음 절차를 따르세요.

1.  **로직 수정**: `utils/rankings.py`, `core/strategy/metrics.py`, `web/app/*`, `web/lib/*`를 우선 확인
    *   가격/환율 문제면 `services/price_service.py`를 함께 확인
    *   KIS ETF 목록/메타데이터/상장일 문제면 `services/reference_data_service.py`를 함께 확인
2.  **검증**:
    *   순위 화면에서 `MA_TYPE`, `MA_MONTHS` 변경 시 즉시 테이블이 갱신되는지 확인
    *   실제 보유 종목이 녹색 행으로 표시되는지 확인
3.  **확인**:
    *   `통합점수`, `추세점수`, `고점점수` 컬럼이 `현재가` 뒤에 순서대로 있고 볼드인지 확인

## 5. 순위 화면의 정의

**"순위(Rank)"**는 계좌의 현재 종목 유니버스에서 `MA_TYPE`, `MA_MONTHS` 기준 추세 점수를 계산한 결과입니다.

### 핵심 원칙
1.  **화면 기준 계산**: 순위는 별도 저장 결과를 읽지 않고, 가격 캐시와 계좌 종목으로 즉시 계산합니다.
2.  **실보유 구분**: 별도 추천 상태는 사용하지 않고, 실제 보유 종목만 행 색상으로 표시합니다.
3.  **정렬 규칙**: 통합점수 내림차순, 통합점수 계산 불가 종목은 맨 아래입니다.
4.  **계좌 종목 직접 관리**: 계좌가 자신의 종목 유니버스를 직접 보유하며, 별도 종목풀 fallback은 사용하지 않습니다.
