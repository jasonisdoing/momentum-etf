# 개발자 가이드 (Developer Guide)

이 문서는 Momentum ETF 시스템의 아키텍처, 데이터 흐름, 그리고 개발 시 반드시 지켜야 할 정합성 원칙을 설명합니다.

## 1. 시스템 아키텍처

### 모듈 구조
*   `logic/recommend/`: 일일 매매 추천 로직
*   `logic/backtest/`: 과거 데이터 시뮬레이션 로직
*   `logic/tune/`: 파라미터 최적화 로직
*   `logic/common/`: 위 3개 모듈이 공유하는 핵심 로직 (시그널, 필터링, 포트폴리오 관리)
*   `utils/`: 데이터 I/O, 로깅 등 유틸리티

### 데이터 파이프라인
1.  **데이터 수집**: `pykrx`, `yfinance` 등을 통해 일별 OHLCV 데이터 수집
2.  **전처리**: 수정주가 반영, 결측치 처리
3.  **시그널 생성**: 이동평균, RSI 등 기술적 지표 계산
4.  **의사결정**: 전략 로직에 따라 매매 신호 생성

## 2. 추천/백테스트 정합성 가이드

> **Critical**: 추천(Recommend)과 백테스트(Backtest)는 **반드시 동일한 로직**으로 동작해야 합니다.

### 파일 매핑
| 추천 (Recommend) | 백테스트 (Backtest) | 역할 |
|-----------------|-------------------|------|
| `logic/recommend/portfolio.py` | `logic/backtest/portfolio_runner.py` | 매매 의사결정 로직 |
| `logic/recommend/pipeline.py` | `logic/backtest/account_runner.py` | 데이터 준비 및 실행 |

### 핵심 일관성 체크리스트

1.  **매도 조건 우선순위**: `CUT_STOPLOSS` → `SELL_RSI` → `SELL_TREND` 순서를 엄수할 것.
2.  **쿨다운 로직**:
    *   매수 쿨다운: 매도 후 `COOLDOWN_DAYS` 동안 재매수 금지
    *   매도 쿨다운: 매수 후 `COOLDOWN_DAYS` 동안 매도 금지 (단, 손절은 예외)
3.  **RSI 과매수 차단**:
    *   과매수 종목은 신규 매수 금지
    *   보유 종목이 과매수 상태면 해당 카테고리 전체 매수 차단
4.  **핵심 보유 종목 (Core Holdings)**:
    *   모든 매도 신호 무시
    *   미보유 시 최우선 순위로 자동 매수
5.  **데이터 기준**:
    *   추천과 백테스트 모두 **전일 종가**를 기준으로 판단
    *   장중 실시간 가격은 의사결정에 반영하지 않음

## 3. 공통 모듈 (`logic/common/`)

로직의 중복을 피하고 일관성을 유지하기 위해 다음 기능들은 반드시 공통 모듈을 사용해야 합니다.

*   `portfolio.py`: 카테고리 중복 체크, 핵심 보유 종목 검증
*   `signals.py`: 매수 시그널 발생 여부, 연속 상승일 계산
*   `filtering.py`: 후보군 필터링 및 정렬

## 4. 테스트 및 검증

코드를 수정할 때는 다음 절차를 따르세요.

1.  **추천 로직 수정**: `logic/recommend/` 수정
2.  **백테스트 로직 수정**: `logic/backtest/`에 동일하게 반영
3.  **검증**:
    *   `python main.py` 실행하여 추천 결과 확인
    *   `python logic/backtest/account_runner.py` 실행하여 백테스트 결과 확인
    *   두 결과가 논리적으로 일치하는지 확인 (로그 비교)
