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
| `logic/recommend/portfolio.py` | `logic/backtest/portfolio.py` | 매매 의사결정 로직 |
| `logic/recommend/pipeline.py` | `logic/backtest/account.py` | 데이터 준비 및 실행 |

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
*   `price.py`: 가격 결정 및 수익률 계산 로직 공통화

## 4. 테스트 및 검증

코드를 수정할 때는 다음 절차를 따르세요.

1.  **추천 로직 수정**: `logic/recommend/` 수정
2.  **검증**:
    *   **추천 실행**: `python recommend.py <account_id>` (예: `python recommend.py us1`)
    *   **결과 확인**: `zaccounts/<account_id>/results/recommend_YYYY-MM-DD.log`
3.  **백테스트 검증 (선택)**:
    *   만약 `logic/common/` 등 공통 모듈을 수정했다면 백테스트도 검증 필요
    *   `python backtest.py <account_id>`
    *   결과 확인: `zaccounts/<account_id>/results/backtest_YYYY-MM-DD.log`

## 5. 추천 시스템의 정의 (Definition of Recommendation)

**"추천(Recommendation)"**은 계좌의 자산 규모(Total Equity)나 현금 잔고(Cash)와는 무관하게, 순수하게 **전략적 신호(Signal)**와 **포트폴리오 슬롯(Slot)** 여부만을 기반으로 생성됩니다.

### 핵심 원칙
1.  **전략적 신호 우선**: 자산이 부족하더라도 전략상 매수 신호(BUY Signal)가 발생하고 슬롯이 비어있으면 `BUY` 추천을 생성합니다.
2.  **자산 독립성**: `current_equity`나 `total_cash` 등 자산 데이터를 로직에 반영하지 않습니다. 따라서 "현금 부족" 등의 사유로 추천이 거절(WAIT)되지 않습니다.
3.  **슬롯 관리**: 유일한 물리적 제약은 `portfolio_topn` (최대 보유 종목 수)에 따른 슬롯 여유분입니다.
4.  **역할 분리**: 실제 매수 가능 여부(예산 부족 등)는 추천 이후의 실행 단계(Execution Layer)나 사용자가 판단할 영역이며, 추천 로직의 책임이 아닙니다.
