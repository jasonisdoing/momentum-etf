# 백테스트

## 실행

특정 종목풀 하나만 실행하려면 파라미터로 넘긴다:

```bash
python backtest/run.py kor_kr
```

인자 없이 실행하면 설정된 모든 종목풀을 순차적으로 실행한다:

```bash
python backtest/run.py
```

## 대상 종목풀

- `kor_kr`
- `kor_us`
- `aus`
- `us`
- `kor`

## 스윕 대상 파라미터

- `TOP_N_HOLD`
- `HOLDING_BONUS_SCORE`
- `FIRST_MA_TYPE`
- `FIRST_MA_MONTHS`
- `SECOND_MA_TYPE`
- `SECOND_MA_MONTHS`

세부 스윕 범위는 프로젝트 루트의 [config.py](../config.py) 안 `BACKTEST_CONFIG`에서 관리한다.

공통 전역값은 같은 파일 상단에서 별도로 관리한다.

```python
BACKTEST_START_DATE = "2023-01-01"
BACKTEST_INITIAL_KRW_AMOUNT = 100_000_000
```

종목풀별 슬리피지 설정은 같은 파일의 `SLIPPAGE_CONFIG`에서 별도로 관리한다.

```python
"kor_kr": {
    "BUY_PCT": 0.0,
    "SELL_PCT": 0.0,
}
```

- `BUY_PCT`: 매수 슬리피지
- `SELL_PCT`: 매도 슬리피지
- 단위는 %
- 매수 체결가는 `시초가 × (1 + BUY_PCT)` 비율로 반영한다.
- 매도 체결가는 `시초가 × (1 - SELL_PCT)` 비율로 반영한다.

## 출력 파일

- `backtest/results/<pool>-backtest_<YYYY-MM-DD>.log`
  - 전체 조합 결과를 CAGR 내림차순으로 기록한다.
  - 실행 중에는 100건마다 중간 결과를 갱신한다.
- `backtest/results/<pool>-backtest_details_<YYYY-MM-DD>.log`
  - 최종 1등 조합만 다시 1회 시뮬레이션하여 거래일별 상세 보유 내역을 기록한다.
  - 각 거래일 표의 첫 row는 항상 CASH이며, 당일 SELL 종목도 함께 남긴다. 상태값은 CASH, HOLD, BUY, SELL, WAIT 로 표기된다. (WAIT은 TOP_N에 들었으나 현금 부족 등으로 미배분된 종목)
  - 상세 표에는 `점수`, `추세1`, `추세2`가 함께 기록되며, 각 값은 해당 거래를 결정한 신호일 기준이다.
  - 각 거래일 헤더에는 `총자산`, `현금`, `평가수익`, `누적수익`이 함께 표시된다.

## 현재 백테스트 가정

- 백테스트 첫 체결일은 전 거래일 종가 신호를 사용해 시초가(Open)에 진입
- 이후에도 전일 종가 신호를 기준으로 다음 거래일 시초가(Open)에 체결
- 매수/매도 체결가는 종목풀별 슬리피지를 반영한다
- 기존 보유 종목은 상위 N에 남아 있으면 비중을 줄이지 않음
- 신규 진입은 `현금 우선 분할 진입 방식` 사용
  - 매도 후 확보한 현금으로 신규 진입 종목에 1차 균등 배분
  - 1차 배분 후 남는 잔액은 점수순으로 추가 소진하되, 종목당 `slot_target` 상한을 넘기지 않음
- 매수는 단주만 허용하며, 남는 자금은 현금으로 유지
- `HOLDING_BONUS_SCORE`는 백테스트 내부에서만 적용

## 모듈 구조

- `run.py`
  - CLI 엔트리 포인트
- `config.py` (프로젝트 루트)
  - 종목풀별 스윕 설정
- `engine.py`
  - 조합 실행, 시뮬레이션, 결과 파일 생성
- `core/strategy/scoring.py`
  - 랭킹과 백테스트가 공용으로 쓰는 점수 계산 엔진
