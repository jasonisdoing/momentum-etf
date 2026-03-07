# 전략 로직 상세 (Strategy Logic)

이 문서는 현재 시스템에서 사용하는 전략(`MAPS`, `HR`)의 의사결정 로직을 설명합니다.

## 1. 공통 실행 원칙

* 추천(`recommend.py`)과 백테스트(`backtest.py`)는 동일 백테스트 엔진(`core/backtest`)을 사용합니다.
* 매매 판단은 당일 종가 기준으로 계산하고, 체결은 다음 거래일 가격 규칙(슬리피지 포함)으로 시뮬레이션합니다.
* 리밸런싱/교체/신규매수는 `REBALANCE_MODE` 규칙을 따릅니다.
* 필수 설정이 누락되면 fallback 없이 명시적 에러로 중단됩니다.

## 2. MAPS 전략

`MAPS`는 이동평균 기반 점수로 종목 상대 순위를 계산합니다.

### 핵심 파라미터

* `TOPN`: 보유 종목 수
* `MA_MONTH`: 이동평균 기간(개월)
* `MA_TYPE`: 이동평균 종류
* `COOLDOWN`: 매수/매도 후 반대 거래 제한 거래일 수
* `REBALANCE_MODE`: 리밸런싱 주기

### 동작 요약

1. 각 종목의 MAPS 점수를 계산
2. 보유 우선 + 점수 순으로 정렬
3. `TOPN` 기준으로 보유/대기 상태 결정
4. 쿨다운/리밸런싱 규칙을 적용해 실행 가능 주문만 확정

## 3. HR 전략

`HR`은 계좌 종목 리스트(`stock_meta`)의 `weight`를 목표 비중으로 사용하는 고정 비중 전략입니다.

### 핵심 파라미터

* `REBALANCE_MODE`: 리밸런싱 주기
* 종목별 `weight`: MongoDB 종목 메타의 비중 값(합계 100 기준)

### 동작 요약

1. 종목 리스트에서 `weight`를 읽어 목표 비중 구성
2. 시작 시점에 목표 비중으로 초기 포지션 구성
3. 리밸런싱 시점마다 목표 비중으로 재조정

### 검증 규칙

* `weight`는 숫자/양수여야 함
* 합계는 100(레거시 1.0도 허용)이어야 함

## 4. 튜닝 로직

* `MAPS` 튜닝: `TOPN`, `MA_MONTH`, `MA_TYPE`, `COOLDOWN`, `REBALANCE_MODE`
* `HR` 튜닝: `REBALANCE_MODE`만 탐색
* 최적 결과는 계좌 `config.json`의 `strategy`에 반영됩니다.
  * 공통 파라미터: `strategy.COMMON`
  * MAPS 전용 파라미터: `strategy.MAPS`
  * HR은 전용 파라미터가 없으면 `strategy.HR` 블록을 생략할 수 있습니다.
