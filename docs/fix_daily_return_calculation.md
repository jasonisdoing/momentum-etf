# 일간 수익률 계산 버그 수정

## 문제 상황

티커 442580의 일간 수익률이 네이버 API의 `changeRate`와 불일치하는 문제가 발생했습니다.

### 증상
- 추천 시스템 표시: +1.32%
- 네이버 API `changeRate`: -0.38%
- 실시간 가격 (`nowVal`): 36,720원

### 원인

`logic/recommend/pipeline.py`의 일간 수익률 계산 로직에서 잘못된 기준 가격을 사용했습니다.

**문제가 있던 코드 (Line 769-770):**
```python
# 실시간 가격이 있으면 전일 종가 대비 계산
if realtime_price and market_prev and market_prev > 0:
    daily_pct = ((realtime_price / market_prev) - 1.0) * 100
```

**변수 정의:**
- `market_latest`: 캐시의 마지막 종가 (= 전일 종가, 10/30)
- `market_prev`: 캐시의 마지막에서 두 번째 종가 (= 전전일 종가, 10/29)
- `realtime_price`: 네이버 API의 실시간 가격 (= 당일 가격, 10/31)

**문제:**
실시간 가격(10/31)을 전전일 종가(10/29)와 비교하여 일간 수익률을 계산했습니다.

```
잘못된 계산: (36,720 / 36,240 - 1) * 100 = +1.32%
             (10/31)   (10/29)
```

올바른 계산은 실시간 가격(10/31)을 전일 종가(10/30)와 비교해야 합니다.

```
올바른 계산: (36,720 / 36,860 - 1) * 100 = -0.38%
             (10/31)   (10/30)
```

## 해결 방법

`market_prev` 대신 `market_latest`를 사용하도록 수정했습니다.

**수정된 코드:**
```python
# 실시간 가격이 있으면 전일 종가 대비 계산
if realtime_price and market_latest and market_latest > 0:
    daily_pct = ((realtime_price / market_latest) - 1.0) * 100
```

## 검증

### 수정 전
```
Cache data:
  10/29 Close (market_prev): 36,240
  10/30 Close (market_latest): 36,860

Realtime price (10/31): 36,720

OLD calculation (using market_prev):
  (36720 / 36240.0 - 1) * 100 = +1.32%  ✗
```

### 수정 후
```
NEW calculation (using market_latest):
  (36720 / 36860.0 - 1) * 100 = -0.38%  ✓

Naver API changeRate: -0.38%
Match: ✓
```

## 영향 범위

- **파일**: `logic/recommend/pipeline.py` (Line 769)
- **영향**: 한국 시장(KOR) ETF의 실시간 일간 수익률 계산
- **백테스트**: 영향 없음 (백테스트는 실시간 가격을 사용하지 않고 캐시 데이터만 사용)

## 관련 이슈

- 캐시 파일 위치: `data/stocks/cache/kor/{ticker}.pkl`
- 캐시는 정상적으로 작동하며, 전일(10/30)까지의 데이터를 포함하고 있음
- 문제는 캐시가 아닌 일간 수익률 계산 로직에 있었음

## 날짜: 2025-10-31
