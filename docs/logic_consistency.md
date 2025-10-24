# 추천/백테스트 로직 일관성 가이드

> **중요:** 추천과 백테스트는 동일한 매매 로직을 사용해야 합니다. 한 쪽을 수정할 때 반드시 다른 쪽도 확인하세요.

## 📂 파일 매핑

| 추천 (Recommend) | 백테스트 (Backtest) | 역할 |
|-----------------|-------------------|------|
| `logic/recommend/portfolio.py` | `logic/backtest/portfolio_runner.py` | 매매 의사결정 로직 |
| `logic/recommend/pipeline.py` | `logic/backtest/account_runner.py` | 데이터 준비 및 실행 |

---

## ✅ 동일해야 하는 핵심 로직 (7개)

### 1. 매도 조건 판단 순서

**우선순위:** CUT_STOPLOSS → SELL_RSI → SELL_TREND

| 항목 | 추천 | 백테스트 |
|------|------|---------|
| 파일 | `portfolio.py` | `portfolio_runner.py` |
| 위치 | 190-198줄 | 663-668줄 |

**조건:**
- **CUT_STOPLOSS:** `holding_return_pct <= stop_loss_threshold`
- **SELL_RSI:** `rsi_score <= rsi_sell_threshold`
- **SELL_TREND:** `current_price < ma_value`

**코드 예시:**
```python
# 손절 체크
if holding_return_pct <= stop_loss_threshold:
    decision = "CUT_STOPLOSS"
# RSI 과매수 체크
elif rsi_score <= rsi_sell_threshold:
    decision = "SELL_RSI"
# 추세 이탈 체크
elif current_price < ma_value:
    decision = "SELL_TREND"
```

---

### 2. 쿨다운 로직

**규칙:** `days_since < cooldown_days` → 거래 차단

| 항목 | 추천 | 백테스트 |
|------|------|---------|
| 파일 | `portfolio.py` | `portfolio_runner.py` |
| 매수 후 매도 쿨다운 | 63-111줄 (`_calculate_cooldown_blocks`) | 654줄 (`sell_block_until`) |
| 매도 후 매수 쿨다운 | 63-111줄 (`_calculate_cooldown_blocks`) | 768줄 (`buy_block_until`) |

**COOLDOWN_DAYS=1 의미:**
- **당일 (0일):** 반대 거래 차단
- **다음날 (1일 이상):** 반대 거래 허용

**구현 차이 (결과는 동일):**
- **추천:** 매도 결정 후 쿨다운 체크 → HOLD로 전환
- **백테스트:** 매도 전 쿨다운 사전 체크 → 매도 로직 진입 안 함

---

### 3. SELL_RSI 카테고리 차단

**규칙:** RSI 과매수로 매도되는 카테고리는 같은 날 매수 금지

| 항목 | 추천 | 백테스트 |
|------|------|---------|
| 파일 | `portfolio.py` | `portfolio_runner.py` |
| 위치 | 484-491줄 | 696-699줄 |

**동작:**
1. SELL_RSI 상태인 종목의 카테고리를 `sell_rsi_categories_today`에 추가
2. 같은 날 해당 카테고리의 다른 종목 매수 차단

**예시:**
```python
# 442580 (미국반도체) SELL_RSI
# → "미국반도체" 카테고리 차단
# → 446770 (미국반도체) 매수 불가
```

---

### 4. RSI 과매수 경고 카테고리 차단 ⭐

**규칙:** 쿨다운으로 아직 매도하지 못했지만 RSI 과매수인 종목의 카테고리도 차단

| 항목 | 추천 | 백테스트 |
|------|------|---------|
| 파일 | `portfolio.py` | `portfolio_runner.py` |
| 위치 | 493-499줄 | 436-445줄 |

**동작:**
1. 보유 중인 종목의 RSI가 과매수 임계값 이하인지 체크
2. 쿨다운 때문에 아직 매도하지 못한 경우에도 카테고리 차단
3. 같은 날 해당 카테고리의 다른 종목 매수 차단

**예시:**
```python
# 457990 (태양광) 보유일 0일, RSI 14
# → 쿨다운으로 매도 불가
# → "태양광" 카테고리 차단
# → 같은 카테고리 다른 종목 매수 불가
```

---

### 5. 핵심 보유 종목 (CORE_HOLDINGS)

**규칙:**
- 매도 신호 무시 (강제 HOLD_CORE)
- 미보유 시 자동 매수
- TOPN에 포함

| 항목 | 추천 | 백테스트 |
|------|------|---------|
| 파일 | `portfolio.py` | `portfolio_runner.py` |
| 매도 무시 | 400-414줄 | 670-672줄 |
| 자동 매수 | 416-454줄 | 734-759줄 |

**동작:**
```python
# 핵심 보유 종목은 SELL_RSI, SELL_TREND 등 모든 매도 신호 무시
if ticker in core_holdings:
    if decision in {"SELL_TREND", "SELL_RSI", "CUT_STOPLOSS"}:
        decision = "HOLD_CORE"
```

---

### 6. 시장 레짐 필터

**규칙:**
- 리스크 오프 시 매수 중단 또는 축소
- 매도는 항상 활성화 (RSI 과매수 매도 포함)

| 항목 | 추천 | 백테스트 |
|------|------|---------|
| 파일 | `portfolio.py` | `portfolio_runner.py` |
| 위치 | 458-473줄 | 447-470줄 |

**설정:**
- `MARKET_REGIME_RISK_OFF_EQUITY_RATIO`: 리스크 오프 시 투자 비율 (0-100%)
  - 100: 정상 투자
  - 90: 90% 투자 (10% 현금)
  - 0: 전량 매도

---

### 7. 카테고리 중복 제한

**규칙:** `MAX_PER_CATEGORY` 설정값만큼만 보유

| 항목 | 추천 | 백테스트 |
|------|------|---------|
| 파일 | `portfolio.py`, `pipeline.py` | `portfolio_runner.py` |
| 위치 | 334-342줄, 1163-1166줄 | 783-785줄 |

**동작:**
```python
# MAX_PER_CATEGORY = 1
# → 같은 카테고리 종목은 1개만 보유 가능
# → 이미 보유 중이면 같은 카테고리 다른 종목 매수 불가
```

---

## 🔧 수정 시 절차

### 1단계: 수정 전 확인
- [ ] 위 7개 로직 중 어떤 것을 수정하는지 확인
- [ ] 해당 로직의 파일 매핑 확인 (추천 ↔ 백테스트)

### 2단계: 양쪽 수정
- [ ] 추천 파일 수정
- [ ] 백테스트 파일도 동일하게 수정
- [ ] 코드 비교 (조건, 순서, 로직 동일한지)

### 3단계: 테스트
```bash
# 추천 실행
python recommend.py k1

# 백테스트 실행
python backtest.py k1

# 튜닝 실행 (선택)
python tune.py k1
```

### 4단계: 결과 비교
- [ ] 추천 결과 확인 (BUY, SELL, HOLD 상태)
- [ ] 백테스트 CAGR 확인
- [ ] 로그에서 쿨다운, RSI 카테고리 차단 메시지 확인

---

## ⚠️ 주의사항

### 구현 방식은 달라도 결과는 동일해야 함

**예시: 쿨다운 체크**
- **추천:** 매도 결정 후 쿨다운 체크 → HOLD로 전환
- **백테스트:** 매도 전 쿨다운 사전 체크 → 매도 로직 진입 안 함
- **결과:** 둘 다 쿨다운 기간 동안 매도하지 않음 ✅

### 데이터 소스 차이는 정상

- **추천:** 실시간 가격, NAV, iNAV 괴리율
- **백테스트:** 과거 OHLCV 데이터, 슬리피지 적용
- **이유:** 목적이 다름 (실시간 추천 vs 과거 시뮬레이션)

---

## 📝 체크리스트 요약

**매도 로직 수정 시:**
- [ ] `portfolio.py` 190-198줄 수정
- [ ] `portfolio_runner.py` 663-668줄 동일하게 수정
- [ ] 순서 확인: CUT_STOPLOSS → SELL_RSI → SELL_TREND

**쿨다운 로직 수정 시:**
- [ ] `portfolio.py` 63-111줄 수정
- [ ] `portfolio_runner.py` 654줄, 768줄 동일하게 수정
- [ ] `days_since < cooldown_days` 조건 확인

**RSI 카테고리 차단 수정 시:**
- [ ] `portfolio.py` 484-499줄 수정
- [ ] `portfolio_runner.py` 436-445줄, 696-699줄 동일하게 수정
- [ ] `sell_rsi_categories_today` 로직 확인

**핵심 보유 종목 수정 시:**
- [ ] `portfolio.py` 400-454줄 수정
- [ ] `portfolio_runner.py` 670-672줄, 734-759줄 동일하게 수정
- [ ] 매도 무시 + 자동 매수 로직 확인

**시장 레짐 필터 수정 시:**
- [ ] `portfolio.py` 458-473줄 수정
- [ ] `portfolio_runner.py` 447-470줄 동일하게 수정
- [ ] `risk_off_equity_ratio` 설정 확인

**카테고리 중복 제한 수정 시:**
- [ ] `portfolio.py` 334-342줄, `pipeline.py` 1163-1166줄 수정
- [ ] `portfolio_runner.py` 783-785줄 동일하게 수정
- [ ] `MAX_PER_CATEGORY` 설정 확인

---

## 🚀 향후 개선 (선택사항)

### 공통 함수 추출 고려 시점

다음 상황이 발생하면 공통화를 고려하세요:

1. **매도 로직을 3번 이상 수정해야 할 때**
2. **새로운 전략(예: MAPS2)을 추가할 때**
3. **로직 불일치로 인한 버그가 실제로 발생했을 때**

### 공통화 후보

1. **매도 조건 판단 함수**
   ```python
   # logic/common_decisions.py
   def determine_sell_decision(...) -> Optional[str]
   ```

2. **쿨다운 상태 계산 함수**
   ```python
   # strategies/maps/cooldown.py
   def calculate_cooldown_status(...) -> Dict[str, Any]
   ```

3. **SELL_RSI 카테고리 수집 함수**
   ```python
   # logic/common_decisions.py
   def collect_sell_rsi_categories(...) -> Set[str]
   ```

---

## 📚 참고

- **메모리:** "추천/백테스트 로직 일관성 체크리스트" 참고
- **코드:** `logic/recommend/portfolio.py`, `logic/backtest/portfolio_runner.py`
- **설정:** `data/settings/account/k1.json`
