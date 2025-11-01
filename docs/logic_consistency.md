# 추천/백테스트 로직 일관성 가이드

> **중요:** 추천과 백테스트는 동일한 매매 로직을 사용해야 합니다. 한 쪽을 수정할 때 반드시 다른 쪽도 확인하세요.

## 📂 파일 매핑

| 추천 (Recommend) | 백테스트 (Backtest) | 퍼포먼스 (Performance) | 역할 |
|-----------------|-------------------|----------------------|------|
| `logic/recommend/portfolio.py` | `logic/backtest/portfolio_runner.py` | - | 매매 의사결정 로직 |
| `logic/recommend/pipeline.py` | `logic/backtest/account_runner.py` | - | 데이터 준비 및 실행 |
| ❌ **없음** | `portfolio_runner.py` (23-163줄, 1303-1305줄) | `performance.py` (245-326줄) | 균등 비중 리밸런싱 |

---

## ✅ 동일해야 하는 핵심 로직 (7개)

> **참고:** 리밸런싱 로직은 백테스트/퍼포먼스에만 존재하며, 추천 시스템에는 없습니다. 자세한 내용은 [8. 리밸런싱 (백테스트/퍼포먼스 전용)](#8-리밸런싱-백테스트퍼포먼스-전용)을 참고하세요.

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

**설정:**
- 시장 레짐 감지는 대시보드 참고용으로만 사용
- 추천/백테스트 실행 로직에서는 항상 100% 투자

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

### 8. 리밸런싱 (백테스트/퍼포먼스 전용)

**규칙:** 매일 균등 비중 리밸런싱 체크 및 실행

| 항목 | 백테스트 | 퍼포먼스 | 추천 |
|------|---------|---------|------|
| 파일 | `portfolio_runner.py` | `performance.py` | ❌ **없음** |
| 리밸런싱 함수 | 23-163줄 (`_rebalance_portfolio_equal_weight`) | 250-326줄 (인라인) | - |
| 조건 체크 | 1303-1305줄 | 245-247줄 | - |

**리밸런싱 조건:**
```python
# 매일 체크
trades_occurred = bool(매수매도_발생)
should_rebalance = trades_occurred or max_weight_diff > rebalance_threshold
```

**동작:**
1. **매일 비중 계산**: 각 보유 종목의 현재 비중과 목표 비중(`100% / PORTFOLIO_TOPN`) 비교
2. **리밸런싱 조건 판단**:
   - 매수/매도 발생 **OR**
   - 비중 편차가 `REBALANCE_THRESHOLD` (기본 0.3%) 초과
3. **균등 비중 리밸런싱 실행** (최대 5회 반복):
   - **1단계**: 과다 보유 종목 매도 (현금 확보)
   - **2단계**: 과소 보유 종목 매수 (비례 배분)
   - 목표 현금: 총자산의 1%

**파라미터:**
- `REBALANCE_THRESHOLD`: 리밸런싱 임계값 (기본 0.3%)
- `PORTFOLIO_TOPN`: 포트폴리오 최대 종목 수 (목표 비중 = 100% / TOPN)

**백테스트 레이블:**
- `[리밸런싱 매수]`: 과소 보유 종목 추가 매수
- `[리밸런싱 매도]`: 과다 보유 종목 일부 매도

#### ❓ 왜 추천에는 리밸런싱 로직이 없나요?

**이유:**
1. **정보 부족**: 추천 시스템은 실제 보유 수량 정보가 없어 정확한 비중 계산 불가
2. **역할 분리**: 
   - **추천**: 어떤 종목을 사고팔지 판단 (매수/매도 신호)
   - **백테스트**: 과거 데이터로 전략 검증 (리밸런싱 효과 측정)
   - **퍼포먼스**: 실제 거래 기반 성과 계산 (리밸런싱 반영)
3. **실행 시스템**: 실제 리밸런싱은 별도 실행 시스템에서 처리

**호환성:**
- `run_portfolio_recommend()` 함수는 `rebalance_threshold` 파라미터를 받지만 사용하지 않음 (호환성 유지)

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
- [ ] 대시보드 표시용으로만 사용
- [ ] 추천/백테스트 실행 로직에는 영향 없음 (항상 100% 투자)

**카테고리 중복 제한 수정 시:**
- [ ] `portfolio.py` 334-342줄, `pipeline.py` 1163-1166줄 수정
- [ ] `portfolio_runner.py` 783-785줄 동일하게 수정
- [ ] `MAX_PER_CATEGORY` 설정 확인

**리밸런싱 로직 수정 시:**
- [ ] `portfolio_runner.py` 23-163줄, 1303-1305줄 수정
- [ ] `performance.py` 245-326줄 동일하게 수정
- [ ] ⚠️ **추천 시스템은 수정하지 않음** (리밸런싱 로직 없음)
- [ ] `REBALANCE_THRESHOLD` 파라미터 확인
- [ ] 매일 체크 조건: `trades_occurred or max_weight_diff > rebalance_threshold`

---

## 📦 공통 함수 (logic/common/)

추천과 백테스트에서 공통으로 사용하는 헬퍼 함수들입니다.

### 포트폴리오 관리 (`logic/common/portfolio.py`)

| 함수 | 설명 | 사용처 |
|------|------|--------|
| `get_held_categories_excluding_sells()` | 매도 예정 종목을 제외한 보유 카테고리 계산 | 추천, 백테스트 |
| `should_exclude_from_category_count()` | 카테고리 카운트 제외 여부 확인 | 추천, 백테스트 |
| `get_sell_states()` | 매도 상태 집합 반환 | 추천, 백테스트 |
| `get_hold_states()` | 보유 상태 집합 반환 (매도 예정 포함) | 추천, 백테스트 |
| `count_current_holdings()` | 현재 물리적 보유 종목 수 계산 | 추천, 백테스트 |
| `validate_core_holdings()` | 핵심 보유 종목 검증 | 추천, 백테스트 |
| `check_buy_candidate_filters()` | 매수 후보 필터링 체크 | 추천, 백테스트 |
| `calculate_buy_budget()` | 매수 예산 계산 | 백테스트 |
| `calculate_held_categories()` | 보유 카테고리 계산 | 백테스트 |
| `calculate_held_categories_from_holdings()` | holdings dict에서 카테고리 계산 | 추천 |
| `track_sell_rsi_categories()` | SELL_RSI 카테고리 추적 | 추천, 백테스트 |
| `calculate_held_count()` | 보유 종목 수 계산 | 백테스트 |
| `validate_portfolio_topn()` | TOPN 값 검증 | 추천, 백테스트 |

### 시그널 처리 (`logic/common/signals.py`)

| 함수 | 설명 | 사용처 |
|------|------|--------|
| `has_buy_signal()` | 매수 시그널 여부 확인 | 추천, 백테스트 |
| `calculate_consecutive_days()` | 연속 보유 일수 계산 | 추천 |
| `get_buy_signal_streak()` | 매수 시그널 연속 일수 계산 | 추천 |

### 필터링 (`logic/common/filtering.py`)

| 함수 | 설명 | 사용처 |
|------|------|--------|
| `select_candidates_by_category()` | 카테고리별 후보 선택 | 추천, 백테스트 |
| `sort_decisions_by_order_and_score()` | 의사결정 정렬 | 추천 |
| `filter_category_duplicates()` | 카테고리 중복 필터링 | 추천, 백테스트 |

---

## 🚀 향후 개선 (선택사항)

### 추가 공통화 후보

다음 상황이 발생하면 공통화를 고려하세요:

1. **매도 로직을 3번 이상 수정해야 할 때**
2. **새로운 전략(예: MAPS2)을 추가할 때**
3. **로직 불일치로 인한 버그가 실제로 발생했을 때**

**후보 함수:**

1. **매도 조건 판단 함수**
   ```python
   # logic/common/decisions.py
   def determine_sell_decision(...) -> Optional[str]
   ```

2. **쿨다운 상태 계산 함수**
   ```python
   # logic/common/cooldown.py
   def calculate_cooldown_status(...) -> Dict[str, Any]
   ```

3. **SELL_RSI 카테고리 수집 함수**
   ```python
   # logic/common/decisions.py
   def collect_sell_rsi_categories(...) -> Set[str]
   ```

---

## 📋 핵심 함수 시그니처

### 추천 함수

```python
def run_portfolio_recommend(
    account_id: str,
    country_code: str,
    base_date: pd.Timestamp,
    strategy_rules: Any,
    data_by_tkr: Dict[str, Any],
    holdings: Dict[str, Dict[str, float]],
    etf_meta: Dict[str, Any],
    full_etf_meta: Dict[str, Any],
    current_equity: float,
    total_cash: float,
    pairs: List[Tuple[str, str]],
    consecutive_holding_info: Dict[str, Dict],
    trade_cooldown_info: Dict[str, Dict[str, Optional[pd.Timestamp]]],
    cooldown_days: int,
    rsi_sell_threshold: float = 10.0,
    rebalance_threshold: float = 0.3,  # 호환성 유지 (사용하지 않음)
) -> List[Dict[str, Any]]
```

### 백테스트 함수

```python
def run_portfolio_backtest(
    stocks: List[Dict],
    initial_capital: float = 100_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
    top_n: int = 10,
    date_range: Optional[List[str]] = None,
    country: str = "kor",
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
    ma_period: int = 20,
    ma_type: str = "SMA",
    replace_threshold: float = 0.0,
    stop_loss_pct: float = -10.0,
    cooldown_days: int = 5,
    rsi_sell_threshold: float = 10.0,
    rebalance_threshold: float = 0.3,  # 균등 비중 리밸런싱 임계값
    core_holdings: Optional[List[str]] = None,
    quiet: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    missing_ticker_sink: Optional[Set[str]] = None,
) -> Dict[str, pd.DataFrame]
```

### 퍼포먼스 함수

```python
def calculate_actual_performance(
    account_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    country_code: str = "kor",
    portfolio_topn: int = 12,
    rebalance_threshold: float = 0.3,  # 균등 비중 리밸런싱 임계값
) -> Optional[Dict[str, Any]]
```

---

## 📚 참고

- **메모리:** "추천/백테스트 로직 일관성 체크리스트" 참고
- **코드:** `logic/recommend/portfolio.py`, `logic/backtest/portfolio_runner.py`
- **공통 함수:** `logic/common/portfolio.py`, `logic/common/signals.py`, `logic/common/filtering.py`
- **설정:** `data/settings/account/k1.json`
