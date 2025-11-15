# 추천/백테스트 로직 일관성 가이드

> **중요:** 추천과 백테스트는 동일한 매매 로직을 사용해야 합니다. 한 쪽을 수정할 때 반드시 다른 쪽도 확인하세요.

## 📂 파일 매핑

| 추천 (Recommend) | 백테스트 (Backtest) | 역할 |
|-----------------|-------------------|------|
| `logic/recommend/portfolio.py` | `logic/backtest/portfolio_runner.py` | 매매 의사결정 로직 |
| `logic/recommend/pipeline.py` | `logic/backtest/account_runner.py` | 데이터 준비 및 실행 |

---

## 🎯 후보군·설정 일관성

1. **대표 ETF 필터링**
   - `utils/stock_list_io.get_etfs()` 가 추천·백테스트·튜닝·캐시 갱신의 공통 진입점이다.
   - 카테고리별로 `3_month_earn_rate` 가 가장 높은 1개만 유지하고, `TBD` 카테고리와 계좌별 벤치마크 티커는 무조건 포함한다.
   - 어느 경로든 티커 목록이 다르면 안 되므로, 새로운 기능 추가 시 반드시 이 함수를 사용한다.

2. **최소 점수( `MIN_BUY_SCORE` ) 허들**
   - `StrategyRules.min_buy_score` 를 추천·백테스트·튜닝에서 모두 동일하게 참조한다.
   - 점수가 임계값 미만이면 `WAIT` 상태와 `최소 X점수 미만` 문구가 표시되며, 매수/교체 로직이 바로 차단된다.
   - 튜닝 탐색 공간에도 `MIN_BUY_SCORE` 값이 포함되어야 하며, 결과 요약/중간 보고서에 해당 값이 기록돼야 한다.

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
| `calculate_buy_budget()` | 총자산/TOPN 기반 균등 매수 예산 | 백테스트 |
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
    prefetched_metrics: Optional[Mapping[str, Dict[str, Any]]] = None,
    price_store: Optional[MemmapPriceStore] = None,
    trading_calendar: Sequence[pd.Timestamp],
    ma_period: int = 20,
    ma_type: str = "SMA",
    replace_threshold: float = 0.0,
    stop_loss_pct: float = -10.0,
    cooldown_days: int = 5,
    rsi_sell_threshold: float = 10.0,
    core_holdings: Optional[List[str]] = None,
    quiet: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    missing_ticker_sink: Optional[Set[str]] = None,
    *,
    min_buy_score: float,
) -> Dict[str, pd.DataFrame]
```

- `trading_calendar`는 필수이며, `date_range` 전체를 덮는 거래일 리스트를 호출자가 프리패치 단계에서 준비해 전달해야 한다. 내부에서는 더 이상 `get_trading_days()`로 보조 조회를 하지 않는다.
- `prefetched_data`/`prefetched_metrics`/`price_store`도 반드시 준비된 상태여야 하며, 백테스트 중에는 원본 데이터 소스(Mongo/pykrx)를 호출하지 않는다. 부족 데이터가 발견되면 즉시 실패한다.

## 📚 참고

- **메모리:** "추천/백테스트 로직 일관성 체크리스트" 참고
- **코드:** `logic/recommend/portfolio.py`, `logic/backtest/portfolio_runner.py`
- **공통 함수:** `logic/common/portfolio.py`, `logic/common/signals.py`, `logic/common/filtering.py`
- **설정:** `zsettings/account/k1.json`
