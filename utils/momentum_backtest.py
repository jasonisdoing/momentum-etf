"""모멘텀 전략 주간 리밸런싱 백테스트.

방식
----
- **매주 재선정한다.** 판정과 체결을 분리한다: **주 마지막 거래일 종가**까지의
  데이터로 선정을 계산하고, **다음 주 첫 거래일 시가**에 체결한다. 한 주 종가를
  모두 보고 판정한 뒤 주말 동안 검토할 시간을 두는 리듬이며, 종가로 판정한 것을
  같은 종가에 체결하는 동시성 편향(look-ahead)도 없다.
- 선정은 꾸준한 모멘텀 점수(연율화 상대기울기 × R²) 순 — 화면 선정과 같은
  ``rank_candidates`` 를 써서 두 화면이 항상 일치한다.
- 슬롯 고정 비중: 종목당 1/top_n 로 배분하고, 자격 종목이 top_n 보다 적으면
  빈 슬롯은 현금으로 남긴다.
- **주중 매도**(simulate_intraweek_exits): 보유 종목이 보유 자격(장기 이격 > 0 &
  단기 이격 >= 0, hold_eligible_mask 와 동일)을 잃으면 다음 거래일 시가에 판다.
  판 슬롯은 다음 주 교체까지 현금이다 — 주중 재매수는 하지 않는다.
- 슬리피지는 편도(%)로, 리밸런싱에서 실제 매매되는 금액 전체에 부과한다:
  편출 전량 매도 + 편입 1/N 매수 + **유지 종목의 1/N 재조정 매매**(한 달간
  흘러간 비중과 목표 1/N 의 차이)까지 포함 — 완전 리밸런싱 모델과 비용이 일치한다.
- 벤치마크: 종목풀 설정(DB `pool_settings`)에 등록된 벤치마크 티커를 그대로 쓴다.
  미설정이면 fallback 없이 에러다.

한계(화면에 명시): 현재 종목풀 기준이라 상장폐지·풀 이탈 종목이 빠진
생존 편향이 있다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils.momentum_service import (
    REBALANCE_MODE_HOLD,
    REBALANCE_MODE_WEEKLY,
    IntraweekSeriesCache,
    adr_gate_blocked,
    available_backtest_months,
    benchmark_info,
    load_benchmark_close,
    load_price_frames,
    load_settings,
    load_universe,
    pool_info,
    rank_candidates,
    select_candidates,
    select_top,
    simulate_intraweek_exits,
    validate_settings,
    week_last_trading_day,
    week_rebalance_pair,
)
from utils.perf_metrics import daily_return_metrics
from utils.pool_settings_store import get_pool_slippage
from utils.pool_signal_backtest_service import get_max_backtest_months
from utils.price_series import positive_prices
from utils.share_allocation import ShareTarget, allocate_integer_shares, backtest_initial_capital
from utils.trade_stats import summarize_trades


def _rebalance_dates(benchmark_close: pd.Series, months: int) -> list[pd.Timestamp]:
    """주 교체일(각 주의 첫 거래일) 목록 — 최근 ``months`` 개월 구간.

    판정은 그 직전 거래일(= 전주 마지막 거래일) 종가다. 마지막 항목은 최신 거래일이라
    진행 중인 구간의 성과까지 보여준다.
    """
    index = benchmark_close.index
    week_ends = index.to_series().groupby(index.to_period("W")).min().tolist()
    start_bound = index[-1] - pd.DateOffset(months=months)
    dates = [stamp for stamp in week_ends if stamp >= start_bound]
    if index[-1] not in dates:
        dates.append(index[-1])
    if len(dates) < 2:
        raise ValueError(f"백테스트 {months}개월에 필요한 데이터가 부족합니다.")
    return dates


def _open_series(frame: pd.DataFrame) -> pd.Series | None:
    """체결가로 쓰는 시가 시계열. 시가가 없는 데이터면 None.

    0 은 거래정지 칸이라 결측으로 돌린다 — 그대로 두면 체결가 0 에 판 것이 되어
    그 구간이 -100% 로 잡힌다(국내 개별주 캐시에 실제로 들어 있다).
    """
    if frame is None or "Open" not in frame.columns:
        return None
    series = positive_prices(frame["Open"]).dropna()
    return series if not series.empty else None


def _open_return(opens: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    """시가 → 시가 수익률. 해당일 시가가 없으면 None(임의 보정하지 않는다)."""
    try:
        start_price = opens.asof(start)
        end_price = opens.asof(end)
    except Exception:
        return None
    if pd.isna(start_price) or pd.isna(end_price) or float(start_price) <= 0:
        return None
    return float(end_price) / float(start_price) - 1.0


def run_backtest(
    months: int,
    settings: dict[str, Any] | None = None,
    *,
    include_daily: bool,
    tuning_only: bool,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """월간 리밸런싱 백테스트. 월별 전략 vs 벤치마크 수익률을 반환한다.

    성과 지표와 주간 표는 항상 같은 일별 곡선에서 계산한다. ``include_daily`` 는 계산 여부가
    아니라 응답에 수천 개의 일별 행을 포함할지만 정한다. ``tuning_only`` 는 튜닝 지표에
    필요한 일별 전략 수익률·거래 통계만 만들고 화면용 주간 표·예정 행·벤치마크를 생략한다.
    동작이 달라지는 값이라 기본값을 두지 않는다.

    ``context`` 는 튜닝처럼 같은 풀로 여러 설정을 연달아 돌릴 때 쓰는 공유 캐시다
    (universe·frames·benchmark_close·정제 시계열·판정일별 후보). 비어 있는
    항목은 여기서 채워 넣으므로 빈 dict 를 넘기고 재사용하면 된다. 후보는 단기·장기
    이평에만 의존하므로 (판정일, 단기, 장기) 키로 캐시한다.
    """
    context = context if context is not None else {}
    intraweek_series_cache: IntraweekSeriesCache = context.setdefault("intraweek_series_cache", {})
    max_months = get_max_backtest_months()
    if not isinstance(months, int) or not 1 <= months <= max_months:
        raise ValueError(f"'months' 는 1~{max_months} 사이의 정수여야 합니다.")
    if settings is None:
        settings = load_settings()
    settings = validate_settings(settings)
    pool = str(settings["pool"])

    if context.get("universe") is None:
        context["universe"] = load_universe(pool)
    universe = context["universe"]
    name_by_ticker = {row["ticker"]: row["name"] for row in universe}

    def holding_label(ticker: str) -> str:
        """편입·편출 표시용 `종목명(티커)`. 이름을 모르면 티커만 쓴다."""
        name = name_by_ticker.get(ticker)
        return f"{name}({ticker})" if name else ticker

    if context.get("frames") is None:
        context["frames"] = load_price_frames(universe)
    frames = context["frames"]
    if context.get("benchmark_close") is None:
        context["benchmark_close"] = load_benchmark_close(pool)
    benchmark_close = context["benchmark_close"]

    # 실제 한계는 종목풀 데이터가 정한다 — 판정일 여유까지 반영해 여기서 다시 막는다.
    pool_max = available_backtest_months(benchmark_close, int(settings["long_ma_days"]))
    if months > pool_max:
        raise ValueError(
            f"장기 이평선 기준으로 이 종목풀은 최대 {pool_max}개월까지 백테스트할 수 있습니다 (요청 {months}개월)."
        )

    dates = _rebalance_dates(benchmark_close, months)

    # 판정 시점 = 각 교체일(체결일)의 직전 거래일. 벤치마크 달력 기준.
    bench_index = benchmark_close.index
    signal_dates: list[pd.Timestamp] = []
    for date in dates[:-1]:
        prior = bench_index[bench_index < date]
        if len(prior) == 0:
            raise ValueError("판정 기준일(직전 거래일)을 구할 수 없습니다 — 데이터가 부족합니다.")
        signal_dates.append(prior[-1])

    # '예정' 행 — 다음 교체일의 판정일(= 이번 주 마지막 거래일) 종가가 이미 확정됐으면
    # 그 판정으로 뽑힐 종목을 미리 보여준다.
    country = pool_info(pool)["country"]
    pending_signal: pd.Timestamp | None = None
    next_rebalance: pd.Timestamp | None = None
    last_cached = dates[-1]
    next_monday = (last_cached - pd.Timedelta(days=int(last_cached.weekday())) + pd.Timedelta(weeks=1)).normalize()
    for monday in (last_cached - pd.Timedelta(days=int(last_cached.weekday())), next_monday):
        pair = week_rebalance_pair(country, monday)
        if pair is None:
            continue
        rebalance_day, signal_day = pair
        if rebalance_day <= last_cached:
            continue  # 이미 지난 교체일
        next_rebalance = rebalance_day
        if signal_day <= last_cached:
            pending_signal = bench_index[bench_index <= signal_day][-1]
        break

    # 교체 시점별 후보 (모멘텀은 가격 캐시만 쓰므로 즉시 계산된다)
    candidates_by_date: list[list[dict[str, Any]]] = []
    all_signal_dates = signal_dates + ([pending_signal] if pending_signal is not None else [])
    # 표시용 ADR — 일간·주간 행에 붙인다(게이트 유무와 무관, 레짐 시장이 없으면 빈 시리즈).
    from utils.momentum_service import adr_market_of_pool, load_adr_series

    adr_market = adr_market_of_pool(pool)
    adr_series = load_adr_series(adr_market) if adr_market else pd.Series(dtype=float)

    def adr_at(stamp: pd.Timestamp) -> float | None:
        if adr_series.empty:
            return None
        value = adr_series.asof(pd.Timestamp(stamp))
        return round(float(value), 1) if pd.notna(value) else None

    candidate_cache: dict = context.setdefault("candidate_cache", {})
    ma_key = (int(settings["short_ma_days"]), int(settings["long_ma_days"]))
    # ADR 게이트는 **캐시 밖**에서 판정한다. 게이트는 후보 내용을 바꾸지 않고 "그 주를 통째로
    # 쉬는지"만 정하므로, 캐시에는 게이트 무관 후보를 두고 차단된 판정일만 빈 목록을 쓴다.
    # (거래대금 하한 때 겪은 튜닝 캐시 전염 — 첫 조합의 필터 결과가 다른 조합에 재사용 — 예방)
    ungated_settings = {**settings, "adr_floor": None}
    for signal_date in all_signal_dates:
        if adr_gate_blocked(settings, signal_date):
            candidates_by_date.append([])
            continue
        cache_key = (signal_date, *ma_key)
        if cache_key not in candidate_cache:
            candidate_cache[cache_key] = select_candidates(
                universe,
                frames,
                ungated_settings,
                as_of=signal_date,
                series_cache=intraweek_series_cache,
            )
        # 아래 단계가 후보 dict 를 손대더라도 캐시는 그대로 남게 복사본을 준다.
        candidates_by_date.append([dict(item) for item in candidate_cache[cache_key]])

    # 슬리피지는 종목풀 설정을 단일 소스로 쓴다 — 매수·매도 편도값을 각각 적용한다.
    buy_slippage_pct, sell_slippage_pct = get_pool_slippage(pool)
    buy_slippage, sell_slippage = buy_slippage_pct / 100.0, sell_slippage_pct / 100.0
    top_n = int(settings["top_n"])  # validate_settings 가 선택한 풀의 TOP_N_HOLD 로 채운다

    # 일간 표용 — 보유 구간 안에서 매일의 동일가중 포트폴리오 수익률.
    # 가격은 한 번만 정제해 재사용한다(구간마다 다시 정제하면 느리다).
    clean_closes: dict[str, pd.Series] = {}
    clean_opens: dict[str, pd.Series] = {}
    clean_benchmark: pd.Series | None = None
    if include_daily or not tuning_only:
        if "clean_closes" not in context:
            context["clean_closes"] = {
                ticker: positive_prices(frame["Close"]).dropna() for ticker, frame in frames.items()
            }
            opens_by: dict[str, pd.Series] = {}
            for ticker, frame in frames.items():
                opens = _open_series(frame)
                if opens is not None:
                    opens_by[ticker] = opens
            context["clean_opens"] = opens_by
            context["clean_benchmark"] = pd.to_numeric(benchmark_close, errors="coerce").dropna()
        clean_closes = context["clean_closes"]
        clean_opens = context["clean_opens"]
        clean_benchmark = context["clean_benchmark"]

    def opens_for(ticker: str) -> pd.Series | None:
        """정제 시가 — 일별 계산 컨텍스트가 있으면 전 구간에서 같은 시리즈를 재사용한다."""
        cached = clean_opens.get(ticker)
        if cached is not None:
            return cached
        frame = frames.get(ticker)
        return _open_series(frame) if frame is not None else None

    daily: list[dict[str, Any]] = []
    # 날짜 -> 그날의 성장배수. 구간 경계일(교체일)은 두 구간이 함께 쓰므로 곱해서 합친다.
    daily_growth: dict[str, float] = {}

    weekly: list[dict[str, Any]] = []
    # 주간 표용 매매 이벤트 — 체결일 기준. 주 행이 이걸로 그 주의 편입·편출을 만든다.
    trade_events: list[dict[str, Any]] = []
    # 체결 목록 — 편입~편출 한 쌍이 한 행. 아직 안 판 종목은 보유중 행으로 남는다.
    open_positions: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []

    def _open_price(ticker: str, day: pd.Timestamp) -> float | None:
        opens = opens_for(ticker)
        if opens is None:
            return None
        value = opens.asof(day)
        return float(value) if pd.notna(value) else None

    def _open_trade(ticker: str, day: pd.Timestamp) -> None:
        price = _open_price(ticker, day)
        if price:
            open_positions[ticker] = {"entry_date": day, "entry_price": price}

    def _open_position_row(ticker: str, position: dict[str, Any]) -> dict[str, Any]:
        """아직 안 판 종목 — 마지막 종가로 평가한 보유중 행."""
        frame = frames.get(ticker)
        close = (
            pd.to_numeric(frame["Close"], errors="coerce").dropna()
            if frame is not None and not frame.empty and "Close" in frame.columns
            else None
        )
        price = float(close.iloc[-1]) if close is not None and not close.empty else None
        return {
            "ticker": ticker,
            "name": name_by_ticker.get(ticker, ticker),
            "entry_date": position["entry_date"].strftime("%Y-%m-%d"),
            "entry_price": round(position["entry_price"], 4),
            "exit_date": None,
            "exit_price": round(price, 4) if price is not None else None,
            "return_pct": round((price / position["entry_price"] - 1) * 100, 2) if price is not None else None,
            "days": int((dates[-1] - position["entry_date"]).days),
            "reason": "보유중",
        }

    def _close_trade(ticker: str, day: pd.Timestamp, reason: str) -> None:
        position = open_positions.pop(ticker, None)
        price = _open_price(ticker, day)
        if position is None or not price:
            return
        trades.append(
            {
                "ticker": ticker,
                "name": name_by_ticker.get(ticker, ticker),
                "entry_date": position["entry_date"].strftime("%Y-%m-%d"),
                "entry_price": round(position["entry_price"], 4),
                "exit_date": day.strftime("%Y-%m-%d"),
                "exit_price": round(price, 4),
                "return_pct": round((price / position["entry_price"] - 1) * 100, 2),
                "days": int((day - position["entry_date"]).days),
                "reason": reason,
            }
        )

    # 벤치마크 시가 — 첫 구간의 시작가로 쓴다(전략이 그날 시가에 체결하므로 조건을 맞춘다).
    bench_open = pd.Series(dtype=float)
    if not tuning_only:
        from utils.benchmark_curve import load_benchmark_frame

        _bench_frame = load_benchmark_frame(pool)
        bench_open = (
            positive_prices(_bench_frame["Open"]).dropna() if "Open" in _bench_frame.columns else pd.Series(dtype=float)
        )
    # 교체 규칙 — 'hold' 면 순위가 밀려도 후보 자격이 남은 종목을 계속 들고 간다.
    hold_while_eligible = str(settings.get("rebalance_mode") or REBALANCE_MODE_WEEKLY) == REBALANCE_MODE_HOLD
    previous_holdings: set[str] = set()
    # 직전 보유 구간의 종목별 성장배수(1+수익률) — 교체 시점의 드리프트 비중 계산용.
    previous_growth: dict[str, float] = {}
    # 굴리는 자산 — 정수 주수 배분의 예산이다. 시작 자본은 통화별 상수(config).
    # 성과 지표는 예전처럼 배수로 내므로 이 값 자체는 결과에 안 나온다.
    equity = backtest_initial_capital(settings["pool"])

    for position in range(len(dates) - 1):
        start = dates[position]
        end = dates[position + 1]
        # 판정은 교체일 직전 거래일(signal_dates), 체결·보유 구간은 교체일 시가(start→end).
        scored = rank_candidates(candidates_by_date[position])
        if hold_while_eligible:
            # 자격 유지형 — 후보에 남아 있으면 순위와 무관하게 계속 들고 가고, 자격을 잃어
            # 빈 자리만 그 시점 상위 후보로 채운다. '자격' 판정은 따로 두지 않는다 —
            # `candidates_by_date` 가 이미 후보 필터를 통과한 목록이라 거기 남아 있는지가
            # 곧 자격이다(ADR 게이트에 걸린 주는 목록이 비어 전량 현금으로 간다).
            ranked = [item["ticker"] for item in scored]
            kept = [ticker for ticker in ranked if ticker in previous_holdings]
            free = max(top_n - len(kept), 0)
            holdings = kept + [ticker for ticker in ranked if ticker not in previous_holdings][:free]
        else:
            holdings = [item["ticker"] for item in select_top(scored, top_n)]
        holdings_set = set(holdings)
        added_tickers = sorted(holdings_set - previous_holdings)
        removed_tickers = sorted(previous_holdings - holdings_set)
        # 슬롯 고정 비중 — 선정이 top_n 보다 적으면 빈 슬롯은 현금으로 남긴다.
        target_weight = 1.0 / top_n

        # ── 주중 매도 — 자격 상실 종목은 다음 거래일 시가 매도 (선정 화면과 같은 함수).
        # 완결된 주는 마지막 판정일(교체일 직전)을 스캔에서 제외한다 — 그 판정의 체결은
        # 주 교체가 대신하기 때문이다(이중 계산 방지). 마지막 구간은 끝까지 스캔해
        # 마지막 종가 판정분을 '다음 거래일 매도 예정'으로 남긴다.
        scan_days = bench_index[(bench_index >= start) & (bench_index < end)]
        exits: list[dict[str, Any]] = []
        if len(scan_days) >= 2:
            exits = simulate_intraweek_exits(
                frames,
                settings,
                holdings_set,
                bench_index,
                start,
                scan_days[-2],
                series_cache=intraweek_series_cache,
            )
        exited_tickers = {x["ticker"] for x in exits}
        sell_date_by_ticker = {x["ticker"]: x["sell_date"] for x in exits}
        # 이 구간을 끝까지 들고 가는 종목 — 다음 교체·표시가 이 기준을 쓴다.
        survivors = holdings_set - exited_tickers

        for ticker in added_tickers:
            trade_events.append({"date": start, "action": "add", "ticker": ticker})
            _open_trade(ticker, start)
        for ticker in removed_tickers:
            trade_events.append({"date": start, "action": "remove", "ticker": ticker, "reason": "주간 교체"})
            _close_trade(ticker, start, "주간 교체")
        for exit_info in exits:
            reason = exit_info.get("reason") or "주중 이탈"
            trade_events.append(
                {"date": exit_info["sell_date"], "action": "remove", "ticker": exit_info["ticker"], "reason": reason}
            )
            _close_trade(exit_info["ticker"], exit_info["sell_date"], reason)

        # ── 교체 매매 금액(포트폴리오 대비 비율) ──
        # 유지 종목도 매주 1/N 로 재조정하므로, 드리프트 비중과 목표의 차이가 전부 매매다.
        # 현금 슬롯(__CASH__)은 비중 분모에는 들어가지만 매매 비용은 없다.
        if position > 0:
            growth = {ticker: previous_growth.get(ticker, 1.0) for ticker in previous_holdings}
            if "__CASH__" in previous_growth:
                growth["__CASH__"] = previous_growth["__CASH__"]
            total_growth = sum(growth.values()) or 1.0
            drifted = {ticker: value / total_growth for ticker, value in growth.items()}
            sell_notional = sum(drifted[t] for t in removed_tickers) + sum(
                max(drifted[t] - target_weight, 0.0) for t in holdings_set & previous_holdings
            )
            buy_notional = sum(target_weight for t in added_tickers) + sum(
                max(target_weight - drifted[t], 0.0) for t in holdings_set & previous_holdings
            )
        else:
            sell_notional = 0.0
            buy_notional = len(holdings) * target_weight  # 첫 구간은 투자분만 신규 매수
        # 주중 매도도 매매 금액에 넣는다 (슬롯당 1/N).
        sell_notional += len(exits) * target_weight
        cost = buy_slippage * buy_notional + sell_slippage * sell_notional

        # 보유 구간 — 주중 매도된 종목은 매도 체결일 시가까지만 수익이 발생한다.
        period_returns: dict[str, float] = {}
        for ticker in holdings:
            opens = opens_for(ticker)
            if opens is None:
                continue
            value = _open_return(opens, start, sell_date_by_ticker.get(ticker, end))
            if value is not None:
                period_returns[ticker] = value
        # ── 슬롯 비중 — **정수 주수**로 정한다 (운용 현황·다른 백테스트와 같은 함수).
        #
        # 예전에는 보유 종목마다 정확히 1/N 을 줬는데, 실제로는 소수점 주식을 못 사서
        # 1주 값에 걸린 만큼이 현금으로 남는다. 비싼 종목일수록·자본이 작을수록 커지는
        # 마찰이라, 1/N 로 두면 백테스트가 실제로 못 내는 성과를 낸다.
        # 자본은 지금까지 굴린 자산(equity)이다 — 정수 마찰은 자본 크기에 달렸다.
        unit = equity / top_n if top_n else 0.0
        # 가격은 period_returns 와 **같은 소스**(frames)에서 읽는다. clean_opens 는
        # include_daily 일 때만 채워져서, 그걸 쓰면 일간 탭을 안 볼 때 비중이 통째로 비었다.
        entry_price_by_ticker: dict[str, float] = {}
        for ticker in holdings:
            opens = opens_for(ticker)
            if opens is None:
                continue
            entry = opens.asof(start)
            if pd.notna(entry) and float(entry) > 0:
                entry_price_by_ticker[ticker] = float(entry)
        quantities = allocate_integer_shares(
            [ShareTarget(key=t, target_amount=unit, price=p) for t, p in entry_price_by_ticker.items()],
            budget=equity,
        )
        # 실제 비중 — 남는 현금은 수익률 0 인 슬롯이 된다(분모를 top_n 으로 두던 것과 같은 효과).
        weight_by_ticker = {
            ticker: quantities.get(ticker, 0) * price / equity
            for ticker, price in entry_price_by_ticker.items()
            if equity > 0
        }

        if not holdings:
            gross: float | None = 0.0  # 전량 현금인 주
        elif period_returns:
            gross = sum(weight_by_ticker.get(t, 0.0) * value for t, value in period_returns.items())
        else:
            gross = None  # 보유 종목의 가격 데이터가 전혀 없음 — 데이터 문제를 그대로 드러낸다

        strategy_pct = (gross - cost) * 100.0 if gross is not None else None
        # ── 일간 행 ──
        # 이 구간(start→end)의 보유 종목은 고정이다. 체결이 시가이므로 **start 시가**를 1 로
        # 두고, 마지막 날(또는 주중 매도일)은 그날 **시가**로 끊는다 — 종가로 재면 교체일
        # 당일 수익이 통째로 빠져 일별 곡선이 구간 수익률과 어긋난다.
        # start 는 직전 구간의 end 와 같은 날이라 날짜가 겹친다. 겹치는 날은 곱해서 하나로
        # 합친다(직전 구간의 '전일 종가 → 시가' 와 이번 구간의 '시가 → 종가').
        # 교체 비용은 구간 첫날에 한 번 반영한다(주간 계산과 같은 방식).
        window = bench_index[(bench_index >= start) & (bench_index <= end)]
        if len(window) > 0:

            def position_curve(ticker: str) -> pd.Series | None:
                """구간 시작 시가를 1 로 둔 종목별 가치. 매도일부터는 매도 시가로 고정한다."""
                closes = clean_closes.get(ticker)
                opens = clean_opens.get(ticker)
                if closes is None or opens is None:
                    return None
                base = opens.asof(start)
                if pd.isna(base) or float(base) <= 0:
                    return None
                curve = closes.reindex(window, method="ffill") / float(base)
                # 이 종목이 나가는 날 — 주중 매도일이 없으면 구간 종료일(교체일)이다.
                stop_day = sell_date_by_ticker.get(ticker, end)
                exit_price = opens.asof(stop_day)
                if pd.isna(exit_price) or float(exit_price) <= 0:
                    return curve
                return curve.where(window < stop_day, float(exit_price) / float(base))

            # 일별 곡선도 같은 정수 주수 비중으로 합친다 — 주간 수익률과 기준이 갈리면
            # 두 탭의 값이 어긋난다.
            curves = {ticker: position_curve(ticker) for ticker in holdings}
            weighted = [
                curve * weight_by_ticker.get(ticker, 0.0) for ticker, curve in curves.items() if curve is not None
            ]
            invested = sum(weight_by_ticker.get(ticker, 0.0) for ticker, curve in curves.items() if curve is not None)
            if weighted:
                # 투자되지 않은 몫은 현금(가치 1 고정).
                portfolio = pd.concat(weighted, axis=1).sum(axis=1) + float(1.0 - invested)
            elif not holdings:
                portfolio = pd.Series(1.0, index=window)  # 전량 현금인 주 — 변동 없음
            else:
                portfolio = None

            if portfolio is not None:
                for day_index, day in enumerate(window):
                    value = portfolio.iloc[day_index]
                    prior = 1.0 if day_index == 0 else portfolio.iloc[day_index - 1]
                    if pd.isna(value) or pd.isna(prior) or float(prior) <= 0:
                        continue
                    growth = float(value) / float(prior)
                    if day_index == 0:
                        growth -= cost  # 교체 매매 비용은 체결일에 한 번
                    key = day.strftime("%Y-%m-%d")
                    daily_growth[key] = daily_growth.get(key, 1.0) * growth

        # 월간·연간 표는 화면이 `daily` 를 **달력 월**로 잘라 만든다. 여기서 교체 구간 기준
        # 월별을 따로 만들지 않는다 — 구간이 월을 걸치면 앞달치가 뒷달로 넘어가서, 같은 기간이
        # 신고가·합성 화면과 다른 숫자로 보였다(2026-07: -9.20% vs -16.62%).

        # 다음 구간의 배분 예산 — 이번 구간 성과를 반영한다.
        if strategy_pct is not None:
            equity *= 1.0 + strategy_pct / 100.0

        previous_holdings = survivors
        previous_growth = {ticker: 1.0 + period_returns.get(ticker, 0.0) for ticker in previous_holdings}
        # 현금 = 빈 슬롯 + 주중 매도 슬롯(근사 1.0) — 다음 교체일 비중 분모에 남긴다.
        cash_multiplier = float(top_n - len(previous_holdings))
        if cash_multiplier > 0:
            previous_growth["__CASH__"] = cash_multiplier

    # ── 이미 체결된 최신 교체 — 마지막 캐시 거래일이 그 주의 교체일이면 그날 시가에
    # 체결이 끝났다. 구간 루프는 이 날을 구간 끝으로만 보므로 여기서 반영한다.
    current_holdings = previous_holdings
    if not tuning_only:
        last_pair = week_rebalance_pair(country, last_cached - pd.Timedelta(days=int(last_cached.weekday())))
        if last_pair is not None and last_pair[0] == last_cached:
            selection = {
                item["ticker"]
                for item in select_top(
                    rank_candidates(
                        select_candidates(
                            universe,
                            frames,
                            settings,
                            as_of=bench_index[bench_index < last_cached][-1],
                            series_cache=intraweek_series_cache,
                        )
                    ),
                    top_n,
                )
            }
            for ticker in sorted(selection - previous_holdings):
                trade_events.append({"date": last_cached, "action": "add", "ticker": ticker})
                _open_trade(ticker, last_cached)
            for ticker in sorted(previous_holdings - selection):
                trade_events.append({"date": last_cached, "action": "remove", "ticker": ticker})
                _close_trade(ticker, last_cached, "주간 교체")
            current_holdings = selection

    # ── 일간 행 조립 — 구간별로 모은 성장배수를 날짜순 행으로 편다.
    # 벤치마크·참조는 구간과 무관하므로 전체 시계열에서 한 번에 일별 변동률을 만든다
    # (구간마다 정규화하면 경계일이 두 번 세어진다).
    if daily_growth:
        span_index = bench_index[(bench_index >= dates[0]) & (bench_index <= dates[-1])]

        def _daily_changes(source: pd.Series | None) -> dict[str, float]:
            """일별 변동률(%) — **첫날은 그날 시가 대비**다.

            전략은 시작일 시가에 체결하므로 벤치마크도 같은 시점에서 출발해야 한다.
            예전에는 시작일 **직전 거래일**부터 시계열을 잡아, 첫 행이 '전날 종가 → 시작일
            종가' 였다. 그 하루가 통째로 여분이라 연간·월간 표의 합계가 총수익과 어긋났다
            (VOO 24개월 기준 -2.9%p).
            """
            if source is None:
                return {}
            series = source.reindex(span_index, method="ffill")
            changes = series.pct_change()
            first_day = span_index[0]
            base = bench_open.get(first_day)
            if base is not None and pd.notna(base) and float(base) > 0 and pd.notna(series.iloc[0]):
                changes.iloc[0] = float(series.iloc[0]) / float(base) - 1.0
            return {day.strftime("%Y-%m-%d"): float(v) * 100.0 for day, v in changes.items() if pd.notna(v)}

        bench_changes = {} if tuning_only else _daily_changes(clean_benchmark)
        # 소수 6자리 — 화면은 2자리로 보여주지만, 연간·월간·주간 표는 이 값을 **복리로 합성**한다.
        # 2자리로 잘라 보내면 하루치 오차(최대 0.005%p)가 250일 쌓여 합계가 총수익과 어긋난다
        # (12개월 기준 전략 0.5%p·벤치마크 0.9%p 차이가 났다).
        for key in sorted(daily_growth):
            row = {
                "date": key,
                "strategy_pct": round((daily_growth[key] - 1.0) * 100.0, 6),
            }
            if not tuning_only:
                row.update(
                    {
                        "benchmark_pct": round(bench_changes[key], 6) if key in bench_changes else None,
                        # 그날의 시장 ADR — 게이트 이해용 표시값(레짐 시장 없는 풀은 None).
                        "adr": adr_at(pd.Timestamp(key)),
                    }
                )
            daily.append(row)

    if tuning_only:
        return {"daily": list(reversed(daily)), **summarize_trades(trades)}

    # ── 주간 행 — 달력 주(월~일) 단위. 기준일은 그 주 마지막 거래일, 수익률은 그 주의
    # 성과, 편입·편출은 그 주에 체결된 매매다.
    if daily:

        def _week_monday(stamp: pd.Timestamp) -> pd.Timestamp:
            return (stamp - pd.Timedelta(days=int(stamp.weekday()))).normalize()

        def _compound_daily(values: list[float | None]) -> float | None:
            usable = [value for value in values if value is not None]
            if not usable:
                return None
            growth = 1.0
            for value in usable:
                growth *= 1.0 + value / 100.0
            return round((growth - 1.0) * 100.0, 2)

        buckets: dict[pd.Timestamp, dict[str, Any]] = {}
        for row in daily:
            stamp = pd.Timestamp(row["date"])
            bucket = buckets.setdefault(_week_monday(stamp), {"end": stamp, "strategy": [], "benchmark": []})
            bucket["end"] = max(bucket["end"], stamp)
            bucket["strategy"].append(row["strategy_pct"])
            bucket["benchmark"].append(row["benchmark_pct"])

        events_sorted = sorted(trade_events, key=lambda event: event["date"])
        event_index = 0
        holdings_running = 0
        for key in sorted(buckets):
            bucket = buckets[key]
            added_labels: list[str] = []
            removed_labels: list[str] = []
            exited_labels: list[str] = []  # 주중 이탈 — 교체 편출과 구분해 보여준다
            while event_index < len(events_sorted) and events_sorted[event_index]["date"] <= bucket["end"]:
                event = events_sorted[event_index]
                if event["action"] == "add":
                    holdings_running += 1
                    added_labels.append(holding_label(event["ticker"]))
                else:
                    holdings_running -= 1
                    reason = str(event.get("reason") or "주간 교체")
                    if reason == "주간 교체":
                        removed_labels.append(holding_label(event["ticker"]))
                    else:
                        exited_labels.append(f"{holding_label(event['ticker'])} · {reason}")
                event_index += 1
            weekly.append(
                {
                    # 기준일은 **그 주 마지막 거래일**이다. 진행 중인 주는 캐시가 거기까지밖에
                    # 없어 예전에는 화요일 같은 날짜가 찍혔다 — 완결된 주와 규칙이 달라
                    # 표에서 주가 하나 더 있는 것처럼 보였다.
                    "week_end": week_last_trading_day(country, bucket["end"]),
                    # 판정일(주 마지막 거래일)의 시장 ADR — 다음 주 게이트를 결정한 값.
                    "adr": adr_at(bucket["end"]),
                    "strategy_pct": _compound_daily(bucket["strategy"]),
                    "benchmark_pct": _compound_daily(bucket["benchmark"]),
                    # 교체 직후(주 시작) 종목 수 — 주중 이탈로 줄면 화면이 "5 → 0" 로 보여준다.
                    "holdings_start": holdings_running + len(exited_labels),
                    "holdings_count": holdings_running,
                    "turnover_pct": round(len(added_labels) / top_n * 100.0, 1),
                    "added": added_labels,
                    "removed": removed_labels,
                    "exited": exited_labels,
                }
            )

    # ── 예정 행 — 다음 주. 매매가 없어도 '보유 N 유지'가 보이도록 항상 붙인다.
    if pending_signal is not None:
        # 다음 교체 판정일 종가가 이미 확정됐다 — 그 선정을 그대로 보여준다.
        pending_holdings = {item["ticker"] for item in select_top(rank_candidates(candidates_by_date[-1]), top_n)}
        pending_added = sorted(pending_holdings - current_holdings)
        pending_removed = sorted(current_holdings - pending_holdings)
        pending_count = len(pending_holdings)
    else:
        # 판정일이 아직 오지 않았다 — 마지막 종가로 판정한 주중 매도 예정만 반영한다.
        pending_exits = simulate_intraweek_exits(
            frames,
            settings,
            current_holdings,
            bench_index,
            last_cached,
            last_cached,
            series_cache=intraweek_series_cache,
        )
        pending_added = []
        pending_removed = sorted(x["ticker"] for x in pending_exits)
        pending_count = len(current_holdings) - len(pending_removed)
        pending_exit_reason = {x["ticker"]: x.get("reason") or "주중 이탈" for x in pending_exits}

    # 예정이 속한 주 — 판정일 종가가 확정됐으면 **다음 교체 뒤의 주**, 아직이면 지금
    # 진행 중인 주다. 후자를 다음 주로 찍으면 내일 시가에 팔 종목이 한 주 뒤 행에 실려
    # 실제 체결 주와 어긋난다.
    pending_week_end = week_last_trading_day(country, next_rebalance if pending_signal is not None else last_cached)
    pending_exited = (
        []
        if pending_signal is not None
        else [f"{holding_label(t)} · {pending_exit_reason[t]} 예정" for t in pending_removed]
    )
    # 진행 중인 주는 그 주 행이 이미 있다 — 새 행을 만들지 않고 예정을 얹는다.
    current_week_row = weekly[-1] if weekly and weekly[-1]["week_end"] == pending_week_end else None
    if current_week_row is not None:
        current_week_row["holdings_count"] = pending_count
        current_week_row["exited"] = [*current_week_row["exited"], *pending_exited]
        current_week_row["is_pending"] = True
    else:
        weekly.append(
            {
                "week_end": pending_week_end,
                "strategy_pct": None,
                "benchmark_pct": None,
                "holdings_start": pending_count + len(pending_removed) if pending_signal is None else pending_count,
                "holdings_count": pending_count,
                "turnover_pct": round(len(pending_added) / top_n * 100.0, 1),
                "added": [holding_label(t) for t in pending_added],
                "removed": [holding_label(t) for t in pending_removed] if pending_signal is not None else [],
                "exited": pending_exited,
                "is_pending": True,
            }
        )

    def _summarize_daily(key: str) -> tuple[float, float | None, float | None, float | None]:
        returns = pd.Series(
            [row[key] for row in daily if row.get(key) is not None],
            index=pd.to_datetime([row["date"] for row in daily if row.get(key) is not None]),
            dtype=float,
        )
        metrics = daily_return_metrics(returns)
        return (
            round(float(metrics["total_pct"]), 2),
            round(float(metrics["mdd_pct"]), 2),
            round(float(metrics["sortino"]), 2) if metrics["sortino"] is not None else None,
            round(float(metrics["cagr_pct"]), 1) if metrics["cagr_pct"] is not None else None,
        )

    strategy_total, strategy_mdd, strategy_sortino, strategy_cagr = _summarize_daily("strategy_pct")
    benchmark_total, benchmark_mdd, benchmark_sortino, benchmark_cagr = _summarize_daily("benchmark_pct")

    return {
        "start_date": dates[0].strftime("%Y-%m-%d"),
        "end_date": dates[-1].strftime("%Y-%m-%d"),
        "months": months,
        "strategy_total_pct": strategy_total,
        "benchmark_total_pct": benchmark_total,
        "strategy_mdd_pct": strategy_mdd,
        "benchmark_mdd_pct": benchmark_mdd,
        "strategy_sortino": strategy_sortino,
        "benchmark_sortino": benchmark_sortino,
        "strategy_cagr_pct": strategy_cagr,
        "benchmark_cagr_pct": benchmark_cagr,
        "benchmark_name": benchmark_info(pool)["name"],
        "benchmark_ticker": benchmark_info(pool)["ticker"],
        # 주간 표 — 매매 내역(편입·편출·교체율·보유 수)을 담는다. 최신 주가 위.
        "weekly": list(reversed(weekly)),
        # 일간 표 — 최신 날짜가 위로 오게 뒤집는다
        "daily": list(reversed(daily)) if include_daily else [],
        # 체결 목록 — 보유중(청산 전) 행이 위, 그 아래는 청산일 최신순.
        "trades": [
            _open_position_row(ticker, position)
            for ticker, position in sorted(open_positions.items(), key=lambda item: item[1]["entry_date"], reverse=True)
        ]
        + sorted(trades, key=lambda trade: trade["exit_date"], reverse=True),
        # 거래 수·승률·평균 손익 — 세 전략이 같은 공용 계산을 쓴다(청산분만 센다).
        **summarize_trades(trades),
    }
