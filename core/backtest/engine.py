"""
포트폴리오 백테스트 실행 모듈

전략 중립적인 포트폴리오 백테스트 로직을 제공합니다.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from config import BACKTEST_SLIPPAGE, REBALANCE_BUFFER, WEIGHT_MAX, WEIGHT_MIN
from core.backtest.price import calculate_trade_price
from core.strategy.metrics import process_ticker_data
from core.strategy.weight_allocator import calculate_score_weights, should_rebalance
from utils.logger import get_app_logger
from utils.settings_loader import get_country_precision

logger = get_app_logger()


def _floor_quantity(quantity: float, qty_precision: int) -> float:
    if quantity <= 0:
        return 0.0
    if qty_precision <= 0:
        return float(int(quantity))
    factor = 10**qty_precision
    return float(int(quantity * factor)) / factor


def _calculate_bootstrap_buy_price(
    *,
    open_values: Any,
    close_values: Any,
    country_code: str,
) -> float:
    open0 = open_values[0] if len(open_values) > 0 else float("nan")
    close0 = close_values[0] if len(close_values) > 0 else float("nan")
    if pd.notna(open0):
        base_price = float(open0)
    elif pd.notna(close0):
        base_price = float(close0)
    else:
        return 0.0
    slippage_config = BACKTEST_SLIPPAGE.get(country_code, BACKTEST_SLIPPAGE.get("kor", {}))
    slippage_pct = float(slippage_config.get("buy_pct", 0.0) or 0.0)
    return base_price * (1 + slippage_pct / 100.0)


def check_is_rebalance_day(
    dt: pd.Timestamp,
    next_dt: pd.Timestamp | None,
    rebalance_mode: str,
    trading_calendar: pd.DatetimeIndex | None = None,
) -> bool:
    """주어진 dt가 거래 신호일인지 판별합니다.

    DAILY는 당일 종가 기준으로 다음 거래일 체결을 유지합니다.
    주간/월간 계열은 실제 체결일의 직전 거래일을 신호일로 사용합니다.
    """
    if trading_calendar is not None:
        trading_calendar = (
            pd.DatetimeIndex(trading_calendar)
            if not isinstance(trading_calendar, pd.DatetimeIndex)
            else trading_calendar
        )

    def _is_execution_day(target_dt: pd.Timestamp, target_next_dt: pd.Timestamp | None) -> bool:
        if rebalance_mode == "DAILY":
            return True

        if rebalance_mode == "WEEKLY":
            if target_next_dt is not None:
                if target_next_dt.isocalendar()[:2] != target_dt.isocalendar()[:2]:
                    return True
            elif trading_calendar is not None:
                try:
                    cal_idx = trading_calendar.get_loc(target_dt)
                    if cal_idx + 1 < len(trading_calendar):
                        if trading_calendar[cal_idx + 1].isocalendar()[:2] != target_dt.isocalendar()[:2]:
                            return True
                except (KeyError, IndexError, AttributeError):
                    pass
            return False

        elif rebalance_mode == "TWICE_A_MONTH":
            import calendar

            dt_friday = target_dt + pd.Timedelta(days=4 - target_dt.weekday())
            days_in_month = calendar.monthrange(dt_friday.year, dt_friday.month)[1]
            fridays = [d for d in range(1, days_in_month + 1) if dt_friday.replace(day=d).weekday() == 4]
            if len(fridays) >= 5:
                target_fridays = [fridays[2], fridays[4]]
            else:
                target_fridays = [fridays[1], fridays[3]]

            if dt_friday.day in target_fridays:
                if target_next_dt is not None:
                    if target_next_dt.isocalendar()[:2] != target_dt.isocalendar()[:2]:
                        return True
                elif trading_calendar is not None:
                    try:
                        cal_idx = trading_calendar.get_loc(target_dt)
                        if cal_idx + 1 < len(trading_calendar):
                            if trading_calendar[cal_idx + 1].isocalendar()[:2] != target_dt.isocalendar()[:2]:
                                return True
                    except (KeyError, IndexError, AttributeError):
                        pass
            return False

        elif rebalance_mode == "MONTHLY":
            if target_next_dt is not None:
                if target_next_dt.month != target_dt.month:
                    return True
            elif trading_calendar is not None:
                try:
                    cal_idx = trading_calendar.get_loc(target_dt)
                    if cal_idx + 1 < len(trading_calendar):
                        if trading_calendar[cal_idx + 1].month != target_dt.month:
                            return True
                except (KeyError, IndexError, AttributeError):
                    pass
            return False

        elif rebalance_mode == "QUARTERLY":
            target_months = {3, 6, 9, 12}
            if target_dt.month in target_months:
                if target_next_dt is not None:
                    if target_next_dt.month != target_dt.month:
                        return True
                elif trading_calendar is not None:
                    try:
                        cal_idx = trading_calendar.get_loc(target_dt)
                        if cal_idx + 1 < len(trading_calendar):
                            if trading_calendar[cal_idx + 1].month != target_dt.month:
                                return True
                    except (KeyError, IndexError, AttributeError):
                        pass
            return False

        return False

    if rebalance_mode == "DAILY":
        return True

    if trading_calendar is None:
        return False

    try:
        cal_idx = trading_calendar.get_loc(dt)
        if isinstance(cal_idx, slice):
            cal_idx = int(cal_idx.stop) - 1
        next_signal_idx = int(cal_idx) + 1
        if next_signal_idx >= len(trading_calendar):
            return False
        execution_dt = trading_calendar[next_signal_idx]
        execution_next_dt = (
            trading_calendar[next_signal_idx + 1] if next_signal_idx + 1 < len(trading_calendar) else None
        )
    except (KeyError, IndexError, AttributeError, TypeError, ValueError):
        return False

    return _is_execution_day(execution_dt, execution_next_dt)


def _find_next_rebalance_signal_offset(
    *,
    dt: pd.Timestamp,
    rebalance_mode: str,
    trading_calendar: pd.DatetimeIndex,
) -> int | None:
    """현재 날짜 이후 다음 리밸런싱 신호일까지 남은 거래일 수를 반환합니다."""
    try:
        cal_idx = trading_calendar.get_loc(dt)
        if isinstance(cal_idx, slice):
            cal_idx = int(cal_idx.start)
        current_idx = int(cal_idx)
    except (KeyError, IndexError, AttributeError, TypeError, ValueError):
        return None

    for future_idx in range(current_idx + 1, len(trading_calendar)):
        signal_dt = trading_calendar[future_idx]
        next_dt = trading_calendar[future_idx + 1] if future_idx + 1 < len(trading_calendar) else None
        if check_is_rebalance_day(
            dt=signal_dt,
            next_dt=next_dt,
            rebalance_mode=rebalance_mode,
            trading_calendar=trading_calendar,
        ):
            return future_idx - current_idx
    return None


def _collect_metrics_by_ticker(
    *,
    stocks: list[dict],
    prefetched_data: dict[str, pd.DataFrame] | None,
    prefetched_metrics: Mapping[str, dict[str, Any]] | None,
    ma_days: int,
    ma_type: str,
    enable_data_sufficiency_check: bool,
    missing_ticker_sink: set[str] | None,
    quiet: bool,
) -> dict[str, dict[str, Any]]:
    """프리패치된 가격 데이터에서 종목별 지표를 준비합니다."""

    metrics_by_ticker: dict[str, dict[str, Any]] = {}
    tickers_to_process = [stock["ticker"] for stock in stocks]

    for ticker in tickers_to_process:
        df = prefetched_data.get(ticker) if prefetched_data else None
        if df is None:
            raise RuntimeError(f"[백테스트] '{ticker}' 데이터가 프리패치에 없습니다. 튜닝 프리패치 단계를 확인하세요.")

        precomputed_entry = prefetched_metrics.get(ticker) if prefetched_metrics else None
        ticker_metrics = process_ticker_data(
            ticker,
            df,
            ma_days=ma_days,
            ma_type=ma_type,
            precomputed_entry=precomputed_entry,
            enable_data_sufficiency_check=enable_data_sufficiency_check,
        )
        if ticker_metrics:
            metrics_by_ticker[ticker] = ticker_metrics

    missing_metrics = [ticker for ticker in tickers_to_process if ticker not in metrics_by_ticker]
    if missing_metrics:
        missing_set = {
            str(ticker).strip().upper() for ticker in missing_metrics if isinstance(ticker, str) and str(ticker).strip()
        }
        if missing_ticker_sink is not None:
            missing_ticker_sink.update(missing_set)
        else:
            logger.warning("가격 데이터 부족으로 제외된 종목: %s", ", ".join(sorted(missing_set)))

    if not quiet:
        logger.info(f"[백테스트] metrics_by_ticker: {len(metrics_by_ticker)}개 종목 처리 완료")

    return metrics_by_ticker


def _build_union_index(
    *,
    metrics_by_ticker: Mapping[str, dict[str, Any]],
    core_start_date: pd.Timestamp | None,
    quiet: bool,
) -> pd.DatetimeIndex:
    """종목별 거래일을 합쳐 백테스트 기준 인덱스를 구성합니다."""

    union_index = pd.DatetimeIndex([])
    for ticker_metrics in metrics_by_ticker.values():
        union_index = union_index.union(ticker_metrics["close"].index)

    if union_index.empty:
        return union_index

    if core_start_date:
        union_index = union_index[union_index >= core_start_date]
        if not quiet:
            logger.info(
                f"[백테스트] union_index: {len(union_index)}일 (core_start_date={core_start_date.strftime('%Y-%m-%d')})"
            )

    return union_index


def _prepare_metric_arrays(
    *,
    metrics_by_ticker: dict[str, dict[str, Any]],
    union_index: pd.DatetimeIndex,
) -> None:
    """일별 루프에서 바로 쓸 수 있도록 시계열 캐시를 채웁니다."""

    for ticker_metrics in metrics_by_ticker.values():
        close_series = ticker_metrics["close"].reindex(union_index)
        open_series = ticker_metrics["open"].reindex(union_index)
        ma_series = ticker_metrics["ma"].reindex(union_index)
        ma_score_series = ticker_metrics["ma_score"].reindex(union_index)

        ticker_metrics["close_series"] = close_series
        ticker_metrics["close_values"] = close_series.to_numpy()
        ticker_metrics["open_series"] = open_series
        ticker_metrics["open_values"] = open_series.to_numpy()
        ticker_metrics["available_mask"] = close_series.notna().to_numpy()
        ticker_metrics["ma_values"] = ma_series.to_numpy()
        ticker_metrics["ma_score_values"] = ma_score_series.to_numpy()

        buy_signal_series = ticker_metrics["buy_signal_days"].reindex(union_index).fillna(0).astype(int)
        ticker_metrics["buy_signal_series"] = buy_signal_series
        ticker_metrics["buy_signal_values"] = buy_signal_series.to_numpy()


def _initialize_bootstrap_positions(
    *,
    metrics_by_ticker: Mapping[str, dict[str, Any]],
    cash: float,
    country_code: str,
    qty_precision: int,
) -> tuple[dict[str, dict[str, float]], float, bool]:
    """첫 거래일 기준으로 전체 유니버스를 초기 편입합니다."""

    position_state = {
        ticker: {
            "shares": 0.0,
            "avg_cost": 0.0,
        }
        for ticker in metrics_by_ticker.keys()
    }
    if not metrics_by_ticker:
        return position_state, cash, False

    initial_candidates: list[tuple[float, str]] = []
    for ticker, ticker_metrics in metrics_by_ticker.items():
        score0 = ticker_metrics["ma_score_values"][0]
        initial_candidates.append((float(score0) if not pd.isna(score0) else 0.0, ticker))

    initial_candidates.sort(reverse=True)
    selected = [ticker for _, ticker in initial_candidates]
    if not selected:
        return position_state, cash, False

    bootstrap_scores = {}
    for ticker in selected:
        score0 = metrics_by_ticker[ticker]["ma_score_values"][0]
        bootstrap_scores[ticker] = float(score0) if not pd.isna(score0) else 0.0

    bootstrap_weights = calculate_score_weights(
        bootstrap_scores,
        min_weight=float(WEIGHT_MIN) / 100.0,
        max_weight=float(WEIGHT_MAX) / 100.0,
    )

    remaining_cash = float(cash)
    bootstrap_initialized = False
    for ticker in selected:
        if remaining_cash <= 0:
            break
        ticker_metrics = metrics_by_ticker[ticker]
        buy_price = _calculate_bootstrap_buy_price(
            open_values=ticker_metrics["open_values"],
            close_values=ticker_metrics["close_values"],
            country_code=country_code,
        )
        if buy_price <= 0:
            continue
        budget = float(cash) * bootstrap_weights.get(ticker, 1.0 / len(selected))
        budget = min(budget, remaining_cash)
        qty = _floor_quantity(budget / buy_price, qty_precision) if buy_price > 0 else 0.0
        trade_amount = qty * buy_price
        if qty <= 0 or trade_amount > remaining_cash + 1e-9:
            continue
        position_state[ticker]["shares"] = qty
        position_state[ticker]["avg_cost"] = buy_price
        remaining_cash -= trade_amount
        bootstrap_initialized = True

    return position_state, max(remaining_cash, 0.0), bootstrap_initialized


def run_portfolio_backtest(
    stocks: list[dict],
    initial_capital: float = 100_000_000.0,
    core_start_date: pd.Timestamp | None = None,
    top_n: int = 10,
    bucket_map: dict[str, int] | None = None,
    date_range: list[str] | None = None,
    country: str = "kor",
    prefetched_data: dict[str, pd.DataFrame] | None = None,
    prefetched_metrics: Mapping[str, dict[str, Any]] | None = None,
    trading_calendar: Sequence[pd.Timestamp] | None = None,
    ma_days: int = 20,
    ma_type: str = "SMA",
    strategy: str = "PORTFOLIO",
    rebalance_mode: str = "TWICE_A_MONTH",
    target_weights: Mapping[str, float] | None = None,
    quiet: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    missing_ticker_sink: set[str] | None = None,
    enable_data_sufficiency_check: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    이동평균 기반 모멘텀 전략으로 포트폴리오 백테스트를 실행합니다.

    Args:
        stocks: 백테스트할 종목 목록
        initial_capital: 초기 자본금
        core_start_date: 백테스트 시작일
        top_n: 계좌 전체 종목 수
        date_range: 백테스트 기간 [시작일, 종료일]
        country: 시장 국가 코드 (예: kor)
        prefetched_data: 미리 로드된 가격 데이터
        ma_days: 이동평균 기간
        rebalance_mode: 리밸런싱 모드 (WEEKLY, MONTHLY, QUARTERLY, TWICE_A_MONTH)

    Returns:
        Dict[str, pd.DataFrame]: 종목별 백테스트 결과
    """

    country_code = (country or "").strip().lower() or "kor"
    qty_precision = int(get_country_precision(country_code).get("qty_precision", 0))

    def _log(message: str) -> None:
        if quiet:
            logger.debug(message)
        else:
            logger.info(message)

    from core.backtest.portfolio import validate_universe_size

    validate_universe_size(top_n)

    metrics_by_ticker = _collect_metrics_by_ticker(
        stocks=stocks,
        prefetched_data=prefetched_data,
        prefetched_metrics=prefetched_metrics,
        ma_days=ma_days,
        ma_type=ma_type,
        enable_data_sufficiency_check=enable_data_sufficiency_check,
        missing_ticker_sink=missing_ticker_sink,
        quiet=quiet,
    )

    union_index = _build_union_index(
        metrics_by_ticker=metrics_by_ticker,
        core_start_date=core_start_date,
        quiet=quiet,
    )

    if union_index.empty:
        return {}

    if union_index.empty:
        logger.warning(
            f"[백테스트] union_index가 비어있습니다. core_start_date={core_start_date}, "
            f"metrics_by_ticker={len(metrics_by_ticker)}"
        )
        return {}

    _prepare_metric_arrays(
        metrics_by_ticker=metrics_by_ticker,
        union_index=union_index,
    )

    # 시뮬레이션 상태 변수 초기화
    # 각 종목별 직전 유효 가격을 추적 (데이터 공백 시 패딩용)
    last_prices = {ticker: 0.0 for ticker in metrics_by_ticker.keys()}
    cash = float(initial_capital)
    daily_records_by_ticker = {ticker: [] for ticker in metrics_by_ticker.keys()}
    out_cash = []
    if trading_calendar is None:
        raise RuntimeError("trading_calendar must be provided to run_portfolio_backtest.")
    trading_calendar_idx = (
        pd.DatetimeIndex(trading_calendar) if not isinstance(trading_calendar, pd.DatetimeIndex) else trading_calendar
    )

    # 일별 루프를 돌며 시뮬레이션을 실행합니다.
    total_days = len(union_index)
    _log(f"[백테스트] 총 {total_days}일의 데이터를 처리합니다... 리밸런싱 모드: {rebalance_mode}")

    position_state, cash, bootstrap_initialized = _initialize_bootstrap_positions(
        metrics_by_ticker=metrics_by_ticker,
        cash=cash,
        country_code=country_code,
        qty_precision=qty_precision,
    )

    # 이전 리밸런싱 인덱스 추적 변수는 제거함
    for i, dt in enumerate(union_index):
        next_dt = union_index[i + 1] if i < total_days - 1 else None

        # 리밸런싱 날짜 판별
        is_rebalance_day = check_is_rebalance_day(
            dt=dt, next_dt=next_dt, rebalance_mode=rebalance_mode, trading_calendar=trading_calendar_idx
        )
        if i == 0 and not bootstrap_initialized:
            is_rebalance_day = True

        # 진행률 표시 (10% 단위로)
        if i % max(1, total_days // 10) == 0 or i == total_days - 1:
            progress_pct = int((i + 1) / total_days * 100)
            _log(f"[백테스트] 진행률: {progress_pct}% ({i + 1}/{total_days}일)")
        if progress_callback is not None:
            progress_callback(i + 1, total_days)

        # 디버깅: 첫 3일만 로그
        if i < 3 and not quiet:
            logger.info(f"[백테스트] Day {i}: {dt}, metrics_by_ticker={len(metrics_by_ticker)}")

        # 당일 시작 시점 보유 수량 스냅샷(순매수/순매도 판단용)
        today_prices: dict[str, float] = {}
        score_today: dict[str, float] = {}
        eligible_scores_today: dict[str, float] = {}
        buy_signal_today: dict[str, int] = {}

        for ticker, ticker_metrics in metrics_by_ticker.items():
            available = bool(ticker_metrics["available_mask"][i])
            price_val = ticker_metrics["close_values"][i]
            price_float = float(price_val) if not pd.isna(price_val) else float("nan")
            today_prices[ticker] = price_float

            score_val = ticker_metrics["ma_score_values"][i]
            buy_signal_val = ticker_metrics["buy_signal_values"][i]

            score_today[ticker] = float(score_val) if not pd.isna(score_val) else 0.0
            if available:
                eligible_scores_today[ticker] = score_today[ticker]
            buy_signal_today[ticker] = int(buy_signal_val) if not pd.isna(buy_signal_val) else 0

            today_prices[ticker] = price_float

        target_weights_today = {ticker: 0.0 for ticker in metrics_by_ticker}
        if eligible_scores_today:
            computed_weights = calculate_score_weights(
                eligible_scores_today,
                min_weight=float(WEIGHT_MIN) / 100.0,
                max_weight=float(WEIGHT_MAX) / 100.0,
            )
            target_weights_today.update(computed_weights)

        # 현재 총 보유 자산 가치를 계산합니다.
        current_holdings_value = 0
        for held_ticker, held_state in position_state.items():
            if held_state.get("shares", 0) > 0:
                price_h = today_prices.get(held_ticker)
                if pd.notna(price_h):
                    current_holdings_value += held_state["shares"] * price_h

        # --- 1. 기본 정보 및 출력 행 생성 ---
        for ticker, ticker_metrics in metrics_by_ticker.items():
            snapshot = position_state[ticker]
            price = today_prices.get(ticker)
            available_today = ticker_metrics["available_mask"][i] and not pd.isna(price)
            ma_value = ticker_metrics["ma_values"][i]
            score_value = score_today.get(ticker, 0.0)
            filter_value = ""

            decision_out = "HOLD"

            note = ""
            # 정적인 문구 제거 ("점수 미달", "점수 없음" 등)

            ma_val = ticker_metrics["ma_values"][i]
            ma_value = float(ma_val) if not pd.isna(ma_val) else float("nan")
            score_value = score_today.get(ticker, 0.0)
            filter_value = buy_signal_today.get(ticker, 0)

            if available_today:
                last_prices[ticker] = price  # 유효한 가격 업데이트
                pv_value = snapshot["shares"] * price
                record = {
                    "date": dt,
                    "price": price,
                    "shares": snapshot["shares"],
                    "pv": pv_value,
                    "decision": decision_out,
                    "avg_cost": snapshot["avg_cost"],
                    "trade_amount": 0.0,
                    "trade_shares": 0.0,
                    "trade_profit": 0.0,
                    "trade_pl_pct": 0.0,
                    "note": note,
                    "pending_action": None,
                    "execute_on": None,
                    "pending_reason": None,
                    "signal1": ma_value if not pd.isna(ma_value) else None,
                    "signal2": None,
                    "score": score_value if not pd.isna(score_value) else None,
                    "target_weight": target_weights_today.get(ticker),
                    "filter": filter_value,
                }
            else:
                avg_cost = snapshot["avg_cost"]
                # 데이터가 없는 날(휴장일 등)에는 직전 유효 가격 사용 (없으면 avg_cost 후순위)
                display_price = last_prices.get(ticker, 0.0)
                if display_price <= 0:
                    display_price = avg_cost if pd.notna(avg_cost) else 0.0

                pv_value = snapshot["shares"] * display_price
                record = {
                    "date": dt,
                    "price": display_price,
                    "shares": snapshot["shares"],
                    "pv": pv_value,
                    "decision": "HOLD",
                    "avg_cost": avg_cost,
                    "trade_amount": 0.0,
                    "trade_shares": 0.0,
                    "trade_profit": 0.0,
                    "trade_pl_pct": 0.0,
                    "note": "데이터 없음",
                    "pending_action": None,
                    "execute_on": None,
                    "pending_reason": None,
                    "signal1": ma_value if not pd.isna(ma_value) else None,
                    "signal2": None,
                    "score": score_value if not pd.isna(score_value) else None,
                    "target_weight": target_weights_today.get(ticker),
                    "filter": filter_value,
                }

            daily_records_by_ticker[ticker].append(record)

        # --- 비중 재조정(Weight Realignment) - 리밸런싱 날에만 수행 ---
        if is_rebalance_day:
            # 당일 비중 조정 전에 총 보유 자산을 다시 계산합니다.
            total_rebalance_equity = cash
            for held_ticker, held_state in position_state.items():
                if held_state["shares"] > 0:
                    price_h = today_prices.get(held_ticker)
                    if pd.notna(price_h) and price_h > 0:
                        total_rebalance_equity += held_state["shares"] * price_h

            # 스코어 비례 목표 비중 계산
            if eligible_scores_today:
                rebalance_target_weights = calculate_score_weights(
                    eligible_scores_today,
                    min_weight=float(WEIGHT_MIN) / 100.0,
                    max_weight=float(WEIGHT_MAX) / 100.0,
                )

                # 현재 비중 계산
                current_weights: dict[str, float] = {}
                for t, s in position_state.items():
                    if s["shares"] > 0:
                        p = today_prices.get(t, 0.0)
                        if pd.notna(p) and p > 0:
                            current_weights[t] = (s["shares"] * p) / total_rebalance_equity

                # 버퍼 체크: 비중 차이가 버퍼 이내인 종목은 매매 유보
                rebalance_needed = should_rebalance(
                    current_weights,
                    rebalance_target_weights,
                    float(REBALANCE_BUFFER) / 100.0,
                )

            if eligible_scores_today:
                # 4-1. Trim (비중 축소)
                for ticker, state in position_state.items():
                    if state["shares"] > 0:
                        price = today_prices.get(ticker)
                        if pd.isna(price) or price <= 0:
                            continue

                        # 버퍼 이내면 매매 유보
                        if not rebalance_needed.get(ticker, False):
                            continue

                        current_val = state["shares"] * price
                        target_weight_for_ticker = rebalance_target_weights.get(ticker, 1.0 / top_n)
                        target_val_for_ticker = total_rebalance_equity * target_weight_for_ticker
                        if current_val > target_val_for_ticker:
                            excess_val = current_val - target_val_for_ticker

                            # 다음날 시초가 + 슬리피지로 매도 가격 계산 (실제 거래 가격)
                            sell_price = calculate_trade_price(
                                i,
                                total_days,
                                metrics_by_ticker[ticker]["open_values"],
                                metrics_by_ticker[ticker]["close_values"],
                                country_code,
                                is_buy=False,
                            )
                            if sell_price <= 0:
                                continue

                            sell_qty = _floor_quantity(excess_val / sell_price, qty_precision)
                            # 최소 1주 이상 매도 가능할 때
                            if sell_qty > 0:
                                sell_amount = sell_qty * sell_price

                                # 상태 업데이트
                                cash += sell_amount
                                state["shares"] -= sell_qty

                                current_weight_pct = (
                                    (current_val / total_rebalance_equity) * 100.0
                                    if total_rebalance_equity > 0
                                    else 0.0
                                )
                                target_weight_pct = target_weight_for_ticker * 100.0
                                trim_note = (
                                    f"[예정] 1거래일 후 비중조절 - "
                                    f"{current_weight_pct:.1f}% => {target_weight_pct:.1f}%"
                                )

                                if (
                                    daily_records_by_ticker[ticker]
                                    and daily_records_by_ticker[ticker][-1]["date"] == dt
                                ):
                                    row = daily_records_by_ticker[ticker][-1]
                                    existing_note = row.get("note", "")
                                    # 리밸런스 비중 축소는 부분 조정이므로 상태는 변경하지 않습니다.
                                    row["note"] = f"{trim_note} | {existing_note}" if existing_note else trim_note
                                    row["shares"] = state["shares"]
                                    row["pv"] = state["shares"] * price
                                    row["pending_action"] = "SELL_REBALANCE"
                                    row["execute_on"] = next_dt
                                    row["pending_reason"] = "비중 조정"
                                    # 만약 이미 거래 기록이 있다면 금액 합산, 없다면 추가
                                    if "trade_amount" in row and row["trade_amount"]:
                                        row["trade_amount"] += sell_amount
                                    else:
                                        row["trade_amount"] = sell_amount
                                    if "trade_shares" in row and row["trade_shares"]:
                                        row["trade_shares"] += sell_qty
                                    else:
                                        row["trade_shares"] = sell_qty

            # 4-2. Top-up (비중 확대)는 당일 전체에 적용되는 `PHASE 3` 추가매수 로직에서
            # 남은 현금을 모두 사용해 부족한 종목들을 채우므로 여기서 별도 진행하지 않아도 되나,
            # 명시적인 [비중조절] 퍼센트 노트를 위해 Phase 3 노트 부분만 조금 수정 (아래 참고)

        # --- PHASE 3: 추가 매수 (남은 현금으로 부족한 종목 채우기) ---
        # 비중 조절용 추가 매수는 is_rebalance_day 일 때만 동작하게 합니다.
        if is_rebalance_day and cash > 0:
            total_equity = cash + current_holdings_value

            # 스코어 비례 목표 비중 계산 (Trim과 동일한 비중 사용)
            topup_target_weights: dict[str, float] = {}
            topup_rebalance_needed: dict[str, bool] = {}
            if eligible_scores_today:
                topup_target_weights = calculate_score_weights(
                    eligible_scores_today,
                    min_weight=float(WEIGHT_MIN) / 100.0,
                    max_weight=float(WEIGHT_MAX) / 100.0,
                )
                current_weights_topup: dict[str, float] = {}
                for t, s in position_state.items():
                    if s["shares"] > 0:
                        p = today_prices.get(t, 0.0)
                        if pd.notna(p) and p > 0:
                            current_weights_topup[t] = (s["shares"] * p) / total_equity if total_equity > 0 else 0.0
                topup_rebalance_needed = should_rebalance(
                    current_weights_topup,
                    topup_target_weights,
                    float(REBALANCE_BUFFER) / 100.0,
                )

            # 목표비중 미만 종목 찾기
            # 이미 보유 중인 종목뿐 아니라, 새로 상장되어 아직 0주인 종목도 포함해야 합니다.
            underweight_tickers = []
            for ticker, state in position_state.items():
                price_now = today_prices.get(ticker, 0)
                if pd.isna(price_now) or price_now <= 0:
                    continue

                target_weight = topup_target_weights.get(ticker, 0.0)
                if target_weight <= 0:
                    continue

                # 0주 종목은 버퍼 체크 없이 신규 편입 대상으로 봅니다.
                if state["shares"] > 0 and not topup_rebalance_needed.get(ticker, False):
                    continue

                current_value = state["shares"] * price_now
                current_weight = current_value / total_equity if total_equity > 0 else 0.0
                target_value = total_equity * target_weight
                gap = target_value - current_value

                if gap > 0 and current_weight < target_weight:
                    underweight_tickers.append((ticker, gap, current_value))

            # 부족분 큰 순서로 정렬
            underweight_tickers.sort(key=lambda x: x[1], reverse=True)

            # 순서대로 채우기
            for ticker_to_topup, gap, current_value in underweight_tickers:
                if cash <= 0:
                    break

                price = today_prices.get(ticker_to_topup)
                if pd.isna(price) or price <= 0:
                    continue

                # 다음날 시초가 + 슬리피지로 매수 가격 계산
                topup_price = calculate_trade_price(
                    i,
                    total_days,
                    metrics_by_ticker[ticker_to_topup]["open_values"],
                    metrics_by_ticker[ticker_to_topup]["close_values"],
                    country_code,
                    is_buy=True,
                )
                if topup_price <= 0:
                    continue

                # 매수 가능 금액: min(gap, cash)
                topup_budget = min(gap, cash)
                topup_qty = _floor_quantity(topup_budget / topup_price, qty_precision) if topup_price > 0 else 0.0

                if topup_qty > 0:
                    topup_amount = topup_qty * topup_price

                    if topup_amount <= cash + 1e-9:
                        # 추가 매수 실행
                        ticker_state = position_state[ticker_to_topup]
                        old_shares = ticker_state["shares"]
                        old_avg_cost = ticker_state["avg_cost"]

                        cash -= topup_amount
                        current_holdings_value += topup_amount
                        ticker_state["shares"] += topup_qty

                        # 평균 단가 재계산
                        total_cost = old_shares * old_avg_cost + topup_amount
                        ticker_state["avg_cost"] = total_cost / ticker_state["shares"]

                        # 레코드 업데이트 (기존 레코드에 추가 매수 표시)
                        if (
                            daily_records_by_ticker[ticker_to_topup]
                            and daily_records_by_ticker[ticker_to_topup][-1]["date"] == dt
                        ):
                            row = daily_records_by_ticker[ticker_to_topup][-1]
                            existing_decision = row.get("decision", "")
                            existing_note = row.get("note", "")

                            # 상태는 HOLD를 유지하고 문구로만 비중 조절 매수를 표시
                            if existing_decision == "HOLD":
                                current_weight_pct = (current_weight * 100.0) if total_equity > 0 else 0.0
                                target_weight_pct = target_weight * 100.0
                                topup_note = (
                                    f"[예정] 1거래일 후 비중조절 - "
                                    f"{current_weight_pct:.1f}% => {target_weight_pct:.1f}%"
                                )
                                row["note"] = f"{topup_note} | {existing_note}" if existing_note else topup_note
                                row["pending_action"] = "BUY"
                                row["execute_on"] = next_dt
                                row["pending_reason"] = "비중 조정"

                            # 수량/금액 업데이트
                            row["shares"] = ticker_state["shares"]
                            row["pv"] = ticker_state["shares"] * price
                            row["avg_cost"] = ticker_state["avg_cost"]

                            # 거래금액 누적
                            if "trade_amount" in row and row["trade_amount"]:
                                row["trade_amount"] += topup_amount
                            else:
                                row["trade_amount"] = topup_amount
                            if "trade_shares" in row and row["trade_shares"]:
                                row["trade_shares"] += topup_qty
                            else:
                                row["trade_shares"] = topup_qty

        # --- 다음 리밸런싱 예정 비중조절 문구 보강 ---
        next_rebalance_offset = (
            1
            if is_rebalance_day
            else _find_next_rebalance_signal_offset(
                dt=dt,
                rebalance_mode=rebalance_mode,
                trading_calendar=trading_calendar_idx,
            )
        )
        if next_rebalance_offset is not None:
            total_equity_for_note = cash
            for held_ticker, held_state in position_state.items():
                if held_state["shares"] <= 0:
                    continue
                held_price = today_prices.get(held_ticker)
                if pd.isna(held_price) or held_price <= 0:
                    held_price = last_prices.get(held_ticker, 0.0)
                if held_price > 0:
                    total_equity_for_note += held_state["shares"] * held_price

            if total_equity_for_note > 0:
                for ticker, rows in daily_records_by_ticker.items():
                    if ticker == "CASH" or not rows or rows[-1]["date"] != dt:
                        continue
                    row = rows[-1]
                    if row.get("pending_action"):
                        continue

                    shares_now = float(row.get("shares", 0.0) or 0.0)
                    if shares_now <= 0:
                        continue

                    target_weight = target_weights_today.get(ticker)
                    if target_weight is None:
                        continue

                    held_price = row.get("price")
                    held_price_float = float(held_price) if pd.notna(held_price) else 0.0
                    if held_price_float <= 0:
                        held_price_float = float(last_prices.get(ticker, 0.0) or 0.0)
                    if held_price_float <= 0:
                        continue

                    current_value = shares_now * held_price_float
                    current_weight = current_value / total_equity_for_note
                    if abs(current_weight - target_weight) <= float(REBALANCE_BUFFER) / 100.0:
                        continue

                    existing_note = str(row.get("note") or "").strip()
                    if existing_note:
                        continue

                    row["note"] = (
                        f"[예정] {next_rebalance_offset}거래일 후 비중조절 - "
                        f"{current_weight * 100.0:.1f}% => {target_weight * 100.0:.1f}%"
                    )

        out_cash.append(
            {
                "date": dt,
                "price": 1.0,
                "cash": cash,
                "shares": 0,
                "pv": cash,
                "decision": "HOLD",
                "note": "",  # CASH는 문구 없음
                "pending_action": None,
                "execute_on": None,
                "pending_reason": None,
            }
        )

    total_records = sum(len(v) for v in daily_records_by_ticker.values())
    expected_records = len(metrics_by_ticker) * len(union_index)
    if not quiet:
        logger.info(
            f"[백테스트] daily_records_by_ticker: {len(daily_records_by_ticker)}개 종목, "
            f"총 {total_records}개 레코드 (예상: {expected_records}개)"
        )

    result: dict[str, pd.DataFrame] = {}
    for ticker_symbol, records in daily_records_by_ticker.items():
        if records:
            result[ticker_symbol] = pd.DataFrame(records).set_index("date")
    if out_cash:
        result["CASH"] = pd.DataFrame(out_cash).set_index("date")

    return result
