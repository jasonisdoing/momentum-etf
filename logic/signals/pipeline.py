"""Signals pipeline.

This module contains the end-to-end signal generation pipeline, including
generate_signal_report() and the CLI-facing main() entrypoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd

# optional deps
try:
    import pytz  # type: ignore
except Exception:  # pragma: no cover
    pytz = None

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None

from logic.signals.formatting import (
    _get_header_money_formatter,
    _load_display_precision,
    _load_precision_all,
)
from logic.signals.logger import get_signal_logger
from logic.signals.schedule import (
    is_market_open,
    determine_target_date_for_scheduler as _determine_target_date_for_scheduler,
)
from logic.signals.benchmarks import _is_trading_day
from utils.account_registry import (
    get_account_info,
    get_account_file_settings,
    get_strategy_rules_for_account,
    get_common_file_settings,
)
from utils.db_manager import (
    save_signal_report_to_db,
    get_trades_on_date,
    get_portfolio_snapshot,
    get_previous_portfolio_snapshot,
)
from utils.report import format_kr_money, render_table_eaw, format_aud_money
from utils.stock_list_io import get_etfs
from logic.momentum import (
    generate_daily_signals_for_portfolio,
    DECISION_CONFIG,
    COIN_ZERO_THRESHOLD,
)
from logic.strategies.momentum.shared import SIGNAL_TABLE_HEADERS
from logic.strategies.momentum.labeler import compute_net_trade_note
from logic.signals.history import (
    calculate_consecutive_holding_info,
    calculate_trade_cooldown_info,
)


@dataclass
class SignalReportData:
    portfolio_data: Dict
    data_by_tkr: Dict
    total_holdings_value: float
    datestamps: List
    pairs: List[Tuple[str, str]]
    base_date: pd.Timestamp
    regime_info: Optional[Dict]
    full_etf_meta: Dict
    etf_meta: Dict
    failed_tickers_info: Dict
    realtime_prices: Dict[str, Optional[float]]


@dataclass
class SignalExecutionResult:
    report_date: datetime
    summary_data: Dict[str, Any]
    header_line: str
    detail_headers: List[str]
    detail_rows: List[List[str]]
    detail_extra_lines: List[str]
    decision_config: Dict[str, Any]


def _resolve_previous_close(close_series: pd.Series, base_date: pd.Timestamp) -> float:
    """Return most recent close before base_date (or 0.0 if unavailable)."""
    if close_series is None or close_series.empty:
        return 0.0
    try:
        closes_until_base = close_series.loc[:base_date]
    except Exception:  # noqa: E722
        closes_until_base = close_series[close_series.index <= base_date]

    if closes_until_base.empty:
        return 0.0

    last_idx = closes_until_base.index[-1]
    if pd.notna(last_idx) and pd.Timestamp(last_idx).normalize() == base_date.normalize():
        closes_until_base = closes_until_base.iloc[:-1]

    if closes_until_base.empty:
        if len(close_series) >= 2:
            candidate = close_series.iloc[-2]
            return float(candidate) if pd.notna(candidate) else 0.0
        return 0.0

    candidate = closes_until_base.iloc[-1]
    return float(candidate) if pd.notna(candidate) else 0.0


def _load_ticker_data(
    tkr: str,
    country: str,
    required_months: int,
    base_date: pd.Timestamp,
    df_full: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Load single ticker OHLCV up to base_date, optionally from prefetched df_full."""
    if df_full is None:
        from utils.data_loader import fetch_ohlcv

        df = fetch_ohlcv(tkr, country=country, months_back=required_months, base_date=base_date)
    else:
        df = df_full[df_full.index <= base_date].copy()

    if df is None or df.empty:
        return None
    return df


def _calculate_indicators(args: Tuple) -> Tuple[str, Dict[str, Any]]:
    """Calculate moving average based indicators for given df."""
    from utils.indicators import calculate_moving_average_signals

    (
        tkr,
        df,
        base_date,
        ma_period,
        realtime_price,
        is_realtime_only,
    ) = args

    if df is None:
        return tkr, {"error": "PROCESS_ERROR"}

    if realtime_price is not None and pd.notna(realtime_price):
        df.loc[base_date, "Close"] = realtime_price

    if not is_realtime_only and (df is None or len(df) < ma_period):
        return tkr, {"error": "INSUFFICIENT_DATA"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]

    close_prices = df["Close"]

    moving_average, buy_signal_active, consecutive_buy_days = calculate_moving_average_signals(
        close_prices, ma_period
    )

    return tkr, {
        "df": df,
        "close": close_prices,
        "ma": moving_average,
        "buy_signal_days": consecutive_buy_days,
        "ma_period": ma_period,
        "unadjusted_close": df.get("unadjusted_close"),
    }


def _build_universe(
    country: str, account: str, base_date: pd.Timestamp
) -> Tuple[Dict[str, Dict], Dict[str, Dict], List[Tuple[str, str]]]:
    """Build complete universe: holdings, meta, and processing pairs."""
    all_etfs_from_file = get_etfs(country)
    full_etf_meta = {etf["ticker"]: etf for etf in all_etfs_from_file}

    portfolio_data = get_portfolio_snapshot(
        country, account=account, date_str=base_date.strftime("%Y-%m-%d")
    )
    if not portfolio_data:
        raise ValueError(f"'{base_date.strftime('%Y-%m-%d')}' 날짜의 포트폴리오 스냅샷을 찾을 수 없습니다.")

    holdings = _normalize_holdings(portfolio_data.get("holdings", []))

    # include sold tickers on base date
    sold_tickers_today = set()
    trades_on_base_date = get_trades_on_date(country, account, base_date)
    for trade in trades_on_base_date:
        if trade["action"] == "SELL":
            ticker = trade["ticker"]
            sold_tickers_today.add(ticker)
            if ticker not in full_etf_meta:
                full_etf_meta[ticker] = {
                    "ticker": ticker,
                    "name": trade.get("name", ""),
                    "category": "",
                }
            if ticker not in holdings:
                holdings[ticker] = {"name": trade.get("name", ""), "shares": 0, "avg_cost": 0.0}

    all_tickers_for_processing = set(holdings.keys()) | set(full_etf_meta.keys())
    pairs: List[Tuple[str, str]] = []
    for tkr in sorted(list(all_tickers_for_processing)):
        name = full_etf_meta.get(tkr, {}).get("name") or holdings.get(tkr, {}).get("name") or ""
        pairs.append((tkr, name))

    if country == "coin":
        allowed_tickers = {etf["ticker"] for etf in all_etfs_from_file}
        pairs = [(t, n) for t, n in pairs if t in allowed_tickers or t in sold_tickers_today]

    return holdings, full_etf_meta, pairs


def _fetch_and_prepare_data(
    country: str,
    account: str,
    portfolio_settings: Dict,
    date_str: Optional[str],
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
    *,
    deterministic: bool = False,
    no_realtime: bool = False,
) -> Optional[SignalReportData]:
    """Fetch OHLCV and compute indicators for universe.

    Returns SignalReportData or None when infeasible.
    """
    logger = get_signal_logger()
    try:
        initial_date_ts = pd.to_datetime(portfolio_settings["initial_date"]).normalize()
        base_date = pd.to_datetime(date_str).normalize()
    except (ValueError, TypeError, AttributeError):
        raise ValueError(f"날짜 형식 변환 실패: {date_str}")

    if base_date < initial_date_ts:
        print(
            f"정보: 요청된 날짜({base_date.strftime('%Y-%m-%d')})가 계좌 시작일({initial_date_ts.strftime('%Y-%m-%d')}) 이전이므로 현황을 계산하지 않습니다."
        )
        return None

    # 1) universe
    try:
        holdings, etf_meta, pairs = _build_universe(country, account, base_date)
    except ValueError as e:
        print(f"오류: {e}")
        return None

    logger.info(
        "[%s/%s] Universe built for %s: holdings=%d, meta=%d, total_pairs=%d",
        country.upper(),
        account,
        base_date.strftime("%Y-%m-%d"),
        len(holdings),
        len(etf_meta),
        len(pairs),
    )

    # realtime usage decision
    today_cal = (
        pd.Timestamp.now(tz="Asia/Seoul").normalize()
        if deterministic
        else pd.Timestamp.now().normalize()
    )
    use_realtime = False
    if base_date.date() == today_cal.date():
        if country == "coin":
            use_realtime = True
        elif country == "kor":
            if pytz:
                try:
                    seoul_tz = pytz.timezone("Asia/Seoul")
                    now_local = datetime.now(seoul_tz)
                    if _is_trading_day(country, now_local) and now_local.hour >= 9:
                        use_realtime = True
                except Exception:
                    pass
        else:  # aus
            use_realtime = is_market_open(country)

    # Override by flags
    if deterministic or no_realtime:
        use_realtime = False
        print(
            f"-> [DEBUG] use_realtime overridden to False (deterministic={deterministic}, no_realtime={no_realtime})"
        )

    try:
        common = get_common_file_settings()
        regime_filter_enabled = common["MARKET_REGIME_FILTER_ENABLED"]
        regime_ma_period = common["MARKET_REGIME_FILTER_MA_PERIOD"]
    except (SystemExit, KeyError, ValueError, TypeError) as e:
        print(f"오류: 공통 설정을 불러오는 중 문제가 발생했습니다: {e}")
        return None

    ma_period = portfolio_settings["ma_period"]
    max_ma_period = max(ma_period, regime_ma_period if regime_filter_enabled else 0)
    required_days = max_ma_period + 5
    required_months = (required_days // 22) + 2

    # realtime prices
    realtime_prices: Dict[str, Optional[float]] = {}
    if use_realtime:
        if country == "kor":
            print("-> 장중 또는 장 마감 직후입니다. 네이버 금융에서 실시간 시세를 가져옵니다.")
        elif country == "coin":
            print("-> 실시간 시세를 가져옵니다 (코인).")
        else:
            print("-> 장중입니다. 실시간 시세를 가져옵니다.")

        print("-> 실시간 가격 일괄 조회 시작...")

        def _fetch_realtime_price(tkr_local: str) -> Optional[float]:
            from utils.data_loader import (
                fetch_naver_realtime_price,
                fetch_bithumb_realtime_price,
                fetch_au_realtime_price,
            )

            if country == "kor":
                return fetch_naver_realtime_price(tkr_local)
            elif country == "coin":
                return fetch_bithumb_realtime_price(tkr_local)
            elif country == "aus":
                return fetch_au_realtime_price(tkr_local)
            return None

        for tkr, _ in pairs:
            rt_price = _fetch_realtime_price(tkr)
            if rt_price is not None:
                realtime_prices[tkr] = rt_price
                print(f"   [DEBUG] realtime {tkr} -> {rt_price}")
        print(f"-> 실시간 가격 조회 완료 ({len(realtime_prices)}/{len(pairs)}개 성공).")

    # regime filter
    regime_info = None
    if regime_filter_enabled:
        if "MARKET_REGIME_FILTER_TICKER" not in common:
            print("오류: 공통 설정에 MARKET_REGIME_FILTER_TICKER 값이 없습니다.")
            return None
        regime_ticker = str(common["MARKET_REGIME_FILTER_TICKER"])

        from utils.data_loader import fetch_ohlcv

        df_regime = fetch_ohlcv(
            regime_ticker,
            country=country,
            months_range=[required_months, 0],
            base_date=base_date,
        )

        if df_regime is not None and not df_regime.empty:
            try:
                df_regime.index = pd.to_datetime(df_regime.index).normalize()
                df_regime = df_regime[~df_regime.index.duplicated(keep="last")]
            except Exception:
                pass

            if use_realtime and yf:
                try:
                    ticker_obj = yf.Ticker(regime_ticker)
                    hist = ticker_obj.history(period="2d", interval="15m", auto_adjust=True)
                    if not hist.empty:
                        latest_price = hist["Close"].iloc[-1]
                        df_regime.loc[base_date, "Close"] = latest_price
                        print(f"-> 시장 레짐 필터({regime_ticker}) 실시간 가격 적용: {latest_price:,.2f}")
                except Exception as e:
                    print(f"-> 경고: 시장 레짐 필터({regime_ticker}) 실시간 가격 조회 실패: {e}")

        if df_regime is not None and not df_regime.empty and len(df_regime) >= regime_ma_period:
            df_regime["MA"] = df_regime["Close"].rolling(window=regime_ma_period).mean()
            current_price = df_regime["Close"].iloc[-1]
            current_ma = df_regime["MA"].iloc[-1]
            if pd.notna(current_price) and pd.notna(current_ma) and current_ma > 0:
                proximity_pct = ((current_price / current_ma) - 1) * 100
                is_risk_off = current_price < current_ma
                regime_info = {
                    "ticker": regime_ticker,
                    "price": current_price,
                    "ma": current_ma,
                    "proximity_pct": proximity_pct,
                    "is_risk_off": is_risk_off,
                }

    data_by_tkr: Dict[str, Dict[str, Any]] = {}
    total_holdings_value = 0.0
    datestamps: List[pd.Timestamp] = []
    failed_tickers_info: Dict[str, str] = {}

    desc = "과거 데이터 처리" if prefetched_data else "종목 데이터 로딩"
    print(f"-> {desc} 시작... (총 {len(pairs)}개 종목)")

    for i, (tkr, _) in enumerate(pairs):
        try:
            df_full = prefetched_data.get(tkr) if prefetched_data else None
            df = _load_ticker_data(tkr, country, required_months, base_date, df_full)
            is_realtime_only = False

            if df is not None and not df.empty and len(df) < ma_period:
                is_realtime_only = True

            if df is None:
                rt_price = realtime_prices.get(tkr)
                if rt_price is not None and pd.notna(rt_price):
                    print(f"\n-> 정보: {tkr}의 과거 데이터는 없지만 실시간 가격({rt_price})이 있어 처리를 계속합니다.")
                    df = pd.DataFrame([{"Close": rt_price}], index=[base_date])
                    is_realtime_only = True
                else:
                    failed_tickers_info[tkr] = "INSUFFICIENT_DATA"
                    continue

            _, result = _calculate_indicators(
                (tkr, df, base_date, ma_period, realtime_prices.get(tkr), is_realtime_only)
            )

            if "error" in result:
                failed_tickers_info[tkr] = result["error"]
            else:
                data_by_tkr[tkr] = result

        except Exception as exc:
            print(f"\n-> 경고: {tkr} 데이터 처리 중 오류 발생: {exc}")
            failed_tickers_info[tkr] = "PROCESS_ERROR"
            logger.error("[%s] %s data processing error", country, tkr, exc_info=True)

        print(f"\r   {desc} 진행: {i + 1}/{len(pairs)}", end="", flush=True)

    print(f"\n-> {desc} 완료.")

    # final combine
    print("\n-> 최종 데이터 조합 및 계산 시작...")
    for tkr, _ in pairs:
        result = data_by_tkr.get(tkr)
        if not result:
            continue

        c0 = float(result["close"].iloc[-1])
        if pd.isna(c0) or c0 <= 0:
            failed_tickers_info[tkr] = "INVALID_PRICE"
            continue

        m = result["ma"].iloc[-1]
        today_cal = pd.Timestamp.now().normalize()
        date_for_prev_close = today_cal if base_date.date() > today_cal.date() else base_date
        prev_close = _resolve_previous_close(result["close"], date_for_prev_close)

        ma_score = 0.0
        if pd.notna(m) and m > 0:
            ma_score = round(((c0 / m) - 1.0) * 100, 1)
        print(
            f"[DEBUG_SCORE] {tkr}: price={c0}, MA={m}, score={ma_score} (formula=round(((price/MA)-1)*100,1))"
        )
        try:
            tail_ma = result["ma"].iloc[-5:].tolist()
        except Exception:
            tail_ma = []
        print(
            f"[DEBUG] {tkr}: base={base_date.date()} price={c0} MA={m} prev_close={prev_close} ma_period={result['ma_period']} tail_MA={tail_ma}"
        )
        buy_signal_days_today = (
            result["buy_signal_days"].iloc[-1] if not result["buy_signal_days"].empty else 0
        )

        sh = float((holdings.get(tkr) or {}).get("shares") or 0.0)
        ac = float((holdings.get(tkr) or {}).get("avg_cost") or 0.0)
        total_holdings_value += sh * c0
        datestamps.append(result["df"].index[-1])

        data_by_tkr[tkr] = {
            "price": c0,
            "prev_close": prev_close,
            "s1": m,
            "s2": result["ma_period"],
            "score": ma_score,
            "filter": buy_signal_days_today,
            "shares": sh,
            "avg_cost": ac,
            "df": result["df"],
        }

    fail_counts: Dict[str, int] = {}
    for reason in failed_tickers_info.values():
        fail_counts[reason] = fail_counts.get(reason, 0) + 1

    logger.info(
        "[%s] signal data summary for %s: processed=%d, failures=%s",
        country.upper(),
        base_date.strftime("%Y-%m-%d"),
        len(data_by_tkr),
        fail_counts or "{}",
    )

    portfolio_data = get_portfolio_snapshot(country, account=account, date_str=date_str)
    return SignalReportData(
        portfolio_data=portfolio_data,
        data_by_tkr=data_by_tkr,
        total_holdings_value=total_holdings_value,
        datestamps=datestamps,
        pairs=pairs,
        base_date=base_date,
        regime_info=regime_info,
        full_etf_meta=etf_meta,
        etf_meta=etf_meta,
        failed_tickers_info=failed_tickers_info,
        realtime_prices=realtime_prices,
    )


def _build_header_line(
    country,
    account: str,
    portfolio_data,
    current_equity,
    data_by_tkr,
    base_date,
    portfolio_settings: Dict,
    summary_data: Dict,
) -> Tuple[str, pd.Timestamp, str]:
    """Build header line for report."""
    money_formatter = _get_header_money_formatter(country)

    if country == "coin":
        held_count = sum(
            1
            for v in portfolio_data.get("holdings", [])
            if float(v.get("shares", 0)) > COIN_ZERO_THRESHOLD
        )
    else:
        held_count = sum(
            1 for v in portfolio_data.get("holdings", []) if float(v.get("shares", 0)) > 0
        )

    portfolio_topn = portfolio_settings.get("portfolio_topn", 0) if portfolio_settings else 0

    summary_data["held_count"] = held_count
    summary_data["portfolio_topn"] = portfolio_topn

    from utils.notification import build_summary_line_from_summary_data

    header_body = build_summary_line_from_summary_data(
        summary_data, money_formatter, use_html=True, prefix=None
    )

    initial_date = summary_data.get("initial_date")
    if initial_date and base_date >= initial_date:
        try:
            from utils.data_loader import get_trading_days

            trading_days_count = len(
                get_trading_days(
                    initial_date.strftime("%Y-%m-%d"), base_date.strftime("%Y-%m-%d"), country
                )
            )
            since_str = f"(Since {initial_date.strftime('%Y-%m-%d')})"
            header_body += (
                f' | <span style="color:blue">{trading_days_count} 거래일째</span> {since_str}'
            )
        except Exception:
            pass

    today_cal = pd.Timestamp.now().normalize()
    label_date = base_date
    if base_date.date() < today_cal.date():
        day_label = "기준일"
    else:
        day_label = "다음 거래일" if base_date.date() > today_cal.date() else "오늘"

    return header_body, label_date, day_label


def _normalize_holdings(raw_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Normalize holdings list to ticker-keyed map."""
    normalized: Dict[str, Dict[str, float]] = {}
    for item in raw_items or []:
        tkr = item.get("ticker")
        if not tkr:
            continue
        normalized[str(tkr)] = {
            "name": item.get("name", ""),
            "shares": float(item.get("shares", 0.0) or 0.0),
            "avg_cost": float(item.get("avg_cost", 0.0) or 0.0),
        }
    return normalized


def _apply_live_balance_to_holdings(
    portfolio_data: Dict[str, Any],
    balance: Dict[str, Any],
    universe_pairs: List[Tuple[str, str]],
) -> Dict[str, float]:
    """Apply live balance quantities into holdings and return per-ticker shares map."""

    def _parse_balance(value: Any) -> float:
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return 0.0

    holdings_list = list(portfolio_data.get("holdings") or [])
    holdings_by_ticker: Dict[str, Dict[str, Any]] = {}
    live_share_map: Dict[str, float] = {}

    for entry in holdings_list:
        ticker = str(entry.get("ticker") or "").upper()
        if not ticker:
            continue
        holdings_by_ticker[ticker] = entry

    name_lookup = {str(t).upper(): n for t, n in universe_pairs}

    seen: set[str] = set()
    updated_holdings: List[Dict[str, Any]] = []

    for ticker, entry in holdings_by_ticker.items():
        live_key_upper = f"total_{ticker.upper()}"
        live_key_lower = f"total_{ticker.lower()}"
        live_amount = 0.0
        if live_key_upper in balance:
            live_amount = _parse_balance(balance[live_key_upper])
        elif live_key_lower in balance:
            live_amount = _parse_balance(balance[live_key_lower])

        if live_amount <= COIN_ZERO_THRESHOLD:
            live_amount = 0.0

        entry = dict(entry)
        entry["shares"] = live_amount
        updated_holdings.append(entry)
        seen.add(ticker.upper())
        live_share_map[ticker.upper()] = live_amount

    for key, value in balance.items():
        if not isinstance(key, str) or not key.lower().startswith("total_"):
            continue
        symbol = key.split("_", 1)[-1].upper()
        if symbol in {"KRW", "P"}:
            continue
        if symbol in seen:
            continue

        live_amount = _parse_balance(value)
        if live_amount <= COIN_ZERO_THRESHOLD:
            continue

        updated_holdings.append(
            {
                "ticker": symbol,
                "name": name_lookup.get(symbol, symbol),
                "shares": live_amount,
                "avg_cost": 0.0,
            }
        )
        seen.add(symbol)
        live_share_map[symbol] = live_amount

    portfolio_data["holdings"] = updated_holdings
    return live_share_map


def _calculate_portfolio_summary(
    country: str,
    account: str,
    portfolio_data: Dict,
    current_equity: float,
    data_by_tkr: Dict,
    base_date: pd.Timestamp,
    portfolio_settings: Dict,
) -> Dict[str, Any]:
    """Compute portfolio summary stats and returns."""
    from utils.transaction_manager import get_transactions_up_to_date
    from utils.account_registry import get_account_info
    from utils.data_loader import get_aud_to_krw_rate, get_usd_to_krw_rate

    account_info = get_account_info(account)
    # 통화 기본값: 국가 기준으로 안전한 기본값을 사용 (aus→AUD, 그 외→KRW)
    if account_info:
        currency = str(account_info.get("currency") or "").upper()
    else:
        currency = ""
    if not currency:
        currency = "AUD" if country == "aus" else "KRW"

    total_holdings = sum(d["shares"] * d["price"] for d in data_by_tkr.values() if d["shares"] > 0)

    if country == "aus":
        intl_info = portfolio_data.get("international_shares")
        if isinstance(intl_info, dict):
            try:
                total_holdings += float(intl_info.get("value", 0.0))
            except (TypeError, ValueError):
                pass

    total_cash = float(current_equity) - float(total_holdings)

    equity_for_cum_calc = current_equity
    if country == "aus" and total_holdings > 1 and current_equity > 1:
        if (current_equity / total_holdings) > 10:
            equity_for_cum_calc = total_holdings

    initial_capital_from_file = (
        float(portfolio_settings.get("initial_capital_krw", 0)) if portfolio_settings else 0.0
    )
    initial_date = (
        pd.to_datetime(portfolio_settings.get("initial_date"))
        if portfolio_settings and portfolio_settings.get("initial_date")
        else None
    )

    aud_krw_rate = None
    usd_krw_rate = None
    if currency == "AUD":
        aud_krw_rate = get_aud_to_krw_rate()
        # Fallback: AUD/KRW = USD/KRW * AUD/USD
        if not aud_krw_rate and yf:
            try:
                usd_krw_rate = get_usd_to_krw_rate()
            except Exception:
                usd_krw_rate = None
            try:
                if usd_krw_rate:
                    ticker_obj = yf.Ticker("AUDUSD=X")
                    hist = ticker_obj.history(period="2d")
                    if not hist.empty:
                        aud_usd = float(hist["Close"].iloc[-1])
                        if aud_usd and aud_usd > 0:
                            aud_krw_rate = usd_krw_rate * aud_usd
            except Exception:
                pass
        if aud_krw_rate:
            current_equity *= aud_krw_rate
            total_holdings *= aud_krw_rate
            total_cash *= aud_krw_rate
            equity_for_cum_calc *= aud_krw_rate
    elif currency == "USD":
        usd_krw_rate = get_usd_to_krw_rate()
        if usd_krw_rate:
            current_equity *= usd_krw_rate
            total_holdings *= usd_krw_rate
            equity_for_cum_calc *= usd_krw_rate

    injections = get_transactions_up_to_date(country, account, base_date, "capital_injection")
    withdrawals = get_transactions_up_to_date(country, account, base_date, "cash_withdrawal")
    total_injections = sum(inj.get("amount", 0.0) for inj in injections)
    total_withdrawals = sum(wd.get("amount", 0.0) for wd in withdrawals)

    adjusted_capital_base = initial_capital_from_file + total_injections
    adjusted_equity = equity_for_cum_calc + total_withdrawals

    cum_ret_pct = (
        ((adjusted_equity / adjusted_capital_base) - 1.0) * 100.0
        if adjusted_capital_base > 0
        else 0.0
    )
    cum_profit_loss = adjusted_equity - adjusted_capital_base

    prev_domestic_holdings_value = sum(
        d["shares"] * d["prev_close"]
        for d in data_by_tkr.values()
        if d.get("shares", 0) > 0 and d.get("prev_close", 0) > 0
    )

    prev_international_value = 0.0
    international_shares_value = None
    international_change_pct = 0.0

    if country == "aus":
        intl_info = portfolio_data.get("international_shares")
        if isinstance(intl_info, dict):
            try:
                international_shares_value = float(intl_info.get("value", 0.0))
            except (TypeError, ValueError):
                international_shares_value = 0.0
            try:
                international_change_pct = float(intl_info.get("change_pct", 0.0))
            except (TypeError, ValueError):
                international_change_pct = 0.0

            denominator = 1.0 + (international_change_pct / 100.0)
            if denominator > 0:
                prev_international_value = international_shares_value / denominator
            else:
                prev_international_value = 0.0

    if currency == "AUD" and aud_krw_rate:
        prev_domestic_holdings_value *= aud_krw_rate
        prev_international_value *= aud_krw_rate
        if international_shares_value is not None:
            international_shares_value *= aud_krw_rate

    prev_total_holdings = prev_domestic_holdings_value + prev_international_value
    prev_equity_from_holdings = prev_total_holdings + total_cash if prev_total_holdings > 0 else 0.0

    prev_equity = prev_equity_from_holdings
    if account:
        try:
            prev_snapshot = get_previous_portfolio_snapshot(
                country, base_date.to_pydatetime(), account
            )
        except Exception:
            prev_snapshot = None

        if prev_snapshot:
            prev_snapshot_equity = float(prev_snapshot.get("total_equity", 0.0) or 0.0)
            if currency == "AUD" and aud_krw_rate:
                prev_snapshot_equity *= aud_krw_rate
            if prev_snapshot_equity > 0:
                prev_equity = prev_snapshot_equity

    day_ret_pct = (
        ((current_equity / prev_equity) - 1.0) * 100.0 if prev_equity and prev_equity > 0 else 0.0
    )
    day_profit_loss = current_equity - prev_equity if prev_equity else 0.0

    total_acquisition_cost = sum(
        d["shares"] * d["avg_cost"] for d in data_by_tkr.values() if d["shares"] > 0
    )

    if country == "aus":
        intl_info = portfolio_data.get("international_shares")
        if isinstance(intl_info, dict):
            international_shares_value = float(intl_info.get("value", 0.0))
            change_pct = float(intl_info.get("change_pct", 0.0))
            cost = (
                international_shares_value / (1 + change_pct / 100)
                if (1 + change_pct / 100) != 0
                else 0
            )
            total_acquisition_cost += cost

    if currency == "AUD" and aud_krw_rate:
        total_acquisition_cost *= aud_krw_rate

    eval_ret_pct = (
        ((total_holdings / total_acquisition_cost) - 1.0) * 100.0
        if total_acquisition_cost > 0
        else 0.0
    )
    eval_profit_loss = total_holdings - total_acquisition_cost

    return {
        "principal": adjusted_capital_base,
        "total_equity": current_equity,
        "total_holdings_value": total_holdings,
        "total_cash": total_cash,
        "daily_profit_loss": day_profit_loss,
        "daily_return_pct": day_ret_pct,
        "eval_profit_loss": eval_profit_loss,
        "eval_return_pct": eval_ret_pct,
        "cum_profit_loss": cum_profit_loss,
        "cum_return_pct": cum_ret_pct,
        "initial_date": initial_date,
        "aud_krw_rate": aud_krw_rate,
        "usd_krw_rate": usd_krw_rate,
    }


def _get_calculation_message_lines(warnings: List[str]) -> List[str]:
    message_lines: List[str] = []
    if warnings:
        max_warnings = 10
        message_lines.append("- 경고:")
        for i, warning in enumerate(warnings):
            if i < max_warnings:
                message_lines.append(f"  ⚠️ {warning}")
        if len(warnings) > max_warnings:
            message_lines.append(f"  ... 외 {len(warnings) - max_warnings}건의 경고가 더 있습니다.")
    return message_lines


def generate_signal_report(
    account: str,
    date_str: Optional[str] = None,
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
    *,
    deterministic: bool = False,
    no_realtime: bool = False,
) -> Optional[Tuple[str, List[str], List[List[str]], pd.Timestamp, List[str], Dict[str, Any]]]:
    """Generate today's signal report for an account."""
    logger = get_signal_logger()
    account_info = get_account_info(account)
    if not account_info:
        raise ValueError(f"등록되지 않은 계좌입니다: {account}")

    country = str(account_info.get("country") or "").strip()
    if not country:
        raise ValueError(f"'{account}' 계좌에 국가 정보가 없습니다.")

    need_signal = account_info.get("need_signal", True)
    strategy_rules = get_strategy_rules_for_account(account)

    # 1) determine date
    if date_str:
        try:
            target_date = pd.to_datetime(date_str).normalize()
        except (ValueError, TypeError):
            raise ValueError(f"잘못된 날짜 형식입니다: {date_str}")
    else:
        target_date = _determine_target_date_for_scheduler(country)

    if country != "coin":
        if not _is_trading_day(country, target_date.to_pydatetime()):
            raise ValueError(f"휴장일({target_date.strftime('%Y-%m-%d')})에는 시그널을 생성할 수 없습니다.")

    effective_date_str = target_date.strftime("%Y-%m-%d")

    # 2) load settings
    try:
        account_settings = get_account_file_settings(account)
        strategy_dict = strategy_rules.to_dict()
        portfolio_settings = {**account_settings, **strategy_dict}
        portfolio_settings["country"] = country
    except SystemExit as e:
        print(str(e))
        return None

    # 3) data
    print(f"\n데이터 로드 (기준일: {effective_date_str})...")
    result = _fetch_and_prepare_data(
        country, account, portfolio_settings, effective_date_str, prefetched_data
    )

    if result is None:
        print("오류: 시그널 생성에 필요한 데이터를 로드하지 못했습니다.")
        return None

    portfolio_data = result.portfolio_data
    data_by_tkr = result.data_by_tkr
    total_holdings_value = result.total_holdings_value
    pairs = result.pairs
    base_date = result.base_date
    etf_meta = result.etf_meta
    full_etf_meta = result.full_etf_meta
    failed_tickers_info = result.failed_tickers_info
    realtime_prices = result.realtime_prices or {}

    logger.info(
        "[%s] decision build starting: pairs=%d, successes=%d, failures=%d",
        country.upper(),
        len(pairs),
        len(data_by_tkr),
        len(failed_tickers_info),
    )

    hard_failure_reasons = ["FETCH_FAILED", "PROCESS_ERROR"]
    fetch_failed_tickers = [
        tkr for tkr, reason in failed_tickers_info.items() if reason in hard_failure_reasons
    ]
    insufficient_data_tickers = [
        tkr for tkr, reason in failed_tickers_info.items() if reason == "INSUFFICIENT_DATA"
    ]

    if insufficient_data_tickers:
        logger.info(
            "[%s] tickers skipped due to insufficient data: %s",
            country.upper(),
            ",".join(sorted(insufficient_data_tickers)),
        )

    if fetch_failed_tickers:
        raise ValueError(f"PRICE_FETCH_FAILED:{','.join(sorted(list(set(fetch_failed_tickers))))}")

    warning_messages_for_slack: List[str] = []
    if insufficient_data_tickers:
        name_map = {tkr: name for tkr, name in pairs}
        for tkr in sorted(insufficient_data_tickers):
            name = name_map.get(tkr, tkr)
            warning_messages_for_slack.append(f"{name}({tkr}): 데이터 기간이 부족하여 계산에서 제외됩니다.")

    slack_message_lines = _get_calculation_message_lines(warning_messages_for_slack)

    holdings = _normalize_holdings(portfolio_data.get("holdings", []))
    current_equity = float(portfolio_data.get("total_equity", 0.0))
    equity_date = portfolio_data.get("equity_date")

    international_shares_value = 0.0
    if country == "aus":
        intl_info = portfolio_data.get("international_shares")
        if isinstance(intl_info, dict):
            try:
                international_shares_value = float(intl_info.get("value", 0.0))
            except (TypeError, ValueError):
                international_shares_value = 0.0
    new_equity_candidate = total_holdings_value + international_shares_value

    today_norm = pd.Timestamp.now().normalize()
    base_norm = base_date.normalize() if isinstance(base_date, pd.Timestamp) else today_norm
    allow_live_balance = country == "coin" and base_norm >= today_norm

    if country == "coin" and allow_live_balance:
        try:
            from scripts.snapshot_bithumb_balances import (
                _fetch_bithumb_balance_dict as fetch_bithumb_balance_dict,
            )

            bal = fetch_bithumb_balance_dict()
            if bal:
                krw_balance = float(bal.get("total_krw", 0.0) or 0.0)
                p_balance = float(bal.get("total_P", 0.0) or 0.0)

                live_share_map: Dict[str, float] = {}
                try:
                    live_share_map = _apply_live_balance_to_holdings(portfolio_data, bal, pairs)
                except Exception as exc:
                    logger.warning(
                        "[%s/%s] Failed to apply live balance to holdings: %s",
                        country.upper(),
                        account,
                        exc,
                    )
                    live_share_map = {}

                if live_share_map:
                    for ticker_upper, shares in live_share_map.items():
                        entry = data_by_tkr.get(ticker_upper)
                        if entry is None and ticker_upper.lower() in data_by_tkr:
                            entry = data_by_tkr.get(ticker_upper.lower())
                        if entry is None and ticker_upper.upper() in data_by_tkr:
                            entry = data_by_tkr.get(ticker_upper.upper())

                        if entry is not None:
                            entry["shares"] = shares
                        else:
                            price = realtime_prices.get(ticker_upper)
                            if price is None:
                                from utils.data_loader import fetch_bithumb_realtime_price

                                price = fetch_bithumb_realtime_price(ticker_upper)
                            if price and price > 0:
                                data_by_tkr[ticker_upper] = {
                                    "price": price,
                                    "prev_close": price,
                                    "s1": 0.0,
                                    "s2": 0.0,
                                    "score": 0.0,
                                    "filter": 0,
                                    "shares": shares,
                                    "avg_cost": 0.0,
                                    "df": pd.DataFrame([{"Close": price}], index=[base_date]),
                                }

                    total_holdings_value = sum(
                        max(0.0, float(entry.get("shares", 0.0) or 0.0))
                        * max(0.0, float(entry.get("price", 0.0) or 0.0))
                        for entry in data_by_tkr.values()
                    )

                    holdings = _normalize_holdings(portfolio_data.get("holdings", []))

                new_equity_candidate = (
                    total_holdings_value + international_shares_value + krw_balance + p_balance
                )
        except Exception as e:
            logger.warning("Bithumb 잔액 조회 실패. 평가금액 자동 보정 시 코인 가치만 반영됩니다. (%s)", e)
    elif country == "coin":
        new_equity_candidate = current_equity

    is_carried_forward = (
        equity_date
        and base_date
        and pd.to_datetime(equity_date).normalize() != base_date.normalize()
    )

    final_equity = current_equity
    updated_by = None
    old_equity_for_log = current_equity

    if is_carried_forward:
        final_equity = current_equity
        updated_by = "스케줄러(이월)"
    else:
        should_autocorrect = False
        autocorrect_reason = ""
        if country == "coin" and allow_live_balance:
            if abs(new_equity_candidate - current_equity) > 1e-9:
                should_autocorrect = True
                autocorrect_reason = "보정"
        elif new_equity_candidate > 0 and (
            new_equity_candidate > current_equity or current_equity == 0
        ):
            should_autocorrect = True
            autocorrect_reason = "보정"

        if should_autocorrect:
            final_equity = new_equity_candidate
            updated_by = f"스케줄러({autocorrect_reason})"

    if updated_by:
        from utils.db_manager import save_daily_equity

        is_data_to_save = None
        if country == "aus":
            is_data_to_save = portfolio_data.get("international_shares")

        save_success = save_daily_equity(
            country,
            account,
            base_date.to_pydatetime(),
            final_equity,
            is_data_to_save,
            updated_by=updated_by,
        )

        if save_success:
            if "보정" in updated_by:
                if abs(final_equity - old_equity_for_log) >= 1.0:
                    log_msg = f"평가금액 자동 보정: {old_equity_for_log:,.0f}원 -> {final_equity:,.0f}원"
                    print(f"-> {log_msg}")
                    diff = final_equity - old_equity_for_log
                    diff_str = f"{'+' if diff > 0 else ''}{format_kr_money(diff)}"
                    slack_message_lines.append(
                        "평가금액 보정: "
                        f"{format_kr_money(old_equity_for_log)} => {format_kr_money(final_equity)} ({diff_str})"
                    )
            logger.info(
                "[%s/%s] Daily equity updated by %s on %s: %0.2f",
                country.upper(),
                account,
                updated_by,
                base_date.strftime("%Y-%m-%d"),
                final_equity,
            )

            current_equity = final_equity
            portfolio_data["total_equity"] = final_equity
        else:
            logger.error(
                "[%s/%s] daily_equities 저장 실패: %s",
                country.upper(),
                account,
                base_date.strftime("%Y-%m-%d"),
            )

    held_categories = set()
    for tkr, d in holdings.items():
        if float(d.get("shares", 0.0)) > 0:
            category = etf_meta.get(tkr, {}).get("category")
            if category and category != "TBD":
                held_categories.add(category)

    total_holdings_value += international_shares_value

    summary_data = _calculate_portfolio_summary(
        country, account, portfolio_data, current_equity, data_by_tkr, base_date, portfolio_settings
    )
    header_line, label_date, day_label = _build_header_line(
        country,
        account,
        portfolio_data,
        current_equity,
        data_by_tkr,
        base_date,
        portfolio_settings,
        summary_data,
    )

    if insufficient_data_tickers:
        name_map = {tkr: name for tkr, name in pairs}
        warning_messages: List[str] = []
        for tkr in sorted(insufficient_data_tickers):
            name = name_map.get(tkr, tkr)
            warning_messages.append(f"{name}({tkr}): 데이터 기간이 부족하여 계산에서 제외됩니다.")

        if warning_messages:
            full_warning_str = "<br>".join(
                [f"<span style='color:orange;'>⚠️ {msg}</span>" for msg in warning_messages]
            )
            header_line += f"<br>{full_warning_str}"

    if not need_signal:
        warning_str = '<br><span style="color:orange;">⚠️ 이 계좌는 시그널 생성을 하지 않도록 설정되어 있습니다.</span>'
        header_line += warning_str

    held_tickers = [tkr for tkr, v in holdings.items() if float((v or {}).get("shares") or 0.0) > 0]
    consecutive_holding_info = calculate_consecutive_holding_info(
        held_tickers, country, account, label_date
    )
    for tkr, d in data_by_tkr.items():
        if float(d.get("shares", 0.0)) > 0:
            buy_date = consecutive_holding_info.get(tkr, {}).get("buy_date")
            if (
                buy_date
                and isinstance(d.get("df"), pd.DataFrame)
                and not d["df"].empty
                and isinstance(d["df"].index, pd.DatetimeIndex)
            ):
                try:
                    buy_date_norm = pd.to_datetime(buy_date).normalize()
                    df_holding_period = d["df"][d["df"].index >= buy_date_norm]
                    if not df_holding_period.empty and "High" in df_holding_period.columns:
                        peak_high = df_holding_period["High"].max()
                        current_price = d["price"]
                        if pd.notna(peak_high) and peak_high > 0 and pd.notna(current_price):
                            d["drawdown_from_peak"] = ((current_price / peak_high) - 1.0) * 100.0
                except Exception:
                    pass

    cooldown_days = int(account_settings.get("cooldown_days", 0))

    try:
        common = get_common_file_settings()
        stop_loss_raw = float(common["HOLDING_STOP_LOSS_PCT"])
        stop_loss = -abs(stop_loss_raw)
    except (SystemExit, KeyError, ValueError, TypeError) as e:
        print(f"오류: 공통 설정을 불러오는 중 문제가 발생했습니다: {e}")
        return None

    total_cash = float(current_equity) - float(total_holdings_value)

    all_tickers_for_cooldown: List[str] = sorted({tkr for tkr, _ in pairs}.union(held_tickers))
    trade_cooldown_info = calculate_trade_cooldown_info(
        all_tickers_for_cooldown, country, account, label_date
    )

    decisions = generate_daily_signals_for_portfolio(
        country=country,
        account=account,
        base_date=base_date,
        portfolio_settings=portfolio_settings,
        strategy_rules=strategy_rules,
        data_by_tkr=data_by_tkr,
        holdings=holdings,
        etf_meta=etf_meta,
        full_etf_meta=full_etf_meta,
        regime_info=result.regime_info,
        current_equity=current_equity,
        total_cash=total_cash,
        pairs=pairs,
        consecutive_holding_info=consecutive_holding_info,
        stop_loss=stop_loss,
        COIN_ZERO_THRESHOLD=COIN_ZERO_THRESHOLD,
        DECISION_CONFIG=DECISION_CONFIG,
        trade_cooldown_info=trade_cooldown_info,
        cooldown_days=cooldown_days,
    )

    if not need_signal:
        for decision in decisions:
            state = decision["state"]
            is_recommendation = DECISION_CONFIG.get(state, {}).get("is_recommendation", False)

            if is_recommendation:
                tkr = decision["tkr"]
                holding_info = holdings.get(tkr, {})
                sh = float(holding_info.get("shares", 0.0))
                is_effectively_held = (sh > COIN_ZERO_THRESHOLD) if country == "coin" else (sh > 0)

                new_state = "HOLD" if is_effectively_held else "WAIT"
                decision["state"] = new_state
                decision["row"][4] = new_state

            decision["row"][-1] = ""

    trades_on_base_date = get_trades_on_date(country, account, label_date)
    executed_buys_today = {
        trade["ticker"]
        for trade in trades_on_base_date
        if str(trade.get("action", "")).upper() == "BUY"
    }
    sell_trades_today: Dict[str, List[Dict[str, Any]]] = {}
    buy_trades_today_map: Dict[str, List[Dict[str, Any]]] = {}
    for trade in trades_on_base_date:
        action = str(trade.get("action", "")).upper()
        if action == "SELL":
            tkr = trade["ticker"]
            if tkr not in sell_trades_today:
                sell_trades_today[tkr] = []
            sell_trades_today[tkr].append(trade)
        elif action == "BUY":
            tkr = trade["ticker"]
            if tkr not in buy_trades_today_map:
                buy_trades_today_map[tkr] = []
            buy_trades_today_map[tkr].append(trade)

    prev_holdings_map: Dict[str, float] = {}
    try:
        ps = locals().get("prev_snapshot")
        if isinstance(ps, dict) and isinstance(ps.get("holdings"), list):
            prev_holdings_map = {
                str(h.get("ticker")): float(h.get("shares", 0.0) or 0.0)
                for h in ps.get("holdings", [])
                if isinstance(h, dict)
            }
    except Exception:
        prev_holdings_map = {}

    for decision in decisions:
        tkr = decision["tkr"]
        overrides = compute_net_trade_note(
            country=country,
            tkr=tkr,
            data_by_tkr=data_by_tkr,
            buy_trades_today_map=buy_trades_today_map,
            sell_trades_today_map=sell_trades_today,
            prev_holdings_map=prev_holdings_map,
            COIN_ZERO_THRESHOLD=COIN_ZERO_THRESHOLD,
        )
        if not overrides:
            continue
        state = overrides.get("state")
        row4 = overrides.get("row4")
        note = overrides.get("note")
        if state:
            decision["state"] = state
        if row4:
            decision["row"][4] = row4
        if note is not None:
            decision["row"][-1] = note

    wait_decisions = [d for d in decisions if d["state"] == "WAIT"]
    other_decisions = [d for d in decisions if d["state"] != "WAIT"]

    # WAIT 항목 개수 제한 제거: 모든 WAIT 항목을 포함하여 정렬 기준(order -> score)에 따라 함께 정렬합니다.
    decisions = other_decisions + wait_decisions

    def sort_key(decision_dict):
        state = decision_dict["state"]
        score = decision_dict["score"]
        tkr = decision_dict["tkr"]
        order = DECISION_CONFIG.get(state, {}).get("order", 99)
        sort_value = -score
        return (order, sort_value, tkr)

    decisions.sort(key=sort_key)

    rows_sorted: List[List[Any]] = []
    for i, decision_dict in enumerate(decisions, 1):
        row = decision_dict["row"]
        row[0] = i
        rows_sorted.append(row)

    international_shares_data = None
    if country == "aus":
        international_shares_data = portfolio_data.get("international_shares")

    if country == "aus" and international_shares_data:
        is_value = international_shares_data.get("value", 0.0)
        is_change_pct = international_shares_data.get("change_pct", 0.0)
        is_weight_pct = (is_value / current_equity) * 100.0 if current_equity > 0 else 0.0

        special_row = [
            0,
            "IS",
            "International Shares",
            "-",
            "HOLD",
            "-",
            "-",
            is_value,
            0.0,
            "1",
            is_value,
            is_change_pct,
            is_weight_pct,
            "-",
            "-",
            "-",
            "International Shares",
        ]

        rows_sorted.insert(0, special_row)
        for i, row in enumerate(rows_sorted, 1):
            row[0] = i

    headers = list(SIGNAL_TABLE_HEADERS)

    state_counts: Dict[str, int] = {}
    for row in rows_sorted:
        state = row[4]
        state_counts[state] = state_counts.get(state, 0) + 1

    logger.info(
        "[%s] signal report ready: rows=%d state_counts=%s",
        country.upper(),
        len(rows_sorted),
        state_counts,
    )

    return (header_line, headers, rows_sorted, base_date, slack_message_lines, summary_data)


def main(
    account: str,
    date_str: Optional[str] = None,
    *,
    deterministic: bool = False,
    no_realtime: bool = False,
) -> Optional[SignalExecutionResult]:
    """Run signal generation and return a structured result for notifications/UI."""
    if not account:
        raise ValueError("account is required for signal generation")

    account_info = get_account_info(account)
    if not account_info:
        raise ValueError(f"등록되지 않은 계좌입니다: {account}")

    country = str(account_info.get("country") or "").strip()
    if not country:
        raise ValueError(f"'{account}' 계좌에 국가 정보가 없습니다.")

    result = generate_signal_report(account, date_str)

    if not result:
        return None

    (
        header_line,
        headers,
        rows_sorted,
        report_base_date,
        slack_message_lines,
        summary_data,
    ) = result

    # Save report to DB for later UI retrieval
    try:
        save_signal_report_to_db(
            country,
            account,
            report_base_date.to_pydatetime(),
            (header_line, headers, rows_sorted),
            summary_data,
        )
    except Exception:
        pass

    # Console formatting (kept for parity with previous behavior)
    col_indices: Dict[str, int] = {}
    try:
        score_header_candidates = ["점수", "모멘텀점수", "MA스코어"]
        for h in score_header_candidates:
            if h in headers:
                col_indices["score"] = headers.index(h)
                break
        col_indices["day_ret"] = headers.index("일간수익률")
        col_indices["cum_ret"] = headers.index("누적수익률")
        col_indices["weight"] = headers.index("비중")
    except (ValueError, KeyError):
        pass

    display_rows: List[List[Any]] = []
    prec = _load_display_precision()
    p_daily = max(0, int(prec.get("daily_return_pct", 2)))
    p_cum = max(0, int(prec.get("cum_return_pct", 2)))
    p_w = max(0, int(prec.get("weight_pct", 2)))

    all_prec = _load_precision_all()
    cprec = (all_prec.get("country") or {}).get(country, {}) if isinstance(all_prec, dict) else {}
    curmap = (all_prec.get("currency") or {}) if isinstance(all_prec, dict) else {}
    stock_ccy = str(cprec.get("stock_currency", "KRW")) if isinstance(cprec, dict) else "KRW"
    qty_p = int(cprec.get("stock_qty_precision", 0)) if isinstance(cprec, dict) else 0
    if isinstance(cprec, dict) and ("stock_amt_precision" in cprec):
        amt_p = int(cprec.get("stock_amt_precision", 0))
    else:
        amt_p = int(((curmap.get(stock_ccy) or {}).get("precision", 0)))

    try:
        col_price = headers.index("현재가")
    except ValueError:
        col_price = None
    try:
        col_amount = headers.index("금액")
    except ValueError:
        col_amount = None

    for row in rows_sorted:
        display_row = list(row)

        idx = col_indices.get("score")
        if idx is not None:
            val = display_row[idx]
            if isinstance(val, (int, float)):
                display_row[idx] = f"{val:.1f}"
            elif val is None or not str(val).strip():
                display_row[idx] = "-"

        idx = col_indices.get("day_ret")
        if idx is not None:
            val = display_row[idx]
            if isinstance(val, (int, float)):
                display_row[idx] = ("{:+." + str(p_daily) + "f}%").format(val)
            else:
                display_row[idx] = "-"

        idx = col_indices.get("cum_ret")
        if idx is not None:
            val = display_row[idx]
            if isinstance(val, (int, float)):
                display_row[idx] = ("{:+." + str(p_cum) + "f}%").format(val)
            else:
                display_row[idx] = "-"

        idx = col_indices.get("weight")
        if idx is not None:
            val = display_row[idx]
            if isinstance(val, (int, float)):
                display_row[idx] = ("{:." + str(p_w) + "f}%").format(val)
            else:
                display_row[idx] = "-"

        idx = col_indices.get("shares")
        if idx is not None:
            val = display_row[idx]
            if isinstance(val, (int, float)):
                if qty_p > 0:
                    s = f"{float(val):.{qty_p}f}".rstrip("0").rstrip(".")
                    display_row[idx] = s if s != "" else "0"
                else:
                    display_row[idx] = f"{int(round(val)):,d}"
            else:
                display_row[idx] = val

        if col_price is not None and isinstance(display_row[col_price], (int, float)):
            fmt = ("{:, ." + str(amt_p) + "f}") if amt_p > 0 else "{:, .0f}"
            fmt = fmt.replace(" ", "")
            display_row[col_price] = fmt.format(float(display_row[col_price]))
        if col_amount is not None and isinstance(display_row[col_amount], (int, float)):
            fmt = ("{:, ." + str(amt_p) + "f}") if amt_p > 0 else "{:, .0f}"
            fmt = fmt.replace(" ", "")
            display_row[col_amount] = fmt.format(float(display_row[col_amount]))

        display_rows.append(display_row)

    aligns = [
        "right",  # #
        "right",  # 티커
        "left",  # 종목명
        "left",  # 카테고리
        "center",  # 상태
        "left",  # 매수일자
        "right",  # 보유일
        "right",  # 현재가
        "right",  # 일간수익률
        "right",  # 보유수량
        "right",  # 금액
        "right",  # 누적수익률
        "right",  # 비중
        "right",  # 고점대비
        "right",  # 점수
        "center",  # 지속
        "left",  # 문구
    ]

    render_table_eaw(headers, display_rows, aligns=aligns)

    summary_line_plain = None
    try:
        from utils.notification import build_summary_line_from_summary_data

        summary_line_plain = build_summary_line_from_summary_data(
            summary_data, _get_header_money_formatter(country), use_html=False, prefix=None
        )
    except Exception:
        # Fallback: KRW formatter
        total_equity = float(summary_data.get("total_equity", 0.0) or 0.0)
        summary_line_plain = f"금액: {format_kr_money(total_equity)}"

    print("\n" + (summary_line_plain or ""))

    cash_amount = float(summary_data.get("total_cash", 0.0) or 0.0)
    total_equity = float(summary_data.get("total_equity", 0.0) or 0.0)

    try:
        idx_ticker = headers.index("티커")
        idx_amount = headers.index("금액")
    except ValueError:
        idx_ticker = idx_amount = None

    breakdown_items: List[tuple[float, str]] = []
    if idx_ticker is not None and idx_amount is not None:
        for row in rows_sorted:
            amount = row[idx_amount]
            try:
                value = float(amount)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            ticker = row[idx_ticker]
            breakdown_items.append((value, ticker))

    breakdown_items.sort(key=lambda x: x[0], reverse=True)

    if breakdown_items or cash_amount:
        print("보유 자산 구성:")
        ticker_name_map: Dict[str, str] = {}
        try:
            for item in get_etfs(country) or []:
                code = item.get("ticker")
                if code:
                    ticker_name_map[str(code)] = item.get("name", "")
        except Exception:
            ticker_name_map = {}

        money_fmt = format_aud_money if country == "aus" else format_kr_money
        for value, ticker in breakdown_items:
            name_lookup = ticker_name_map.get(ticker) or ticker
            display_name = (
                f"{ticker}({name_lookup})" if name_lookup and name_lookup != ticker else ticker
            )
            print(f"  - {display_name}: {money_fmt(value)}")
        print(f"  - 현금: {money_fmt(cash_amount)}")
        print(f"  = 합계: {money_fmt(total_equity)}")

    return SignalExecutionResult(
        report_date=report_base_date.to_pydatetime(),
        summary_data=summary_data,
        header_line=header_line,
        detail_headers=headers,
        detail_rows=rows_sorted,
        detail_extra_lines=slack_message_lines,
        decision_config=DECISION_CONFIG,
    )


__all__ = ["main", "generate_signal_report"]
