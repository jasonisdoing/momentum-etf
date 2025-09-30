"""국가 기반 간소화 시그널 파이프라인."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

from utils.settings_loader import (
    CountrySettingsError,
    get_country_precision,
    get_country_settings,
    get_strategy_rules,
)
from logic.strategies.momentum.constants import DECISION_CONFIG, DECISION_MESSAGES
from utils.stock_list_io import get_etfs
from utils.data_loader import fetch_ohlcv, get_latest_trading_day


@dataclass
class _TickerMeta:
    ticker: str
    name: str
    category: str


@dataclass
class _TickerScore:
    meta: _TickerMeta
    price: float
    prev_close: float
    daily_pct: float
    score: float
    streak: int


def _iter_universe(country: str) -> Iterable[_TickerMeta]:
    for item in get_etfs(country) or []:
        ticker = item.get("ticker")
        if not ticker:
            continue
        name = str(item.get("name", ticker))
        category = str(item.get("category", "-") or "-")
        yield _TickerMeta(ticker=ticker, name=name, category=category)


def _fetch_dataframe(
    ticker: str, *, country: str, ma_period: int, base_date: Optional[pd.Timestamp]
) -> Optional[pd.DataFrame]:
    months_back = max(6, ma_period // 4)
    df = fetch_ohlcv(ticker, country=country, months_back=months_back, base_date=base_date)
    if df is None or df.empty:
        return None
    df = df.dropna(subset=["Close"])
    if df.empty or len(df) <= ma_period:
        return None
    return df


def _calc_metrics(df: pd.DataFrame, ma_period: int) -> Optional[_TickerScore]:
    close = df["Close"].astype(float)
    ma = close.rolling(ma_period).mean()
    if ma.isna().all():
        return None

    latest_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else latest_close
    ma_value = float(ma.iloc[-1]) if not pd.isna(ma.iloc[-1]) else 0.0
    if ma_value <= 0 or latest_close <= 0:
        return None

    daily_pct = ((latest_close / prev_close) - 1.0) * 100 if prev_close else 0.0
    score = ((latest_close / ma_value) - 1.0) * 100

    streak = 0
    for price, ma_entry in zip(reversed(close.iloc[-ma_period:]), reversed(ma.iloc[-ma_period:])):
        if pd.isna(ma_entry) or price < ma_entry:
            break
        streak += 1

    return latest_close, prev_close, daily_pct, score, streak


def _build_score(meta: _TickerMeta, metrics) -> _TickerScore:
    price, prev_close, daily_pct, score, streak = metrics
    return _TickerScore(
        meta=meta,
        price=float(price),
        prev_close=float(prev_close),
        daily_pct=float(round(daily_pct, 2)),
        score=float(round(score, 2)),
        streak=int(streak),
    )


def _resolve_base_date(country: str, date_str: Optional[str]) -> pd.Timestamp:
    if date_str:
        try:
            base = pd.to_datetime(date_str).normalize()
        except Exception as exc:
            raise ValueError(f"잘못된 날짜 형식입니다: {date_str}") from exc
    else:
        base = get_latest_trading_day(country)
    return base.normalize()


def _apply_precision(value: float, precision: int) -> float | int:
    if precision <= 0:
        return int(round(value))
    return round(value, precision)


def _resolve_state_phrase(state: str) -> str:
    state_key = (state or "").upper()
    if state_key == "BUY":
        return DECISION_MESSAGES.get("NEW_BUY", "")
    return ""


def _resolve_state_order(state: str) -> int:
    state_key = (state or "").upper()
    cfg = DECISION_CONFIG.get(state_key, {})
    return int(cfg.get("order", 99))


def generate_country_signal_report(country: str, date_str: Optional[str] = None) -> List[dict]:
    """국가 단위 추천 종목 리스트를 반환합니다."""
    if not country:
        raise ValueError("country 인자가 필요합니다.")
    country = country.strip().lower()

    base_date = _resolve_base_date(country, date_str)

    try:
        strategy_rules = get_strategy_rules(country)
    except CountrySettingsError as exc:
        raise ValueError(str(exc)) from exc

    ma_period = int(strategy_rules.ma_period)
    portfolio_topn = int(strategy_rules.portfolio_topn)

    precision_cfg = get_country_precision(country) or {}
    price_precision = int(precision_cfg.get("price_precision", 0))
    daily_pct_precision = (
        int(precision_cfg.get("daily_pct_precision", 2))
        if "daily_pct_precision" in precision_cfg
        else 2
    )
    score_precision = (
        int(precision_cfg.get("score_precision", 2)) if "score_precision" in precision_cfg else 2
    )

    scored: List[_TickerScore] = []
    for meta in _iter_universe(country):
        df = _fetch_dataframe(
            meta.ticker, country=country, ma_period=ma_period, base_date=base_date
        )
        if df is None:
            continue
        metrics = _calc_metrics(df, ma_period)
        if metrics is None:
            continue
        scored.append(_build_score(meta, metrics))

    if not scored:
        return []

    scored.sort(key=lambda item: (-item.score, item.meta.ticker))

    max_per_category_raw = None
    try:
        country_settings = get_country_settings(country)
        strategy_cfg = (
            country_settings.get("strategy", {}) if isinstance(country_settings, dict) else {}
        )
        max_per_category_raw = strategy_cfg.get("MAX_PER_CATEGORY") or strategy_cfg.get(
            "max_per_category"
        )
    except CountrySettingsError:
        max_per_category_raw = None

    if max_per_category_raw is None:
        try:
            from utils.account_registry import (
                get_common_file_settings,
            )  # avoid cycle at import time

            common_cfg = get_common_file_settings() or {}
            max_per_category_raw = common_cfg.get("MAX_PER_CATEGORY")
        except Exception:
            max_per_category_raw = None
    try:
        max_per_category = int(max_per_category_raw) if max_per_category_raw is not None else None
    except (TypeError, ValueError):
        max_per_category = None
    if max_per_category is not None and max_per_category <= 0:
        max_per_category = None

    category_counts: Dict[str, int] = {}

    results: List[dict] = []
    for item in scored:
        category_key = item.meta.category or "-"
        if max_per_category is not None:
            current_count = category_counts.get(category_key, 0)
            if current_count >= max_per_category:
                continue

        next_rank = len(results) + 1
        state = "BUY" if next_rank <= portfolio_topn else "WAIT"
        price_value = _apply_precision(item.price, price_precision)
        daily_pct_value = round(item.daily_pct, daily_pct_precision)
        score_value = round(item.score, score_precision)

        results.append(
            {
                "rank": next_rank,
                "ticker": item.meta.ticker,
                "name": item.meta.name,
                "category": item.meta.category,
                "state": state,
                "price": price_value,
                "daily_pct": daily_pct_value,
                "score": score_value,
                "streak": item.streak,
                "base_date": base_date.strftime("%Y-%m-%d"),
                "holding_days": None,
                "phrase": _resolve_state_phrase(state),
                "state_order": _resolve_state_order(state),
            }
        )

        if max_per_category is not None:
            category_counts[category_key] = category_counts.get(category_key, 0) + 1

    # 상태 정렬 기준은 DECISION_CONFIG order -> rank 순으로 유지
    results.sort(key=lambda row: (row.get("state_order", 99), row["rank"]))

    for idx, row in enumerate(results, start=1):
        row["rank"] = idx
        row.pop("state_order", None)

    return results


__all__ = ["generate_country_signal_report"]
