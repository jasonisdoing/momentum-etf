"""국가 기반 간소화 추천 파이프라인."""

from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import pandas as pd

# 데이터 디렉토리 경로 설정
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "stocks"
from utils.settings_loader import (
    CountrySettingsError,
    get_country_precision,
    get_country_settings,
    get_strategy_rules,
)
from logic.strategies.maps.constants import DECISION_CONFIG, DECISION_MESSAGES
from utils.stock_list_io import get_etfs
from utils.data_loader import fetch_ohlcv, get_latest_trading_day
from logic.recommend.history import calculate_consecutive_holding_info


@dataclass
class _TickerMeta:
    ticker: str
    name: str
    category: str


@dataclass
class _TickerScore:
    meta: _TickerMeta
    price: float
    prev_close: float  # 이전 종가 추가
    daily_pct: float
    score: float
    streak: int
    category: str  # 카테고리 정보 추가


def _iter_universe(country: str, universe: str) -> Iterator[_TickerMeta]:
    """Load universe from JSON file and yield ticker metadata."""
    filepath = DATA_DIR / f"{universe.lower()}.json"
    with filepath.open("r") as f:
        data = json.load(f)

    for category_group in data:
        # 카테고리 정보 가져오기 (리스트나 집합인 경우 첫 번째 항목 사용)
        category = category_group.get("category", "")
        if isinstance(category, (list, set, tuple)):
            category = next(iter(category), "")  # 첫 번째 항목 가져오기
        # 문자열로 변환하고 양쪽 공백 제거
        category = str(category).strip() if category is not None else ""

        # tickers 리스트가 있는지 확인
        if "tickers" not in category_group or not isinstance(category_group["tickers"], list):
            print(f"경고: 유효한 티커 리스트를 찾을 수 없습니다. 카테고리: {category}")
            continue

        for ticker_info in category_group["tickers"]:
            if not isinstance(ticker_info, dict) or "ticker" not in ticker_info:
                print(f"경고: 티커 정보가 올바르지 않습니다. 카테고리: {category}")
                continue

            ticker = str(ticker_info["ticker"]).strip()
            name = str(ticker_info.get("name", ticker)).strip()

            # 디버깅을 위해 카테고리 정보 출력
            if not category:
                print(f"경고: {ticker}에 대한 카테고리 정보가 없습니다.")

            # _TickerMeta 객체 생성 및 반환
            yield _TickerMeta(ticker=ticker, name=name, category=category)


def _fetch_dataframe(
    ticker: str, *, country: str, ma_period: int, base_date: Optional[pd.Timestamp]
) -> Optional[pd.DataFrame]:
    try:
        # 더 긴 기간의 데이터를 가져오기 위해 months_back를 늘립니다.
        months_back = max(12, ma_period)  # 최소 1년치 데이터 요청

        df = fetch_ohlcv(ticker, country=country, months_back=months_back, base_date=base_date)

        if df is None or df.empty:
            print(f"경고: {ticker}에 대한 데이터를 가져오지 못했습니다.")
            return None

        # Close 컬럼이 없으면 에러 메시지와 함께 None 반환
        if "Close" not in df.columns:
            print(f"경고: {ticker}에 대한 종가(Close) 데이터가 없습니다.")
            return None

        # Close가 NaN인 행 제거
        df = df.dropna(subset=["Close"])

        if df.empty:
            print(f"경고: {ticker}에 대한 유효한 데이터가 없습니다.")
            return None

        # 데이터가 충분하지 않아도 계속 진행 (나중에 _calc_metrics에서 처리)
        return df

    except Exception as e:
        print(f"경고: {ticker} 데이터 처리 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return None


def _calc_metrics(df: pd.DataFrame, ma_period: int) -> Optional[tuple]:
    try:
        # 'Close' 또는 'Adj Close' 중 사용 가능한 컬럼 선택
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        close = df[price_col].astype(float)

        # 이동평균 계산 (최소 1개 데이터로도 계산 가능하도록 min_periods=1 설정)
        ma = close.rolling(window=ma_period, min_periods=1).mean()

        latest_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else latest_close
        ma_value = float(ma.iloc[-1]) if not pd.isna(ma.iloc[-1]) else latest_close

        # 0 이하 값이면 기본값 설정
        if ma_value <= 0:
            ma_value = latest_close if latest_close > 0 else 1.0

        if latest_close <= 0:
            return None

        # 일간 수익률 계산 (이전 종가가 없거나 0 이하면 0%로 처리)
        daily_pct = 0.0
        if prev_close and prev_close > 0:
            daily_pct = ((latest_close / prev_close) - 1.0) * 100

        # 점수 계산 (이동평균 대비 수익률, % 단위)
        score = 0.0
        if ma_value > 0:
            score = ((latest_close / ma_value) - 1.0) * 100

        # 점수가 매우 작으면 0.01%로 처리 (0점 방지)
        if abs(score) < 0.01 and score != 0:
            score = 0.01 if score > 0 else -0.01

        # 연속 상승일 계산
        streak = 0
        for price, ma_entry in zip(
            reversed(close.iloc[-ma_period:]), reversed(ma.iloc[-ma_period:])
        ):
            if pd.isna(ma_entry) or pd.isna(price) or price < ma_entry:
                break
            streak += 1

        # 보유일 계산 (최대 20일로 제한)
        holding_days = min(streak, 20) if streak > 0 else 0

        return latest_close, prev_close, daily_pct, score, holding_days
    except Exception as e:
        print(f"경고: 메트릭 계산 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return None


def _build_score(meta: _TickerMeta, metrics) -> _TickerScore:
    # 메트릭이 없는 경우 기본값 반환
    if metrics is None:
        print(f"경고: {meta.ticker}에 대한 메트릭이 없습니다.")
        return _TickerScore(
            meta=meta,
            price=0.0,
            prev_close=0.0,
            daily_pct=0.0,
            score=0.0,
            streak=0,
            category="",  # 카테고리를 빈 문자열로 초기화
        )

    try:
        price, prev_close, daily_pct, score, holding_days = metrics

        # 점수가 매우 작은 경우 0으로 처리하지 않도록 합니다.
        if abs(score) < 0.01 and score != 0:
            score = 0.01 if score > 0 else -0.01

        # 카테고리 정보가 없는 경우 빈 문자열로 설정
        category = ""
        if hasattr(meta, "category") and meta.category is not None:
            # 카테고리가 집합이나 리스트인 경우 첫 번째 항목을 사용
            if isinstance(meta.category, (set, list)):
                category = str(next(iter(meta.category), "")) if meta.category else ""
            else:
                category = str(meta.category)

        # 점수가 None이 아니면 그대로 사용, None이면 0.0으로 설정
        final_score = float(round(score, 2)) if score is not None else 0.0

        return _TickerScore(
            meta=meta,
            price=float(price) if price is not None else 0.0,
            prev_close=float(prev_close) if prev_close is not None else 0.0,
            daily_pct=float(round(daily_pct, 2)) if daily_pct is not None else 0.0,
            score=final_score,
            streak=int(holding_days) if holding_days is not None else 0,
            category=category.strip(),  # 공백 제거
        )
    except Exception as e:
        print(f"경고: {meta.ticker} 점수 생성 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        # 오류 발생 시 기본값 반환
        return _TickerScore(
            meta=meta,
            price=0.0,
            prev_close=0.0,
            daily_pct=0.0,
            score=0.0,
            streak=0,
            category=meta.category if hasattr(meta, "category") else "",
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
    # 국가별로 적절한 universe 파일명 사용 (예: 'kor' 또는 'aus')
    universe = country.lower()
    for meta in _iter_universe(country, universe):
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

    # trades 기반 보유일 계산
    holding_days_by_ticker: Dict[str, int] = {}
    as_of_dt = base_date.to_pydatetime()
    now_dt = datetime.now()
    try:
        if now_dt.date() > as_of_dt.date():
            as_of_dt = now_dt
    except Exception:
        pass

    try:
        holding_info = calculate_consecutive_holding_info(
            [item.meta.ticker for item in scored], country, as_of_dt
        )
    except Exception as exc:
        print(f"경고: 보유일 계산 중 오류 발생: {exc}")
        holding_info = {}

    if holding_info:
        as_of_date = as_of_dt.date()
        for ticker, info in holding_info.items():
            days = 0
            buy_date = info.get("buy_date") if isinstance(info, dict) else None
            if isinstance(buy_date, datetime):
                delta = (as_of_date - buy_date.date()).days
                days = max(delta + 1, 0)
            holding_days_by_ticker[ticker] = days

    # 점수에 따라 정렬 (높은 순)
    scored.sort(
        key=lambda item: (
            -item.score if item.score is not None else float("-inf"),
            item.meta.ticker,
        )
    )

    max_per_category_raw = None
    try:
        country_settings = get_country_settings(country)
        strategy_cfg = country_settings.get("strategy") or {}
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
        # 카테고리가 집합이거나 리스트인 경우 첫 번째 항목을 사용하거나 빈 문자열로 처리
        category = item.meta.category
        if isinstance(category, (set, list)):
            category = next(iter(category), "") if category else ""
        category_key = str(category) if category else "-"
        if max_per_category is not None:
            current_count = category_counts.get(category_key, 0)
            if current_count >= max_per_category:
                continue

        next_rank = len(results) + 1
        state = "BUY" if next_rank <= portfolio_topn else "WAIT"
        price_value = _apply_precision(item.price, price_precision)
        daily_pct_value = round(item.daily_pct, daily_pct_precision)
        score_value = round(item.score, score_precision)

        # Use item.category instead of item.meta.category
        holding_days_val = holding_days_by_ticker.get(item.meta.ticker, 0)

        results.append(
            {
                "rank": next_rank,
                "ticker": item.meta.ticker,
                "name": item.meta.name,
                "category": item.category,  # Updated to use item.category
                "state": state,
                "price": price_value,
                "daily_pct": daily_pct_value,
                "score": score_value,
                "streak": item.streak,
                "base_date": base_date.strftime("%Y-%m-%d"),
                "holding_days": holding_days_val,
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
