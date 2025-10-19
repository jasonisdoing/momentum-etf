"""국가 기반 간소화 추천 파이프라인."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# 데이터 디렉토리 경로 설정
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "stocks"
from utils.settings_loader import (
    AccountSettingsError,
    get_account_settings,
    get_strategy_rules,
    load_common_settings,
)
from strategies.maps.constants import DECISION_CONFIG, DECISION_MESSAGES, DECISION_NOTES
from logic.common import sort_decisions_by_order_and_score, filter_category_duplicates
from strategies.maps.history import (
    calculate_consecutive_holding_info,
    calculate_trade_cooldown_info,
)
from utils.stock_list_io import get_etfs
from utils.trade_store import list_open_positions
from utils.data_loader import (
    fetch_ohlcv,
    prepare_price_data,
    get_latest_trading_day,
    get_next_trading_day,
    count_trading_days,
)
from utils.db_manager import get_db_connection
from logic.recommend.market import get_market_regime_status_info
from utils.logger import get_app_logger

logger = get_app_logger()


@dataclass
class RecommendationReport:
    account_id: str
    country_code: str
    base_date: pd.Timestamp
    recommendations: List[Dict[str, Any]]
    report_date: datetime
    summary_data: Optional[Dict[str, Any]] = None
    header_line: Optional[str] = None
    detail_headers: Optional[List[str]] = None
    detail_rows: Optional[List[List[Any]]] = None
    detail_extra_lines: Optional[List[str]] = None
    decision_config: Dict[str, Any] = None


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
    category: str
    ma_value: float = 0.0


def _load_full_etf_meta(country_code: str) -> Dict[str, Dict[str, Any]]:
    """Load metadata for all ETFs including recommend_disabled ones."""

    file_path = DATA_DIR / f"{country_code}.json"
    if not file_path.exists():
        return {}

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("전체 ETF 메타 로드 실패: %s", exc)
        return {}

    meta_map: Dict[str, Dict[str, Any]] = {}
    if not isinstance(data, list):
        return meta_map

    for block in data:
        if not isinstance(block, dict):
            continue

        raw_category = block.get("category")
        if isinstance(raw_category, (list, set, tuple)):
            raw_category = next(iter(raw_category), "") if raw_category else ""
        category_name = str(raw_category or "TBD").strip() or "TBD"

        tickers = block.get("tickers") or []
        if not isinstance(tickers, list):
            continue

        for item in tickers:
            if not isinstance(item, dict):
                continue

            ticker = str(item.get("ticker") or "").strip().upper()
            if not ticker:
                continue

            raw_item_category = item.get("category", category_name)
            if isinstance(raw_item_category, (list, set, tuple)):
                raw_item_category = next(iter(raw_item_category), "") if raw_item_category else ""
            item_category = str(raw_item_category or category_name or "TBD").strip() or "TBD"

            name = str(item.get("name") or ticker).strip() or ticker

            meta_map[ticker] = {
                "ticker": ticker,
                "name": name,
                "category": item_category,
            }

    return meta_map


def _fetch_dataframe(
    ticker: str,
    *,
    country: str,
    ma_period: int,
    base_date: Optional[pd.Timestamp],
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Optional[pd.DataFrame]:
    try:
        if prefetched_data and ticker in prefetched_data:
            df = prefetched_data[ticker]
        else:
            months_back = max(12, ma_period)  # 최소 1년치 데이터 요청
            df = fetch_ohlcv(
                ticker,
                country=country,
                months_back=months_back,
                base_date=base_date,
            )

        if df is None or df.empty:
            logger.warning("%s에 대한 데이터를 가져오지 못했습니다.", ticker)
            return None

        if "Close" not in df.columns:
            logger.warning("%s에 대한 종가(Close) 데이터가 없습니다.", ticker)
            return None

        df = df.dropna(subset=["Close"])

        if df.empty:
            logger.warning("%s에 대한 유효한 데이터가 없습니다.", ticker)
            return None

        return df

    except Exception as e:
        logger.warning("%s 데이터 처리 중 오류 발생: %s", ticker, e)
        import traceback

        traceback.print_exc()
        return None


def _calc_metrics(df: pd.DataFrame, ma_period: int) -> Optional[tuple]:
    try:
        # 'Close' 또는 'Adj Close' 중 사용 가능한 컬럼 선택
        if "unadjusted_close" in df.columns:
            raw_close = df["unadjusted_close"].astype(float)
        else:
            raw_close = df["Close"].astype(float)

        if "Adj Close" in df.columns and not df["Adj Close"].isnull().all():
            price_series = df["Adj Close"].astype(float)
        else:
            price_series = raw_close

        raw_close = raw_close.dropna()
        price_series = price_series.dropna()

        if raw_close.empty or price_series.empty:
            return None

        # 이동평균 계산 (최소 1개 데이터로도 계산 가능하도록 min_periods=1 설정)
        ma = price_series.rolling(window=ma_period, min_periods=1).mean()

        latest_close = float(price_series.iloc[-1])
        prev_close = float(price_series.iloc[-2]) if len(price_series) >= 2 else latest_close
        ma_value = float(ma.iloc[-1]) if not pd.isna(ma.iloc[-1]) else latest_close

        # 0 이하 값이면 기본값 설정
        if ma_value <= 0:
            ma_value = latest_close if latest_close > 0 else 1.0

        if latest_close <= 0:
            return None

        # 일간 수익률 계산 (이전 종가가 없거나 0 이하면 0%로 처리)
        daily_pct = 0.0
        raw_prev_close = float(raw_close.iloc[-2]) if len(raw_close) >= 2 else float(raw_close.iloc[-1])
        raw_latest_close = float(raw_close.iloc[-1])
        if raw_prev_close and raw_prev_close > 0:
            daily_pct = ((raw_latest_close / raw_prev_close) - 1.0) * 100

        # 점수 계산 (이동평균 대비 수익률, % 단위)
        score = 0.0
        if ma_value > 0:
            score = ((latest_close / ma_value) - 1.0) * 100

        # 점수가 매우 작으면 0.01%로 처리 (0점 방지)
        if abs(score) < 0.01 and score != 0:
            score = 0.01 if score > 0 else -0.01

        # 연속 상승일 계산
        streak = 0
        for price, ma_entry in zip(reversed(price_series.iloc[-ma_period:]), reversed(ma.iloc[-ma_period:])):
            if pd.isna(ma_entry) or pd.isna(price) or price < ma_entry:
                break
            streak += 1

        # 보유일 계산 (최대 20일로 제한)
        holding_days = min(streak, 20) if streak > 0 else 0

        return latest_close, prev_close, daily_pct, score, holding_days
    except Exception as e:
        logger.exception("메트릭 계산 중 오류 발생: %s", e)
        return None


def _build_score(meta: _TickerMeta, metrics) -> _TickerScore:
    # 메트릭이 없는 경우 기본값 반환
    if metrics is None:
        logger.warning("%s에 대한 메트릭이 없습니다.", meta.ticker)
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
            # 카테고리가 리스트인 경우 첫 번째 항목을 사용
            if isinstance(meta.category, (set, list)):
                category = str(next(iter(meta.category), "")) if meta.category else ""
            else:
                category = str(meta.category)

        # 점수가 None이 아니면 그대로 사용, None이면 0.0으로 설정
        final_score = float(round(score, 2)) if score is not None else 0.0

        # 이동평균 값 계산 (점수를 이용해서 근사치 계산)
        ma_value = price * (1 - score / 100) if price > 0 else 0.0

        return _TickerScore(
            meta=meta,
            price=float(price) if price is not None else 0.0,
            prev_close=float(prev_close) if prev_close is not None else 0.0,
            daily_pct=float(round(daily_pct, 2)) if daily_pct is not None else 0.0,
            score=final_score,
            streak=int(holding_days) if holding_days is not None else 0,
            category=category.strip(),  # 공백 제거
            ma_value=ma_value,
        )
    except Exception as e:
        logger.exception("%s 점수 생성 중 오류 발생: %s", meta.ticker, e)
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


def _resolve_base_date(account_id: str, date_str: Optional[str]) -> pd.Timestamp:
    if date_str:
        try:
            base = pd.to_datetime(date_str).normalize()
        except Exception as exc:
            raise ValueError(f"잘못된 날짜 형식입니다: {date_str}") from exc
    else:
        account_settings = get_account_settings(account_id)
        country_code = (account_settings.get("country_code") or account_id).strip().lower()
        today_norm = pd.Timestamp.now().normalize()
        latest_trading_day = get_latest_trading_day(country_code)
        latest_norm = latest_trading_day.normalize()

        if latest_norm >= today_norm:
            base = latest_norm
        else:
            next_trading_day = get_next_trading_day(country_code, reference_date=today_norm)
            base = next_trading_day if next_trading_day is not None else latest_norm

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


def _join_phrase_parts(*parts: Optional[str]) -> str:
    """Join non-empty phrase components with a separator."""

    cleaned: List[str] = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if text:
            cleaned.append(text)
    return " | ".join(cleaned)


def _format_sell_replace_phrase(phrase: str, *, etf_meta: Dict[str, Dict[str, Any]]) -> str:
    if not phrase or "교체매도" not in phrase:
        return phrase

    ratio_match = re.search(r"손익률\s+[+-]?[0-9.,]+%", phrase)
    ticker_matches = re.findall(r"([A-Za-z0-9:]+)\(으\)로 교체", phrase)

    if not ratio_match or not ticker_matches:
        return phrase

    target_ticker = ticker_matches[-1]
    ratio_text = ratio_match.group(0)
    target_meta = etf_meta.get(target_ticker) or etf_meta.get(target_ticker.upper()) or {}
    target_name = target_meta.get("name") or target_ticker

    return f"교체매도 {ratio_text} - {target_name}({target_ticker})로 교체"


def _append_risk_off_suffix(phrase: str, ratio: Optional[int]) -> str:
    if ratio is None:
        return phrase
    try:
        ratio_int = int(ratio)
    except (TypeError, ValueError):
        return phrase
    if not (0 <= ratio_int <= 100) or ratio_int >= 100:
        return phrase
    phrase_str = str(phrase or "")
    if "시장위험회피" in phrase_str:
        return phrase_str
    suffix = f"❗시장위험회피 비중조절❗ (보유목표 {ratio_int}%)"
    return f"{phrase_str} | {suffix}" if phrase_str else suffix


def _normalize_buy_date(value: Any) -> Optional[pd.Timestamp]:
    """Convert various buy date formats into a normalized pandas Timestamp."""

    if value in (None, "", "-"):
        return None
    try:
        ts = pd.to_datetime(value)
    except Exception:
        return None

    if pd.isna(ts):
        return None

    if getattr(ts, "tzinfo", None) is not None:
        try:
            ts = ts.tz_convert(None)  # type: ignore[attr-defined]
        except AttributeError:
            ts = ts.tz_localize(None)  # type: ignore[attr-defined]
    return ts.normalize()


def _resolve_buy_price(
    ticker_data: Dict[str, Any],
    buy_date: Optional[pd.Timestamp],
    *,
    fallback_price: Optional[float] = None,
) -> Optional[float]:
    """Pick the closest available closing price on or before the buy date."""

    if buy_date is None:
        return fallback_price

    close_series = ticker_data.get("close")
    if not isinstance(close_series, pd.Series) or close_series.empty:
        return fallback_price

    series = close_series.dropna().copy()
    if series.empty:
        return fallback_price

    try:
        index = pd.to_datetime(series.index)
    except Exception:
        return fallback_price

    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)  # type: ignore[attr-defined]
    index = index.normalize()
    normalized_series = pd.Series(series.values, index=index)
    normalized_series = normalized_series.sort_index()
    normalized_series = normalized_series[~normalized_series.index.duplicated(keep="last")]

    if normalized_series.empty:
        return fallback_price

    prior_or_same = normalized_series.loc[normalized_series.index <= buy_date]
    if not prior_or_same.empty:
        return float(prior_or_same.iloc[-1])

    after = normalized_series.loc[normalized_series.index >= buy_date]
    if not after.empty:
        return float(after.iloc[0])

    return fallback_price


def _compute_trailing_return(
    close_series: pd.Series,
    periods_back: int,
) -> float:
    """Compute percentage return using close price N trading days ago."""

    if not isinstance(close_series, pd.Series) or close_series.empty:
        return 0.0
    valid = close_series.dropna()
    if valid.empty:
        return 0.0

    if len(valid) <= periods_back:
        return 0.0

    try:
        latest_price = float(valid.iloc[-1])
        prev_price = float(valid.iloc[-(periods_back + 1)])
    except (IndexError, TypeError, ValueError):
        return 0.0

    if prev_price <= 0:
        return 0.0

    return round(((latest_price / prev_price) - 1.0) * 100.0, 2)


def _fetch_trades_for_date(account_id: str, base_date: pd.Timestamp) -> List[Dict[str, Any]]:
    """Retrieve trades executed on the given base_date."""

    db = get_db_connection()
    if db is None:
        return []

    start = base_date.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
    # 최신 추천 실행 시 실제 거래 시간이 기준일 다음 날일 수도 있으므로 현재 시각까지 확장
    end = max(start + timedelta(days=1), datetime.utcnow())

    account_norm = (account_id or "").strip().lower()

    cursor = db.trades.find(
        {
            "account": account_norm,
            "deleted_at": {"$exists": False},
            "executed_at": {"$gte": start, "$lt": end},
        },
        projection={"ticker": 1, "action": 1, "name": 1, "_id": 0},
    )

    trades: List[Dict[str, Any]] = []
    for doc in cursor:
        trades.append(
            {
                "ticker": str(doc.get("ticker") or "").upper(),
                "action": str(doc.get("action") or "").upper(),
                "name": str(doc.get("name") or ""),
            }
        )
    return trades


def generate_account_recommendation_report(account_id: str, date_str: Optional[str] = None) -> RecommendationReport:
    """계정 단위 추천 종목 리스트를 반환합니다."""
    if not account_id:
        raise ValueError("account_id 인자가 필요합니다.")
    account_id = account_id.strip().lower()

    base_date = _resolve_base_date(account_id, date_str)

    try:
        strategy_rules = get_strategy_rules(account_id)
        account_settings = get_account_settings(account_id)
    except AccountSettingsError as exc:
        raise ValueError(str(exc)) from exc

    country_code = account_settings.get("country_code")

    ma_period = int(strategy_rules.ma_period)
    portfolio_topn = int(strategy_rules.portfolio_topn)

    strategy_cfg = account_settings.get("strategy", {}) or {}
    if not isinstance(strategy_cfg, dict):
        strategy_cfg = {}

    if "tuning" in strategy_cfg or "static" in strategy_cfg:
        strategy_static = strategy_cfg.get("static") if isinstance(strategy_cfg.get("static"), dict) else {}
        strategy_tuning = strategy_cfg.get("tuning") if isinstance(strategy_cfg.get("tuning"), dict) else {}
    else:
        strategy_static = strategy_cfg
        strategy_tuning = strategy_cfg

    # 검증은 get_account_strategy_sections에서 이미 완료됨 - 바로 사용
    max_per_category = int(strategy_static["MAX_PER_CATEGORY"])
    rsi_sell_threshold = int(strategy_tuning["OVERBOUGHT_SELL_THRESHOLD"])
    regime_filter_equity_ratio = int(strategy_static["MARKET_REGIME_RISK_OFF_EQUITY_RATIO"])

    # ETF 목록 가져오기
    etf_universe = get_etfs(country_code) or []
    logger.info(
        "[%s] 추천 Universe 로딩 완료: %d개 종목 데이터 준비",
        account_id.upper(),
        len(etf_universe),
    )
    disabled_tickers = {str(stock.get("ticker") or "").strip().upper() for stock in etf_universe if not bool(stock.get("recommend_enabled", True))}
    pairs = [(stock.get("ticker"), stock.get("name")) for stock in etf_universe if stock.get("ticker")]

    # 실제 포트폴리오 데이터 준비
    holdings: Dict[str, Dict[str, float]] = {}
    try:
        # 현재 미매도 포지션만 조회
        open_positions = list_open_positions(account_id)
        if open_positions:
            for position in open_positions:
                ticker = (position.get("ticker") or "").strip().upper()
                if not ticker:
                    continue
                exec_at = position.get("executed_at")
                buy_date = None
                if exec_at is not None:
                    try:
                        buy_date = pd.to_datetime(exec_at).strftime("%Y-%m-%d")
                    except Exception:
                        buy_date = None
                holdings[ticker] = {
                    "buy_date": buy_date,
                }

        # 예외적으로 포지션이 비어 있을 경우를 대비해 기존 BUY 집계를 백업으로 사용
        if not holdings:
            db = get_db_connection()
            if db is not None:
                pipeline = [
                    {"$match": {"account": account_id, "action": "BUY"}},
                    {"$group": {"_id": "$ticker"}},
                    {"$project": {"ticker": "$_id", "_id": 0}},
                ]
                holdings_tickers = [item["ticker"] for item in db.trades.aggregate(pipeline)]
                for ticker in holdings_tickers:
                    ticker_norm = (ticker or "").strip().upper()
                    if not ticker_norm:
                        continue
                    holdings[ticker_norm] = {
                        "buy_date": None,
                    }

        logger.debug("계산된 holdings: %d개 종목", len(holdings))
    except Exception as e:
        logger.error("포트폴리오 데이터 조회 실패: %s", e)
        holdings = {}

    # 연속 보유 정보 계산
    consecutive_holding_info = calculate_consecutive_holding_info(list(holdings.keys()), account_id, base_date.to_pydatetime())

    # 현재 자산/현금 정보 (임시값 - 실제 계산 필요)
    current_equity = 100_000_000  # 임시값
    total_cash = 100_000_000  # 임시값

    # 각 티커의 현재 데이터 준비 (실제 OHLCV 데이터 사용)
    tickers_all = [stock.get("ticker") for stock in etf_universe if stock.get("ticker")]
    prefetched_data: Dict[str, pd.DataFrame] = {}
    months_back = max(12, ma_period)
    warmup_days = int(max(ma_period, 1) * 1.5)

    common_settings = load_common_settings()
    cache_seed_raw = (common_settings or {}).get("CACHE_START_DATE")
    prefetch_start_dt = base_date - pd.DateOffset(months=months_back)
    if cache_seed_raw:
        try:
            cache_seed_dt = pd.to_datetime(cache_seed_raw).normalize()
            if cache_seed_dt > prefetch_start_dt:
                prefetch_start_dt = cache_seed_dt
        except Exception:
            pass
    start_date = prefetch_start_dt.strftime("%Y-%m-%d")
    end_date = base_date.strftime("%Y-%m-%d")
    logger.info(
        "[%s] 가격 데이터 로딩 시작 (기간 %s~%s, 대상 %d개)",
        account_id.upper(),
        start_date,
        end_date,
        len(tickers_all),
    )
    fetch_start = time.perf_counter()
    prefetched_data, missing_prefetch = prepare_price_data(
        tickers=tickers_all,
        country=country_code,
        start_date=start_date,
        end_date=end_date,
        warmup_days=warmup_days,
    )
    logger.info(
        "[%s] 가격 데이터 로딩 완료 (%.1fs)",
        account_id.upper(),
        time.perf_counter() - fetch_start,
    )
    missing_logged = set(missing_prefetch)
    if missing_prefetch:
        logger.warning(
            "[%s] 다음 종목의 가격 데이터를 확보하지 못해 제외합니다: %s",
            account_id.upper(),
            ", ".join(sorted(missing_logged)),
        )

    data_by_tkr = {}
    missing_data_tickers: List[str] = list(missing_prefetch)
    for stock in etf_universe:
        ticker = stock["ticker"]
        # 실제 데이터 가져오기
        df = _fetch_dataframe(
            ticker,
            country=country_code,
            ma_period=ma_period,
            base_date=base_date,
            prefetched_data=prefetched_data,
        )
        if df is not None and not df.empty:
            # 최신 가격 정보
            latest_close = df["Close"].iloc[-1]
            prev_close = df["Close"].iloc[-2] if len(df) > 1 else latest_close
            latest_data_date = pd.to_datetime(df.index[-1]).normalize()
            base_norm = base_date.normalize()

            daily_pct = 0.0
            if prev_close and prev_close > 0:
                daily_pct = ((latest_close / prev_close) - 1.0) * 100

            if base_norm > latest_data_date:
                prev_close = latest_close
                daily_pct = 0.0

            # MAPS 전략 계산
            from utils.indicators import calculate_ma_score
            from logic.common import get_buy_signal_streak

            # 이동평균 계산
            moving_average = df["Close"].rolling(window=ma_period).mean()

            # 점수 계산
            ma_score_series = calculate_ma_score(df["Close"], moving_average, normalize=False)
            score = ma_score_series.iloc[-1] if not ma_score_series.empty else 0.0

            # 지속일 계산 (점수 기반)
            consecutive_buy_days = get_buy_signal_streak(score, ma_score_series)

            # RSI 전략 계산 (strategies/rsi/recommend.py에서 처리)
            from strategies.rsi.recommend import calculate_rsi_for_ticker

            rsi_score = calculate_rsi_for_ticker(df["Close"])

            # RSI 계산 실패 시 디버깅 로그
            if rsi_score == 0.0 and len(df["Close"]) < 15:
                logger.warning(f"[RSI] {ticker} 데이터 부족: {len(df['Close'])}개 (최소 15개 필요)")

            recent_prices = df["Close"].tail(15)
            trend_prices = [round(float(val), 6) for val in recent_prices.tolist()] if not recent_prices.empty else []

            data_by_tkr[ticker] = {
                "price": latest_close,
                "prev_close": prev_close,
                "daily_pct": round(daily_pct, 2),
                "close": df["Close"],  # 백테스트용 close 데이터 추가
                "s1": moving_average.iloc[-1] if not moving_average.empty else None,
                "s2": None,
                "score": score,
                "rsi_score": rsi_score,
                "filter": consecutive_buy_days,
                "ret_1w": _compute_trailing_return(df["Close"], 5),
                "ret_2w": _compute_trailing_return(df["Close"], 10),
                "ret_3w": _compute_trailing_return(df["Close"], 15),
                "trend_prices": trend_prices,
            }
        else:
            missing_data_tickers.append(ticker)

    if missing_data_tickers:
        extra_missing = set(missing_data_tickers) - missing_logged
        if extra_missing:
            logger.warning(
                "[%s] 분석 중 추가로 제외된 종목: %s",
                account_id.upper(),
                ", ".join(sorted(extra_missing)),
            )
        missing_logged.update(missing_data_tickers)

    regime_info = None
    regime_filter_enabled = True
    try:
        common_settings = load_common_settings()
    except Exception as exc:
        logger.warning("시장 레짐 공통 설정 로드 실패: %s", exc)
        common_settings = None
    else:
        regime_filter_enabled = bool((common_settings or {}).get("MARKET_REGIME_FILTER_ENABLED", True))
        common_ratio_value = (common_settings or {}).get("MARKET_REGIME_RISK_OFF_EQUITY_RATIO")
        # 공통 설정에 값이 있으면 사용 (검증은 이미 완료됨)
        if regime_filter_equity_ratio is None and common_ratio_value is not None:
            regime_filter_equity_ratio = int(common_ratio_value)

    if regime_filter_enabled:
        try:
            regime_info_candidate, _ = get_market_regime_status_info()
        except Exception as exc:
            logger.warning("시장 레짐 정보 계산 실패: %s", exc)
        else:
            if regime_info_candidate:
                regime_info_candidate["risk_off_equity_ratio"] = regime_filter_equity_ratio
                regime_info = regime_info_candidate

    # 쿨다운 정보 계산
    trade_cooldown_info = calculate_trade_cooldown_info(
        [stock["ticker"] for stock in etf_universe],
        account_id,
        base_date.to_pydatetime(),
        country_code=country_code,
    )

    # generate_daily_recommendations_for_portfolio 호출
    try:
        from strategies.maps import safe_generate_daily_recommendations_for_portfolio

        decision_start = time.perf_counter()
        actual_cooldown_days = int(strategy_tuning["COOLDOWN_DAYS"])
        logger.info(
            "[%s] 추천 계산 시작 (보유 %d개, 후보 %d개, cooldown_days=%d)",
            account_id.upper(),
            len(holdings),
            len(data_by_tkr),
            actual_cooldown_days,
        )
        decisions = safe_generate_daily_recommendations_for_portfolio(
            account_id=account_id,
            country_code=country_code,
            base_date=base_date,
            strategy_rules=strategy_rules,
            data_by_tkr=data_by_tkr,
            holdings=holdings,
            etf_meta={stock["ticker"]: stock for stock in etf_universe},
            full_etf_meta={stock["ticker"]: stock for stock in etf_universe},
            regime_info=regime_info,
            current_equity=current_equity,
            total_cash=total_cash,
            pairs=pairs,
            consecutive_holding_info=consecutive_holding_info,
            trade_cooldown_info=trade_cooldown_info,
            cooldown_days=actual_cooldown_days,
            risk_off_equity_ratio=regime_filter_equity_ratio,
            rsi_sell_threshold=rsi_sell_threshold,
        )
        logger.info(
            "[%s] 추천 계산 완료 (%.1fs, 결과 %d개)",
            account_id.upper(),
            time.perf_counter() - decision_start,
            len(decisions) if isinstance(decisions, list) else -1,
        )
    except Exception as exc:
        logger.error("generate_daily_recommendations_for_portfolio 실행 중 오류: %s", exc)
        return []

    # 당일 SELL 트레이드를 결과에 추가하여 SOLD 상태로 노출
    trades_today = _fetch_trades_for_date(account_id, base_date)
    sold_entries: List[Dict[str, Any]] = []
    buy_traded_today: set[str] = set()
    for trade in trades_today:
        action = (trade.get("action") or "").strip().upper()
        ticker = (trade.get("ticker") or "").strip().upper()
        if not ticker:
            continue

        if action == "SELL":
            if ticker in holdings:
                # 여전히 보유 중이면 SOLD로 표시하지 않음
                continue

            existing = next((d for d in decisions if d.get("tkr") == ticker), None)
            if existing:
                existing["state"] = "SOLD"
                if existing.get("row"):
                    existing["row"][4] = "SOLD"
                    existing["row"][-1] = DECISION_MESSAGES["SOLD"]
                existing["buy_signal"] = False
                continue

            name = trade.get("name") or ticker
            ticker_data = data_by_tkr.get(ticker, {})
            if not ticker_data:
                meta_info = next(
                    (stock for stock in etf_universe if stock.get("ticker", "").upper() == ticker),
                    None,
                )
                if meta_info:
                    name = meta_info.get("name") or name
                else:
                    logger.warning("SOLD 종목 메타데이터 없음: %s", ticker)
                    name = ticker

            price_val = ticker_data.get("price", 0.0)
            daily_pct_val = (
                ticker_data.get("daily_pct", 0.0)
                if "daily_pct" in ticker_data
                else (
                    ((ticker_data.get("price", 0.0) / ticker_data.get("prev_close", 1.0)) - 1.0) * 100
                    if ticker_data.get("prev_close", 0.0) > 0
                    else 0.0
                )
            )
            score_val = float(ticker_data.get("score", 0.0) or 0.0)

            sold_entries.append(
                {
                    "state": "SOLD",
                    "tkr": ticker,
                    "score": score_val,
                    "buy_signal": False,
                    "row": [
                        0,
                        ticker,
                        name,
                        "-",
                        "SOLD",
                        "-",
                        price_val,
                        daily_pct_val,
                        0,
                        0.0,
                        0.0,
                        0.0,
                        "-",
                        score_val,
                        "-",
                        DECISION_MESSAGES["SOLD"],
                    ],
                }
            )

        elif action == "BUY":
            buy_traded_today.add(ticker)

        else:
            continue

    # 결과 포맷팅
    etf_meta_map: Dict[str, Dict[str, Any]] = {}
    for stock in etf_universe:
        ticker = stock.get("ticker")
        if not ticker:
            continue
        upper = str(ticker).upper()
        etf_meta_map[upper] = {
            "ticker": upper,
            "name": stock.get("name") or upper,
            "category": stock.get("category") or "TBD",
        }

    # 추천 비활성 티커도 메타데이터 보완용으로 포함한다
    full_meta_map = _load_full_etf_meta(country_code)
    for ticker, meta in full_meta_map.items():
        upper_ticker = ticker.upper()
        if upper_ticker not in etf_meta_map:
            etf_meta_map[upper_ticker] = {
                "ticker": upper_ticker,
                "name": meta.get("name") or upper_ticker,
                "category": meta.get("category") or "TBD",
            }

    disabled_note = DECISION_NOTES.get("NO_RECOMMEND", "추천 제외")
    results = []
    for decision in decisions:
        ticker = decision["tkr"]
        raw_state = decision["state"]
        phrase = decision["row"][-1] if decision["row"] else ""

        is_currently_held = ticker in holdings

        state = raw_state
        if is_currently_held and raw_state in {"WAIT"}:
            state = "HOLD"

        new_buy_phrase = DECISION_MESSAGES.get("NEW_BUY", "✅ 신규 매수")

        if new_buy_phrase in str(phrase):
            state = "BUY"

        phrase = _format_sell_replace_phrase(phrase, etf_meta=etf_meta_map)

        meta_info = etf_meta_map.get(ticker) or {}
        name = meta_info.get("name", ticker)
        category = meta_info.get("category", "TBD")
        ticker_upper = str(ticker).upper()
        recommend_enabled = ticker_upper not in disabled_tickers

        # 보유일 계산
        base_norm = base_date.normalize()
        holding_days_val = 0
        latest_buy_date_norm: Optional[pd.Timestamp] = None
        if ticker in holdings:
            current_date = pd.Timestamp.now().normalize()
            raw_buy_date = consecutive_holding_info.get(ticker, {}).get("buy_date")
            if raw_buy_date:
                buy_timestamp = pd.to_datetime(raw_buy_date).normalize()
                if pd.notna(buy_timestamp):
                    latest_buy_date_norm = buy_timestamp
                    if buy_timestamp <= current_date:
                        holding_days_val = count_trading_days(
                            country_code,
                            buy_timestamp,
                            current_date,
                        )
        elif ticker in consecutive_holding_info:
            buy_date = consecutive_holding_info[ticker].get("buy_date")
            if buy_date:
                buy_timestamp = pd.to_datetime(buy_date).normalize()
                if pd.notna(buy_timestamp) and buy_timestamp <= base_norm:
                    latest_buy_date_norm = buy_timestamp
                    holding_days_val = count_trading_days(
                        country_code,
                        buy_timestamp,
                        base_norm,
                    )

        if latest_buy_date_norm is None:
            fallback_buy_date = holdings.get(ticker, {}).get("buy_date") if ticker in holdings else None
            fallback_norm = _normalize_buy_date(fallback_buy_date)
            if fallback_norm is not None:
                latest_buy_date_norm = fallback_norm

        # 상태 및 문구 재정의
        bought_today = False
        if latest_buy_date_norm is not None and latest_buy_date_norm >= base_norm:
            bought_today = True

        # 당일 매수 체결된 종목 처리
        if ticker in buy_traded_today:
            state = "HOLD"
            new_phrase = DECISION_MESSAGES.get("NEWLY_ADDED", "🆕 신규 편입")
            phrase = _append_risk_off_suffix(new_phrase, decision.get("risk_off_target_ratio"))
            if holding_days_val == 0:
                holding_days_val = 1
        # 추천에 따라 오늘 신규 매수해야 할 종목
        elif state in {"BUY", "BUY_REPLACE"}:
            phrase_str = str(phrase)
            risk_off_ratio = decision.get("risk_off_target_ratio")

            if state == "BUY_REPLACE":
                replacement_note = phrase_str if phrase_str else ""
                combined_phrase = _join_phrase_parts(DECISION_MESSAGES.get("NEW_BUY", "✅ 신규 매수"), replacement_note)
                phrase = _append_risk_off_suffix(combined_phrase, risk_off_ratio)
            else:  # state == "BUY"
                base_new_phrase = DECISION_MESSAGES.get("NEW_BUY", "✅ 신규 매수")
                if "시장위험회피" in phrase_str or phrase_str == DECISION_NOTES.get("RISK_OFF_TRIM"):
                    chosen_phrase = base_new_phrase
                elif phrase_str:
                    chosen_phrase = phrase_str
                else:
                    chosen_phrase = base_new_phrase
                phrase = _append_risk_off_suffix(chosen_phrase, risk_off_ratio)
            if holding_days_val == 0:
                holding_days_val = 1
        # 이미 보유 중인 종목이 오늘 신규 편입된 경우
        elif is_currently_held and bought_today:
            state = "HOLD"
            new_phrase = DECISION_MESSAGES.get("NEWLY_ADDED", "🆕 신규 편입")
            phrase = _append_risk_off_suffix(new_phrase, decision.get("risk_off_target_ratio"))
            if holding_days_val == 0:
                holding_days_val = 1

        ticker_data = data_by_tkr.get(ticker, {})
        price_val = ticker_data.get("price", 0.0)
        daily_pct_val = (
            ticker_data.get("daily_pct", 0.0)
            if "daily_pct" in ticker_data
            else (
                ((ticker_data.get("price", 0.0) / ticker_data.get("prev_close", 1.0)) - 1.0) * 100 if ticker_data.get("prev_close", 0.0) > 0 else 0.0
            )
        )
        score_val = decision.get("score", 0.0)

        evaluation_pct_val: float = 0.0
        if holding_days_val and holding_days_val > 0 and is_currently_held:
            buy_date_raw = consecutive_holding_info.get(ticker, {}).get("buy_date")
            if not buy_date_raw:
                buy_date_raw = holdings.get(ticker, {}).get("buy_date")

            buy_date_norm = _normalize_buy_date(buy_date_raw)
            if buy_date_norm is None and bought_today:
                buy_date_norm = base_date.normalize()

            buy_price = _resolve_buy_price(
                ticker_data,
                buy_date_norm,
                fallback_price=float(price_val) if price_val else None,
            )

            if buy_price and buy_price > 0 and price_val:
                evaluation_pct_val = round(((float(price_val) / buy_price) - 1.0) * 100, 2)

        ret_1w = ticker_data.get("ret_1w", 0.0)
        ret_2w = ticker_data.get("ret_2w", 0.0)
        ret_3w = ticker_data.get("ret_3w", 0.0)

        filter_days = decision.get("filter")
        if filter_days is None:
            filter_days_row = decision.get("row") or []
            if len(filter_days_row) >= 16:
                try:
                    filter_days = int(str(filter_days_row[15]).replace("일", "")) if filter_days_row[15] not in ("-", None) else 0
                except Exception:
                    filter_days = 0
            else:
                filter_days = 0

        streak_val = int(filter_days)

        if not recommend_enabled:
            if state in {"BUY", "BUY_REPLACE"}:
                state = "WAIT"
            phrase = disabled_note

        rsi_score_val = decision.get("rsi_score", 0.0)

        result_entry = {
            "rank": len(results) + 1,
            "ticker": ticker,
            "name": name,
            "category": category,
            "state": state,
            "price": price_val,
            "daily_pct": daily_pct_val,
            "evaluation_pct": evaluation_pct_val,
            "return_1w": ret_1w,
            "return_2w": ret_2w,
            "return_3w": ret_3w,
            "trend_prices": ticker_data.get("trend_prices", []),
            "score": score_val,
            "rsi_score": rsi_score_val,
            "streak": streak_val,
            "base_date": base_date.strftime("%Y-%m-%d"),
            "holding_days": holding_days_val,
            "phrase": phrase,
            "state_order": DECISION_CONFIG.get(state, {}).get("order", 99),
            "recommend_enabled": recommend_enabled,
        }

        results.append(result_entry)

    # BUY 종목 생성: 상위 점수의 WAIT 종목들을 BUY로 변경
    wait_items = [item for item in results if item["state"] == "WAIT" and item.get("recommend_enabled", True)]
    # MAPS 점수 기반 정렬
    wait_items.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # 카테고리 보유 제한이 있는 경우, 동일 카테고리 수를 체크 (매도 예정 종목 제외)
    from logic.common import should_exclude_from_category_count

    category_counts: Dict[str, int] = {}
    category_counts_normalized: Dict[str, int] = {}
    category_limit = max_per_category if max_per_category and max_per_category > 0 else 1
    for item in results:
        # 매도 예정 종목은 카테고리 카운트에서 제외
        if not should_exclude_from_category_count(item["state"]) and item["state"] in {"HOLD", "BUY", "BUY_REPLACE"}:
            category_raw = item.get("category")
            category = str(category_raw or "").strip()
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
            category_key = _normalize_category_value(category_raw)
            if category_key:
                category_counts_normalized[category_key] = category_counts_normalized.get(category_key, 0) + 1

    current_holdings_count = len(holdings)
    sell_state_set = {"SELL_TREND", "SELL_REPLACE", "CUT_STOPLOSS", "SELL_RSI"}
    buy_state_set = {"BUY", "BUY_REPLACE"}
    planned_sell_count = sum(1 for item in results if item["state"] in sell_state_set)
    planned_buy_count = sum(1 for item in results if item["state"] in buy_state_set)

    projected_holdings = current_holdings_count - planned_sell_count + planned_buy_count
    additional_buy_slots = max(0, portfolio_topn - projected_holdings)

    promoted = 0
    for item in wait_items:
        if promoted >= additional_buy_slots:
            break

        category_raw = item.get("category")
        category = str(category_raw or "").strip()
        category_key = _normalize_category_value(category_raw)

        # 카테고리 중복 체크 시, 매도 예정 종목은 제외
        # 같은 카테고리의 매도 예정 종목이 있으면 해당 카테고리 슬롯이 비게 됨
        sell_in_same_category = sum(
            1 for r in results if r["state"] in sell_state_set and _normalize_category_value(r.get("category")) == category_key
        )
        effective_category_count = category_counts_normalized.get(category_key, 0) - sell_in_same_category

        if category_key and effective_category_count >= category_limit:
            # 카테고리 중복인 경우 BUY로 변경하지 않고 WAIT 상태 유지
            # filter_category_duplicates에서 필터링됨
            continue

        # RSI 과매수 종목 매수 차단
        # rsi_sell_threshold는 계좌별 설정에서 로드됨 (이 함수 외부에서 처리)
        # 여기서는 이미 portfolio.py에서 처리되므로 추가 체크 불필요
        pass

        item["state"] = "BUY"
        item["phrase"] = DECISION_MESSAGES.get("NEW_BUY", "✅ 신규 매수")
        promoted += 1

        if category:
            category_counts[category] = category_counts.get(category, 0) + 1
        if category_key:
            category_counts_normalized[category_key] = category_counts_normalized.get(category_key, 0) + 1

        # 신규 매수로 전환된 종목은 holdings 정보가 없으므로 기본값 추가
        holdings.setdefault(
            item["ticker"],
            {
                "buy_date": base_date.strftime("%Y-%m-%d"),
            },
        )

    sell_state_set = {
        "SELL_TREND",
        "SELL_REPLACE",
        "CUT_STOPLOSS",
    }
    buy_state_set = {"BUY", "BUY_REPLACE"}

    planned_sell_count = sum(1 for item in results if (item.get("state") or "").upper() in sell_state_set)
    planned_buy_count = sum(1 for item in results if (item.get("state") or "").upper() in buy_state_set)
    projected_holdings = current_holdings_count - planned_sell_count + planned_buy_count

    if projected_holdings > portfolio_topn:
        logger.debug(
            "Projected holdings (%d) exceed portfolio_topn(%d); trim logic removed",
            projected_holdings,
            portfolio_topn,
        )

    # rank를 MAPS 점수 순서대로 재설정
    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    for i, item in enumerate(results, 1):
        item["rank"] = i

    # 최종 state_order 재계산 및 상태 정렬 (백테스트와 동일한 기준 사용)
    for item in results:
        state_key = (item.get("state") or "").upper()
        item["state_order"] = DECISION_CONFIG.get(state_key, {}).get("order", 99)

    # 공통 정렬 함수 사용
    sort_decisions_by_order_and_score(results)

    # sort 후 rank 재설정
    for i, item in enumerate(results, 1):
        item["rank"] = i

    # 카테고리별 최고 점수만 표시 (교체 매매 제외)
    results = filter_category_duplicates(results, category_key_getter=_normalize_category_value)

    # 점수가 음수인 종목 제외
    results = [item for item in results if item.get("score", 0.0) >= 0]

    # rank 재설정
    for i, item in enumerate(results, 1):
        item["rank"] = i

    detail_headers = [
        "순위",
        "티커",
        "종목명",
        "카테고리",
        "상태",
        "보유일",
        "일간(%)",
        "평가(%)",
        "현재가",
        "1주(%)",
        "2주(%)",
        "3주(%)",
        "점수",
        "지속",
        "문구",
    ]

    detail_rows: List[List[Any]] = []
    for item in results:
        detail_rows.append(
            [
                item.get("rank", 0),
                item.get("ticker"),
                item.get("name"),
                item.get("category"),
                item.get("state"),
                item.get("holding_days"),
                item.get("daily_pct"),
                item.get("evaluation_pct"),
                item.get("price"),
                item.get("return_1w"),
                item.get("return_2w"),
                item.get("return_3w"),
                item.get("score"),
                item.get("streak"),
                item.get("phrase", ""),
            ]
        )

    report = RecommendationReport(
        account_id=account_id,
        country_code=country_code,
        base_date=base_date,
        recommendations=results,
        report_date=datetime.now(),
        summary_data=None,
        header_line=None,
        detail_headers=detail_headers,
        detail_rows=detail_rows,
        detail_extra_lines=None,
        decision_config=DECISION_CONFIG,
    )

    return report


# 하위 호환: 기존 함수명을 그대로 제공
generate_country_recommendation_report = generate_account_recommendation_report


__all__ = [
    "generate_account_recommendation_report",
    "generate_country_recommendation_report",
]


def _normalize_category_value(category: Optional[str]) -> Optional[str]:
    """Normalize category strings for comparison."""
    if category is None:
        return None
    category_str = str(category).strip()
    if not category_str:
        return None
    return category_str.upper()
