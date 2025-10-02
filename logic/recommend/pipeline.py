"""국가 기반 간소화 추천 파이프라인."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
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
from logic.strategies.maps.constants import DECISION_CONFIG, DECISION_MESSAGES, DECISION_NOTES
from logic.strategies.maps.shared import sort_decisions_by_order_and_score
from utils.stock_list_io import get_etfs
from utils.trade_store import list_open_positions
from logic.recommend.history import (
    calculate_consecutive_holding_info,
    calculate_trade_cooldown_info,
)
from utils.data_loader import fetch_ohlcv, get_latest_trading_day
from utils.db_manager import get_db_connection


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


def _load_full_etf_meta(country: str) -> Dict[str, Dict[str, Any]]:
    """Load metadata for all ETFs including recommend_disabled ones."""

    file_path = DATA_DIR / f"{country}.json"
    if not file_path.exists():
        return {}

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[경고] 전체 ETF 메타 로드 실패: {exc}")
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


def _fetch_trades_for_date(country: str, base_date: pd.Timestamp) -> List[Dict[str, Any]]:
    """Retrieve trades executed on the given base_date."""

    db = get_db_connection()
    if db is None:
        return []

    start = base_date.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
    # 최신 추천 실행 시 실제 거래 시간이 기준일 다음 날일 수도 있으므로 현재 시각까지 확장
    end = max(start + timedelta(days=1), datetime.utcnow())

    cursor = db.trades.find(
        {
            "country": country,
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


def generate_country_signal_report(country: str, date_str: Optional[str] = None) -> List[dict]:
    """국가 단위 추천 종목 리스트를 반환합니다."""
    if not country:
        raise ValueError("country 인자가 필요합니다.")
    country = country.strip().lower()

    base_date = _resolve_base_date(country, date_str)

    try:
        strategy_rules = get_strategy_rules(country)
        country_settings = get_country_settings(country)
    except CountrySettingsError as exc:
        raise ValueError(str(exc)) from exc

    ma_period = int(strategy_rules.ma_period)
    portfolio_topn = int(strategy_rules.portfolio_topn)
    replace_threshold = float(strategy_rules.replace_threshold)

    # 손절매 비율은 common_settings에서 가져옴
    try:
        from utils.account_registry import get_common_file_settings

        common_settings = get_common_file_settings() or {}
        stop_loss_pct = -abs(float(common_settings.get("HOLDING_STOP_LOSS_PCT", 10.0)))
        max_per_category = int(common_settings.get("MAX_PER_CATEGORY", 0) or 0)
    except Exception:
        stop_loss_pct = -10.0
        max_per_category = 0

    # ETF 목록 가져오기
    etf_universe = get_etfs(country)
    pairs = [(stock["ticker"], stock["name"]) for stock in etf_universe]

    # 실제 포트폴리오 데이터 준비
    holdings: Dict[str, Dict[str, float]] = {}
    try:
        # 현재 미매도 포지션만 조회
        open_positions = list_open_positions(country)
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
                    {"$match": {"country": country, "action": "BUY"}},
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

        print(f"계산된 holdings: {len(holdings)}개 종목")
    except Exception as e:
        print(f"포트폴리오 데이터 조회 실패: {e}")
        holdings = {}

    # 연속 보유 정보 계산
    consecutive_holding_info = calculate_consecutive_holding_info(
        list(holdings.keys()), country, base_date.to_pydatetime()
    )

    # 현재 자산/현금 정보 (임시값 - 실제 계산 필요)
    current_equity = 100_000_000  # 임시값
    total_cash = 100_000_000  # 임시값

    # 각 티커의 현재 데이터 준비 (실제 OHLCV 데이터 사용)
    data_by_tkr = {}
    for stock in etf_universe:
        ticker = stock["ticker"]
        # 실제 데이터 가져오기
        df = _fetch_dataframe(ticker, country=country, ma_period=ma_period, base_date=base_date)
        if df is not None and not df.empty:
            # 최신 가격 정보
            latest_close = df["Close"].iloc[-1]
            prev_close = df["Close"].iloc[-2] if len(df) > 1 else latest_close
            daily_pct = 0.0
            if prev_close and prev_close > 0:
                daily_pct = ((latest_close / prev_close) - 1.0) * 100

            # 이동평균 신호 계산
            from utils.indicators import calculate_moving_average_signals, calculate_ma_score

            (
                moving_average,
                buy_signal_active,
                consecutive_buy_days,
            ) = calculate_moving_average_signals(df["Close"], ma_period)
            ma_score_series = calculate_ma_score(df["Close"], moving_average)
            score = ma_score_series.iloc[-1] if not ma_score_series.empty else 0.0

            data_by_tkr[ticker] = {
                "price": latest_close,
                "prev_close": prev_close,
                "daily_pct": round(daily_pct, 2),
                "close": df["Close"],  # 백테스트용 close 데이터 추가
                "s1": moving_average.iloc[-1] if not moving_average.empty else None,
                "s2": None,
                "score": score,
                "filter": int(consecutive_buy_days.iloc[-1])
                if not consecutive_buy_days.empty
                else 0,
            }
        else:
            # 데이터가 없을 경우 기본값
            data_by_tkr[ticker] = {
                "price": 0.0,
                "prev_close": 0.0,
                "daily_pct": 0.0,
                "close": pd.Series(),  # 빈 Series
                "s1": None,
                "s2": None,
                "score": 0.0,
                "filter": 0,
                "drawdown_from_peak": None,
            }

    # 전략 설정
    portfolio_settings = {}
    regime_info = None

    # 쿨다운 정보 계산
    trade_cooldown_info = calculate_trade_cooldown_info(
        [stock["ticker"] for stock in etf_universe], country, base_date.to_pydatetime()
    )

    # generate_daily_recommendations_for_portfolio 호출
    try:
        from logic.strategies.maps import safe_generate_daily_recommendations_for_portfolio

        decisions = safe_generate_daily_recommendations_for_portfolio(
            country=country,
            base_date=base_date,
            portfolio_settings=portfolio_settings,
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
            stop_loss=stop_loss_pct,
            DECISION_CONFIG=DECISION_CONFIG,
            trade_cooldown_info=trade_cooldown_info,
            cooldown_days=int(country_settings.get("strategy", {}).get("COOLDOWN_DAYS", 5)),
            max_per_category=max_per_category,
        )
    except Exception as exc:
        print(f"generate_daily_recommendations_for_portfolio 실행 중 오류: {exc}")
        return []

    # 당일 SELL 트레이드를 결과에 추가하여 SOLD 상태로 노출
    trades_today = _fetch_trades_for_date(country, base_date)
    sold_entries: List[Dict[str, Any]] = []
    for trade in trades_today:
        if trade.get("action") != "SELL":
            continue
        ticker = (trade.get("ticker") or "").strip().upper()
        if not ticker:
            continue
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
                print(f"[경고] SOLD 종목 메타데이터 없음: {ticker}")
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

    # Include recommend_disabled tickers for metadata fallback
    full_meta_map = _load_full_etf_meta(country)
    for ticker, meta in full_meta_map.items():
        upper_ticker = ticker.upper()
        if upper_ticker not in etf_meta_map:
            etf_meta_map[upper_ticker] = {
                "ticker": upper_ticker,
                "name": meta.get("name") or upper_ticker,
                "category": meta.get("category") or "TBD",
            }

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
        # 보유일 계산
        holding_days_val = 0
        if ticker in holdings:
            # 실제 현재 날짜를 사용해서 보유일 계산
            current_date = pd.Timestamp.now().date()
            raw_buy_date = consecutive_holding_info.get(ticker, {}).get("buy_date")
            if raw_buy_date:
                buy_timestamp = pd.to_datetime(raw_buy_date)
                if pd.notna(buy_timestamp):
                    buy_date = buy_timestamp.date()
                    holding_days_val = (current_date - buy_date).days + 1
        elif ticker in consecutive_holding_info:
            buy_date = consecutive_holding_info[ticker].get("buy_date")
            if buy_date and buy_date <= base_date:
                holding_days_val = (base_date - pd.to_datetime(buy_date).normalize()).days + 1

        # 당일 신규 편입 종목은 최소 1일 보유로 표시 및 문구 유지
        new_buy_phrase = DECISION_MESSAGES.get("NEW_BUY", "✅ 신규 매수")
        bought_today = False
        if holding_days_val == 0:
            if raw_state in {"BUY", "BUY_REPLACE"}:
                holding_days_val = 1
                bought_today = True
            elif is_currently_held:
                holding_days_val = 1
                bought_today = True
        if bought_today and is_currently_held:
            phrase = new_buy_phrase

        ticker_data = data_by_tkr.get(ticker, {})
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
        score_val = decision.get("score", 0.0)

        filter_days = decision.get("filter")
        if filter_days is None:
            filter_days_row = decision.get("row") or []
            if len(filter_days_row) >= 16:
                try:
                    filter_days = (
                        int(str(filter_days_row[15]).replace("일", ""))
                        if filter_days_row[15] not in ("-", None)
                        else 0
                    )
                except Exception:
                    filter_days = 0
            else:
                filter_days = 0

        streak_val = int(filter_days or 0)

        results.append(
            {
                "rank": len(results) + 1,
                "ticker": ticker,
                "name": name,
                "category": category,
                "state": state,
                "price": price_val,
                "daily_pct": daily_pct_val,
                "score": score_val,
                "streak": streak_val,
                "base_date": base_date.strftime("%Y-%m-%d"),
                "holding_days": holding_days_val,
                "phrase": phrase,
                "state_order": DECISION_CONFIG.get(state, {}).get("order", 99),
            }
        )

    # BUY 종목 생성: 상위 점수의 WAIT 종목들을 BUY로 변경
    wait_items = [
        item
        for item in results
        if item["state"] == "WAIT" and item.get("phrase") != DECISION_NOTES.get("CATEGORY_DUP")
    ]
    wait_items.sort(key=lambda x: x["score"], reverse=True)

    # 카테고리 보유 제한이 있는 경우, 동일 카테고리 수를 체크
    category_counts = {}
    if max_per_category and max_per_category > 0:
        for item in results:
            if item["state"] in {"HOLD", "BUY", "BUY_REPLACE"}:
                category = str(item.get("category") or "").strip()
                if category:
                    category_counts[category] = category_counts.get(category, 0) + 1

    current_holdings_count = len(holdings)
    sell_state_set = {"SELL_TREND", "SELL_REPLACE", "CUT_STOPLOSS", "SELL_REGIME_FILTER"}
    buy_state_set = {"BUY", "BUY_REPLACE"}
    planned_sell_count = sum(1 for item in results if item["state"] in sell_state_set)
    planned_buy_count = sum(1 for item in results if item["state"] in buy_state_set)

    projected_holdings = current_holdings_count - planned_sell_count + planned_buy_count
    additional_buy_slots = max(0, portfolio_topn - projected_holdings)

    for i, item in enumerate(wait_items[:additional_buy_slots]):
        item["state"] = "BUY"
        item["phrase"] = DECISION_MESSAGES.get("NEW_BUY", "✅ 신규 매수")
        # 신규 매수로 전환된 종목은 holdings 정보가 없으므로 기본값 추가
        holdings.setdefault(
            item["ticker"],
            {
                "buy_date": base_date.strftime("%Y-%m-%d"),
            },
        )

    # rank를 점수 순서대로 재설정
    results.sort(key=lambda x: x["score"], reverse=True)
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

    return results


__all__ = ["generate_country_signal_report"]
