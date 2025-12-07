"""
추천 파이프라인 오케스트레이터.

이 모듈은 데이터 로딩, 전처리, 그리고 `logic.recommend.portfolio`를 이용한 추천 생성을 조율합니다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from logic.recommend.portfolio import run_portfolio_recommend
from strategies.maps.history import (
    calculate_consecutive_holding_info,
    calculate_trade_cooldown_info,
)
from utils.data_loader import prepare_price_data
from utils.db_manager import list_open_positions
from utils.logger import get_app_logger
from utils.settings_loader import get_account_settings, get_strategy_rules
from utils.stock_list_io import get_etfs as load_universe

logger = get_app_logger()


@dataclass
class RecommendationReport:
    account_id: str
    country_code: str
    base_date: pd.Timestamp
    recommendations: list[dict[str, Any]]
    report_date: datetime
    summary_data: dict[str, Any] | None = None
    header_line: str | None = None
    detail_headers: list[str] | None = None
    detail_rows: list[list[Any]] | None = None
    detail_extra_lines: list[str] | None = None
    decision_config: dict[str, Any] = None


def generate_account_recommendation_report(account_id: str, date_str: str | None = None) -> RecommendationReport:
    """계정 단위 추천 종목 리스트를 반환합니다."""
    start_time = time.time()

    # 1. 기준일 설정
    if date_str and date_str.lower() != "auto":
        base_date = pd.Timestamp(date_str)
    else:
        # 최근 거래일 자동 계산 (단순화: 오늘 날짜 사용 후 데이터 로더에서 처리하게 할 수도 있음)
        # 여기서는 오늘을 기준으로 함
        base_date = pd.Timestamp.now().normalize()
        # 실제로는 get_trading_days 등으로 유효한 전 영업일을 찾아야 할 수 있음.
        # recommend.py가 auto 처리를 하기도 함.

    # 2. 계정 설정 로드
    account_config = get_account_settings(account_id)
    country_code = account_config.get("country_code", "kor").lower()

    # 3. 전략 규칙 로드

    strategy_rules = get_strategy_rules(account_id)  # 계정별 오버라이드 포함 로드

    # 4. 유니버스 로드
    universe = load_universe(account_id)
    # universe는 list[dict] 형태 (ticker, name, category, etc.)
    etf_meta = {u["ticker"]: u for u in universe}

    # 5. 보유 종목 로드
    # 파일에서 보유량 로드 (또는 API)
    holdings_raw = list_open_positions(account_id)
    holdings = {}
    for pos in holdings_raw:
        tkr = pos.get("ticker")
        if tkr:
            # list_open_positions returns Open Positions (Last Action == BUY)
            # Quantity/Cost info is not in list_open_positions results in this version of db_manager.
            # We treat them as held. Defaulting quantity to 1.0 if not available.
            holdings[tkr] = {"shares": 1.0, "avg_cost": 0.0}

    # 6. 가격 데이터 프리패치
    # pairs 구성
    recommend_universe = account_config.get("universe", [])
    if not recommend_universe:
        # 설정이 없으면 전체 유니버스
        recommend_universe = [u["ticker"] for u in universe]

    # 보유 종목도 데이터 로딩 대상에 포함
    tickers_to_fetch = set(recommend_universe) | set(holdings.keys())

    logger.info(f"[{account_id.upper()}] 가격 데이터 로딩 시작 (대상 {len(tickers_to_fetch)}개)")

    # 충분한 기간의 데이터 로딩 (MA 계산 등을 위해)
    # pipeline/backtest 등에서 사용하는 기간 로직 참조
    lookback_days = 400
    start_date_fetch = base_date - pd.Timedelta(days=lookback_days)

    start_str = start_date_fetch.strftime("%Y-%m-%d")
    end_str = base_date.strftime("%Y-%m-%d")

    logger.info(f"Preparing price data for {account_id} (Country: {country_code})")
    price_data_map, missings = prepare_price_data(
        tickers=list(tickers_to_fetch),
        country=country_code,
        start_date=start_str,
        end_date=end_str,
        account_id=account_id,
        allow_remote_fetch=True,
    )
    logger.info(f"[{account_id.upper()}] 가격 데이터 로딩 완료")

    # 7. 지표 계산 및 데이터 가공
    from strategies.maps.metrics import process_ticker_data
    from utils.data_loader import fetch_naver_etf_inav_snapshot

    # [KOR] Fetch Real-time NAV/Price snapshot from Naver (for Disparity)
    nav_snapshot = {}
    if country_code == "kor":
        logger.info(f"[{account_id.upper()}] Naver API에서 실시간 NAV/괴리율 조회 중...")
        try:
            nav_snapshot = fetch_naver_etf_inav_snapshot(list(tickers_to_fetch))
            logger.info(f"[{account_id.upper()}] {len(nav_snapshot)}개 종목의 NAV 정보 수신 완료")
        except Exception as e:
            logger.warning(f"Naver NAV 조회 실패: {e}")

    data_by_tkr: dict[str, Any] = {}

    # 전체 etf_tickers 집합 (ETF 식별용 - Unused)
    # all_etf_tickers = {u["ticker"] for u in universe if u.get("type") == "etf"}

    for tkr, df in price_data_map.items():
        if df.empty:
            continue

        metrics = process_ticker_data(
            tkr,
            df,
            ma_period=strategy_rules.ma_period,
            ma_type=strategy_rules.ma_type,
            min_buy_score=strategy_rules.min_buy_score,
        )

        if metrics is None and tkr in holdings:
            # [CRITICAL fallback for held items with insufficient data]
            # 전략 지표 계산엔 실패했으나(데이터 부족 등), 보유 중인 종목이라면
            # 현재가와 NAV 정보라도 표시해야 함.
            if not df.empty:
                last_row = df.iloc[-1]
                current_price_fallback = float(last_row["Close"])

                # Try NAV fallback
                nav_fallback = None
                dev_fallback = None

                # Check Naver Snapshot first
                tkr_norm_fb = tkr.strip().upper()
                if tkr_norm_fb in nav_snapshot:
                    snap_fb = nav_snapshot[tkr_norm_fb]
                    nav_fallback = snap_fb.get("nav")
                    snap_price_fb = snap_fb.get("nowVal")
                    if snap_price_fb and snap_price_fb > 0:
                        # For consistency, maybe use closing price?
                        # But for fallback, getting any valid price is better.
                        pass

                    if nav_fallback and nav_fallback > 0:
                        dev_fallback = ((current_price_fallback / nav_fallback) - 1.0) * 100.0
                    if dev_fallback is None:
                        dev_fallback = snap_fb.get("deviation")

                # Check DataFrame columns if Naver failed
                if nav_fallback is None:
                    if "NAV" in last_row:
                        nav_fallback = float(last_row["NAV"])
                    elif "순자산가치(NAV)" in last_row:
                        nav_fallback = float(last_row["순자산가치(NAV)"])

                    if nav_fallback and nav_fallback > 0:
                        dev_fallback = (current_price_fallback / nav_fallback - 1.0) * 100.0
                    elif "괴리율" in last_row:
                        dev_fallback = float(last_row["괴리율"])

                # Calculate returns using raw dataframe
                close_series = df["Close"]
                cur_idx = len(close_series) - 1

                def _calc_ret_fallback(days: int) -> float:
                    if cur_idx - days >= 0:
                        prev_val = float(close_series.iloc[cur_idx - days])
                        if prev_val > 0:
                            return (current_price_fallback / prev_val) - 1.0
                    return 0.0

                row_data = {
                    "price": current_price_fallback,
                    "prev_close": float(df.iloc[-2]["Close"]) if len(df) > 1 else current_price_fallback,  # Approx
                    "score": 0.0,  # No score available
                    "rsi_score": 0.0,
                    "s1": 0.0,
                    "ma_value": 0.0,
                    "ma_period": strategy_rules.ma_period,
                    "close": df["Close"],
                    "filter": 0,
                    "drawdown_from_peak": 0.0,
                    "nav_price": nav_fallback,
                    "price_deviation": dev_fallback,
                    "trend_prices": df["Close"].iloc[-60:].tolist() if not df.empty else [],
                    "return_1w": _calc_ret_fallback(5),
                    "return_2w": _calc_ret_fallback(10),
                    "return_1m": _calc_ret_fallback(20),
                    "return_3m": _calc_ret_fallback(60),
                    "drawdown_from_high": 0.0,
                }
                data_by_tkr[tkr] = row_data
            continue

        if metrics:
            # 포트폴리오 로직에서 필요한 마지막 날짜(기준일) 데이터 추출
            try:
                idx = metrics["close"].index.get_indexer([base_date], method="pad")[0]
                if idx == -1:
                    continue
                dt_found = metrics["close"].index[idx]

                # 너무 오래된 데이터면 스킵 (예: 1주일 이상 갭)
                if (base_date - dt_found).days > 7:
                    logger.warning(f"{tkr}: 데이터가 너무 오래됨 ({dt_found.date()})")
                    continue

                # Extended Metrics Calculation
                close_series = metrics["close"]
                current_price = float(close_series.iloc[idx])

                def _get_ret(days_back: int) -> float:
                    if idx - days_back >= 0:
                        prev = float(close_series.iloc[idx - days_back])
                        if prev > 0:
                            return (current_price / prev) - 1.0
                    return 0.0

                # MDD / Drawdown from Peak (Recent 1 year)
                lookback_window = 252  # 1 year roughly
                start_idx = max(0, idx - lookback_window)
                window_max = close_series.iloc[start_idx : idx + 1].max()
                drawdown_from_peak = (current_price / window_max - 1.0) if window_max > 0 else 0.0

                # NAV and Disparity (KOR only)
                nav_val = None
                price_deviation_val = None

                # 1. Try Naver Snapshot first (Priority)
                tkr_norm = tkr.strip().upper()
                if tkr_norm in nav_snapshot:
                    snap = nav_snapshot[tkr_norm]
                    nav_val = snap.get("nav")
                    snap_price = snap.get("nowVal")

                    # Update current price with realtime if available and valid
                    if snap_price and snap_price > 0:
                        # Optional: override current_price for display consistency?
                        # For now, we only use it for Disparity calculation correctness relative to NAV
                        # But user might prefer recommendation to use closing price for signals.
                        # We will use Snapshot for NAV/Disparity, but keep 'price' as closing price for strategy.
                        # However, for Disparity display, it makes sense to use the pair (RealPrice, RealNAV).

                        if nav_val and nav_val > 0:
                            price_deviation_val = ((snap_price / nav_val) - 1.0) * 100.0

                    if price_deviation_val is None:
                        price_deviation_val = snap.get("deviation")

                # 2. Fallback to pykrx cached data if Naver failed
                if nav_val is None:
                    raw_df = metrics.get("df")
                    if raw_df is not None and not raw_df.empty:
                        try:
                            raw_row = raw_df.loc[dt_found]
                            if isinstance(raw_row, pd.DataFrame):
                                raw_row = raw_row.iloc[-1]

                            if "NAV" in raw_row:
                                nav_val = float(raw_row["NAV"])
                            elif "순자산가치(NAV)" in raw_row:
                                nav_val = float(raw_row["순자산가치(NAV)"])

                            if nav_val and nav_val > 0:
                                price_deviation_val = (current_price / nav_val - 1.0) * 100.0
                            elif "괴리율" in raw_row:
                                price_deviation_val = float(raw_row["괴리율"])
                        except KeyError:
                            pass

                row_data = {
                    "price": current_price,
                    "prev_close": float(metrics["close"].iloc[idx - 1]) if idx > 0 else 0.0,
                    "score": float(metrics["ma_score"].iloc[idx]),
                    "rsi_score": float(metrics["rsi_score"].iloc[idx]) if "rsi_score" in metrics else 0.0,
                    "s1": float(metrics["ma"].iloc[idx]),
                    "ma_value": float(metrics["ma"].iloc[idx]),
                    "ma_period": strategy_rules.ma_period,
                    "close": metrics["close"],  # Series for history check
                    "filter": int(metrics["buy_signal_days"].iloc[idx]),  # buy_signal_days
                    "drawdown_from_peak": drawdown_from_peak * 100.0,  # Convert to % for display compatibility
                    # Extended Fields for UI
                    "return_1w": _get_ret(5),
                    "return_2w": _get_ret(10),
                    "return_1m": _get_ret(20),
                    "return_3m": _get_ret(60),
                    "drawdown_from_high": drawdown_from_peak
                    * 100.0,  # Same as drawdown_from_peak but consistent naming
                    "nav_price": nav_val,
                    "price_deviation": price_deviation_val,
                    "trend_prices": close_series.iloc[max(0, idx - 60) : idx + 1].tolist(),  # Last 3 months trend
                }
                data_by_tkr[tkr] = row_data
            except Exception as e:
                logger.error(f"{tkr} 데이터 처리 중 오류: {e}")
                continue

    # 8. 추가 정보 로드 (Holding period, Cooldown info)
    consecutive_holding_info = calculate_consecutive_holding_info(
        list(holdings.keys()), account_id, base_date.to_pydatetime()
    )

    trade_cooldown_info = calculate_trade_cooldown_info(list(tickers_to_fetch), account_id, base_date.to_pydatetime())

    # 9. 추천 실행
    pairs = [(t, "") for t in recommend_universe]  # (ticker, reason) format expected by portfolio

    recommendations = run_portfolio_recommend(
        account_id=account_id,
        country_code=country_code,
        base_date=base_date,
        strategy_rules=strategy_rules,
        data_by_tkr=data_by_tkr,
        holdings=holdings,
        etf_meta=etf_meta,
        full_etf_meta=etf_meta,
        pairs=pairs,
        consecutive_holding_info=consecutive_holding_info,
        trade_cooldown_info=trade_cooldown_info,
        cooldown_days=account_config.get("cooldown_days", 1),
        rsi_sell_threshold=account_config.get("rsi_sell_threshold") or 100.0,
    )

    # 10. 보고서 포맷팅
    # RecommendationReport 생성
    summary_data = {
        "total_count": len(recommendations),
        "buy_count": sum(1 for r in recommendations if r["state"] == "BUY"),
        "sell_count": sum(1 for r in recommendations if "SELL" in r["state"]),
    }

    # Table headers etc
    detail_headers = [
        "순위",
        "티커",
        "종목명",
        "카테고리",
        "상태",
        "보유일",
        "현재가",
        "일간수익률",
        "보유",
        "평가금액",
        "수익률",
        "MDD",
        "점수",
        "필터",
        "문구",
    ]

    detail_rows = []
    for i, rec in enumerate(recommendations, 1):
        row = rec["row"]
        row[0] = i  # 순위 업데이트
        detail_rows.append(row)

    elapsed = time.time() - start_time
    logger.info(f"[{account_id.upper()}] 추천 완료 (소요 {elapsed:.1f}초)")

    return RecommendationReport(
        account_id=account_id,
        country_code=country_code,
        base_date=base_date,
        recommendations=recommendations,
        report_date=datetime.now(),
        summary_data=summary_data,
        detail_headers=detail_headers,
        detail_rows=detail_rows,
        decision_config=None,
    )
