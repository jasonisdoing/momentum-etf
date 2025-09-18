import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import pandas as pd
from pymongo import DESCENDING

import settings as global_settings

try:
    import pytz
except ImportError:
    pytz = None

from utils.data_loader import (
    fetch_ohlcv,
    get_trading_days,
)

# New structure imports
from utils.db_manager import (
    get_app_settings,
    get_common_settings,
    get_db_connection,
    get_portfolio_snapshot,
    get_previous_portfolio_snapshot,
    get_trades_on_date,
    save_status_report_to_db,
)
from utils.report import (
    format_aud_money,
    format_aud_price,
    format_kr_money,
    render_table_eaw,
)
from utils.stock_list_io import get_etfs

try:
    from pykrx import stock as _stock
except ImportError:
    _stock = None

try:
    import yfinance as yf
except ImportError:
    yf = None

# 슬랙 알림에 사용될 매매 결정(decision) 코드별 표시 설정을 관리합니다.
# - display_name: 슬랙 메시지에 표시될 그룹 헤더
# - order: 그룹 표시 순서 (낮을수록 위)
# - is_recommendation: True이면 @channel 알림을 유발하는 '추천'으로 간주
# - show_return: True이면 메시지에 '수익률' 정보를 포함
DECISION_CONFIG = {
    # 보유  (알림 없음)
    "HOLD": {
        "display_name": "<💼 보유>",
        "order": 1,
        "is_recommendation": False,
        "show_return": True,
    },
    # 매도 추천 (알림 발생)
    "CUT_STOPLOSS": {
        "display_name": "<🚨 손절매도>",
        "order": 10,
        "is_recommendation": True,
        "show_return": False,
    },
    "SELL_TREND": {
        "display_name": "<📉 추세이탈 매도>",
        "order": 11,
        "is_recommendation": True,
        "show_return": False,
    },
    "SELL_REPLACE": {
        "display_name": "<🔄 교체매도>",
        "order": 12,
        "is_recommendation": True,
        "show_return": False,
    },
    # 매수 추천 (알림 발생)
    "BUY_REPLACE": {
        "display_name": "<🔄 교체매수>",
        "order": 20,
        "is_recommendation": True,
        "show_return": True,
    },
    "BUY": {
        "display_name": "<🚀 신규매수>",
        "order": 21,
        "is_recommendation": True,
        "show_return": True,
    },
    # 거래 완료 (알림 없음)
    "SOLD": {
        "display_name": "<✅ 매도 완료>",
        "order": 40,
        "is_recommendation": False,
        "show_return": False,
    },
    # 보유 및 대기 (알림 없음)
    "WAIT": {
        "display_name": "<⏳ 대기>",
        "order": 50,
        "is_recommendation": False,
        "show_return": False,
    },
}

# 코인 보유 수량에서 0으로 간주할 임계값 (거래소의 dust 처리)
COIN_ZERO_THRESHOLD = 1e-9


_STATUS_LOGGER = None


def get_status_logger() -> logging.Logger:
    """로그 파일(콘솔 출력 없이)에 기록하는 status 전용 로거를 반환합니다."""
    global _STATUS_LOGGER
    if _STATUS_LOGGER:
        return _STATUS_LOGGER

    logger = logging.getLogger("status.detail")
    if not logger.handlers:
        project_root = Path(__file__).resolve().parent
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    _STATUS_LOGGER = logger
    return logger


def get_next_trading_day(country: str, start_date: pd.Timestamp) -> pd.Timestamp:
    """주어진 날짜 또는 그 이후의 가장 가까운 거래일을 반환합니다."""
    if country == "coin":
        return start_date
    try:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = (start_date + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        days = get_trading_days(start_str, end_str, country)
        for d in days:
            if d.date() >= start_date.date():
                return pd.Timestamp(d).normalize()
    except Exception:
        pass
    # 폴백: 토/일이면 다음 월요일, 평일이면 그대로
    wd = start_date.weekday()
    delta = 0 if wd < 5 else (7 - wd)
    return (start_date + pd.Timedelta(days=delta)).normalize()


@dataclass
class StatusReportData:
    portfolio_data: Dict
    data_by_tkr: Dict
    total_holdings_value: float
    datestamps: List
    pairs: List[Tuple[str, str]]
    base_date: pd.Timestamp
    regime_info: Optional[Dict]
    etf_meta: Dict
    failed_tickers_info: Dict
    description: str


def get_market_regime_status_string() -> Optional[str]:
    """
    S&P 500 지수를 기준으로 현재 시장 레짐 상태를 계산하여 HTML 문자열로 반환합니다.
    """
    # 공통 설정 로드 (DB)
    common = get_common_settings()
    if not common:
        # 설정이 없으면 안내 문구를 회색으로 표시하여 사용자에게 알림
        return '<span style="color:grey">시장 상태: 설정 필요</span>'
    try:
        regime_filter_enabled = bool(common.get("MARKET_REGIME_FILTER_ENABLED"))
        if not regime_filter_enabled:
            return '<span style="color:grey">시장 상태: 비활성화</span>'
        regime_ticker = str(common["MARKET_REGIME_FILTER_TICKER"])
        regime_ma_period = int(common["MARKET_REGIME_FILTER_MA_PERIOD"])
    except KeyError:
        return '<span style="color:grey">시장 상태: 설정 필요</span>'
    except (ValueError, TypeError):
        print("오류: 공통 설정의 시장 레짐 필터 값 형식이 올바르지 않습니다.")
        return '<span style="color:grey">시장 상태: 설정 오류</span>'

    # 데이터 로딩에 필요한 기간 계산: 레짐 MA 기간을 만족하도록 동적으로 산정
    # 거래일 기준 대략 22일/월 가정 + 여유 버퍼
    required_days = int(regime_ma_period) + 30
    required_months = max(3, (required_days // 22) + 2)

    # 데이터 조회
    df_regime = fetch_ohlcv(
        regime_ticker,
        country="kor",
        months_range=[required_months, 0],  # country doesn't matter for index
    )
    # 만약 데이터가 부족하면, 기간을 늘려 한 번 더 시도합니다.
    if df_regime is None or df_regime.empty or len(df_regime) < regime_ma_period:
        df_regime = fetch_ohlcv(regime_ticker, country="kor", months_range=[required_months * 2, 0])

    if df_regime is None or df_regime.empty or len(df_regime) < regime_ma_period:
        return '<span style="color:grey">시장 상태: 데이터 부족</span>'

    # 지표 계산
    df_regime["MA"] = df_regime["Close"].rolling(window=regime_ma_period).mean()
    df_regime.dropna(subset=["MA"], inplace=True)

    # --- 최근 투자 중단 기간 찾기 ---
    risk_off_periods_str = ""
    if not df_regime.empty:
        is_risk_off_series = df_regime["Close"] < df_regime["MA"]

        # 모든 완료된 리스크 오프 기간을 찾습니다.
        completed_periods = []
        in_period = False
        start_date = None
        for i, (dt, is_off) in enumerate(is_risk_off_series.items()):
            if is_off and not in_period:
                in_period = True
                start_date = dt
            elif not is_off and in_period:
                in_period = False
                # 리스크 오프 기간의 마지막 날은 is_off가 False가 되기 바로 전날입니다.
                # i > 0 이므로 is_risk_off_series.index[i - 1]은 안전합니다.
                end_date = is_risk_off_series.index[i - 1]
                completed_periods.append((start_date, end_date))
                start_date = None

        if completed_periods:
            # 최근 1개의 중단 기간을 가져옵니다.
            recent_periods = completed_periods[-1:]
            period_strings = [
                f"{start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}"
                for start, end in recent_periods
            ]
            if period_strings:
                risk_off_periods_str = f" (최근 중단: {', '.join(period_strings)})"

    current_price = df_regime["Close"].iloc[-1]
    current_ma = df_regime["MA"].iloc[-1]

    if pd.notna(current_price) and pd.notna(current_ma) and current_ma > 0:
        proximity_pct = ((current_price / current_ma) - 1) * 100
        is_risk_off = current_price < current_ma

        status_text = "위험" if is_risk_off else "안전"
        color = "orange" if is_risk_off else "green"
        return f'시장: <span style="color:{color}">{status_text} ({proximity_pct:+.1f}%)</span>{risk_off_periods_str}'

    return f'<span style="color:grey">시장 상태: 계산 불가</span>{risk_off_periods_str}'


def get_benchmark_status_string(country: str, date_str: Optional[str] = None) -> Optional[str]:
    """
    포트폴리오의 누적 수익률을 벤치마크와 비교하여 초과 성과를 HTML 문자열로 반환합니다.
    가상화폐의 경우, 여러 벤치마크와 비교할 수 있습니다.
    """
    # 1. 설정 로드
    # 함수 내에서 동적으로 import가 필요할 경우, 함수 상단에 배치하여 스코프 문제를 방지합니다.
    from utils.data_loader import fetch_ohlcv

    app_settings = get_app_settings(country)
    if (
        not app_settings
        or "initial_capital" not in app_settings
        or "initial_date" not in app_settings
    ):
        return None

    initial_capital = float(app_settings["initial_capital"])
    initial_date = pd.to_datetime(app_settings["initial_date"])

    if initial_capital <= 0:
        return None

    # 2. 해당 날짜의 포트폴리오 스냅샷 로드
    portfolio_data = get_portfolio_snapshot(country, date_str)
    if not portfolio_data:
        return None

    current_equity = float(portfolio_data.get("total_equity", 0.0))
    base_date = pd.to_datetime(portfolio_data["date"]).normalize()

    # --- 데이터 오염 방지를 위한 평가금액 재계산 ---
    # DB의 평가금액이 오염되었을 수 있으므로, 보유 종목의 현재가 합계를 직접 계산하여 비교합니다.
    # 이는 비효율적이지만, 데이터 정합성을 보장하기 위한 방어적 코드입니다.
    equity_for_calc = current_equity
    if country == "aus":
        holdings = portfolio_data.get("holdings", [])
        recalculated_holdings_value = 0.0

        for h in holdings:
            df = fetch_ohlcv(h["ticker"], country=country, months_back=1, base_date=base_date)
            if df is not None and not df.empty:
                price = df["Close"].iloc[-1]
                recalculated_holdings_value += h["shares"] * price

        if portfolio_data.get("international_shares"):
            recalculated_holdings_value += portfolio_data["international_shares"].get("value", 0.0)

        # 휴리스틱: DB 평가금액이 재계산된 보유금액보다 10배 이상 크면, 데이터 오염으로 간주합니다.
        if recalculated_holdings_value > 1 and (current_equity / recalculated_holdings_value) > 10:
            equity_for_calc = recalculated_holdings_value  # 현금을 무시하고 보유금액만 사용

    # 3. 포트폴리오 누적 수익률 계산
    portfolio_cum_ret_pct = ((equity_for_calc / initial_capital) - 1.0) * 100.0

    def _calculate_and_format_single_benchmark(
        benchmark_ticker: str,
        benchmark_country: str,
        display_name_override: Optional[str] = None,
    ) -> str:
        """단일 벤치마크와의 비교 문자열을 생성하는 헬퍼 함수입니다."""
        df_benchmark = fetch_ohlcv(
            benchmark_ticker,
            country=benchmark_country,
            date_range=[
                initial_date.strftime("%Y-%m-%d"),
                base_date.strftime("%Y-%m-%d"),
            ],
        )

        # Fallbacks when primary source is unavailable
        if df_benchmark is None or df_benchmark.empty:
            # 1) KRX/KOR fallback via yfinance (e.g., 379800 -> 379800.KS)
            if benchmark_country == "kor" and yf is not None:
                try:
                    y_ticker = benchmark_ticker
                    if benchmark_ticker.isdigit() and len(benchmark_ticker) == 6:
                        y_ticker = f"{benchmark_ticker}.KS"
                    df_y = yf.download(
                        y_ticker,
                        start=initial_date.strftime("%Y-%m-%d"),
                        end=(base_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        progress=False,
                        auto_adjust=True,
                    )
                    if df_y is not None and not df_y.empty:
                        # Normalize columns/index to expected shape
                        if isinstance(df_y.columns, pd.MultiIndex):
                            df_y.columns = df_y.columns.get_level_values(0)
                            df_y = df_y.loc[:, ~df_y.columns.duplicated()]
                        if df_y.index.tz is not None:
                            df_y.index = df_y.index.tz_localize(None)
                        df_benchmark = df_y.rename(columns={"Adj Close": "Close"})
                except (
                    Exception
                ):  # TODO: Refine exception handling (e.g., requests.exceptions.RequestException, ValueError)
                    pass
            # 2) COIN fallback via yfinance (e.g., BTC -> BTC-USD)
            if (
                (df_benchmark is None or df_benchmark.empty)
                and benchmark_country == "coin"
                and yf is not None
            ):
                try:
                    y_ticker = (
                        "BTC-USD"
                        if benchmark_ticker.upper() == "BTC"
                        else f"{benchmark_ticker.upper()}-USD"
                    )
                    df_y = yf.download(
                        y_ticker,
                        start=initial_date.strftime("%Y-%m-%d"),
                        end=(base_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        progress=False,
                        auto_adjust=True,
                    )
                    if df_y is not None and not df_y.empty:
                        if isinstance(df_y.columns, pd.MultiIndex):
                            df_y.columns = df_y.columns.get_level_values(0)
                            df_y = df_y.loc[:, ~df_y.columns.duplicated()]
                        if df_y.index.tz is not None:
                            df_y.index = df_y.index.tz_localize(None)
                        df_benchmark = df_y.rename(columns={"Adj Close": "Close"})
                except Exception:
                    pass

        if df_benchmark is None or df_benchmark.empty:
            return f'<span style="color:grey">벤치마크({benchmark_ticker}) 데이터 조회 실패</span>'

        start_prices = df_benchmark[df_benchmark.index >= initial_date]["Close"]
        if start_prices.empty:
            return '<span style="color:grey">벤치마크 시작 가격 조회 실패</span>'
        benchmark_start_price = start_prices.iloc[0]

        end_prices = df_benchmark[df_benchmark.index <= base_date]["Close"]
        if end_prices.empty:
            return '<span style="color:grey">벤치마크 종료 가격 조회 실패</span>'
        benchmark_end_price = end_prices.iloc[-1]

        if (
            pd.isna(benchmark_start_price)
            or pd.isna(benchmark_end_price)
            or benchmark_start_price <= 0
        ):
            return '<span style="color:grey">벤치마크 가격 정보 오류</span>'

        benchmark_cum_ret_pct = ((benchmark_end_price / benchmark_start_price) - 1.0) * 100.0

        excess_return_pct = portfolio_cum_ret_pct - benchmark_cum_ret_pct
        color = "red" if excess_return_pct > 0 else "blue" if excess_return_pct < 0 else "black"

        from utils.data_loader import fetch_pykrx_name, fetch_yfinance_name

        benchmark_name = display_name_override
        if not benchmark_name:
            if benchmark_country == "kor" and _stock:
                benchmark_name = fetch_pykrx_name(benchmark_ticker)
            elif benchmark_country == "aus":
                benchmark_name = fetch_yfinance_name(benchmark_ticker)
            elif benchmark_country == "coin":
                benchmark_name = benchmark_ticker.upper()

        benchmark_display_name = (
            f" vs {benchmark_name}" if benchmark_name else f" vs {benchmark_ticker}"
        )
        return f'초과성과: <span style="color:{color}">{excess_return_pct:+.2f}%</span>{benchmark_display_name}'

    if country == "coin":
        # 가상화폐의 경우, 두 개의 벤치마크와 비교합니다.
        benchmarks_to_compare = [
            {"ticker": "379800", "country": "kor", "name": "KODEX 미국S&P500"},
            {"ticker": "BTC", "country": "coin", "name": "BTC"},
        ]

        results = []
        for bm in benchmarks_to_compare:
            results.append(
                _calculate_and_format_single_benchmark(bm["ticker"], bm["country"], bm["name"])
            )

        return "<br>".join(results)
    else:
        # 기존 로직 (한국/호주)
        try:
            benchmark_ticker = global_settings.BENCHMARK_TICKERS.get(country)
        except AttributeError:
            print("오류: BENCHMARK_TICKERS 설정이 settings.py 에 정의되어야 합니다.")
            return None
        if not benchmark_ticker:
            return None

        return _calculate_and_format_single_benchmark(benchmark_ticker, country)


def is_market_open(country: str = "kor") -> bool:
    """
    지정된 국가의 주식 시장이 현재 개장 시간인지 확인합니다.
    정확한 공휴일을 반영하여 시간과 요일, 날짜를 모두 판단합니다.
    """
    if not pytz:
        return False  # pytz 없으면 안전하게 False 반환

    if country == "coin":  # 코인은 항상 열려있다고 가정
        return True

    timezones = {"kor": "Asia/Seoul", "aus": "Australia/Sydney"}
    market_hours = {
        "kor": (
            datetime.strptime("09:00", "%H:%M").time(),
            datetime.strptime("15:30", "%H:%M").time(),
        ),
        "aus": (
            datetime.strptime("10:00", "%H:%M").time(),  # 호주 시드니 시간 기준
            datetime.strptime("16:00", "%H:%M").time(),  # 호주 시드니 시간 기준
        ),
    }

    tz_str = timezones.get(country)
    if not tz_str:
        return False

    try:
        local_tz = pytz.timezone(tz_str)
        now_local = datetime.now(local_tz)

        # 1. 거래일인지 확인 (공휴일 포함)
        today_str_for_util = now_local.strftime("%Y-%m-%d")
        is_trading_day_today = bool(
            get_trading_days(today_str_for_util, today_str_for_util, country)
        )

        if not is_trading_day_today:
            return False

        # 2. 개장 시간 확인
        market_open_time, market_close_time = market_hours[country]
        return market_open_time <= now_local.time() <= market_close_time
    except Exception:  # TODO: Refine exception handling
        return False  # 오류 발생 시 안전하게 False 반환


def _determine_target_date_for_scheduler(country: str) -> pd.Timestamp:
    """
    스케줄러 실행 시, 현재 시간에 따라 계산 대상 날짜를 동적으로 결정합니다.
    - 코인: 항상 오늘
    - 주식/ETF: 장 마감 2시간 후부터는 다음 거래일을 계산 대상으로 함.
    """
    if country == "coin":
        # 서버의 시간대가 UTC일 수 있으므로, 한국 시간 기준으로 '오늘'을 결정합니다.
        if pytz:
            try:
                seoul_tz = pytz.timezone("Asia/Seoul")
                return pd.Timestamp.now(seoul_tz).normalize()
            except Exception:
                # pytz가 있으나 타임존을 못찾는 경우 폴백
                return pd.Timestamp.now().normalize()
        else:
            # pytz가 없는 경우, 시스템 시간에 의존
            return pd.Timestamp.now().normalize()

    # 각 시장의 현지 시간과 장 마감 시간을 기준으로 대상 날짜를 결정합니다.
    market_settings = {
        "kor": {"tz": "Asia/Seoul", "close": "15:30"},
        "aus": {"tz": "Australia/Sydney", "close": "16:00"},
    }
    settings = market_settings.get(country)
    if not settings or not pytz:
        # 설정이 없거나 pytz가 없으면 오늘 날짜로 폴백
        return pd.Timestamp.now().normalize()

    try:
        local_tz = pytz.timezone(settings["tz"])
        now_local = datetime.now(local_tz)
    except Exception:
        now_local = datetime.now()  # 폴백

    today = pd.Timestamp(now_local).normalize()

    # 오늘이 거래일인지 확인
    try:
        today_str = today.strftime("%Y-%m-%d")
        is_trading_today = bool(get_trading_days(today_str, today_str, country))
    except Exception:
        is_trading_today = now_local.weekday() < 5  # 폴백

    if is_trading_today:
        close_time = datetime.strptime(settings["close"], "%H:%M").time()
        close_datetime_naive = datetime.combine(today.date(), close_time)
        # Naive datetime을 localize 해야 시간대 계산이 정확함
        close_datetime_local = local_tz.localize(close_datetime_naive)
        cutoff_datetime_local = close_datetime_local + pd.Timedelta(hours=2)

        if now_local < cutoff_datetime_local:
            # 컷오프 이전: 오늘 날짜를 대상으로 함
            target_date = today
        else:
            # 컷오프 이후: 다음 거래일을 대상으로 함
            start_search_date = today + pd.Timedelta(days=1)
            target_date = get_next_trading_day(country, start_search_date)
    else:
        # 오늘이 거래일이 아님 (주말/공휴일): 다음 거래일을 대상으로 함
        target_date = get_next_trading_day(country, today)
    return target_date


def calculate_consecutive_holding_info(
    held_tickers: List[str], country: str, as_of_date: datetime
) -> Dict[str, Dict]:
    """
    'trades' 컬렉션을 스캔하여 각 티커의 연속 보유 시작일을 계산합니다.
    'buy_date' (연속 보유 시작일)을 포함한 딕셔너리를 반환합니다.
    """
    holding_info = {tkr: {"buy_date": None} for tkr in held_tickers}
    if not held_tickers:
        return holding_info

    db = get_db_connection()
    if db is None:
        print("-> 경고: DB에 연결할 수 없어 보유일 계산을 건너뜁니다.")
        return holding_info

    # 코인은 트레이드가 시각 포함으로 기록되므로, 동일 달력일의 모든 거래를 포함하도록
    # as_of_date 상한을 해당일 23:59:59.999999로 확장합니다.
    # 모든 국가에 대해 동일하게 적용하여, 특정 날짜의 모든 거래를 포함하도록 합니다.
    include_until = as_of_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    for tkr in held_tickers:
        try:
            # 해당 티커의 모든 거래를 날짜 내림차순, 그리고 같은 날짜 내에서는 생성 순서(_id) 내림차순으로 가져옵니다.
            # 이를 통해 동일한 날짜에 발생한 거래의 순서를 정확히 반영하여 연속 보유 기간을 계산합니다.
            trades = list(
                db.trades.find(
                    {
                        "country": country,
                        "ticker": tkr,
                        "date": {"$lte": include_until},
                    },
                    sort=[("date", DESCENDING), ("_id", DESCENDING)],
                )
            )

            if not trades:
                continue

            # 현재 보유 수량을 계산합니다.
            current_shares = 0
            for trade in reversed(trades):  # 시간순으로 반복
                if trade["action"] == "BUY":
                    current_shares += trade["shares"]
                elif trade["action"] == "SELL":
                    current_shares -= trade["shares"]

            # 현재부터 과거로 시간을 거슬러 올라가며 확인합니다.
            buy_date = None
            for trade in trades:  # 날짜 내림차순으로 정렬되어 있음
                if current_shares <= 0:
                    break  # 현재 보유 기간의 시작점을 지났음

                buy_date = trade["date"]  # 잠재적인 매수 시작일
                if trade["action"] == "BUY":
                    current_shares -= trade["shares"]
                elif trade["action"] == "SELL":
                    current_shares += trade["shares"]

            if buy_date:
                holding_info[tkr]["buy_date"] = buy_date
        except Exception as e:
            print(f"-> 경고: {tkr} 보유일 계산 중 오류 발생: {e}")

    return holding_info


def build_pairs_with_holdings(
    pairs: List[Tuple[str, str]], holdings: dict
) -> List[Tuple[str, str]]:
    name_map = {t: n for t, n in pairs if n}
    out_map = {t: n for t, n in pairs}
    # If holdings has tickers not in pairs, add with blank name
    for tkr in holdings.keys():
        if tkr not in out_map:
            out_map[tkr] = name_map.get(tkr, "")
    return [(t, out_map.get(t, "")) for t in out_map.keys()]


def _format_return_for_header(label: str, pct: float, amount: float, formatter: callable) -> str:
    """수익률과 금액을 HTML 색상과 함께 포맷팅합니다."""
    color = "red" if pct > 0 else "blue" if pct < 0 else "black"
    # Streamlit의 st.markdown은 HTML을 지원합니다.
    formatted_amount = formatter(amount)
    return f'{label}: <span style="color:{color}">{pct:+.2f}%({formatted_amount})</span>'


def _load_and_prepare_ticker_data(args):
    """
    단일 티커에 대한 데이터 조회 및 지표 계산을 수행하는 워커 함수입니다.
    """
    # Unpack arguments
    tkr, country, required_months, base_date, ma_period, atr_period_norm, df_full = args
    from utils.indicators import calculate_atr

    if df_full is None:
        from utils.data_loader import fetch_ohlcv

        # df_full이 제공되지 않으면, 네트워크를 통해 데이터를 새로 조회합니다.
        df = fetch_ohlcv(
            tkr, country=country, months_range=[required_months, 0], base_date=base_date
        )
    else:
        # df_full이 제공되면, base_date까지의 데이터만 잘라서 사용합니다.
        df = df_full[df_full.index <= base_date].copy()

    if df is None:
        return tkr, {"error": "FETCH_FAILED"}

    if len(df) < max(ma_period, atr_period_norm):
        return tkr, {"error": "INSUFFICIENT_DATA"}

    # yfinance가 가끔 MultiIndex 컬럼을 반환하는 경우에 대비하여,
    # 컬럼을 단순화하고 중복을 제거합니다.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    ma = close.rolling(window=ma_period).mean()
    atr = calculate_atr(df, period=atr_period_norm)

    buy_signal_active = close > ma
    buy_signal_days = (
        buy_signal_active.groupby((buy_signal_active != buy_signal_active.shift()).cumsum())
        .cumsum()
        .fillna(0)
        .astype(int)
    )

    return tkr, {
        "df": df,
        "close": close,
        "ma": ma,
        "atr": atr,
        "buy_signal_days": buy_signal_days,
        "ma_period": ma_period,
    }


def _fetch_and_prepare_data(
    country: str,
    date_str: Optional[str],
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Optional[StatusReportData]:
    """
    주어진 종목 목록에 대해 OHLCV 데이터를 조회하고,
    신호 계산에 필요한 보조지표(이동평균, ATR 등)를 계산합니다.
    """
    logger = get_status_logger()

    # 설정을 불러옵니다.
    app_settings = get_app_settings(country)
    if not app_settings or "ma_period" not in app_settings:
        print(
            f"오류: '{country}' 국가의 전략 파라미터(MA 기간)가 설정되지 않았습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요."
        )
        return None

    try:
        ma_period = int(app_settings["ma_period"])
    except (ValueError, TypeError):
        print(f"오류: '{country}' 국가의 MA 기간 설정이 올바르지 않습니다.")
        return None

    request_label = date_str or "auto"
    logger.info(
        "[%s] status data preparation started (input date=%s)", country.upper(), request_label
    )

    # 현황 조회 시, 날짜가 지정되지 않으면 항상 오늘 날짜를 기준으로 조회합니다.
    if date_str is None:
        target_date = _determine_target_date_for_scheduler(country)
        date_str = target_date.strftime("%Y-%m-%d")

    portfolio_data = get_portfolio_snapshot(country, date_str)
    if not portfolio_data:
        print(
            f"오류: '{country}' 국가의 '{date_str}' 날짜에 대한 포트폴리오 스냅샷을 DB에서 찾을 수 없습니다. 거래 내역이 없거나 DB 연결에 문제가 있을 수 있습니다."
        )
        logger.warning("[%s] portfolio snapshot missing for %s", country.upper(), date_str)
        return None
    try:
        # DB에서 가져온 date는 스냅샷의 기준일이 됩니다.
        base_date = pd.to_datetime(portfolio_data["date"]).normalize()
    except (ValueError, TypeError):
        print("경고: 포트폴리오 스냅샷에서 날짜를 추출할 수 없습니다. 현재 날짜를 사용합니다.")
        base_date = pd.Timestamp.now().normalize()

    logger.info(
        "[%s] portfolio snapshot loaded for %s (holdings=%d)",
        country.upper(),
        base_date.strftime("%Y-%m-%d"),
        len(portfolio_data.get("holdings", [])),
    )

    # 콘솔 로그에 국가/날짜를 포함하여 표시
    try:
        print(f"{country}/{base_date.strftime('%Y-%m-%d')} 현황을 계산합니다")
    except Exception:
        pass

    holdings = {
        item["ticker"]: {
            "name": item.get("name", ""),
            "shares": item.get("shares", 0),
            "avg_cost": item.get("avg_cost", 0.0),
        }
        for item in portfolio_data.get("holdings", [])
        if item.get("ticker")
    }

    # DB에서 종목 목록을 가져와 전체 유니버스를 구성합니다.
    etfs_from_file = get_etfs(country)
    etf_meta = {etf["ticker"]: etf for etf in etfs_from_file}

    # 오늘 판매된 종목을 추가합니다.
    sold_tickers_today = set()
    trades_on_base_date = get_trades_on_date(country, base_date)
    for trade in trades_on_base_date:
        if trade["action"] == "SELL":
            sold_tickers_today.add(trade["ticker"])
            # etf_meta에 없는 경우 추가 (이름은 나중에 채워질 수 있음)
            if trade["ticker"] not in etf_meta:
                etf_meta[trade["ticker"]] = {
                    "ticker": trade["ticker"],
                    "name": trade.get("name", ""),
                    "category": "",
                }
            # holdings에 없는 경우 추가 (shares=0으로)
            if trade["ticker"] not in holdings:
                holdings[trade["ticker"]] = {
                    "name": trade.get("name", ""),
                    "shares": 0,
                    "avg_cost": 0.0,
                }

    # 모든 티커를 포함하도록 pairs를 재구성합니다.
    all_tickers_for_processing = set(holdings.keys()) | set(etf_meta.keys())
    pairs = []
    for tkr in all_tickers_for_processing:
        name = etf_meta.get(tkr, {}).get("name") or holdings.get(tkr, {}).get("name") or ""
        pairs.append((tkr, name))

    logger.info(
        "[%s] gathered universe: holdings=%d, meta=%d, total_pairs=%d",
        country.upper(),
        len(holdings),
        len(etf_meta),
        len(pairs),
    )

    # 국가별로 다른 포맷터 사용
    def _fetch_realtime_price(tkr):
        from utils.data_loader import fetch_naver_realtime_price

        return fetch_naver_realtime_price(tkr) if country == "kor" else None

    # 실시간 가격 조회는 포트폴리오 기준일이 오늘일 경우에만 시도합니다.
    today_cal = pd.Timestamp.now().normalize()
    market_is_open = is_market_open(country) and base_date.date() == today_cal.date()
    if market_is_open and base_date.date() == today_cal.date():
        if country == "kor":
            print("-> 장중입니다. 네이버 금융에서 실시간 시세를 가져옵니다 (비공식, 지연 가능).")

    # --- 신호 계산 (공통 설정에서) ---
    common = get_common_settings()
    if not common:
        print("오류: 공통 설정이 DB에 없습니다. '설정' 탭에서 값을 저장해주세요.")
        return None
    try:
        atr_period_norm = int(common["ATR_PERIOD_FOR_NORMALIZATION"])
        regime_filter_enabled = bool(common["MARKET_REGIME_FILTER_ENABLED"])
        regime_ma_period = int(common["MARKET_REGIME_FILTER_MA_PERIOD"])
    except KeyError as e:
        print(f"오류: 공통 설정 '{e.args[0]}' 값이 없습니다.")
        return None
    except (ValueError, TypeError):
        print("오류: 공통 설정 값 형식이 올바르지 않습니다.")
        return None

    # DB에서 종목 유형(ETF/주식) 정보 가져오기
    # 코인은 거래소 잔고 기반 표시이므로, 종목 마스터가 비어 있어도 보유코인을 기준으로 진행합니다.
    if not etfs_from_file and country != "coin":
        print(
            f"오류: 'data/{country}/' 폴더에서 '{country}' 국가의 현황을 계산할 종목을 찾을 수 없습니다."
        )
        return None

    max_ma_period = max(ma_period, regime_ma_period if regime_filter_enabled else 0)
    required_days = max(max_ma_period, atr_period_norm) + 5  # 버퍼 추가
    required_months = (required_days // 22) + 2

    # --- 시장 레짐 필터 데이터 로딩 ---
    regime_info = None
    if regime_filter_enabled:
        if "MARKET_REGIME_FILTER_TICKER" not in common:
            print("오류: 공통 설정에 MARKET_REGIME_FILTER_TICKER 값이 없습니다.")
            return None
        regime_ticker = str(common["MARKET_REGIME_FILTER_TICKER"])

        df_regime = fetch_ohlcv(
            regime_ticker,
            country=country,
            months_range=[required_months, 0],
            base_date=base_date,
        )

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

    data_by_tkr = {}
    total_holdings_value = 0.0
    datestamps = []
    failed_tickers_info = {}

    # 코인도 이제 trades 기반 포트폴리오를 사용합니다. (빗썸 스냅샷→trades 동기화 별도 스크립트)
    if country == "coin":
        # 제외할 특수 심볼 제거 (예: 'P')
        # 종목 마스터에 없는 종목은 처리에서 제외합니다. (단, 오늘 매도된 종목은 포함)
        allowed_tickers = {etf["ticker"] for etf in etfs_from_file}
        pairs = [(t, n) for t, n in pairs if t in allowed_tickers or t in sold_tickers_today]
        logger.info(
            "[%s] coin universe filtered to %d tickers (allowed=%d, sold_today=%d)",
            country.upper(),
            len(pairs),
            len(allowed_tickers),
            len(sold_tickers_today),
        )

    # --- 병렬 데이터 로딩 및 지표 계산 ---
    tasks = []
    for tkr, _ in pairs:
        df_full = prefetched_data.get(tkr) if prefetched_data else None
        tasks.append(
            (
                tkr,
                country,
                required_months,
                base_date,
                ma_period,
                atr_period_norm,
                df_full,
            )
        )

    # 병렬 처리로 데이터 로딩 및 기본 지표 계산
    processed_results = {}
    desc = "과거 데이터 처리" if prefetched_data else "종목 데이터 로딩"
    logger.info(
        "[%s] %s started (tickers=%d)",
        country.upper(),
        desc,
        len(tasks),
    )
    print(f"-> {desc} 시작... (총 {len(tasks)}개 종목)")

    # 직렬 처리로 데이터 로딩 및 기본 지표 계산
    for i, task in enumerate(tasks):
        tkr = task[0]
        try:
            _, result = _load_and_prepare_ticker_data(task)
            processed_results[tkr] = result
        except Exception as exc:
            print(f"\n-> 경고: {tkr} 데이터 처리 중 오류 발생: {exc}")
            processed_results[tkr] = {"error": "PROCESS_ERROR"}
            logger.exception("[%s] %s data processing error", country.upper(), tkr)

        # 진행 상황 표시
        print(f"\r   {desc} 진행: {i + 1}/{len(tasks)}", end="", flush=True)

    print("\n-> 데이터 처리 완료.")
    logger.info("[%s] %s finished", country.upper(), desc)

    # --- 최종 데이터 조합 및 계산 ---
    # --- 최종 데이터 조합 및 계산 ---
    # 이제 `processed_results`를 사용하여 순차적으로 나머지 계산을 수행합니다.
    for tkr, _ in pairs:
        result = processed_results.get(tkr)
        if not result:
            failed_tickers_info[tkr] = "FETCH_FAILED"
            logger.warning("[%s] %s missing result (treated as FETCH_FAILED)", country.upper(), tkr)
            continue

        if "error" in result:
            failed_tickers_info[tkr] = result["error"]
            logger.warning(
                "[%s] %s excluded due to %s",
                country.upper(),
                tkr,
                result["error"],
            )
            continue

        realtime_price = _fetch_realtime_price(tkr) if market_is_open else None
        c0 = float(realtime_price) if realtime_price else float(result["close"].iloc[-1])
        if pd.isna(c0) or c0 <= 0:
            failed_tickers_info[tkr] = "FETCH_FAILED"
            logger.warning(
                "[%s] %s excluded (invalid price: %s, realtime=%s)",
                country.upper(),
                tkr,
                c0,
                bool(realtime_price),
            )
            continue

        prev_close = (
            float(result["close"].iloc[-2])
            if len(result["close"]) >= 2 and pd.notna(result["close"].iloc[-2])
            else 0.0
        )
        m = result["ma"].iloc[-1]
        a = result["atr"].iloc[-1]

        ma_score = (c0 - m) / a if pd.notna(m) and pd.notna(a) and a > 0 else 0.0
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

        logger.debug(
            "[%s] %s processed: shares=%.4f price=%.2f prev_close=%.2f data_points=%d buy_signal_days=%d",
            country.upper(),
            tkr,
            sh,
            c0,
            prev_close,
            len(result["df"]),
            buy_signal_days_today,
        )

    fail_counts: Dict[str, int] = {}
    for reason in failed_tickers_info.values():
        fail_counts[reason] = fail_counts.get(reason, 0) + 1

    logger.info(
        "[%s] status data summary for %s: processed=%d, failures=%s",
        country.upper(),
        base_date.strftime("%Y-%m-%d"),
        len(data_by_tkr),
        fail_counts or "{}",
    )

    return StatusReportData(
        portfolio_data=portfolio_data,
        data_by_tkr=data_by_tkr,
        total_holdings_value=total_holdings_value,
        datestamps=datestamps,
        pairs=pairs,
        base_date=base_date,
        regime_info=regime_info,
        etf_meta=etf_meta,
        failed_tickers_info=failed_tickers_info,
        description=desc,
    )


def _build_header_line(
    country,
    portfolio_data,
    current_equity,
    total_holdings_value,
    data_by_tkr,
    base_date,
):
    """리포트의 헤더 라인을 생성합니다."""
    # 국가별 포맷터 설정
    money_formatter = format_kr_money if country != "aus" else format_aud_money

    # 보유 종목 수
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

    # 해외 주식 가치 포함
    total_holdings = total_holdings_value
    # 코인도 다른 국가와 동일하게 보유금액은 포지션 합으로 계산합니다.

    # 현금
    total_cash = float(current_equity) - float(total_holdings)

    # --- 데이터 오염 방지를 위한 누적 수익률용 평가금액 보정 ---
    equity_for_cum_calc = current_equity
    # 휴리스틱: DB 평가금액이 재계산된 보유금액보다 10배 이상 크면, 데이터 오염으로 간주합니다.
    if country == "aus" and total_holdings > 1 and current_equity > 1:
        if (current_equity / total_holdings) > 10:
            equity_for_cum_calc = total_holdings  # 현금을 무시하고 보유금액만 사용

    # 누적 수익률 및 TopN
    app_settings = get_app_settings(country)
    initial_capital_local = float(app_settings.get("initial_capital", 0)) if app_settings else 0.0
    cum_ret_pct = (
        ((equity_for_cum_calc / initial_capital_local) - 1.0) * 100.0
        if initial_capital_local > 0
        else 0.0
    )
    portfolio_topn = app_settings.get("portfolio_topn", 0) if app_settings else 0

    cum_profit_loss = equity_for_cum_calc - initial_capital_local

    today_cal = pd.Timestamp.now().normalize()

    # 표시 날짜는 항상 계산 기준일(base_date)을 따릅니다.
    label_date = base_date

    # 라벨(오늘, 다음 거래일 등)을 결정합니다.
    if base_date.date() < today_cal.date():
        day_label = "기준일"
    elif base_date.date() > today_cal.date():
        day_label = "다음 거래일"
    else:
        day_label = "오늘"

    # 일간 수익률: 다음 거래일 기준일에는 아직 수익률이 없으므로 0 처리
    if day_label == "다음 거래일":
        day_ret_pct = 0.0
        day_profit_loss = 0.0
    else:
        compare_date = base_date
        prev_snapshot = get_previous_portfolio_snapshot(country, compare_date)
        prev_equity = float(prev_snapshot.get("total_equity", 0.0)) if prev_snapshot else None
        day_ret_pct = (
            ((current_equity / prev_equity) - 1.0) * 100.0
            if prev_equity and prev_equity > 0
            else 0.0
        )
        day_profit_loss = current_equity - prev_equity if prev_equity else 0.0

    # 평가 수익률
    total_aus_etf_acquisition_cost = sum(
        d["shares"] * d["avg_cost"] for d in data_by_tkr.values() if d["shares"] > 0
    )

    # 최종 평가 수익률 계산을 위한 변수 초기화
    final_total_holdings_value = total_holdings_value
    final_total_acquisition_cost = total_aus_etf_acquisition_cost
    eval_ret_pct = (
        ((final_total_holdings_value / final_total_acquisition_cost) - 1.0) * 100.0
        if final_total_acquisition_cost > 0
        else 0.0
    )
    eval_profit_loss = final_total_holdings_value - final_total_acquisition_cost

    # 헤더 문자열 생성
    equity_str = money_formatter(current_equity)
    holdings_str = money_formatter(total_holdings)
    cash_str = money_formatter(total_cash)
    day_ret_str = _format_return_for_header("일간", day_ret_pct, day_profit_loss, money_formatter)
    eval_ret_str = _format_return_for_header(
        "평가", eval_ret_pct, eval_profit_loss, money_formatter
    )
    cum_ret_str = _format_return_for_header("누적", cum_ret_pct, cum_profit_loss, money_formatter)

    # 헤더 본문
    header_body = (
        f"보유종목: {held_count}/{portfolio_topn} | 평가금액: {equity_str} | 보유금액: {holdings_str} | "
        f"현금: {cash_str} | {day_ret_str} | {eval_ret_str} | {cum_ret_str}"
    )

    # 평가금액 경고: 표시 기준일의 평가금액이 없으면 최근 평가금액 날짜를 안내
    equity_date = portfolio_data.get("equity_date") or base_date
    if label_date.normalize() != pd.to_datetime(equity_date).normalize():
        target_date = label_date
        weekday_map = ["월", "화", "수", "목", "금", "토", "일"]
        weekday_str = weekday_map[target_date.weekday()]
        stale_str = pd.to_datetime(equity_date).strftime("%Y-%m-%d")
        warning_msg = f"<br><span style='color:orange;'>⚠️ {target_date.strftime('%Y년 %m월 %d일')}({weekday_str})의 평가금액이 없습니다. 최근({stale_str}) 평가금액으로 현황을 계산합니다.</span>"
        header_body += warning_msg

    return header_body, label_date, day_label


def _notify_calculation_start(
    country: str, num_tickers: int, description: str, warnings: List[str]
):
    """계산 시작과 경고에 대한 슬랙 알림을 보냅니다."""
    try:
        from utils.notify import get_slack_webhook_url, send_slack_message
    except Exception:
        return False

    webhook_url = get_slack_webhook_url(country)
    if not webhook_url:
        return False

    app_type = os.environ.get("APP_TYPE", "SERVER")
    country_kor = {"kor": "한국", "aus": "호주", "coin": "코인"}.get(country, country.upper())

    message_lines = [
        f"[{app_type}][{country_kor}] 계산",
        f"- 대상 종목: {num_tickers}개",
        f"- 계산 내용: {description}",
    ]

    if warnings:
        max_warnings = 10
        message_lines.append("- 경고:")
        for i, warning in enumerate(warnings):
            if i < max_warnings:
                message_lines.append(f"  ⚠️ {warning}")
        if len(warnings) > max_warnings:
            message_lines.append(f"  ... 외 {len(warnings) - max_warnings}건의 경고가 더 있습니다.")

    message = "\n".join(message_lines)

    return send_slack_message(message, webhook_url=webhook_url)


def _notify_equity_update(country: str, old_equity: float, new_equity: float):
    """평가금액 자동 보정 시 슬랙으로 알림을 보냅니다."""
    try:
        from utils.notify import get_slack_webhook_url, send_slack_message
        from utils.report import format_aud_money, format_kr_money
    except Exception:
        return False

    webhook_url = get_slack_webhook_url(country)
    if not webhook_url:
        return False

    app_type = os.environ.get("APP_TYPE", "SERVER")
    country_kor = {"kor": "한국", "aus": "호주", "coin": "코인"}.get(country, country.upper())
    money_formatter = format_aud_money if country == "aus" else format_kr_money

    diff = new_equity - old_equity
    diff_str = f"{'+' if diff > 0 else ''}{money_formatter(diff)}"

    if old_equity > 0:
        # 평가금액 변동(증가/감소)에 따라 다른 레이블을 사용합니다.
        change_label = "증가" if diff >= 0 else "감소"
        message = f"[{app_type}][{country_kor}] 평가금액 {change_label}: {money_formatter(old_equity)} => {money_formatter(new_equity)} ({diff_str})"
    else:
        message = f"[{app_type}][{country_kor}] 신규 평가금액 저장: {money_formatter(new_equity)}"

    return send_slack_message(message, webhook_url=webhook_url)


def generate_status_report(
    country: str = "kor",
    date_str: Optional[str] = None,
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
    notify_start: bool = False,
) -> Optional[Tuple[str, List[str], List[List[str]], pd.Timestamp]]:
    """지정된 전략에 대한 오늘의 현황 데이터를 생성하여 반환합니다."""
    logger = get_status_logger()
    try:
        # 1. 데이터 로드 및 지표 계산
        result = _fetch_and_prepare_data(country, date_str, prefetched_data)
        if result is None:
            return None
    except Exception:
        raise  # 오류를 다시 발생시켜 호출한 쪽에서 처리하도록 함

    portfolio_data = result.portfolio_data
    data_by_tkr = result.data_by_tkr
    total_holdings_value = result.total_holdings_value
    pairs = result.pairs
    base_date = result.base_date
    etf_meta = result.etf_meta
    failed_tickers_info = result.failed_tickers_info
    desc = result.description

    logger.info(
        "[%s] decision build starting: pairs=%d, successes=%d, failures=%d",
        country.upper(),
        len(pairs),
        len(data_by_tkr),
        len(failed_tickers_info),
    )

    # --- 데이터 유효성 검증 및 경고 생성 ---
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

    # 데이터 조회/처리에 실패한 종목이 있으면, 처리를 중단하고 예외를 발생시킵니다.
    if fetch_failed_tickers:
        # 이 예외는 web_app.py에서 처리하여 사용자에게 메시지를 표시합니다.
        raise ValueError(f"PRICE_FETCH_FAILED:{','.join(sorted(list(set(fetch_failed_tickers))))}")

    # --- 현황 계산 시작 알림 ---
    if notify_start:
        warning_messages_for_slack = []
        if insufficient_data_tickers:
            name_map = {tkr: name for tkr, name in pairs}
            for tkr in sorted(insufficient_data_tickers):
                name = name_map.get(tkr, tkr)
                warning_messages_for_slack.append(
                    f"{name}({tkr}): 데이터 기간이 부족하여 계산에서 제외됩니다."
                )
        _notify_calculation_start(country, len(pairs), desc, warning_messages_for_slack)

    current_equity = float(portfolio_data.get("total_equity", 0.0))
    equity_date = portfolio_data.get("equity_date")

    # 자동 보정 로직을 위한 평가금액 결정:
    # 평가금액의 날짜가 기준일(base_date)과 다르면, 기준일의 평가금액은 0으로 간주합니다.
    # 이렇게 하면, 오늘 날짜의 평가금액이 없을 때 과거 값을 가져와도 '신규'로 처리됩니다.
    equity_for_autocorrect = current_equity
    is_stale_equity = (
        equity_date and pd.to_datetime(equity_date).normalize() != base_date.normalize()
    )
    if is_stale_equity:
        equity_for_autocorrect = 0.0

    international_shares_value = 0.0
    if country == "aus":
        intl_info = portfolio_data.get("international_shares")
        if isinstance(intl_info, dict):
            try:
                international_shares_value = float(intl_info.get("value", 0.0))
            except (TypeError, ValueError):
                international_shares_value = 0.0

    # --- 자동 평가금액 보정 로직 ---
    # 보유 종목의 현재가 합(total_holdings_value)이 기록된 평가금액(equity_for_autocorrect)보다 크거나,
    # 평가금액이 0일 경우, 평가금액을 보유 종목 가치 합으로 자동 보정합니다.
    # 이는 현금이 음수로 표시되는 것을 방지하고, 평가금액 미입력 시 초기값을 설정해줍니다.
    # 호주의 경우, 해외 주식 가치도 포함하여 최종 평가금액을 계산합니다.
    new_equity_candidate = total_holdings_value + international_shares_value

    # new_equity_candidate가 0보다 크고, (기존 평가금액보다 크거나, 기존 평가금액이 0일 때)
    if new_equity_candidate > 0 and (
        new_equity_candidate > equity_for_autocorrect or equity_for_autocorrect == 0
    ):
        old_equity = equity_for_autocorrect
        new_equity = new_equity_candidate

        # 보정된 평가금액이 유의미한 차이를 보일 때만 업데이트 및 알림 (부동소수점 오차 방지)
        if abs(new_equity - old_equity) > 1e-9:
            # 1. DB에 새로운 평가금액 저장
            from utils.db_manager import save_daily_equity

            # 호주: international_shares 정보도 함께 저장해야 함
            is_data_to_save = None
            if country == "aus":
                is_data_to_save = portfolio_data.get("international_shares")

            save_daily_equity(
                country,
                base_date.to_pydatetime(),
                new_equity,
                is_data_to_save,
                updated_by="스케줄러",
            )

            # 2. 슬랙 알림 전송
            _notify_equity_update(country, old_equity, new_equity)

            # 3. 현재 실행 컨텍스트에 보정된 값 반영
            current_equity = new_equity
            portfolio_data["total_equity"] = new_equity
            print(f"-> 평가금액 자동 보정: {old_equity:,.0f}원 -> {new_equity:,.0f}원")

    holdings = {
        item["ticker"]: {
            "name": item.get("name", ""),
            "shares": item.get("shares", 0),
            "avg_cost": item.get("avg_cost", 0.0),
        }
        for item in portfolio_data.get("holdings", [])
        if item.get("ticker")
    }

    # 현재 보유 종목의 카테고리 (TBD 제외)
    held_categories = set()
    for tkr, d in holdings.items():
        if float(d.get("shares", 0.0)) > 0:
            category = etf_meta.get(tkr, {}).get("category")
            if category and category != "TBD":
                held_categories.add(category)

    # 2. 헤더 생성
    total_holdings_value += international_shares_value

    header_line, label_date, day_label = _build_header_line(
        country,
        portfolio_data,
        current_equity,
        total_holdings_value,
        data_by_tkr,
        base_date,
    )

    # 데이터 기간이 부족한 종목에 대한 경고 메시지를 헤더에 추가합니다.
    if insufficient_data_tickers:
        name_map = {tkr: name for tkr, name in pairs}
        warning_messages = []
        for tkr in sorted(insufficient_data_tickers):
            name = name_map.get(tkr, tkr)
            warning_messages.append(f"{name}({tkr}): 데이터 기간이 부족하여 계산에서 제외됩니다.")

        if warning_messages:
            full_warning_str = "<br>".join(
                [f"<span style='color:orange;'>⚠️ {msg}</span>" for msg in warning_messages]
            )
            header_line += f"<br>{full_warning_str}"

    # 3. 보유 기간 및 고점 대비 하락률 계산
    held_tickers = [tkr for tkr, v in holdings.items() if float((v or {}).get("shares") or 0.0) > 0]
    # 보유 시작일 계산 기준은 실제 표시 기준일(label_date)과 일치시킵니다.
    consecutive_holding_info = calculate_consecutive_holding_info(held_tickers, country, label_date)
    for tkr, d in data_by_tkr.items():
        if float(d.get("shares", 0.0)) > 0:
            buy_date = consecutive_holding_info.get(tkr, {}).get("buy_date")
            # Drawdown 계산은 시계열이 있는 경우에만 수행 (코인 간소화 경로는 df가 비어있을 수 있음)
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

    app_settings = get_app_settings(country)
    if not app_settings or "portfolio_topn" not in app_settings:
        print(
            f"오류: '{country}' 국가의 최대 보유 종목 수(portfolio_topn)가 설정되지 않았습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요."
        )
        return None

    try:
        denom = int(app_settings["portfolio_topn"])
    except (ValueError, TypeError):
        print("오류: DB의 portfolio_topn 값이 올바르지 않습니다.")
        return None

    # 공통 설정에서 손절 퍼센트 로드
    common = get_common_settings()
    if not common or "HOLDING_STOP_LOSS_PCT" not in common:
        print("오류: 공통 설정에 HOLDING_STOP_LOSS_PCT 값이 없습니다.")
        return None
    try:
        stop_loss_raw = float(common["HOLDING_STOP_LOSS_PCT"])
        # Interpret positive input as a negative threshold (e.g., 10 -> -10)
        stop_loss = -abs(stop_loss_raw)
    except (ValueError, TypeError):
        print("오류: 공통 설정의 HOLDING_STOP_LOSS_PCT 값 형식이 올바르지 않습니다.")
        return None

    if denom <= 0:
        print(f"오류: '{country}' 국가의 최대 보유 종목 수(portfolio_topn)는 0보다 커야 합니다.")
        return None
    # 포지션 비중 가이드라인: 모든 국가 동일 규칙 적용
    min_pos = 1.0 / (denom * 2.0)  # 최소 편입 비중
    max_pos = 1.0 / denom  # 목표/최대 비중 # noqa: F841

    if country == "coin":
        held_count = sum(
            1
            for v in holdings.values()
            if float((v or {}).get("shares") or 0.0) > COIN_ZERO_THRESHOLD
        )
    else:
        held_count = sum(1 for v in holdings.values() if float((v or {}).get("shares") or 0.0) > 0)

    total_cash = float(current_equity) - float(total_holdings_value)

    # 4. 초기 매매 결정 생성
    decisions = []

    def _format_kr_price(p):
        return f"{int(round(p)):,}"

    def _format_kr_ma(p):
        return f"{int(round(p)):,}원"

    # 국가별 포맷터 설정
    if country == "aus":
        price_formatter = format_aud_price
        money_formatter = format_aud_money
    else:  # kor
        money_formatter = format_kr_money
        price_formatter = _format_kr_price

    def format_shares(quantity):
        if country == "coin":
            # 코인: 소수점 8자리까지 표시 (불필요한 0 제거)
            return f"{quantity:,.8f}".rstrip("0").rstrip(".")
        if country == "aus":
            # 호주: 소수점 4자리까지 표시 (불필요한 0 제거)
            return f"{quantity:,.4f}".rstrip("0").rstrip(".")
        return f"{int(quantity):,d}"

    for tkr, name in pairs:
        d = data_by_tkr.get(tkr)

        # 보유 정보는 `holdings` 딕셔너리에서 직접 가져옵니다.
        holding_info = holdings.get(tkr, {})
        sh = float(holding_info.get("shares", 0.0))
        ac = float(holding_info.get("avg_cost", 0.0))

        # 코인의 경우, 아주 작은 잔량(dust)은 보유하지 않은 것으로 간주합니다.
        is_effectively_held = (sh > COIN_ZERO_THRESHOLD) if country == "coin" else (sh > 0)

        # 데이터가 없고, 실질적으로 보유하지도 않은 종목은 건너뜁니다.
        if not d and not is_effectively_held:
            continue

        # 데이터가 없는 보유 종목을 위한 기본값 설정
        if not d:
            d = {
                "price": 0.0,
                "prev_close": 0.0,
                "s1": float("nan"),
                "s2": float("nan"),
                "score": 0.0,
                "filter": 0,
            }

        price = d.get("price", 0.0)
        score = d.get("score", 0.0)

        # 자동 계산된 보유종목의 매수일과 보유일
        buy_signal = False
        state = "HOLD" if is_effectively_held else "WAIT"
        phrase = ""
        if price == 0.0 and is_effectively_held:
            phrase = "가격 데이터 조회 실패"

        # 이 루프의 모든 경로에서 사용되므로, 여기서 초기화합니다.
        buy_date = None
        holding_days = 0
        hold_ret = None

        # 카테고리 중복 확인 및 상태 변경 (BUY 대상에서 제외)
        category = etf_meta.get(tkr, {}).get("category")
        # 실질적으로 보유하지 않은 종목(매수 후보)에 대해서만 카테고리 중복을 확인합니다.
        if (
            not is_effectively_held
            and category
            and category != "TBD"
            and category in held_categories
        ):
            state = "WAIT"  # 카테고리 중복 시 BUY 대상에서 제외하고 WAIT 상태로
            phrase = "카테고리 중복"
            buy_signal = False  # 매수 신호도 비활성화
        else:
            consecutive_info = consecutive_holding_info.get(tkr)
            buy_date = consecutive_info.get("buy_date") if consecutive_info else None

            if buy_date:
                # label_date는 naive timestamp이므로, buy_date도 naive로 만듭니다.
                if hasattr(buy_date, "tzinfo") and buy_date.tzinfo is not None:
                    buy_date = buy_date.tz_localize(None)
                buy_date = pd.to_datetime(buy_date).normalize()

            if is_effectively_held and buy_date and buy_date <= label_date:
                try:
                    # 거래일 기준으로 보유일수 계산 (캐시된 함수 사용)
                    trading_days_in_period = get_trading_days(
                        buy_date.strftime("%Y-%m-%d"),
                        label_date.strftime("%Y-%m-%d"),
                        country,
                    )
                    holding_days = len(trading_days_in_period)
                except Exception as e:
                    print(
                        f"경고: 보유일 계산 중 오류 발생 ({tkr}): {e}. 달력일 기준으로 대체합니다."
                    )
                    # 거래일 계산 실패 시, 달력일 기준으로 계산
                    holding_days = (label_date - buy_date).days + 1

            qty = 0
            # Current holding return
            hold_ret = (
                ((price / ac) - 1.0) * 100.0
                if (is_effectively_held and ac > 0 and pd.notna(price))
                else None
            )
            if is_effectively_held:
                if (
                    stop_loss is not None
                    and ac > 0
                    and hold_ret is not None
                    and hold_ret <= float(stop_loss)
                ):
                    state = "CUT_STOPLOSS"
                    qty = sh
                    prof = (price - ac) * qty if ac > 0 else 0.0
                    phrase = f"가격기반손절 {format_shares(qty)}주 @ {price_formatter(price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"

            # --- 전략별 매수/매도 로직 ---
            if state == "HOLD":  # 아직 매도 결정이 내려지지 않은 경우
                price, ma, _ = d["price"], d["s1"], d["s2"]
                if not pd.isna(price) and not pd.isna(ma) and price < ma:
                    state = "SELL_TREND"  # 결정 코드 # noqa: F841
                    qty = sh
                    prof = (price - ac) * qty if ac > 0 else 0.0
                    tag = "추세이탈(이익)" if hold_ret >= 0 else "추세이탈(손실)"  # noqa: F841
                    phrase = f"{tag} {format_shares(qty)}주 @ {price_formatter(price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"

            elif state == "WAIT":  # 아직 보유하지 않은 경우
                price, ma, _ = d["price"], d["s1"], d["s2"]
                buy_signal_days_today = d["filter"]
                if buy_signal_days_today > 0:
                    buy_signal = True
                    phrase = f"추세진입 ({buy_signal_days_today}일째)"

        amount = sh * price if pd.notna(price) else 0.0
        # 일간 수익률 계산
        prev_close = d.get("prev_close")
        day_ret = 0.0
        # 다음 거래일 화면에서는 아직 일간 수익률이 없으므로 0으로 고정
        if day_label != "다음 거래일":
            if prev_close is not None and prev_close > 0 and pd.notna(price):
                day_ret = ((price / prev_close) - 1.0) * 100.0

        buy_date_display = buy_date.strftime("%Y-%m-%d") if buy_date else "-"
        holding_days_display = str(holding_days) if holding_days > 0 else "-"

        position_weight_pct = (amount / current_equity) * 100.0 if current_equity > 0 else 0.0

        current_row = [
            0,
            tkr,
            state,
            buy_date_display,
            holding_days_display,
            price,
            day_ret,
            sh,
            amount,
            hold_ret if hold_ret is not None else 0.0,
            position_weight_pct,
            (
                f"{d.get('drawdown_from_peak'):.1f}%"
                if d.get("drawdown_from_peak") is not None
                else "-"
            ),  # 고점대비
            d.get("score"),  # raw score 값으로 변경
            f"{d['filter']}일" if d.get("filter") is not None else "-",
            phrase,
        ]
        decisions.append(
            {
                "state": state,
                "weight": position_weight_pct,
                "score": score,
                "tkr": tkr,
                "row": current_row,
                "buy_signal": buy_signal,
            }
        )

    # 5. 신규 매수 및 교체 매매 로직 적용
    # 교체 매매 관련 설정 로드 (임계값은 DB 설정 우선)
    # 국가별 전략 파라미터는 DB에서 필수 제공
    app_settings_for_country = get_app_settings(country)
    if not app_settings_for_country:
        print(
            f"오류: '{country}' 국가의 전략 파라미터가 DB에 없습니다. 웹 앱의 '설정' 탭에서 값을 저장해주세요."
        )
        return None
    # 교체 매매 사용 여부 (bool)
    if "replace_weaker_stock" not in app_settings_for_country:
        print(
            f"오류: '{country}' 국가의 설정에 'replace_weaker_stock'가 없습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요."
        )
        return None
    try:
        replace_weaker_stock = bool(app_settings_for_country["replace_weaker_stock"])
    except Exception:
        print(f"오류: '{country}' 국가의 'replace_weaker_stock' 값이 올바르지 않습니다.")
        return None
    # 하루 최대 교체 수 (int)
    if "max_replacements_per_day" not in app_settings_for_country:
        print(
            f"오류: '{country}' 국가의 설정에 'max_replacements_per_day'가 없습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요."
        )
        return None
    try:
        max_replacements_per_day = int(app_settings_for_country["max_replacements_per_day"])
    except Exception:
        print(f"오류: '{country}' 국가의 'max_replacements_per_day' 값이 올바르지 않습니다.")
        return None
    if "replace_threshold" not in app_settings_for_country:
        print(
            f"오류: '{country}' 국가의 교체 매매 임계값(replace_threshold)이 DB에 없습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요."
        )
        return None
    try:
        replace_threshold = float(app_settings_for_country["replace_threshold"])
    except (ValueError, TypeError):
        print(
            f"오류: '{country}' 국가의 교체 매매 임계값(replace_threshold) 값이 올바르지 않습니다."
        )
        return None
    slots_to_fill = denom - held_count
    if slots_to_fill > 0:
        # 현재 보유 종목의 카테고리 (TBD 제외)
        held_categories = set()
        for tkr, d in data_by_tkr.items():
            if float(d.get("shares", 0.0)) > 0:
                category = etf_meta.get(tkr, {}).get("category")
                if category and category != "TBD":
                    held_categories.add(category)

        # 매수 후보들을 점수 순으로 정렬
        buy_candidates_raw = sorted(
            [a for a in decisions if a.get("buy_signal")],
            key=lambda x: x["score"],
            reverse=True,
        )

        final_buy_candidates = []
        recommended_buy_categories = (
            set()
        )  # New set to track categories for current BUY recommendations

        for cand in buy_candidates_raw:
            category = etf_meta.get(cand["tkr"], {}).get("category")
            # First, check against already held categories (from previous fix)
            if category and category != "TBD" and category in held_categories:
                cand["state"] = "WAIT"
                cand["row"][2] = "WAIT"
                cand["row"][-1] = "카테고리 중복 (보유)" + f" ({cand['row'][-1]})"
                continue  # Skip to next candidate if category is already held

            # Then, check against categories already recommended for BUY in this cycle
            if category and category != "TBD" and category in recommended_buy_categories:
                cand["state"] = "WAIT"
                cand["row"][2] = "WAIT"
                cand["row"][-1] = "카테고리 중복 (추천)" + f" ({cand['row'][-1]})"
                continue  # Skip to next candidate if category is already recommended

            final_buy_candidates.append(cand)
            if category and category != "TBD":
                recommended_buy_categories.add(category)

        buy_candidates = final_buy_candidates  # Use the filtered and processed candidates

        available_cash = total_cash
        buys_made = 0

        for cand in buy_candidates:
            if buys_made >= slots_to_fill:
                # 포트폴리오가 가득 찼으므로 더 이상 매수 불가
                cand["row"][-1] = "포트폴리오 가득 참" + f" ({cand['row'][-1]})"
                continue

            d = data_by_tkr.get(cand["tkr"])
            price = d["price"]

            if price > 0:
                equity = current_equity
                min_val = min_pos * equity
                max_val = max_pos * equity
                budget = min(max_val, available_cash)

                req_qty = 0
                buy_notional = 0.0
                if budget >= min_val and budget > 0:
                    if country in ("coin", "aus"):
                        req_qty = budget / price
                        buy_notional = budget
                    else:
                        # 정수 수량 시장: 예산 내에서 살 수 있는 최대 정수 수량으로, 최소 비중 충족해야 함
                        req_qty = int(budget // price)
                        buy_notional = req_qty * price
                        if req_qty <= 0 or buy_notional + 1e-9 < min_val:
                            req_qty = 0
                            buy_notional = 0.0

                if req_qty > 0 and buy_notional <= available_cash + 1e-9:
                    # 매수 결정
                    cand["state"] = "BUY"
                    cand["row"][2] = "BUY"
                    buy_phrase = f"🚀 매수 {format_shares(req_qty)}주 @ {price_formatter(price)} ({money_formatter(buy_notional)})"
                    original_phrase = cand["row"][-1]
                    cand["row"][-1] = f"{buy_phrase} ({original_phrase})"

                    available_cash -= buy_notional
                    buys_made += 1
                else:
                    cand["row"][-1] = "현금 부족" + f" ({cand['row'][-1]})"
            else:
                cand["row"][-1] = "가격 정보 없음" + f" ({cand['row'][-1]})"

    # --- 교체 매매 로직 (포트폴리오가 가득 찼을 경우) ---
    if held_count >= denom and replace_weaker_stock:
        buy_candidates = sorted(
            [a for a in decisions if a["buy_signal"]],
            key=lambda x: x["score"],
            reverse=True,
        )
        held_stocks = sorted(
            [a for a in decisions if a["state"] == "HOLD"],
            key=lambda x: x["score"],
        )

        num_possible_replacements = min(
            len(buy_candidates), len(held_stocks), max_replacements_per_day
        )

        for k in range(num_possible_replacements):
            best_new = buy_candidates[k]
            weakest_held = held_stocks[k]

            # 교체 조건: 새 후보의 점수가 기존 보유 종목보다 임계값 이상 높을 때
            if best_new["score"] > weakest_held["score"] + replace_threshold:
                # 1. 교체될 종목(매도)의 상태 업데이트
                d_weakest = data_by_tkr.get(weakest_held["tkr"])
                sell_price = float(d_weakest.get("price", 0))
                sell_qty = float(d_weakest.get("shares", 0))
                avg_cost = float(d_weakest.get("avg_cost", 0))

                hold_ret = 0.0
                prof = 0.0
                if avg_cost > 0 and sell_price > 0:
                    hold_ret = ((sell_price / avg_cost) - 1.0) * 100.0
                    prof = (sell_price - avg_cost) * sell_qty

                sell_phrase = f"교체매도 {format_shares(sell_qty)}주 @ {price_formatter(sell_price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'} ({best_new['tkr']}(으)로 교체)"

                weakest_held["state"] = "SELL_REPLACE"
                weakest_held["row"][2] = "SELL_REPLACE"
                weakest_held["row"][-1] = sell_phrase

                # 2. 새로 편입될 종목(매수)의 상태 업데이트
                best_new["state"] = "BUY_REPLACE"
                best_new["row"][2] = "BUY_REPLACE"

                # 매수 수량 및 금액 계산
                sell_value = weakest_held["weight"] / 100.0 * current_equity
                buy_price = float(data_by_tkr.get(best_new["tkr"], {}).get("price", 0))
                if buy_price > 0:
                    if country in ("coin", "aus"):
                        buy_qty = sell_value / buy_price
                    else:
                        # 매도 금액으로 살 수 있는 최대 수량
                        buy_qty = int(sell_value // buy_price)
                    buy_notional = buy_qty * buy_price
                    best_new["row"][
                        -1
                    ] = f"매수 {format_shares(buy_qty)}주 @ {price_formatter(buy_price)} ({money_formatter(buy_notional)}) ({weakest_held['tkr']} 대체)"
                else:
                    best_new["row"][-1] = f"{weakest_held['tkr']}(을)를 대체 (가격정보 없음)"
            else:
                # 점수가 정렬되어 있으므로, 더 이상의 교체는 불가능합니다.
                break

        # 3. 교체되지 않은 나머지 매수 후보들의 상태 업데이트
        for cand in buy_candidates:
            if cand["state"] == "WAIT":  # 아직 매수/교체매수 결정이 안된 경우
                cand["row"][-1] = "포트폴리오 가득 참 (교체대상 아님)"

    # 6. 완료된 거래 표시
    # 기준일에 발생한 거래를 가져와서, 추천에 따라 실행되었는지 확인하는 데 사용합니다.
    # 표시 기준일 기준으로 '완료' 거래를 표시합니다. 다음 거래일이면 거래가 없을 확률이 높음
    trades_on_base_date = get_trades_on_date(country, label_date)
    executed_buys_today = {
        trade["ticker"] for trade in trades_on_base_date if trade["action"] == "BUY"
    }
    sell_trades_today = {}
    for trade in trades_on_base_date:
        if trade["action"] == "SELL":
            tkr = trade["ticker"]
            if tkr not in sell_trades_today:
                sell_trades_today[tkr] = []
            sell_trades_today[tkr].append(trade)

    # 기준일에 실행된 거래가 있다면, 현황 목록에 '완료' 상태를 표시합니다.
    for decision in decisions:
        tkr = decision["tkr"]

        # 오늘 매수했고, 현재 보유 중인 종목
        if tkr in executed_buys_today:
            # 이 종목이 오늘 신규 매수되었음을 표시
            decision["row"][-1] = "✅ 신규 매수"

        # 오늘 매도된 종목 처리
        if tkr in sell_trades_today:
            d = data_by_tkr.get(tkr)
            remaining_shares = float(d.get("shares", 0.0)) if d else 0.0

            # 코인의 경우, 아주 작은 잔량은 0으로 간주합니다.
            is_fully_sold = (
                remaining_shares <= COIN_ZERO_THRESHOLD
                if country == "coin"
                else remaining_shares <= 0
            )

            if not is_fully_sold:
                # 부분 매도: 상태는 HOLD로 유지하고, 문구에만 정보를 추가합니다.
                decision["state"] = "HOLD"
                decision["row"][2] = "HOLD"

                total_sold_shares = sum(trade.get("shares", 0) for trade in sell_trades_today[tkr])

                sell_phrase = f"⚠️ 부분 매도 ({format_shares(total_sold_shares)}주)"

                # 기존 문구와 합칩니다.
                original_phrase = decision["row"][-1]
                # 'HOLD'나 'WAIT' 같은 기본 상태 문구는 덮어씁니다.
                if original_phrase and original_phrase not in ["HOLD", "WAIT", ""]:
                    decision["row"][-1] = f"{sell_phrase}, {original_phrase}"
                else:
                    decision["row"][-1] = sell_phrase
            else:
                # 전체 매도: 상태를 SOLD로 변경합니다.
                decision["state"] = "SOLD"
                decision["row"][2] = "SOLD"
                decision["row"][-1] = "🔚 매도 완료"

    # --- WAIT 종목 수 제한 ---
    # 웹 UI와 슬랙 알림에 표시될 대기(WAIT) 종목의 수를 최대 MAX_WAIT_ITEMS 개로 제한합니다.
    # 점수가 높은 순서대로 상위 MAX_WAIT_ITEMS 개만 남깁니다.
    wait_decisions = [d for d in decisions if d["state"] == "WAIT"]
    other_decisions = [d for d in decisions if d["state"] != "WAIT"]

    MAX_WAIT_ITEMS = 100
    if len(wait_decisions) > MAX_WAIT_ITEMS:
        # 점수(score)가 높은 순으로 정렬합니다. 점수가 없는 경우 0으로 처리합니다.
        wait_decisions_sorted = sorted(
            wait_decisions, key=lambda x: x.get("score", 0.0) or 0.0, reverse=True
        )
        decisions = other_decisions + wait_decisions_sorted[:MAX_WAIT_ITEMS]

    # 7. 최종 정렬
    def sort_key(decision_dict):
        state = decision_dict["state"]
        score = decision_dict["score"]
        tkr = decision_dict["tkr"]

        state_order = {
            "HOLD": 0,
            "CUT_STOPLOSS": 1,
            "SELL_MOMENTUM": 2,
            "SELL_TREND": 3,
            "SELL_REPLACE": 4,
            "SOLD": 5,
            "BUY_REPLACE": 6,
            "BUY": 7,
            "WAIT": 8,
        }
        order = state_order.get(state, 99)

        # 보유/매수/대기 종목 모두 점수가 높은 순으로 정렬
        sort_value = -score
        return (order, sort_value, tkr)

    decisions.sort(key=sort_key)

    rows_sorted = []
    for i, decision_dict in enumerate(decisions, 1):
        row = decision_dict["row"]
        row[0] = i
        rows_sorted.append(row)

    # 호주 시장의 경우, international_shares 정보를 테이블의 최상단에 추가합니다.
    international_shares_data = None
    if country == "aus":
        international_shares_data = portfolio_data.get("international_shares")

    if country == "aus" and international_shares_data:
        is_value = international_shares_data.get("value", 0.0)
        is_change_pct = international_shares_data.get("change_pct", 0.0)
        is_weight_pct = (is_value / current_equity) * 100.0 if current_equity > 0 else 0.0

        special_row = [
            0,  # #
            "IS",  # 티커
            "HOLD",  # 상태
            "-",  # 매수일
            "-",  # 보유
            is_value,  # 현재가
            0.0,  # 일간수익률
            "1",  # 보유수량
            is_value,  # 금액
            is_change_pct,  # 누적수익률
            is_weight_pct,  # 비중
            "-",  # 고점대비
            "-",  # 점수
            "-",  # 지속
            "International Shares",  # 문구
        ]

        rows_sorted.insert(0, special_row)
        # 행을 추가했으므로, 순번을 다시 매깁니다.
        for i, row in enumerate(rows_sorted, 1):
            row[0] = i

    # 8. 최종 결과 반환
    headers = [
        "#",
        "티커",
        "상태",
        "매수일자",
        "보유일",
        "현재가",
        "일간수익률",
        "보유수량",
        "금액",
        "누적수익률",
        "비중",
    ]
    headers.extend(["고점대비", "점수", "지속", "문구"])

    state_counts: Dict[str, int] = {}
    for row in rows_sorted:
        state = row[2]
        state_counts[state] = state_counts.get(state, 0) + 1

    logger.info(
        "[%s] status report ready: rows=%d state_counts=%s",
        country.upper(),
        len(rows_sorted),
        state_counts,
    )

    return (header_line, headers, rows_sorted, base_date)


def main(country: str = "kor", date_str: Optional[str] = None) -> Optional[datetime]:
    """CLI에서 오늘의 현황을 실행하고 결과를 출력/저장합니다."""
    result = generate_status_report(country, date_str, notify_start=True)

    if result:
        header_line, headers, rows_sorted, report_base_date = result
        # Persist status report for use in web app history, if possible.
        try:
            # Use the returned base_date for saving, which is the true date of the report
            save_status_report_to_db(
                country,
                report_base_date.to_pydatetime(),
                (header_line, headers, rows_sorted),
            )
        except Exception:
            pass

        # 슬랙 알림: 현황 전송
        try:
            _maybe_notify_detailed_status(country, header_line, headers, rows_sorted)
        except Exception:
            pass

        # print(rows_sorted)
        return report_base_date.to_pydatetime()

        # --- 콘솔 출력용 포맷팅 ---
        # 웹앱은 raw data (rows_sorted)를 사용하고, 콘솔은 포맷된 데이터를 사용합니다.

        # 컬럼 인덱스 찾기
        col_indices = {}
        try:
            score_header_candidates = ["점수", "모멘텀점수", "MA스코어"]
            for h in score_header_candidates:
                if h in headers:
                    col_indices["score"] = headers.index(h)
                    break
            col_indices["day_ret"] = headers.index("일간수익률")
            col_indices["cum_ret"] = headers.index("누적수익률")
            col_indices["weight"] = headers.index("비중")
            col_indices["shares"] = headers.index("보유수량")
        except (ValueError, KeyError):
            pass  # 일부 컬럼을 못찾아도 괜찮음

        display_rows = []
        for row in rows_sorted:
            display_row = list(row)  # 복사

            # 점수 포맷팅
            idx = col_indices.get("score")
            if idx is not None:
                val = display_row[idx]
                if isinstance(val, (int, float)):
                    display_row[idx] = f"{val:+.2f}"
                else:
                    display_row[idx] = "-"

            # 일간수익률 포맷팅
            idx = col_indices.get("day_ret")
            if idx is not None:
                val = display_row[idx]
                display_row[idx] = f"{val:+.1f}%" if isinstance(val, (int, float)) else "-"

            # 누적수익률 포맷팅
            idx = col_indices.get("cum_ret")
            if idx is not None:
                val = display_row[idx]
                if isinstance(val, (int, float)):
                    # International Shares는 소수점 2자리
                    fmt = "{:+.2f}%" if row[1] == "IS" else "{:+.1f}%"
                    display_row[idx] = fmt.format(val)
                else:
                    display_row[idx] = "-"

            # 비중 포맷팅
            idx = col_indices.get("weight")
            if idx is not None:
                val = display_row[idx]
                display_row[idx] = f"{val:.0f}%" if isinstance(val, (int, float)) else "-"

            # 보유수량 포맷팅 (코인은 소수점 8자리)
            idx = col_indices.get("shares")
            if idx is not None:
                val = display_row[idx]
                if isinstance(val, (int, float)):
                    if country == "coin":
                        s = f"{float(val):.8f}".rstrip("0").rstrip(".")
                        display_row[idx] = s if s != "" else "0"
                    else:
                        display_row[idx] = f"{int(round(val)):,d}"
                else:
                    display_row[idx] = val

            display_rows.append(display_row)

        aligns = [
            "right",  # #
            "right",  # 티커
            "center",  # 상태
            "left",  # 매수일
            "right",  # 보유
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

        print("\n" + header_line)


def _is_trading_day(country: str) -> bool:
    """Return True if today is a trading day for the given country.

    - kor/aus: use trading calendar (fallback: Mon-Fri)
    - coin: always True
    """
    if country == "coin":
        return True
    try:
        # Localize 'today' to country timezone for accuracy
        if pytz:
            tz = pytz.timezone("Asia/Seoul" if country == "kor" else "Australia/Sydney")
            today_local = datetime.now(tz).date()
        else:
            today_local = datetime.now().date()
        start = end = pd.Timestamp(today_local).strftime("%Y-%m-%d")
        days = get_trading_days(start, end, country)
        return any(pd.Timestamp(d).date() == today_local for d in days)
    except Exception:
        # Fallback: Mon-Fri
        if pytz:
            tz = pytz.timezone("Asia/Seoul" if country == "kor" else "Australia/Sydney")
            wd = datetime.now(tz).weekday()
        else:
            wd = datetime.now().weekday()
        return wd < 5


def _maybe_notify_detailed_status(
    country: str,
    header_line: str,
    headers: list,
    rows_sorted: list,
    force: bool = False,
) -> bool:
    """국가별 설정에 따라 슬랙으로 상세 현황 알림을 전송합니다."""
    try:
        from utils.report import format_aud_money, format_aud_price, format_kr_money
        from utils.notify import get_slack_webhook_url, send_slack_message
    except Exception:
        return False

    if not force and not _is_trading_day(country):
        return False

    try:
        # 국가별 포맷터 설정
        if country == "aus":
            price_formatter = format_aud_price
            money_formatter = format_aud_money
        else:  # kor, coin
            money_formatter = format_kr_money

            def price_formatter(p):
                return f"{int(round(p)):,}" if isinstance(p, (int, float)) else str(p)

        def format_shares(quantity):
            if not isinstance(quantity, (int, float)):
                return str(quantity)
            if country == "coin":
                return f"{quantity:,.8f}".rstrip("0").rstrip(".")
            if country == "aus":
                return f"{quantity:.4f}".rstrip("0").rstrip(".")
            return f"{int(quantity):,d}"

        # 상세 알림에서는 시작 알림에서 보낸 경고(데이터 부족 등)를 제외합니다.
        # header_line은 HTML <br> 태그로 경고와 구분됩니다.
        header_line_clean = header_line.split("<br>")[0]

        def _strip_html(s: str) -> str:
            try:
                return re.sub(r"<[^>]+>", "", s)
            except Exception:
                return s

        # --- Parse header_line for caption ---
        # Date
        first_seg = header_line_clean.split("|")[0].strip()
        date_part = first_seg.split(":", 1)[1].strip()
        if "[" in date_part:
            date_part = date_part.split("[")[0].strip()
        date_part = _strip_html(date_part)

        # Holdings count
        hold_seg = next(
            (seg for seg in header_line_clean.split("|") if "보유종목:" in seg),
            "보유종목: -",
        )
        hold_text = _strip_html(hold_seg.split(":", 1)[1].strip())

        # Holdings value
        hold_val_seg = next(
            (seg for seg in header_line_clean.split("|") if "보유금액:" in seg),
            "보유금액: 0",
        )
        hold_val_text = _strip_html(hold_val_seg.split(":", 1)[1].strip())

        # Cash value
        cash_seg = next((seg for seg in header_line_clean.split("|") if "현금:" in seg), "현금: 0")
        cash_text = _strip_html(cash_seg.split(":", 1)[1].strip())

        # Cumulative return
        cum_seg = next(
            (seg for seg in header_line_clean.split("|") if "누적:" in seg),
            "누적: +0.00%(0원)",
        )
        cum_text = _strip_html(cum_seg.split(":", 1)[1].strip())

        # Total equity value
        equity_seg = next(
            (seg for seg in header_line_clean.split("|") if "평가금액:" in seg),
            "평가금액: 0",
        )
        equity_text = _strip_html(equity_seg.split(":", 1)[1].strip())

        # Columns
        idx_ticker = headers.index("티커")
        idx_state = headers.index("상태") if "상태" in headers else None
        idx_price = headers.index("현재가") if "현재가" in headers else None
        idx_shares = headers.index("보유수량") if "보유수량" in headers else None
        idx_amount = headers.index("금액") if "금액" in headers else None
        idx_ret = (
            headers.index("누적수익률")
            if "누적수익률" in headers
            else (headers.index("일간수익률") if "일간수익률" in headers else None)
        )
        idx_score = headers.index("점수") if "점수" in headers else None

        # Names map
        name_map = {}
        try:
            # Use the country parameter to get the correct etfs
            etfs = get_etfs(country) or []
            name_map = {str(s.get("ticker") or "").upper(): str(s.get("name") or "") for s in etfs}
        except Exception:
            pass

        # 호주 'IS' 종목의 이름을 수동으로 지정합니다.
        if country == "aus":
            name_map["IS"] = "International Shares"

        # 1. 데이터를 사전 처리하여 표시할 부분을 만들고 최대 너비를 찾습니다.
        display_parts_list = []
        max_len_name = 0
        max_len_price_col = 0
        max_len_shares_col = 0
        max_len_amount_col = 0
        max_len_return_col = 0
        max_len_score_col = 0

        for row in rows_sorted:
            try:
                num_part = f"[{row[0]}]"
                tkr = str(row[idx_ticker])
                name = name_map.get(tkr.upper(), "")

                # 'IS' 종목은 티커 없이 이름만 표시합니다.
                if country == "aus" and tkr.upper() == "IS":
                    name_part = name
                else:
                    name_part = f"{name}({tkr})" if name else tkr
                full_name_part = f"{num_part} {name_part}"

                stt = (
                    str(row[idx_state]) if (idx_state is not None and idx_state < len(row)) else ""
                )

                price_col = ""
                if idx_price is not None:
                    p = row[idx_price]
                    if isinstance(p, (int, float)):
                        price_col = f"@{price_formatter(p)}"

                shares_col = ""
                if idx_shares is not None:
                    s = row[idx_shares]
                    # 보유한 경우에만 표시
                    if isinstance(s, (int, float)) and s > 1e-9:
                        shares_col = f"{format_shares(s)}주"

                amount_col = ""
                if idx_amount is not None:
                    a = row[idx_amount]
                    if isinstance(a, (int, float)) and a > 1e-9:
                        amount_col = f"{money_formatter(a)}"

                return_col = ""
                if idx_ret is not None:
                    r = row[idx_ret]
                    if isinstance(r, (int, float)) and abs(r) > 0.001:
                        return_col = f"수익 {r:+.2f}%,"

                score_col = ""
                if idx_score is not None:
                    sc = row[idx_score]
                    if isinstance(sc, (int, float)):
                        score_col = f"점수 {float(sc):.2f}"

                parts = {
                    "name": full_name_part,
                    "status": stt,
                    "price_col": price_col,
                    "shares_col": shares_col,
                    "amount_col": amount_col,
                    "return_col": return_col,
                    "score_col": score_col,
                }
                display_parts_list.append(parts)

                max_len_name = max(max_len_name, len(full_name_part))
                max_len_price_col = max(max_len_price_col, len(price_col))
                max_len_shares_col = max(max_len_shares_col, len(shares_col))
                max_len_amount_col = max(max_len_amount_col, len(amount_col))
                max_len_return_col = max(max_len_return_col, len(return_col))
                max_len_score_col = max(max_len_score_col, len(score_col))

            except Exception:
                continue

        # 2. 상태별로 그룹화합니다.
        grouped_parts = {}
        for parts in display_parts_list:
            status = parts["status"]
            if status not in grouped_parts:
                grouped_parts[status] = []
            grouped_parts[status].append(parts)

        # 3. 그룹 헤더와 함께 정렬된 라인을 만듭니다.
        body_lines = []
        # 정렬 순서는 DECISION_CONFIG의 'order' 값을 기준으로 합니다.
        sorted_groups = sorted(
            grouped_parts.items(),
            key=lambda item: DECISION_CONFIG.get(item[0], {"order": 99}).get("order", 99),
        )

        for group_name, parts_in_group in sorted_groups:
            config = DECISION_CONFIG.get(group_name)
            if not config:
                # 설정에 없는 상태(예: SELL_MOMENTUM)에 대한 폴백 처리
                display_name = f"<{group_name}>"
                show_return = group_name == "HOLD"
            else:
                display_name = config["display_name"]
                show_return = config["show_return"]

            if parts_in_group:
                body_lines.append(display_name)
                for parts in parts_in_group:
                    name_part = parts["name"].ljust(max_len_name)
                    price_part = parts["price_col"].ljust(max_len_price_col)
                    shares_part = parts["shares_col"].rjust(max_len_shares_col)
                    amount_part = parts["amount_col"].rjust(max_len_amount_col)
                    score_part = parts["score_col"].ljust(max_len_score_col)

                    if show_return:
                        return_part = parts["return_col"].ljust(max_len_return_col)
                        line = f"{name_part}  {price_part} {shares_part} {amount_part}  {return_part} {score_part}"
                    else:
                        return_part = "".ljust(max_len_return_col)
                        line = f"{name_part}  {price_part} {shares_part} {amount_part}  {return_part} {score_part}"

                    body_lines.append(line.rstrip())
                body_lines.append("")  # 그룹 사이에 빈 줄 추가

        if body_lines and body_lines[-1] == "":
            body_lines.pop()

        # --- Build caption for message ---
        country_kor = {"kor": "한국", "aus": "호주", "coin": "코인"}.get(country, country.upper())

        app_type = os.environ.get("APP_TYPE", "SERVER")
        title_line = f"[{app_type}][{country_kor}] 상세내역"
        equity_line = f"평가금액: {equity_text}, 누적수익 {cum_text}"
        cash_line = f"현금: {cash_text}, 보유금액: {hold_val_text}"
        hold_line = f"보유종목: {hold_text}"
        caption = "\n".join([title_line, equity_line, cash_line, hold_line])

        # --- Send notifications ---
        webhook_url = get_slack_webhook_url(country)
        if not webhook_url:
            return False

        # DECISION_CONFIG에서 is_recommendation=True인 그룹이 하나라도 있으면 @channel 멘션을 포함합니다.
        has_recommendation = False
        for group_name in grouped_parts.keys():
            config = DECISION_CONFIG.get(group_name)
            if config and config.get("is_recommendation", False):
                has_recommendation = True
                break
        slack_mention = "<!channel>\n" if has_recommendation else ""

        if not body_lines:
            # No items to report, just send caption
            slack_sent = send_slack_message(slack_mention + caption, webhook_url=webhook_url)
        else:
            # For Slack, use ``` for code blocks
            slack_message = caption + "\n\n" + "```\n" + "\n".join(body_lines) + "\n```"
            slack_sent = send_slack_message(slack_mention + slack_message, webhook_url=webhook_url)

        return slack_sent
    except Exception:
        return False


if __name__ == "__main__":
    main(country="kor")
