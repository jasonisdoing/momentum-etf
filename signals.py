import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import requests

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import pandas as pd
from pymongo import DESCENDING

import argparse
import settings as global_settings

try:
    import pytz
except ImportError:
    pytz = None

from utils.data_loader import (
    fetch_ohlcv,
    get_trading_days,
    PykrxDataUnavailable,
)

# 신규 구조 모듈 임포트를 정리합니다.
from utils.db_manager import (
    get_db_connection,
    get_portfolio_snapshot,
    get_previous_portfolio_snapshot,
    get_trades_on_date,
    save_signal_report_to_db,
)
from utils.report import (
    format_kr_money,
    render_table_eaw,
)
from utils.stock_list_io import get_etfs
from utils.account_registry import get_account_file_settings, get_common_file_settings
from utils.notify import send_log_to_slack

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
# - show_slack: True이면 슬랙 알림에 해당 그룹을 포함
DECISION_CONFIG = {
    # 보유  (알림 없음)
    "HOLD": {
        "display_name": "<💼 보유>",
        "order": 1,
        "is_recommendation": False,
        "show_slack": True,
    },
    # 매도 추천 (알림 발생)
    "CUT_STOPLOSS": {
        "display_name": "<🚨 손절매도>",
        "order": 10,
        "is_recommendation": True,
        "show_slack": True,
    },
    "SELL_TREND": {
        "display_name": "<📉 추세이탈 매도>",
        "order": 11,
        "is_recommendation": True,
        "show_slack": True,
    },
    "SELL_REPLACE": {
        "display_name": "<🔄 교체매도>",
        "order": 12,
        "is_recommendation": True,
        "show_slack": True,
    },
    "SELL_REBALANCE": {
        "display_name": "<⚖️ 리밸런스 매도>",
        "order": 13,
        "is_recommendation": True,
        "show_slack": True,
    },
    "SELL_INACTIVE": {
        "display_name": "<🗑️ 비활성 매도>",
        "order": 14,
        "is_recommendation": True,
        "show_slack": True,
    },
    "SELL_REGIME_FILTER": {
        "display_name": "<🛡️ 시장위험회피 매도>",
        "order": 15,
        "is_recommendation": True,
        "show_slack": True,
    },
    # 매수 추천 (알림 발생)
    "BUY_REPLACE": {
        "display_name": "<🔄 교체매수>",
        "order": 20,
        "is_recommendation": True,
        "show_slack": True,
    },
    "BUY": {
        "display_name": "<🚀 신규매수>",
        "order": 21,
        "is_recommendation": True,
        "show_slack": True,
    },
    # 거래 완료 (알림 없음)
    "SOLD": {
        "display_name": "<✅ 매도 완료>",
        "order": 40,
        "is_recommendation": False,
        "show_slack": True,
    },
    # 보유 및 대기 (알림 없음)
    "WAIT": {
        "display_name": "<⏳ 대기>",
        "order": 50,
        "is_recommendation": False,
        "show_slack": False,
    },
}

# 코인 보유 수량에서 0으로 간주할 임계값 (거래소의 dust 처리)
COIN_ZERO_THRESHOLD = 1e-9


def _fetch_bithumb_realtime_price(symbol: str) -> Optional[float]:
    symbol = (symbol or "").upper()
    if not symbol or symbol in {"KRW", "P"}:
        return 1.0
    url = f"https://api.bithumb.com/public/ticker/{symbol}_KRW"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and data.get("status") == "0000":
            closing_price = data.get("data", {}).get("closing_price")
            if closing_price is not None:
                return float(str(closing_price).replace(",", ""))
    except Exception:
        return None
    return None


_SIGNAL_LOGGER = None


def _resolve_previous_close(close_series: pd.Series, base_date: pd.Timestamp) -> float:
    """기준일 이전에 존재하는 가장 최근 종가를 반환합니다. (없으면 0.0)"""
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


def get_signal_logger() -> logging.Logger:
    """로그 파일(콘솔 출력 없이)에 기록하는 signal 전용 로거를 반환합니다."""
    global _SIGNAL_LOGGER
    if _SIGNAL_LOGGER:
        return _SIGNAL_LOGGER

    logger = logging.getLogger("signal.detail")
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

    _SIGNAL_LOGGER = logger
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
                return pd.Timestamp(d).normalize()  # noqa: E722
    except Exception:
        pass
    # 폴백: 토/일이면 다음 월요일, 평일이면 그대로
    wd = start_date.weekday()
    delta = 0 if wd < 5 else (7 - wd)
    return (start_date + pd.Timedelta(days=delta)).normalize()


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


def get_market_regime_status_string() -> Optional[str]:
    """
    S&P 500 지수를 기준으로 현재 시장 레짐 상태를 계산하여 HTML 문자열로 반환합니다.
    """
    # 공통 설정 로드 (파일)
    try:
        common = get_common_file_settings()
        regime_filter_enabled = common["MARKET_REGIME_FILTER_ENABLED"]
        if not regime_filter_enabled:
            return '<span style="color:grey">시장 상태: 비활성화</span>'
        regime_ticker = str(common["MARKET_REGIME_FILTER_TICKER"])
        regime_ma_period = int(common["MARKET_REGIME_FILTER_MA_PERIOD"])
    except (SystemExit, KeyError, ValueError, TypeError) as e:
        print(f"오류: 공통 설정을 불러오는 중 문제가 발생했습니다: {e}")
        return '<span style="color:grey">시장 상태: 설정 파일 오류</span>'

    # 데이터 로딩에 필요한 기간 계산: 레짐 MA 기간을 만족하도록 동적으로 산정
    # 거래일 기준 대략 22일/월 가정 + 여유 버퍼
    required_days = int(regime_ma_period) + 30
    required_months = max(3, (required_days // 22) + 2)

    # 데이터 조회
    df_regime = fetch_ohlcv(
        regime_ticker,
        country="kor",
        months_range=[required_months, 0],  # 지수 조회에서는 country 인자가 의미 없습니다.
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
                end_date = is_risk_off_series.index[is_risk_off_series.index.get_loc(dt) - 1]
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


def _normalize_yfinance_df(df_y: pd.DataFrame) -> Optional[pd.DataFrame]:
    """yfinance 다운로드 결과를 표준 형식으로 정규화합니다."""
    if df_y is None or df_y.empty:
        return None

    # 기대한 형태가 되도록 컬럼과 인덱스를 정규화합니다.
    if isinstance(df_y.columns, pd.MultiIndex):
        df_y.columns = df_y.columns.get_level_values(0)
        df_y = df_y.loc[:, ~df_y.columns.duplicated()]
    if df_y.index.tz is not None:
        df_y.index = df_y.index.tz_localize(None)

    # yfinance는 'Adj Close'를 반환하지만, 우리 시스템은 'Close'를 기대합니다.
    # 'Adj Close'가 있으면 'Close'로 이름을 바꾸고, 없으면 'Close'를 그대로 사용합니다.
    if "Adj Close" in df_y.columns:
        df_y = df_y.rename(columns={"Adj Close": "Close"})
    elif "Close" not in df_y.columns:
        # 'Close' 또는 'Adj Close'가 모두 없는 비정상적인 경우
        return None

    return df_y


def _determine_benchmark_country(ticker: str) -> str:
    """벤치마크 티커를 기반으로 국가 코드를 추론합니다."""
    if ticker.isdigit() and len(ticker) == 6:
        return "kor"
    if ".AX" in ticker.upper():
        return "aus"
    # 암호화폐 티커로 추정 (예: BTC, ETH)
    if len(ticker) <= 5 and ticker.isalpha() and ticker.isupper():
        return "coin"
    # 기본값으로 한국 시장을 가정 (S&P500 지수 등)
    return "kor"


def _calculate_single_benchmark(
    benchmark_ticker: str,
    benchmark_name: str,
    benchmark_country: str,
    initial_date: pd.Timestamp,
    base_date: pd.Timestamp,
) -> Dict[str, Any]:
    """단일 벤치마크의 성과를 계산하여 딕셔너리로 반환합니다."""
    base_result = {"ticker": benchmark_ticker, "name": benchmark_name}

    from utils.data_loader import PykrxDataUnavailable, fetch_ohlcv, get_trading_days

    # 방어 코드: base_date가 initial_date보다 이전일 수 없음
    if base_date < initial_date:
        error_msg = f"조회 종료일({base_date.strftime('%Y-%m-%d')})이 시작일({initial_date.strftime('%Y-%m-%d')})보다 빠릅니다."
        base_result["error"] = error_msg
        return base_result

    try:
        df_benchmark = fetch_ohlcv(
            benchmark_ticker,
            country=benchmark_country,
            date_range=[
                initial_date.strftime("%Y-%m-%d"),
                base_date.strftime("%Y-%m-%d"),
            ],
        )
    except PykrxDataUnavailable:
        # 데이터 조회 실패 시, 이전 거래일로 재시도합니다.
        prev_day_search_end = base_date - pd.Timedelta(days=1)
        prev_day_search_start = prev_day_search_end - pd.Timedelta(days=14)

        previous_trading_days = get_trading_days(
            prev_day_search_start.strftime("%Y-%m-%d"),
            prev_day_search_end.strftime("%Y-%m-%d"),
            benchmark_country,
        )

        if previous_trading_days:
            previous_trading_day = previous_trading_days[-1]
            print(
                f"경고: {base_date.date()} 벤치마크 데이터 조회 실패. 이전 거래일({previous_trading_day.date()})로 재시도합니다."
            )
            df_benchmark = fetch_ohlcv(
                benchmark_ticker,
                country=benchmark_country,
                date_range=[
                    initial_date.strftime("%Y-%m-%d"),
                    previous_trading_day.strftime("%Y-%m-%d"),
                ],
            )
        else:
            df_benchmark = None

    # 주요 데이터 소스에서 벤치마크를 가져오지 못했을 때의 폴백 경로입니다.
    if df_benchmark is None or df_benchmark.empty:
        # 코인 지수는 yfinance 심볼(예: BTC -> BTC-USD)로 재시도합니다.
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
                    auto_adjust=True,
                )
                df_benchmark = _normalize_yfinance_df(df_y)
            except Exception:
                pass

    if df_benchmark is None or df_benchmark.empty:
        base_result["error"] = "데이터 조회 실패"
        return base_result

    start_prices = df_benchmark[df_benchmark.index >= initial_date]["Close"]
    if start_prices.empty:
        base_result["error"] = "시작 가격 조회 실패"
        return base_result
    benchmark_start_price = start_prices.iloc[0]

    end_prices = df_benchmark[df_benchmark.index <= base_date]["Close"]
    if end_prices.empty:
        base_result["error"] = "종료 가격 조회 실패"
        return base_result
    benchmark_end_price = end_prices.iloc[-1]

    if pd.isna(benchmark_start_price) or pd.isna(benchmark_end_price) or benchmark_start_price <= 0:
        base_result["error"] = "가격 정보 오류"
        return base_result

    benchmark_cum_ret_pct = ((benchmark_end_price / benchmark_start_price) - 1.0) * 100.0

    base_result["cum_ret_pct"] = benchmark_cum_ret_pct
    base_result["error"] = None
    return base_result


def calculate_benchmark_comparison(
    country: str, account: str, date_str: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    포트폴리오의 누적 수익률을 계좌에 설정된 벤치마크들과 비교하여 결과 리스트를 반환합니다.
    """
    from utils.account_registry import get_account_info

    if not account:
        return None

    # 파일에서 초기 자본/날짜 설정을 로드합니다.
    try:
        file_settings = get_account_file_settings(country, account)
        initial_capital = float(file_settings["initial_capital"])
        initial_date = pd.to_datetime(file_settings["initial_date"])
    except SystemExit as e:
        return [{"name": "벤치마크", "error": str(e)}]

    # accounts.json 파일의 정적 설정(벤치마크 목록)을 가져옵니다.
    account_info = get_account_info(account)

    if not account_info or "benchmarks_tickers" not in account_info:
        return None

    benchmarks_to_compare = account_info["benchmarks_tickers"]
    if not benchmarks_to_compare:
        return None

    if initial_capital <= 0:
        return None

    portfolio_data = get_portfolio_snapshot(country, account=account, date_str=date_str)
    if not portfolio_data:
        return None

    base_date = pd.to_datetime(portfolio_data["date"]).normalize()

    # --- 평가금액 보정 로직 추가 (generate_signal_report와 일관성 유지) ---
    current_equity = float(portfolio_data.get("total_equity", 0.0))
    recalculated_holdings_value = 0.0
    holdings = portfolio_data.get("holdings", [])
    for h in holdings:
        ticker = h.get("ticker")
        shares = h.get("shares", 0.0)
        if not ticker or not shares > 0:
            continue
        # 이 함수는 UI에서 호출되므로, 속도를 위해 간단한 조회를 사용합니다.
        df = fetch_ohlcv(ticker, country=country, months_back=1, base_date=base_date)
        price = 0.0
        if df is not None and not df.empty:
            prices_until_base = df[df.index <= base_date]["Close"]
            if not prices_until_base.empty:
                price = prices_until_base.iloc[-1]
        if price > 0:
            recalculated_holdings_value += shares * price

    # 호주 계좌의 해외 주식 가치 추가
    if country == "aus":
        intl_info = portfolio_data.get("international_shares")
        if isinstance(intl_info, dict):
            recalculated_holdings_value += float(intl_info.get("value", 0.0))

    equity_for_calc = current_equity
    # `generate_signal_report`의 보정 로직과 동일하게 적용
    if recalculated_holdings_value > 0 and (
        current_equity == 0 or recalculated_holdings_value > current_equity
    ):
        equity_for_calc = recalculated_holdings_value

    # 벤치마크 계산 기준일(base_date)이 거래일이 아닌 경우, 그 이전의 가장 가까운 거래일로 보정합니다.
    if country != "coin":
        if not _is_trading_day(country, base_date.to_pydatetime()):
            start_search = (base_date - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
            end_search = base_date.strftime("%Y-%m-%d")
            trading_days = get_trading_days(start_search, end_search, country)
            if trading_days:
                # get_trading_days는 오름차순으로 날짜를 반환하므로, 마지막 날짜가 가장 최근입니다.
                base_date = pd.to_datetime(trading_days[-1]).normalize()

    if initial_date > base_date:
        error_msg = f"초기 기준일({initial_date.strftime('%Y-%m-%d')})이 조회일({base_date.strftime('%Y-%m-%d')})보다 미래입니다."
        return [{"name": "벤치마크", "error": error_msg}]

    portfolio_cum_ret_pct = ((equity_for_calc / initial_capital) - 1.0) * 100.0

    results = []
    for bm_info in benchmarks_to_compare:
        bm_ticker = bm_info.get("ticker")
        bm_name = bm_info.get("name")
        if not bm_ticker or not bm_name:
            continue

        bm_country = _determine_benchmark_country(bm_ticker)

        bm_result = _calculate_single_benchmark(
            benchmark_ticker=bm_ticker,
            benchmark_name=bm_name,
            benchmark_country=bm_country,
            initial_date=initial_date,
            base_date=base_date,
        )
        if bm_result:
            bm_result["ticker"] = bm_ticker
            if not bm_result.get("error"):
                bm_result["excess_return_pct"] = portfolio_cum_ret_pct - bm_result["cum_ret_pct"]
            results.append(bm_result)

    return results if results else None


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
    except Exception:  # TODO: 예외 처리 로직을 세분화합니다.
        return False  # 오류 발생 시 안전하게 False 반환


def _determine_target_date_for_scheduler(country: str) -> pd.Timestamp:
    """
    스케줄러 실행 시, 현재 시간에 따라 계산 대상 날짜를 동적으로 결정합니다.
    - 코인: 항상 오늘
    - 주식/ETF: 장 마감 후 일정 시간(버퍼)까지는 당일을, 그 이후에는 다음 거래일을 계산 대상으로 함.
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
        # 데이터 지연을 고려하여 30분 버퍼를 추가합니다.
        cutoff_datetime_local = close_datetime_local + pd.Timedelta(minutes=30)

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
    held_tickers: List[str], country: str, account: str, as_of_date: datetime
) -> Dict[str, Dict]:
    """
    'trades' 컬렉션을 스캔하여 지정된 계좌의 각 티커별 연속 보유 시작일을 계산합니다.
    N+1 DB 조회를 피하기 위해 모든 종목의 거래를 한 번에 가져옵니다.
    """
    holding_info = {tkr: {"buy_date": None} for tkr in held_tickers}
    if not held_tickers:
        return holding_info

    db = get_db_connection()
    if db is None:
        print("-> 경고: DB에 연결할 수 없어 보유일 계산을 건너뜁니다.")
        return holding_info

    if not account:
        raise ValueError("account is required for calculating holding info")

    # 코인은 트레이드가 시각 포함으로 기록되므로, 동일 달력일의 모든 거래를 포함하도록
    # as_of_date 상한을 해당일 23:59:59.999999로 확장합니다.
    # 모든 국가에 대해 동일하게 적용하여, 특정 날짜의 모든 거래를 포함하도록 합니다.
    include_until = as_of_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    # 1. 모든 보유 종목의 거래 내역을 한 번의 쿼리로 가져옵니다.
    query = {
        "country": country,
        "account": account,
        "ticker": {"$in": held_tickers},
        "date": {"$lte": include_until},
    }
    all_trades = list(
        db.trades.find(
            query,
            sort=[("date", DESCENDING), ("_id", DESCENDING)],
        )
    )

    # 2. 거래 내역을 티커별로 그룹화합니다.
    from collections import defaultdict

    trades_by_ticker = defaultdict(list)
    for trade in all_trades:
        trades_by_ticker[trade["ticker"]].append(trade)

    # 3. 각 티커별로 연속 보유 시작일을 계산합니다.
    for tkr in held_tickers:
        trades = trades_by_ticker.get(tkr)
        if not trades:
            continue

        try:
            # 현재 보유 수량을 계산합니다. (모든 거래의 합)
            current_shares = sum(
                t["shares"] if t["action"] == "BUY" else -t["shares"] for t in trades
            )

            # 현재부터 과거로 시간을 거슬러 올라가며 확인합니다. (trades는 날짜 내림차순으로 정렬되어 있음)
            buy_date = None
            for trade in trades:
                if current_shares <= COIN_ZERO_THRESHOLD:
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
    # 전달받은 인자를 변수로 풀어냅니다.
    (
        tkr,
        country,
        required_months,
        base_date,
        ma_period,
        df_full,
        realtime_price,
    ) = args

    if df_full is None:
        from utils.data_loader import fetch_ohlcv

        # df_full이 제공되지 않으면, 네트워크를 통해 데이터를 새로 조회합니다.
        df = fetch_ohlcv(
            tkr, country=country, months_range=[required_months, 0], base_date=base_date
        )
    else:
        # df_full이 제공되면, base_date까지의 데이터만 잘라서 사용합니다.
        df = df_full[df_full.index <= base_date].copy()

    if df is None or df.empty:
        return tkr, {"error": "INSUFFICIENT_DATA"}

    # 실시간 가격이 있으면, 이를 데이터프레임의 마지막 행으로 추가/업데이트합니다.
    # 이렇게 하면 이동평균 및 추세 신호가 실시간 가격을 반영하여 계산됩니다.
    if realtime_price is not None and pd.notna(realtime_price):
        # .loc를 사용하여 base_date 인덱스에 'Close' 값을 설정합니다.
        # 해당 날짜가 없으면 새로 추가되고, 있으면 업데이트됩니다.
        df.loc[base_date, "Close"] = realtime_price

    if len(df) < ma_period:
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
        "buy_signal_days": buy_signal_days,
        "ma_period": ma_period,
        "unadjusted_close": df.get("unadjusted_close"),
    }


def _normalize_holdings(raw_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """포트폴리오 스냅샷의 보유 종목 리스트를 정규화된 딕셔너리로 변환합니다."""
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


def _lookup_price(
    data_entry: Dict[str, Any], target_date: Optional[pd.Timestamp]
) -> Optional[float]:
    """캐시된 데이터에서 특정 날짜의 종가를 조회합니다."""
    if target_date is None or not data_entry:
        return None
    df_price = data_entry.get("df")
    if df_price is None or df_price.empty:
        return None

    # 데이터프레임 복사본을 만들어 원본 수정을 방지합니다.
    df_local = df_price.copy()

    if isinstance(df_local.columns, pd.MultiIndex):
        df_local.columns = df_local.columns.get_level_values(0)
        df_local = df_local.loc[:, ~df_local.columns.duplicated()]
    if not isinstance(df_local.index, pd.DatetimeIndex):
        df_local.index = pd.to_datetime(df_local.index)

    price = None
    try:
        # 가장 빠른 방법: 날짜로 직접 조회
        row = df_local.loc[target_date]
        price = row.get("Close") if isinstance(row, pd.Series) else row["Close"].iloc[0]
    except KeyError:
        # 해당 날짜에 데이터가 없는 경우 (휴장일 등), 그 이전 가장 최근 데이터를 찾습니다.
        subset = df_local[df_local.index <= target_date]
        if not subset.empty and "Close" in subset.columns:
            price = subset["Close"].iloc[-1]

    # 'Adj Close'에 대한 폴백 로직 (yfinance 등)
    if price is None and "Adj Close" in df_local.columns:
        try:
            row = df_local.loc[target_date, "Adj Close"]
            price = row.iloc[0] if isinstance(row, pd.Series) else row
        except KeyError:
            subset = df_local[df_local.index <= target_date]
            if not subset.empty:
                price = subset["Adj Close"].iloc[-1]

    try:
        return float(price)
    except (TypeError, ValueError, OverflowError):
        return None


def _fetch_and_prepare_data(
    country: str,
    account: str,
    portfolio_settings: Dict,
    date_str: Optional[str],
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,  # noqa: E501
) -> Optional[SignalReportData]:
    """
    주어진 종목 목록에 대해 OHLCV 데이터를 조회하고,
    신호 계산에 필요한 이동평균 기반 지표를 계산합니다.
    """
    logger = get_signal_logger()

    ma_period = portfolio_settings["ma_period"]

    request_label = date_str or "auto"
    logger.info(
        "[%s] signal data preparation started (input date=%s)", country.upper(), request_label
    )

    try:
        initial_date_ts = pd.to_datetime(portfolio_settings["initial_date"]).normalize()
        request_date_ts = pd.to_datetime(date_str).normalize()
    except (ValueError, TypeError, AttributeError):
        print(
            f"오류: 날짜 형식 변환에 실패했습니다. (요청: {date_str}, 시작일: {portfolio_settings.get('initial_date')})"
        )
        return None

    if request_date_ts < initial_date_ts:
        print(
            f"정보: 요청된 날짜({request_date_ts.strftime('%Y-%m-%d')})가 계좌 시작일({initial_date_ts.strftime('%Y-%m-%d')}) 이전이므로 현황을 계산하지 않습니다."
        )
        return None

    portfolio_data = get_portfolio_snapshot(country, account=account, date_str=date_str)

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
    try:
        sample_holding = portfolio_data.get("holdings", [])[:2]
        logger.info("[%s/%s] sample holdings: %s", country.upper(), account, sample_holding)
    except Exception:
        pass

    # 콘솔 로그에 국가/날짜를 포함하여 표시
    try:
        print(f"[{country}/{account}]{base_date.strftime('%Y-%m-%d')} 시그널을 계산합니다")
    except Exception:
        pass

    holdings = _normalize_holdings(portfolio_data.get("holdings", []))

    # DB에서 종목 목록을 가져와 전체 유니버스를 구성합니다.
    all_etfs_from_file = get_etfs(country)
    # is_active 필드가 없는 종목이 있는지 확인합니다.
    for etf in all_etfs_from_file:
        if "is_active" not in etf:
            raise ValueError(
                f"종목 마스터 파일의 '{etf.get('ticker')}' 종목에 'is_active' 필드가 없습니다. 파일을 확인해주세요."
            )
    full_etf_meta = {etf["ticker"]: etf for etf in all_etfs_from_file}
    etfs_from_file = [etf for etf in all_etfs_from_file if etf.get("is_active") is not False]
    etf_meta = {etf["ticker"]: etf for etf in etfs_from_file}

    # 오늘 판매된 종목을 추가합니다.
    sold_tickers_today = set()
    trades_on_base_date = get_trades_on_date(country, account, base_date)
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

    # 실시간 가격을 조회할지 여부를 결정합니다.
    # - 코인: 항상 조회
    # - 한국: 거래일의 장 시작(09시) ~ 자정까지 조회 (DEVELOPMENT_RULES.md 2.2)
    # - 호주: 장중(10:00-16:00)에만 조회
    today_cal = pd.Timestamp.now().normalize()
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
                    pass  # pytz 또는 타임존 오류 시 False 유지
        else:  # aus
            use_realtime = is_market_open(country)

    if use_realtime:
        if country == "kor":
            print("-> 장중 또는 장 마감 직후입니다. 네이버 금융에서 실시간 시세를 가져옵니다.")
        elif country == "coin":
            print("-> 실시간 시세를 가져옵니다 (코인).")
        else:  # aus
            print("-> 장중입니다. 실시간 시세를 가져옵니다.")

    # --- 신호 계산 (공통 설정에서) ---
    try:
        common = get_common_file_settings()
        regime_filter_enabled = common["MARKET_REGIME_FILTER_ENABLED"]
        regime_ma_period = common["MARKET_REGIME_FILTER_MA_PERIOD"]
    except (SystemExit, KeyError, ValueError, TypeError) as e:
        print(f"오류: 공통 설정을 불러오는 중 문제가 발생했습니다: {e}")
        return None

    # DB에서 종목 유형(ETF/주식) 정보 가져오기
    # 코인은 거래소 잔고 기반 표시이므로, 종목 마스터가 비어 있어도 보유코인을 기준으로 진행합니다.
    if not etfs_from_file and country != "coin":
        print(f"오류: 'data/stocks/{country}.json' 파일에서 '{country}' 국가의 현황을 계산할 종목을 찾을 수 없습니다.")
        return None

    max_ma_period = max(ma_period, regime_ma_period if regime_filter_enabled else 0)
    required_days = max_ma_period + 5  # 버퍼 추가
    required_months = (required_days // 22) + 2

    # --- 실시간 가격 일괄 조회 ---
    # 개장 중일 경우, 모든 종목의 실시간 가격을 미리 한 번에 조회합니다.
    # 이 가격은 추세 분석(이동평균 계산)에 사용됩니다.
    realtime_prices: Dict[str, Optional[float]] = {}
    if use_realtime:
        print("-> 실시간 가격 일괄 조회 시작...")

        def _fetch_realtime_price(tkr_local: str) -> Optional[float]:
            from utils.data_loader import fetch_naver_realtime_price

            if country == "kor":
                return fetch_naver_realtime_price(tkr_local)
            if country == "coin":
                return _fetch_bithumb_realtime_price(tkr_local)
            return None

        for tkr, _ in pairs:
            rt_price = _fetch_realtime_price(tkr)
            if rt_price is not None:
                realtime_prices[tkr] = rt_price
        print(f"-> 실시간 가격 조회 완료 ({len(realtime_prices)}/{len(pairs)}개 성공).")

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

        if df_regime is not None and not df_regime.empty:
            # 실시간 가격 조회 및 적용
            if use_realtime and yf:
                try:
                    # yfinance를 사용하여 최근 데이터를 가져옵니다.
                    # 미국 지수는 보통 15분 지연되지만, 장중 추세를 반영하기에 충분합니다.
                    ticker_obj = yf.Ticker(regime_ticker)
                    # "1d" 기간은 때때로 전날 종가만 반환할 수 있으므로 "2d"로 조회하여 최신 데이터를 확보합니다.
                    hist = ticker_obj.history(period="2d", interval="15m", auto_adjust=True)
                    if not hist.empty:
                        latest_price = hist["Close"].iloc[-1]
                        # base_date에 최신 가격을 업데이트/추가합니다.
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

    # --- 데이터 로딩 및 지표 계산 ---
    tasks = []
    for tkr, _ in pairs:
        if not full_etf_meta.get(tkr, {}).get("is_active", True):
            continue
        df_full = prefetched_data.get(tkr) if prefetched_data else None
        tasks.append(
            (
                tkr,
                country,
                required_months,
                base_date,
                ma_period,
                df_full,
                realtime_prices.get(tkr),  # 조회된 실시간 가격 전달
            )
        )

    # 순차 처리로 데이터 로딩 및 기본 지표 계산
    processed_results: Dict[str, Dict[str, Any]] = {}
    desc = "과거 데이터 처리" if prefetched_data else "종목 데이터 로딩"
    logger.info(
        "[%s] %s started (tickers=%d)",
        country.upper(),
        desc,
        len(tasks),
    )
    print(f"-> {desc} 시작... (총 {len(tasks)}개 종목)")

    # 순차 처리로 데이터 로딩 및 기본 지표 계산 (병렬 처리 금지 원칙 준수)
    for i, task in enumerate(tasks):
        tkr = task[0]
        try:
            _, result = _load_and_prepare_ticker_data(task)
            processed_results[tkr] = result
        except PykrxDataUnavailable as exc:
            start_str = exc.start_dt.strftime("%Y-%m-%d")
            end_str = exc.end_dt.strftime("%Y-%m-%d")
            message = f"[{country}/{account}] pykrx 조회 실패 ({start_str}~{end_str}): {exc.detail}"
            logger.error(message)
            try:
                send_log_to_slack(message)
            except Exception:
                pass
            raise
        except Exception as exc:
            print(f"\n-> 경고: {tkr} 데이터 처리 중 오류 발생: {exc}")
            processed_results[tkr] = {"error": "PROCESS_ERROR"}
            logger.error("[%s] %s data processing error", country, tkr)
        # 진행 상황 표시
        print(f"\r   {desc} 진행: {i + 1}/{len(tasks)}", end="", flush=True)

    print("\n-> 종목 데이터 처리 완료.")
    logger.info("[%s] %s finished", country.upper(), desc)

    # --- 최종 데이터 조합 및 계산 ---
    # 이제 `processed_results`를 사용하여 순차적으로 나머지 계산을 수행합니다.
    print("\n-> 최종 데이터 조합 및 계산 시작...")
    for tkr, _ in pairs:
        if not full_etf_meta.get(tkr, {}).get("is_active", True):
            continue
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

        # 현재가는 _load_and_prepare_ticker_data에서 실시간 가격을 반영하여
        # 계산된 'close' 시리즈의 마지막 값을 사용합니다.
        c0 = float(result["close"].iloc[-1])
        if pd.isna(c0) or c0 <= 0:
            failed_tickers_info[tkr] = "FETCH_FAILED"
            logger.error(
                "[%s] %s excluded due to invalid price (price: %s)",
                country.upper(),
                tkr,
                c0,
            )
            continue

        m = result["ma"].iloc[-1]

        # `base_date`가 '다음 거래일'인 경우, `prev_close`는 '어제' 종가를 의미해야 합니다.
        # `result["close"]`는 '오늘'까지의 데이터를 포함하므로, '오늘'을 기준으로 이전 종가를 찾습니다.
        today_cal = pd.Timestamp.now().normalize()
        date_for_prev_close = today_cal if base_date.date() > today_cal.date() else base_date
        prev_close = _resolve_previous_close(result["close"], date_for_prev_close)

        if pd.notna(m) and m > 0:
            ma_score = (c0 / m) - 1.0
        else:
            ma_score = 0.0
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
        "[%s] signal data summary for %s: processed=%d, failures=%s",
        country.upper(),
        base_date.strftime("%Y-%m-%d"),
        len(data_by_tkr),
        fail_counts or "{}",
    )

    return SignalReportData(
        portfolio_data=portfolio_data,
        data_by_tkr=data_by_tkr,
        total_holdings_value=total_holdings_value,
        datestamps=datestamps,
        pairs=pairs,
        base_date=base_date,
        regime_info=regime_info,
        full_etf_meta=full_etf_meta,
        etf_meta=etf_meta,
        failed_tickers_info=failed_tickers_info,
    )


def _build_header_line(
    country,
    account: str,
    portfolio_data,
    current_equity,
    total_holdings_value,
    data_by_tkr,
    base_date,
    portfolio_settings: Dict,
):
    """리포트의 헤더 라인을 생성합니다."""
    from utils.account_registry import get_account_info

    account_info = get_account_info(account)
    currency = account_info.get("currency", "KRW")
    precision = account_info.get("precision", 0)

    def _aud_money_formatter(amount):
        return f"${amount:,.{precision}f}"

    # 국가별 포맷터 설정
    money_formatter = _aud_money_formatter if currency == "AUD" else format_kr_money

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
    initial_capital_local = (
        float(portfolio_settings.get("initial_capital", 0)) if portfolio_settings else 0.0
    )
    initial_date = (
        pd.to_datetime(portfolio_settings.get("initial_date"))
        if portfolio_settings and portfolio_settings.get("initial_date")
        else None
    )
    cum_ret_pct = (
        ((equity_for_cum_calc / initial_capital_local) - 1.0) * 100.0
        if initial_capital_local > 0
        else 0.0
    )
    portfolio_topn = portfolio_settings.get("portfolio_topn", 0) if portfolio_settings else 0

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

    # 일간 수익률 계산
    # '다음 거래일' 리포트의 일간 수익률은 '오늘'의 수익률을 의미합니다.
    # 따라서 이전 스냅샷을 조회하는 기준 날짜를 조정합니다.
    if day_label == "다음 거래일":
        # '다음 거래일' 리포트에서는 일간 수익률을 0으로 표시합니다.
        day_ret_pct = 0.0
        day_profit_loss = 0.0
        prev_equity = None
    else:
        # '오늘' 또는 '과거' 리포트에서는 `base_date`를 기준으로 이전 스냅샷을 가져옵니다.
        compare_date_for_prev = base_date
        prev_snapshot = get_previous_portfolio_snapshot(country, compare_date_for_prev, account)
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
    eval_ret_str = _format_return_for_header("평가", eval_ret_pct, eval_profit_loss, money_formatter)
    cum_ret_str = _format_return_for_header("누적", cum_ret_pct, cum_profit_loss, money_formatter)

    # 헤더 본문
    header_body = (
        f"보유종목: {held_count}/{portfolio_topn} | 평가금액: {equity_str} | 보유금액: {holdings_str} | "
        f"현금: {cash_str} | {day_ret_str} | {eval_ret_str} | {cum_ret_str}"
    )

    # --- N 거래일차 계산 및 추가 ---
    if initial_date and base_date >= initial_date:
        try:
            # get_trading_days는 시작일과 종료일을 포함하여 계산합니다.
            trading_days_count = len(
                get_trading_days(
                    initial_date.strftime("%Y-%m-%d"),
                    base_date.strftime("%Y-%m-%d"),
                    country,
                )
            )
            trading_days_str = f' | <span style="color:blue">{trading_days_count} 거래일차</span>'
            header_body += trading_days_str
        except Exception:
            # 오류 발생 시 거래일차 정보는 추가하지 않습니다.
            pass

    return header_body, label_date, day_label


def _get_calculation_message_lines(num_tickers: int, warnings: List[str]):
    message_lines = [
        f"계산에 이용된 종목의 수: {num_tickers}",
    ]

    if warnings:
        max_warnings = 10
        message_lines.append("- 경고:")
        for i, warning in enumerate(warnings):
            if i < max_warnings:
                message_lines.append(f"  ⚠️ {warning}")
        if len(warnings) > max_warnings:
            message_lines.append(f"  ... 외 {len(warnings) - max_warnings}건의 경고가 더 있습니다.")

    return message_lines


def _get_equity_update_message_line(
    country: str, account: str, old_equity: float, new_equity: float
):
    """평가금액 자동 보정 시 슬랙으로 알림을 보냅니다."""
    from utils.account_registry import get_account_info
    from utils.report import format_kr_money

    account_info = get_account_info(account)
    currency = account_info.get("currency", "KRW")
    precision = account_info.get("precision", 0)

    def _aud_money_formatter(amount):
        return f"${amount:,.{precision}f}"

    money_formatter = _aud_money_formatter if currency == "AUD" else format_kr_money

    diff = new_equity - old_equity
    diff_str = f"{'+' if diff > 0 else ''}{money_formatter(diff)}"

    if old_equity > 0:
        # 평가금액 변동(증가/감소)에 따라 다른 레이블을 사용합니다.
        change_label = "증가" if diff >= 0 else "감소"
        message = f"평가금액 {change_label}: {money_formatter(old_equity)} => {money_formatter(new_equity)} ({diff_str})"
    else:
        message = f"신규 평가금액 저장: {money_formatter(new_equity)}"

    return message


def generate_signal_report(
    country: str,
    account: str,
    date_str: Optional[str] = None,
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Optional[Tuple[str, List[str], List[List[str]], pd.Timestamp, List[str]]]:
    """지정된 전략에 대한 오늘의 매매 신호를 생성하여 리포트로 반환합니다."""
    logger = get_signal_logger()

    # 1. 대상 날짜 결정
    if date_str:
        try:
            target_date = pd.to_datetime(date_str).normalize()
        except (ValueError, TypeError):
            raise ValueError(f"잘못된 날짜 형식입니다: {date_str}")
    else:
        # 날짜가 지정되지 않으면 스케줄러 로직에 따라 동적으로 결정
        target_date = _determine_target_date_for_scheduler(country)

    # 휴장일 검사
    if country != "coin":
        if not _is_trading_day(country, target_date.to_pydatetime()):
            raise ValueError(f"휴장일({target_date.strftime('%Y-%m-%d')})에는 시그널을 생성할 수 없습니다.")

    effective_date_str = target_date.strftime("%Y-%m-%d")

    # 2. 설정을 파일에서 가져옵니다.
    try:
        portfolio_settings = get_account_file_settings(country, account)
    except SystemExit as e:
        print(str(e))
        return None

    # 3. 데이터 로드 및 지표 계산
    # PykrxDataUnavailable 예외는 get_latest_trading_day 로직으로 인해 발생 가능성이 낮아졌습니다.
    # 만약 발생하더라도, 상위 호출자(cli.py, web_app.py)에서 처리하도록 그대로 전달합니다.
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

    warning_messages_for_slack = []
    if insufficient_data_tickers:
        name_map = {tkr: name for tkr, name in pairs}
        for tkr in sorted(insufficient_data_tickers):
            # Check if the ticker is inactive.
            if not full_etf_meta.get(tkr, {}).get("is_active", True):
                # If it's an inactive ticker, we don't need to warn about insufficient data.
                # The SELL_INACTIVE signal will explain its status.
                continue
            name = name_map.get(tkr, tkr)
            warning_messages_for_slack.append(f"{name}({tkr}): 데이터 기간이 부족하여 계산에서 제외됩니다.")
    # 슬랙 메시지를 위한 메시지 만들기 시작
    slack_message_lines = _get_calculation_message_lines(len(pairs), warning_messages_for_slack)

    holdings = _normalize_holdings(portfolio_data.get("holdings", []))
    current_equity = float(portfolio_data.get("total_equity", 0.0))
    equity_date = portfolio_data.get("equity_date")

    # --- 평가금액 이월 및 자동 보정 로직 ---
    # 1. 자동 보정이 필요한지 판단하기 위한 후보 금액 계산
    international_shares_value = 0.0
    if country == "aus":
        intl_info = portfolio_data.get("international_shares")
        if isinstance(intl_info, dict):
            try:
                international_shares_value = float(intl_info.get("value", 0.0))
            except (TypeError, ValueError):
                international_shares_value = 0.0
    new_equity_candidate = total_holdings_value + international_shares_value

    if country == "coin":
        try:
            from scripts.snapshot_bithumb_balances import (
                _fetch_bithumb_balance_dict as fetch_bithumb_balance_dict,
            )

            bal = fetch_bithumb_balance_dict()
            if bal:
                krw_balance = bal.get("total_krw", 0.0)
                p_balance = bal.get("total_P", 0.0)
                new_equity_candidate += krw_balance + p_balance
        except Exception as e:
            logger.warning("Bithumb 잔액 조회 실패. 평가금액 자동 보정 시 코인 가치만 반영됩니다. (%s)", e)

    # 2. 자동 보정 및 이월 조건 확인
    is_carried_forward = (
        equity_date
        and base_date
        and pd.to_datetime(equity_date).normalize() != base_date.normalize()
    )

    # 3. 최종 평가금액 및 DB 저장 여부 결정
    final_equity = current_equity
    updated_by = None
    old_equity_for_log = current_equity

    if is_carried_forward:
        # 휴장일 등: 과거 평가금액을 현재 날짜로 이월만 합니다. 보정(재계산)은 하지 않습니다.
        final_equity = current_equity  # 값은 그대로 유지
        updated_by = "스케줄러(이월)"
    else:
        # 거래일: 자동 보정 로직을 적용합니다.
        should_autocorrect = False
        autocorrect_reason = ""
        if country == "coin":
            # 코인은 항상 최신 잔액으로 덮어씁니다.
            if abs(new_equity_candidate - current_equity) > 1e-9:
                should_autocorrect = True
                autocorrect_reason = "보정"
        elif new_equity_candidate > 0 and (
            new_equity_candidate > current_equity or current_equity == 0
        ):
            # 주식/ETF는 오늘 날짜의 평가금액이 이미 있을 때, 증가하는 경우에만 보정합니다.
            should_autocorrect = True
            autocorrect_reason = "보정"

        if should_autocorrect:
            final_equity = new_equity_candidate
            updated_by = f"스케줄러({autocorrect_reason})"

    # 4. DB에 저장 및 컨텍스트 업데이트
    # '이월'의 경우 평가금액 변동이 없으므로, updated_by가 설정되었는지 여부로 저장 로직을 트리거합니다.
    if updated_by:
        # 이월(휴장일) 또는 보정(거래일) 시 모두 DB에 저장합니다.
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
                # 보정은 금액 변동이 있을 때만 로그를 남깁니다.
                if abs(final_equity - old_equity_for_log) >= 1.0:
                    log_msg = f"평가금액 자동 보정: {old_equity_for_log:,.0f}원 -> {final_equity:,.0f}원"
                    print(f"-> {log_msg}")
                    equity_message_line = _get_equity_update_message_line(
                        country, account, old_equity_for_log, final_equity
                    )
                    slack_message_lines.append(equity_message_line)
                else:  # 이월
                    log_msg = (
                        f"평가금액 이월: {pd.to_datetime(equity_date).strftime('%Y-%m-%d')}의 평가금액 "
                        f"({final_equity:,.0f}원)을 {base_date.strftime('%Y-%m-%d')}으로 저장했습니다."
                    )
                    print(f"-> {log_msg}")

                logger.info(
                    "[%s/%s] Daily equity updated by %s on %s: %0.2f",
                    country.upper(),
                    account,
                    updated_by,
                    base_date.strftime("%Y-%m-%d"),
                    final_equity,
                )

                # 로컬 컨텍스트 업데이트
                current_equity = final_equity
                portfolio_data["total_equity"] = final_equity
            else:
                logger.error(
                    "[%s/%s] daily_equities 저장 실패: %s",
                    country.upper(),
                    account,
                    base_date.strftime("%Y-%m-%d"),
                )

                logger.info(
                    "[%s/%s] Daily equity updated by %s on %s: %0.2f",
                    country.upper(),
                    account,
                    updated_by,
                    base_date.strftime("%Y-%m-%d"),
                    final_equity,
                )

                # 로컬 컨텍스트 업데이트
                current_equity = final_equity
                portfolio_data["total_equity"] = final_equity
        else:
            logger.error(
                "[%s/%s] daily_equities 저장 실패: %s",
                country.upper(),
                account,
                base_date.strftime("%Y-%m-%d"),
            )

    # 현재 보유 종목의 카테고리 (TBD 제외)
    held_categories = set()
    for tkr, d in holdings.items():
        if float(d.get("shares", 0.0)) > 0:
            category = etf_meta.get(tkr, {}).get("category")
            if category and category != "TBD":
                held_categories.add(category)

    # 3. 헤더 생성
    total_holdings_value += international_shares_value

    header_line, label_date, day_label = _build_header_line(
        country,
        account,
        portfolio_data,
        current_equity,
        total_holdings_value,
        data_by_tkr,
        base_date,
        portfolio_settings,
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

    # 4. 보유 기간 및 고점 대비 하락률 계산
    held_tickers = [tkr for tkr, v in holdings.items() if float((v or {}).get("shares") or 0.0) > 0]
    # 보유 시작일 계산 기준은 실제 표시 기준일(label_date)과 일치시킵니다.
    consecutive_holding_info = calculate_consecutive_holding_info(
        held_tickers, country, account, label_date
    )
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

    try:
        denom = int(portfolio_settings["portfolio_topn"])
    except (ValueError, TypeError):
        print("오류: DB의 portfolio_topn 값이 올바르지 않습니다.")
        return None

    # 공통 설정에서 손절 퍼센트 로드
    try:
        common = get_common_file_settings()
        stop_loss_raw = float(common["HOLDING_STOP_LOSS_PCT"])
        # 양수 입력이 들어오더라도 손절 임계값은 음수로 해석합니다 (예: 10 -> -10).
        stop_loss = -abs(stop_loss_raw)
    except (SystemExit, KeyError, ValueError, TypeError) as e:
        print(f"오류: 공통 설정을 불러오는 중 문제가 발생했습니다: {e}")
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

    # 5. 초기 매매 결정 생성
    decisions = []

    from utils.account_registry import get_account_info

    account_info = get_account_info(account)
    currency = account_info.get("currency", "KRW")
    precision = account_info.get("precision", 0)

    def _format_kr_price(p):
        return f"{int(round(p)):,}"

    def _aud_money_formatter(amount):
        return f"${amount:,.{precision}f}"

    def _aud_price_formatter(p):
        return f"${p:,.{precision}f}"

    # 국가별 포맷터 설정
    if currency == "AUD":
        money_formatter = _aud_money_formatter
        price_formatter = _aud_price_formatter
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
        is_active = full_etf_meta.get(tkr, {}).get("is_active", True)
        if price == 0.0 and is_effectively_held:
            phrase = "가격 데이터 조회 실패"

        # 이 루프의 모든 경로에서 사용되므로, 여기서 초기화합니다.
        buy_date = None
        holding_days = 0
        hold_ret = None

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
                print(f"경고: 보유일 계산 중 오류 발생 ({tkr}): {e}. 달력일 기준으로 대체합니다.")
                # 거래일 계산 실패 시, 달력일 기준으로 계산
                holding_days = (label_date - buy_date).days + 1

        qty = 0
        # 현재 보유 포지션의 손익률을 계산합니다.
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
            elif not is_active:
                state = "SELL_INACTIVE"
                qty = sh
                prof = (price - ac) * qty if ac > 0 else 0.0
                phrase = f"비활성 종목 정리 {format_shares(qty)}주 @ {price_formatter(price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"

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
        # '다음 거래일' 리포트에서는 일간 수익률을 0으로 표시합니다.
        is_next_day_report = base_date.date() > pd.Timestamp.now().normalize().date()
        prev_close = d.get("prev_close")
        day_ret = 0.0
        if not is_next_day_report:
            day_ret = (
                ((price / prev_close) - 1.0) * 100.0
                if prev_close is not None and prev_close > 0 and pd.notna(price)
                else 0.0
            )

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

    # 매수/교체매수 후보는 반드시 '종목 마스터(data/stocks/{country}.json)'에 포함된 종목으로 제한합니다.
    # 이는 사용자가 유니버스에서 제외한 종목(예: 당일 매도 후 목록에서 제거)이
    # 다시 매수 후보로 추천되는 것을 방지합니다.
    from utils.stock_list_io import get_etfs

    universe_tickers = {etf["ticker"] for etf in get_etfs(country)}

    # 6. 시장 레짐 필터 및 매매 로직 적용
    is_risk_off = result.regime_info and result.regime_info.get("is_risk_off", False)

    if is_risk_off:
        # 리스크 오프: 모든 보유 종목을 매도하고, 매수 신호를 무시합니다.
        for decision in decisions:
            # 1. 보유 종목 매도 (이미 다른 이유로 매도 결정된 것은 제외)
            if decision["state"] == "HOLD":
                decision["state"] = "SELL_REGIME_FILTER"
                decision["row"][2] = "SELL_REGIME_FILTER"

                # 매도 문구 생성
                d_sell = data_by_tkr.get(decision["tkr"])
                if d_sell:
                    sell_price = float(d_sell.get("price", 0))
                    sell_qty = float(d_sell.get("shares", 0))
                    avg_cost = float(d_sell.get("avg_cost", 0))

                    hold_ret = 0.0
                    prof = 0.0
                    if avg_cost > 0 and sell_price > 0:
                        hold_ret = ((sell_price / avg_cost) - 1.0) * 100.0
                        prof = (sell_price - avg_cost) * sell_qty

                    sell_phrase = f"시장위험회피 매도 {format_shares(sell_qty)}주 @ {price_formatter(sell_price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"
                    decision["row"][-1] = sell_phrase

            # 2. 매수 신호 무시
            if decision.get("buy_signal"):
                decision["buy_signal"] = False
                if decision["state"] == "WAIT":
                    original_phrase = decision["row"][-1]
                    if original_phrase and "추세진입" in original_phrase:
                        decision["row"][-1] = f"시장 위험 회피 ({original_phrase})"
                    else:
                        decision["row"][-1] = "시장 위험 회피"
    else:
        # 리스크 온: 기존 리밸런싱, 신규매수, 교체매매 로직 적용
        # 교체 매매 관련 설정 로드 (임계값은 DB 설정 우선)
        try:
            replace_weaker_stock = bool(portfolio_settings["replace_weaker_stock"])
            replace_threshold = float(portfolio_settings["replace_threshold"])
        except (KeyError, ValueError, TypeError) as e:
            print(f"오류: '{country}' 국가의 교체 매매 설정값이 올바르지 않습니다: {e}")
            return None

        # 리밸런싱 매도 결정 전, 다른 이유로 이미 매도 결정된 종목 수를 파악합니다.
        other_sell_states = {"CUT_STOPLOSS", "SELL_TREND", "SELL_INACTIVE"}
        num_already_selling = sum(1 for d in decisions if d["state"] in other_sell_states)

        # 목표 보유 수(denom)를 맞추기 위해 추가로 매도해야 할 종목 수를 계산합니다.
        num_to_sell_for_rebalance = (held_count - num_already_selling) - denom

        if num_to_sell_for_rebalance > 0:
            # Case 1: 포트폴리오가 목표보다 크므로, 가장 약한 종목을 매도하여 축소
            rebalance_sell_candidates = [d for d in decisions if d["state"] == "HOLD"]
            rebalance_sell_candidates.sort(
                key=lambda x: x.get("score") if pd.notna(x.get("score")) else -float("inf")
            )
            tickers_to_sell = [
                d["tkr"] for d in rebalance_sell_candidates[:num_to_sell_for_rebalance]
            ]

            for decision in decisions:
                if decision["tkr"] in tickers_to_sell:
                    decision["state"] = "SELL_REBALANCE"
                    decision["row"][2] = "SELL_REBALANCE"
                    d_sell = data_by_tkr.get(decision["tkr"])
                    if d_sell:
                        sell_price = float(d_sell.get("price", 0))
                        sell_qty = float(d_sell.get("shares", 0))
                        avg_cost = float(d_sell.get("avg_cost", 0))
                        hold_ret = (
                            ((sell_price / avg_cost) - 1.0) * 100.0
                            if avg_cost > 0 and sell_price > 0
                            else 0.0
                        )
                        prof = (sell_price - avg_cost) * sell_qty if avg_cost > 0 else 0.0
                        sell_phrase = f"리밸런스 매도 {format_shares(sell_qty)}주 @ {price_formatter(sell_price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"
                        decision["row"][-1] = sell_phrase
        else:
            # Case 2: 포트폴리오 크기가 정상이거나 작음 (신규매수 또는 교체매매)
            slots_to_fill = denom - held_count
            if slots_to_fill > 0:
                # 2a: 빈 슬롯이 있으므로 신규 매수
                # ... (The rest of the logic for new buys and replacement buys)
                buy_candidates_raw = sorted(
                    [a for a in decisions if a.get("buy_signal") and a["tkr"] in universe_tickers],
                    key=lambda x: x["score"],
                    reverse=True,
                )
                final_buy_candidates, recommended_buy_categories = [], set()
                for cand in buy_candidates_raw:
                    category = etf_meta.get(cand["tkr"], {}).get("category")
                    if category and category != "TBD":
                        if category in held_categories:
                            # Find which held ticker has this category
                            conflicting_ticker = "???"
                            for held_tkr, held_data in holdings.items():
                                if float(held_data.get("shares", 0.0)) > 0:
                                    held_category = etf_meta.get(held_tkr, {}).get("category")
                                    if held_category == category:
                                        conflicting_ticker = held_tkr
                                        break
                            cand["row"][
                                -1
                            ] = f"카테고리 중복 ({conflicting_ticker} 보유) ({cand['row'][-1]})"
                            continue
                        if category in recommended_buy_categories:
                            cand["row"][-1] = f"카테고리 중복 (추천) ({cand['row'][-1]})"
                            continue
                        recommended_buy_categories.add(category)
                    final_buy_candidates.append(cand)

                available_cash, buys_made = total_cash, 0
                for cand in final_buy_candidates:
                    if buys_made >= slots_to_fill:
                        cand["row"][-1] = f"포트폴리오 가득 참 ({cand['row'][-1]})"
                        continue
                    d, price = data_by_tkr.get(cand["tkr"]), 0
                    if d:
                        price = d.get("price", 0)
                    if price > 0:
                        _, min_val, max_val = (
                            current_equity,
                            min_pos * current_equity,
                            max_pos * current_equity,
                        )
                        budget = min(max_val, available_cash)
                        req_qty, buy_notional = 0, 0.0
                        if budget >= min_val and budget > 0:
                            if country in ("coin", "aus"):
                                req_qty, buy_notional = budget / price, budget
                            else:
                                req_qty = int(budget // price)
                                buy_notional = req_qty * price
                                if req_qty <= 0 or buy_notional + 1e-9 < min_val:
                                    req_qty, buy_notional = 0, 0.0
                        if req_qty > 0 and buy_notional <= available_cash + 1e-9:
                            cand["state"], cand["row"][2] = "BUY", "BUY"
                            buy_phrase = f"🚀 매수 {format_shares(req_qty)}주 @ {price_formatter(price)} ({money_formatter(buy_notional)})"
                            cand["row"][-1] = f"{buy_phrase} ({cand['row'][-1]})"
                            available_cash -= buy_notional
                            buys_made += 1
                        else:
                            cand["row"][-1] = f"현금 부족 ({cand['row'][-1]})"
                    else:
                        cand["row"][-1] = f"가격 정보 없음 ({cand['row'][-1]})"
            else:
                # 2b: 포트폴리오가 가득 찼으므로 교체 매매 고려
                if replace_weaker_stock:
                    buy_candidates = sorted(
                        [
                            a
                            for a in decisions
                            if a.get("buy_signal") and a["tkr"] in universe_tickers
                        ],
                        key=lambda x: x["score"],
                        reverse=True,
                    )
                    held_stocks = sorted(
                        [a for a in decisions if a["state"] == "HOLD"],
                        key=lambda x: x["score"] if pd.notna(x["score"]) else -float("inf"),
                    )
                    for k in range(min(len(buy_candidates), len(held_stocks))):
                        best_new, weakest_held = buy_candidates[k], held_stocks[k]
                        if best_new["state"] != "WAIT" or weakest_held["state"] != "HOLD":
                            continue
                        if (
                            pd.notna(best_new["score"])
                            and pd.notna(weakest_held["score"])
                            and best_new["score"]
                            > weakest_held["score"] + (replace_threshold / 100.0)
                        ):
                            d_weakest = data_by_tkr.get(weakest_held["tkr"])
                            sell_price, sell_qty, avg_cost = (
                                float(d_weakest.get(k, 0)) for k in ["price", "shares", "avg_cost"]
                            )
                            hold_ret = (
                                ((sell_price / avg_cost) - 1.0) * 100.0
                                if avg_cost > 0 and sell_price > 0
                                else 0.0
                            )
                            prof = (sell_price - avg_cost) * sell_qty if avg_cost > 0 else 0.0
                            sell_phrase = f"교체매도 {format_shares(sell_qty)}주 @ {price_formatter(sell_price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'} ({best_new['tkr']}(으)로 교체)"
                            (
                                weakest_held["state"],
                                weakest_held["row"][2],
                                weakest_held["row"][-1],
                            ) = ("SELL_REPLACE", "SELL_REPLACE", sell_phrase)
                            best_new["state"], best_new["row"][2] = "BUY_REPLACE", "BUY_REPLACE"
                            sell_value = weakest_held["weight"] / 100.0 * current_equity
                            buy_price = float(data_by_tkr.get(best_new["tkr"], {}).get("price", 0))
                            if buy_price > 0:
                                buy_qty = (
                                    sell_value / buy_price
                                    if country in ("coin", "aus")
                                    else int(sell_value // buy_price)
                                )
                                buy_notional = buy_qty * buy_price
                                best_new["row"][
                                    -1
                                ] = f"매수 {format_shares(buy_qty)}주 @ {price_formatter(buy_price)} ({money_formatter(buy_notional)}) ({weakest_held['tkr']} 대체)"
                            else:
                                best_new["row"][-1] = f"{weakest_held['tkr']}(을)를 대체 (가격정보 없음)"
                        else:
                            break
    # 최종 정리: 아직 'WAIT' 상태인 종목들의 사유를 명확히 합니다.
    for cand in decisions:
        if cand["state"] == "WAIT":
            # 이미 '현금 부족' 또는 '카테고리 중복' 등의 구체적인 사유가 설정된 경우는 덮어쓰지 않습니다.
            if "추세진입" in cand["row"][-1]:
                cand["row"][-1] = "포트폴리오 가득 참 (교체대상 아님)" + f" ({cand['row'][-1]})"
    # 7. 완료된 거래 표시
    # 기준일에 발생한 거래를 가져와서, 추천에 따라 실행되었는지 확인하는 데 사용합니다.
    # 표시 기준일 기준으로 '완료' 거래를 표시합니다. 다음 거래일이면 거래가 없을 확률이 높음
    trades_on_base_date = get_trades_on_date(country, account, label_date)
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

    # 8. 최종 정렬
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
            "SELL_REBALANCE": 4,
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

    # 9. 최종 결과 반환
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
        "[%s] signal report ready: rows=%d state_counts=%s",
        country.upper(),
        len(rows_sorted),
        state_counts,
    )

    return (header_line, headers, rows_sorted, base_date, slack_message_lines)


def main(
    country: str = "kor",
    account: str = "",
    date_str: Optional[str] = None,
) -> Optional[datetime]:
    """CLI에서 오늘의 매매 신호를 실행하고 결과를 출력/저장합니다."""
    if not account:
        raise ValueError("account is required for signal generation")

    result = generate_signal_report(country, account, date_str)

    if result:
        header_line, headers, rows_sorted, report_base_date, slack_message_lines = result
        # 가능하다면 웹 앱 히스토리에서 사용할 수 있도록 현황 보고서를 저장합니다.
        try:
            # 반환된 base_date는 보고서의 실제 기준일이므로 그대로 저장에 사용합니다.
            save_signal_report_to_db(
                country,
                account,
                report_base_date.to_pydatetime(),
                (header_line, headers, rows_sorted),
            )
        except Exception:
            pass

        # 슬랙 알림: 현황 전송
        try:
            _maybe_notify_detailed_signal(
                country,
                account,
                header_line,
                headers,
                rows_sorted,
                slack_message_lines
                # report_base_date,
            )
        except Exception:
            pass

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
                    display_row[idx] = f"{val * 100:.1f}"
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
        return report_base_date.to_pydatetime()


def _is_trading_day(country: str, a_date: Optional[datetime] = None) -> bool:
    """지정 국가 기준으로 해당 날짜가 거래일이면 True를 반환합니다.
    a_date가 None이면 오늘 날짜를 검사합니다. # noqa: E501

    - kor/aus: 거래소 달력을 사용합니다. 조회 실패 시 안전하게 비거래일(False)로 간주합니다.
    - coin: 항상 True를 반환합니다.
    """
    if country == "coin":
        return True

    check_date = a_date or datetime.now()
    logger = get_signal_logger()

    try:
        # get_trading_days 함수는 문자열 형태의 날짜를 기대합니다.
        start = end = pd.Timestamp(check_date).strftime("%Y-%m-%d")
        days = get_trading_days(start, end, country)
        # 반환된 거래일 목록에 대상 날짜가 포함되어 있는지 확인합니다.
        return any(pd.Timestamp(d).date() == check_date.date() for d in days)
    except Exception as e:
        # 예외가 발생하면 거래일 판별이 불가능하므로, 안전하게 False를 반환하고 경고를 기록합니다.
        # 기존의 평일 기반 폴백은 공휴일을 잘못 판단할 위험이 있습니다.
        logger.warning(
            "[%s] 거래일 판별 중 오류 발생하여 비거래일로 간주합니다: %s. (date: %s)",
            country.upper(),
            e,
            check_date.strftime("%Y-%m-%d"),
        )
        return False


def _maybe_notify_detailed_signal(
    country: str,
    account: str,
    header_line: str,
    headers: list,
    rows_sorted: list,
    slack_message_lines: list[str],
) -> bool:
    """국가별 설정에 따라 슬랙으로 상세 현황 알림을 전송합니다."""
    from utils.notify import get_slack_webhook_url, send_slack_message

    # 사용자가 모든 수동 실행에서 슬랙 알림을 받기를 원하므로, 거래일 확인 로직을 비활성화합니다.
    # 이로 인해 과거 날짜 조회 등 모든 'status' 명령어 실행 시 알림이 전송됩니다.
    # if not _is_trading_day(country, report_date.to_pydatetime() if report_date else None):
    #     return False
    # --- 슬랙 알림 발송 ---
    webhook_info = get_slack_webhook_url(country, account=account)
    if not webhook_info:
        return False
    webhook_url, webhook_name = webhook_info

    from utils.account_registry import get_account_info

    account_info = get_account_info(account)
    currency = account_info.get("currency", "KRW")
    precision = account_info.get("precision", 0)

    def _aud_money_formatter(amount):
        return f"${amount:,.{precision}f}"

    def _aud_price_formatter(p):
        return f"${p:,.{precision}f}" if isinstance(p, (int, float)) else str(p)

    def _kor_coin_price_formatter(p):
        return f"{int(round(p)):,}" if isinstance(p, (int, float)) else str(p)

    # 국가별 포맷터 설정
    if currency == "AUD":
        money_formatter = _aud_money_formatter
        price_formatter = _aud_price_formatter
    else:  # kor, coin
        money_formatter = format_kr_money
        price_formatter = _kor_coin_price_formatter

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

    # --- 헤더 문자열을 파싱하여 캡션 구성 요소로 나눕니다. ---
    # 날짜 정보
    first_seg = header_line_clean.split("|")[0].strip()
    date_part = first_seg.split(":", 1)[1].strip()
    if "[" in date_part:
        date_part = date_part.split("[")[0].strip()
    date_part = _strip_html(date_part)

    # 보유 종목 수
    hold_seg = next(
        (seg for seg in header_line_clean.split("|") if "보유종목:" in seg),
        "보유종목: -",
    )
    hold_text = _strip_html(hold_seg.split(":", 1)[1].strip())

    # 보유 금액
    hold_val_seg = next(
        (seg for seg in header_line_clean.split("|") if "보유금액:" in seg),
        "보유금액: 0",
    )
    hold_val_text = _strip_html(hold_val_seg.split(":", 1)[1].strip())

    # 현금 금액
    cash_seg = next((seg for seg in header_line_clean.split("|") if "현금:" in seg), "현금: 0")
    cash_text = _strip_html(cash_seg.split(":", 1)[1].strip())

    # 누적 수익률 정보
    cum_seg = next(
        (seg for seg in header_line_clean.split("|") if "누적:" in seg),
        "누적: +0.00%(0원)",
    )
    cum_text = _strip_html(cum_seg.split(":", 1)[1].strip())

    # 총 평가 금액
    equity_seg = next(
        (seg for seg in header_line_clean.split("|") if "평가금액:" in seg),
        "평가금액: 0",
    )
    equity_text = _strip_html(equity_seg.split(":", 1)[1].strip())

    # 컬럼 인덱스를 계산합니다.
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

    # 티커와 이름 매핑을 구성합니다.
    name_map = {}
    try:
        # 국가 코드에 맞는 ETF 목록을 불러옵니다.
        etfs = get_etfs(country) or []
        name_map = {str(s.get("ticker") or "").upper(): str(s.get("name") or "") for s in etfs}
    except Exception:
        pass

    # 호주 'IS' 종목은 수동으로 이름을 지정합니다.
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

            stt = str(row[idx_state]) if (idx_state is not None and idx_state < len(row)) else ""

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
                    score_col = f"점수 {float(sc) * 100:.1f}"

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
            display_name = f"<{group_name}>({group_name})"
            show_slack = True  # 알 수 없는 그룹은 일단 표시
        else:
            display_name = f"{config['display_name']}({group_name})"
            show_slack = config.get("show_slack", True)

        if not show_slack:
            continue

        if parts_in_group:
            body_lines.append(display_name)
            # 수익률 컬럼 표시 여부 결정: 보유 또는 매수 관련 상태일 때만 표시
            show_return_col = group_name in ["HOLD", "BUY", "BUY_REPLACE"]
            for parts in parts_in_group:
                name_part = parts["name"].ljust(max_len_name)
                price_part = parts["price_col"].ljust(max_len_price_col)
                shares_part = parts["shares_col"].rjust(max_len_shares_col)
                amount_part = parts["amount_col"].rjust(max_len_amount_col)
                score_part = parts["score_col"].ljust(max_len_score_col)

                if show_return_col:
                    return_part = parts["return_col"].ljust(max_len_return_col)
                    line = f"{name_part}  {price_part} {shares_part} {amount_part}  {return_part} {score_part}"
                else:
                    return_part = "".ljust(max_len_return_col)
                    line = f"{name_part}  {price_part} {shares_part} {amount_part}  {return_part} {score_part}"

                body_lines.append(line.rstrip())
            body_lines.append("")  # 그룹 사이에 빈 줄 추가

    if body_lines and body_lines[-1] == "":
        body_lines.pop()

    # --- 슬랙 메시지의 캡션을 구성합니다. ---

    title_line = f"[{global_settings.APP_TYPE}][{country}/{account}] 시그널"
    test_line = "\n".join(slack_message_lines)
    equity_line = f"평가금액: {equity_text}, 누적수익 {cum_text}"
    cash_line = f"현금: {cash_text}, 보유금액: {hold_val_text}"
    hold_line = f"보유종목: {hold_text}"
    caption = "\n".join([title_line, test_line, equity_line, cash_line, hold_line])

    # DECISION_CONFIG에서 is_recommendation=True인 그룹이 하나라도 있으면 @channel 멘션을 포함합니다.
    has_recommendation = False
    for group_name in grouped_parts.keys():
        config = DECISION_CONFIG.get(group_name)
        if config and config.get("is_recommendation", False):
            has_recommendation = True
            break
    slack_mention = "<!channel>\n" if has_recommendation else ""
    if not body_lines:
        # 상세 항목이 없으면 캡션만 전송합니다.
        slack_sent = send_slack_message(
            slack_mention + caption, webhook_url=webhook_url, webhook_name=webhook_name
        )
    else:
        # 슬랙 코드 블록을 사용하여 표 형태를 유지합니다.
        # slack_message = caption + "\n\n" + "\n".join(slack_message_lines)+ "```\n" + "\n".join(body_lines) + "\n```"
        slack_message = caption + "\n\n" + "```\n" + "\n".join(body_lines) + "\n```"
        slack_sent = send_slack_message(
            slack_mention + slack_message, webhook_url=webhook_url, webhook_name=webhook_name
        )

    return slack_sent


def send_summary_notification(
    country: str,
    account: str,
    report_date: datetime,
    duration: float,
    old_equity: float,
) -> None:
    """작업 완료 요약 슬랙 알림을 전송합니다."""
    from utils.db_manager import get_portfolio_snapshot
    from utils.report import format_kr_money

    try:
        date_str = report_date.strftime("%Y-%m-%d")
        prefix = f"{country}/{account}"

        # Get new equity
        new_snapshot = get_portfolio_snapshot(country, account=account)
        new_equity = float(new_snapshot.get("total_equity", 0.0)) if new_snapshot else 0.0

        # Calculate cumulative return
        try:
            file_settings = get_account_file_settings(country, account)
            initial_capital = float(file_settings.get("initial_capital", 0))
        except SystemExit:
            initial_capital = 0.0  # 알림에서는 조용히 실패 처리

        message = f"[{prefix}/{date_str}] 작업 완료(작업시간: {duration:.1f}초)"
        from utils.account_registry import get_account_info

        account_info = get_account_info(account)
        currency = account_info.get("currency", "KRW")
        precision = account_info.get("precision", 0)

        def _aud_money_formatter(amount):
            return f"${amount:,.{precision}f}"

        money_formatter = _aud_money_formatter if currency == "AUD" else format_kr_money

        if initial_capital > 0:
            cum_ret_pct = ((new_equity / initial_capital) - 1.0) * 100.0
            cum_profit_loss = new_equity - initial_capital
            equity_summary = f"평가금액: {money_formatter(new_equity)}, 누적수익 {cum_ret_pct:+.2f}%({money_formatter(cum_profit_loss)})"
            message += f" | {equity_summary}"

        min_change_threshold = 0.5 if country != "aus" else 0.005
        if abs(new_equity - old_equity) >= min_change_threshold:
            diff = new_equity - old_equity
            change_label = "📈평가금액 증가" if diff > 0 else "📉평가금액 감소"

            if country == "aus" or abs(diff) >= 10_000:
                old_equity_str = money_formatter(old_equity)
                new_equity_str = money_formatter(new_equity)
                diff_str = f"{'+' if diff > 0 else ''}{money_formatter(diff)}"
            else:
                old_equity_str = f"{int(round(old_equity)):,}원"
                new_equity_str = f"{int(round(new_equity)):,}원"
                diff_int = int(round(diff))
                diff_str = (
                    f"{'+' if diff_int > 0 else ''}{diff_int:,}원"
                    if diff_int != 0
                    else f"{diff:+.2f}원"
                )

            equity_change_message = (
                f"{change_label}: {old_equity_str} => {new_equity_str} ({diff_str})"
            )
            message += f" | {equity_change_message}"

        send_log_to_slack(message)
    except Exception as e:
        logging.error(
            f"Failed to send summary notification for {country}/{account}: {e}", exc_info=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="포트폴리오 매매 신호를 계산합니다.")
    parser.add_argument("country", choices=["kor", "aus", "coin"], help="국가 코드")
    parser.add_argument("--account", required=True, help="계좌 코드 (예: m1, a1, b1)")
    parser.add_argument("--date", default=None, help="기준 날짜 (YYYY-MM-DD). 미지정 시 자동 결정")
    args = parser.parse_args()

    import time

    start_time = time.time()

    # 알림에 사용할 이전 평가금액을 미리 가져옵니다.
    old_snapshot = get_portfolio_snapshot(args.country, account=args.account)
    old_equity = float(old_snapshot.get("total_equity", 0.0)) if old_snapshot else 0.0

    report_date = main(country=args.country, account=args.account, date_str=args.date)

    # 요약 알림 전송
    if report_date:
        duration = time.time() - start_time
        send_summary_notification(args.country, args.account, report_date, duration, old_equity)
