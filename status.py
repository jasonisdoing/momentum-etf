import os
import warnings
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logic import settings
from pymongo import DESCENDING

try:
    import pytz
except ImportError:
    pytz = None

# New structure imports
from utils.db_manager import get_db_connection, get_portfolio_snapshot, get_previous_portfolio_snapshot, get_app_settings, get_trades_on_date, get_stocks, get_common_settings
from utils.data_loader import (
    fetch_ohlcv,
    format_aus_ticker_for_yfinance,
    get_trading_days,
)
from utils.report import format_aud_money, format_kr_money, render_table_eaw, format_aud_price

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

try:
    from pykrx import stock as _stock
except Exception:
    _stock = None

try:
    import yfinance as yf
except ImportError:
    yf = None

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
        regime_ticker, country="kor", months_range=[required_months, 0] # country doesn't matter for index
    )
    # 만약 데이터가 부족하면, 기간을 늘려 한 번 더 시도합니다.
    if (df_regime is None or df_regime.empty or len(df_regime) < regime_ma_period):
        df_regime = fetch_ohlcv(
            regime_ticker, country="kor", months_range=[required_months * 2, 0]
        )

    if df_regime is None or df_regime.empty or len(df_regime) < regime_ma_period:
        return '<span style="color:grey">시장 상태: 데이터 부족</span>'

    # 지표 계산
    df_regime["MA"] = df_regime["Close"].rolling(window=regime_ma_period).mean()
    df_regime.dropna(subset=['MA'], inplace=True)

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


def get_benchmark_status_string(country: str) -> Optional[str]:
    """
    포트폴리오의 누적 수익률을 벤치마크와 비교하여 초과 성과를 HTML 문자열로 반환합니다.
    가상화폐의 경우, 여러 벤치마크와 비교할 수 있습니다.
    """
    # 1. 설정 로드
    app_settings = get_app_settings(country)
    if not app_settings or "initial_capital" not in app_settings or "initial_date" not in app_settings:
        return None

    initial_capital = float(app_settings["initial_capital"])
    initial_date = pd.to_datetime(app_settings["initial_date"])

    if initial_capital <= 0:
        return None

    # 2. 최신 포트폴리오 스냅샷 로드
    portfolio_data = get_portfolio_snapshot(country)  # date_str=None to get latest
    if not portfolio_data:
        return None

    current_equity = float(portfolio_data.get("total_equity", 0.0))
    base_date = pd.to_datetime(portfolio_data["date"]).normalize()

    # 3. 포트폴리오 누적 수익률 계산
    portfolio_cum_ret_pct = ((current_equity / initial_capital) - 1.0) * 100.0

    def _calculate_and_format_single_benchmark(benchmark_ticker: str, benchmark_country: str, display_name_override: Optional[str] = None) -> str:
        """단일 벤치마크와의 비교 문자열을 생성하는 헬퍼 함수입니다."""
        df_benchmark = fetch_ohlcv(
            benchmark_ticker,
            country=benchmark_country,
            date_range=[initial_date.strftime("%Y-%m-%d"), base_date.strftime("%Y-%m-%d")],
        )

        if df_benchmark is None or df_benchmark.empty:
            return f'<span style="color:grey">벤치마크({benchmark_ticker}) 데이터 조회 실패</span>'

        start_prices = df_benchmark[df_benchmark.index >= initial_date]["Close"]
        if start_prices.empty:
            return f'<span style="color:grey">벤치마크 시작 가격 조회 실패</span>'
        benchmark_start_price = start_prices.iloc[0]

        end_prices = df_benchmark[df_benchmark.index <= base_date]["Close"]
        if end_prices.empty:
            return f'<span style="color:grey">벤치마크 종료 가격 조회 실패</span>'
        benchmark_end_price = end_prices.iloc[-1]

        if pd.isna(benchmark_start_price) or pd.isna(benchmark_end_price) or benchmark_start_price <= 0:
            return '<span style="color:grey">벤치마크 가격 정보 오류</span>'

        benchmark_cum_ret_pct = ((benchmark_end_price / benchmark_start_price) - 1.0) * 100.0

        excess_return_pct = portfolio_cum_ret_pct - benchmark_cum_ret_pct
        color = "red" if excess_return_pct > 0 else "blue" if excess_return_pct < 0 else "black"

        from utils.data_loader import fetch_yfinance_name, fetch_pykrx_name

        benchmark_name = display_name_override
        if not benchmark_name:
            if benchmark_country == 'kor' and _stock:
                benchmark_name = fetch_pykrx_name(benchmark_ticker)
            elif benchmark_country == 'aus':
                benchmark_name = fetch_yfinance_name(benchmark_ticker)
            elif benchmark_country == 'coin':
                benchmark_name = benchmark_ticker.upper()

        benchmark_display_name = f" vs {benchmark_name}" if benchmark_name else f" vs {benchmark_ticker}"
        return f'초과성과: <span style="color:{color}">{excess_return_pct:+.2f}%</span>{benchmark_display_name}'

    if country == "coin":
        # 가상화폐의 경우, 두 개의 벤치마크와 비교합니다.
        benchmarks_to_compare = [
            {"ticker": "379800", "country": "kor", "name": "KODEX 미국S&P500"},
            {"ticker": "BTC", "country": "coin", "name": "BTC"},
        ]
        
        results = []
        for bm in benchmarks_to_compare:
            results.append(_calculate_and_format_single_benchmark(bm["ticker"], bm["country"], bm["name"]))
        
        return "<br>".join(results)
    else:
        # 기존 로직 (한국/호주)
        try:
            benchmark_ticker = settings.BENCHMARK_TICKERS.get(country)
        except AttributeError:
            print("오류: BENCHMARK_TICKERS 설정이 logic/settings.py 에 정의되어야 합니다.")
            return None
        if not benchmark_ticker:
            return None
        
        return _calculate_and_format_single_benchmark(benchmark_ticker, country)

def is_market_open(country: str = "kor") -> bool:
    """
    지정된 국가의 주식 시장이 현재 개장 시간인지 확인합니다.
    정확한 공휴일은 반영하지 않으며, 시간과 요일만으로 판단합니다.
    """
    if not pytz:
        return False  # pytz 없으면 안전하게 False 반환

    timezones = {"kor": "Asia/Seoul", "aus": "Australia/Sydney"}
    market_hours = {
        "kor": (datetime.strptime("09:00", "%H:%M").time(), datetime.strptime("15:30", "%H:%M").time()),
        "aus": (datetime.strptime("10:00", "%H:%M").time(), datetime.strptime("16:00", "%H:%M").time()),
    }

    tz_str = timezones.get(country)
    if not tz_str:
        return False

    try:
        local_tz = pytz.timezone(tz_str)
        now_local = datetime.now(local_tz)

        # 주말(토, 일) 확인
        if now_local.weekday() >= 5:
            return False

        # 개장 시간 확인
        market_open_time, market_close_time = market_hours[country]
        return market_open_time <= now_local.time() <= market_close_time
    except Exception:
        return False  # 오류 발생 시 안전하게 False 반환


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

    for tkr in held_tickers:
        try:
            # 해당 티커의 모든 거래를 날짜 내림차순, 그리고 같은 날짜 내에서는 생성 순서(_id) 내림차순으로 가져옵니다.
            # 이를 통해 동일한 날짜에 발생한 거래의 순서를 정확히 반영하여 연속 보유 기간을 계산합니다.
            trades = list(db.trades.find(
                {"country": country, "ticker": tkr, "date": {"$lte": as_of_date}},
                sort=[("date", DESCENDING), ("_id", DESCENDING)]
            ))

            if not trades:
                continue

            # 현재 보유 수량을 계산합니다.
            current_shares = 0
            for trade in reversed(trades): # 시간순으로 반복
                if trade['action'] == 'BUY':
                    current_shares += trade['shares']
                elif trade['action'] == 'SELL':
                    current_shares -= trade['shares']
            
            # 현재부터 과거로 시간을 거슬러 올라가며 확인합니다.
            buy_date = None
            for trade in trades: # 날짜 내림차순으로 정렬되어 있음
                if current_shares <= 0:
                    break # 현재 보유 기간의 시작점을 지났음
                
                buy_date = trade['date'] # 잠재적인 매수 시작일
                if trade['action'] == 'BUY':
                    current_shares -= trade['shares']
                elif trade['action'] == 'SELL':
                    current_shares += trade['shares']
            
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
    병렬 처리를 위해 사용됩니다.
    """
    # Unpack arguments
    tkr, country, required_months, base_date, ma_period, atr_period_norm, df_full = args
    from utils.indicators import calculate_atr

    if df_full is None:
        # df_full이 제공되지 않으면, 네트워크를 통해 데이터를 새로 조회합니다.
        df = fetch_ohlcv(
            tkr, country=country, months_range=[required_months, 0], base_date=base_date
        )
    else:
        # df_full이 제공되면, base_date까지의 데이터만 잘라서 사용합니다.
        df = df_full[df_full.index <= base_date].copy()

    if df is None or len(df) < max(ma_period, atr_period_norm):
        return tkr, None

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
        .cumsum().fillna(0).astype(int)
    )

    return tkr, {
        "df": df, "close": close, "ma": ma, "atr": atr,
        "buy_signal_days": buy_signal_days, "ma_period": ma_period,
    }


def _fetch_and_prepare_data(country: str, date_str: Optional[str], prefetched_data: Optional[Dict[str, pd.DataFrame]] = None):
    """
    주어진 종목 목록에 대해 OHLCV 데이터를 조회하고,
    신호 계산에 필요한 보조지표(이동평균, ATR 등)를 계산합니다.
    """
    # 설정을 불러옵니다.
    print(f"현황을 계산합니다.")
    app_settings = get_app_settings(country)
    if not app_settings or "ma_period_etf" not in app_settings or "ma_period_stock" not in app_settings:
        print(f"오류: '{country}' 국가의 전략 파라미터(MA 기간)가 설정되지 않았습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요.")
        return None, None, None, None, None, None, None, None

    try:
        ma_period_etf = int(app_settings["ma_period_etf"])
        ma_period_stock = int(app_settings["ma_period_stock"])
    except (ValueError, TypeError):
        print(f"오류: '{country}' 국가의 MA 기간 설정이 올바르지 않습니다.")
        return None, None, None, None, None, None, None, None

    portfolio_data = get_portfolio_snapshot(country, date_str)
    if not portfolio_data:
        print(
            f"오류: '{country}' 국가의 포트폴리오 스냅샷을 DB에서 찾을 수 없습니다. 웹 앱의 '거래 입력' 또는 '설정' 탭을 통해 데이터를 먼저 생성해주세요."
        )
        return None, None, None, None, None, None, None, None
    try:
        # DB에서 가져온 date는 이미 datetime 객체일 수 있습니다.
        base_date = pd.to_datetime(portfolio_data["date"]).normalize()
    except (ValueError, TypeError):
        print(f"경고: 포트폴리오 스냅샷에서 날짜를 추출할 수 없습니다. 현재 날짜를 사용합니다.")
        base_date = pd.Timestamp.now().normalize()

    holdings = {
        item["ticker"]: {
            "name": item.get("name", ""),
            "shares": item.get("shares", 0),
            "avg_cost": item.get("avg_cost", 0.0),
        }
        for item in portfolio_data.get("holdings", []) if item.get("ticker")
    }

    # DB에서 종목 목록을 가져와 전체 유니버스를 구성합니다.
    stocks_from_db = get_stocks(country)
    stock_meta = {stock['ticker']: stock for stock in stocks_from_db}
    static_pairs = [(stock['ticker'], stock['name']) for stock in stocks_from_db]
    pairs = build_pairs_with_holdings(static_pairs, holdings)

    # 국가별로 다른 포맷터 사용
    header_money_formatter = format_kr_money
    if country == "aus":
        # 호주: 가격은 AUD, 금액은 KRW로 표시
        price_formatter = format_aud_price
        money_formatter = format_aud_money
        ma_formatter = format_aud_price
    else:
        # 원화(KRW) 형식으로 가격을 포맷합니다.
        money_formatter = format_kr_money
        price_formatter = lambda p: f"{int(round(p)):,}"
        ma_formatter = lambda p: f"{int(round(p)):,}원"

    # 실시간 가격 조회를 위한 헬퍼 함수
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
        return None, None, None, None, None, None, None, None
    try:
        atr_period_norm = int(common["ATR_PERIOD_FOR_NORMALIZATION"])
        regime_filter_enabled = bool(common["MARKET_REGIME_FILTER_ENABLED"])
        regime_ma_period = int(common["MARKET_REGIME_FILTER_MA_PERIOD"])
    except KeyError as e:
        print(f"오류: 공통 설정 '{e.args[0]}' 값이 없습니다.")
        return None, None, None, None, None, None, None, None
    except (ValueError, TypeError):
        print("오류: 공통 설정 값 형식이 올바르지 않습니다.")
        return None, None, None, None, None, None, None, None

    # DB에서 종목 유형(ETF/주식) 정보 가져오기
    if not stocks_from_db:
        print(f"오류: '{country}_stocks' 컬렉션에서 현황을 계산할 종목을 찾을 수 없습니다.")
        return None, None, None, None, None, None, None, None
    etf_tickers_status = {stock['ticker'] for stock in stocks_from_db if stock.get('type') == 'etf'}

    max_ma_period = max(ma_period_etf, ma_period_stock, regime_ma_period if regime_filter_enabled else 0)
    required_days = max(max_ma_period, atr_period_norm) + 5  # 버퍼 추가
    required_months = (required_days // 22) + 2

    # --- 시장 레짐 필터 데이터 로딩 ---
    regime_info = None
    if regime_filter_enabled:
        if "MARKET_REGIME_FILTER_TICKER" not in common:
            print("오류: 공통 설정에 MARKET_REGIME_FILTER_TICKER 값이 없습니다.")
            return None, None, None, None, None, None, None, None
        regime_ticker = str(common["MARKET_REGIME_FILTER_TICKER"])
        
        df_regime = fetch_ohlcv(
            regime_ticker, country=country, months_range=[required_months, 0], base_date=base_date
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

    # --- 병렬 데이터 로딩 및 지표 계산 ---
    tasks = []
    for tkr, _ in pairs:
        ma_period = ma_period_etf if tkr in etf_tickers_status else ma_period_stock
        df_full = prefetched_data.get(tkr) if prefetched_data else None
        tasks.append((tkr, country, required_months, base_date, ma_period, atr_period_norm, df_full))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_load_and_prepare_ticker_data, task) for task in tasks]
        
        desc = "과거 데이터 처리" if prefetched_data else "종목 데이터 로딩"
        for future in tqdm(as_completed(futures), total=len(tasks), desc=desc):
            tkr, result = future.result()
            if not result:
                continue
    
            realtime_price = _fetch_realtime_price(tkr) if market_is_open else None
            c0 = float(realtime_price) if realtime_price else float(result["close"].iloc[-1])
            if pd.isna(c0): continue
    
            prev_close = float(result["close"].iloc[-2]) if len(result["close"]) >= 2 and pd.notna(result["close"].iloc[-2]) else 0.0
            m = result["ma"].iloc[-1]
            a = result["atr"].iloc[-1]
    
            ma_score = (c0 - m) / a if pd.notna(m) and pd.notna(a) and a > 0 else 0.0
            buy_signal_days_today = result["buy_signal_days"].iloc[-1] if not result["buy_signal_days"].empty else 0
    
            sh = float((holdings.get(tkr) or {}).get("shares") or 0.0)
            ac = float((holdings.get(tkr) or {}).get("avg_cost") or 0.0)
            total_holdings_value += sh * c0
            datestamps.append(result["df"].index[-1])
    
            data_by_tkr[tkr] = {
                "price": c0, "prev_close": prev_close, "s1": m, "s2": result["ma_period"],
                "score": ma_score, "filter": buy_signal_days_today,
                "shares": sh, "avg_cost": ac, "df": result["df"]
            }
    
    return portfolio_data, data_by_tkr, total_holdings_value, datestamps, pairs, base_date, regime_info, stock_meta

def _build_header_line(country, portfolio_data, current_equity, total_holdings_value, data_by_tkr, base_date):
    """리포트의 헤더 라인을 생성합니다."""
    # 국가별 포맷터 설정
    money_formatter = format_kr_money if country != 'aus' else format_aud_money

    # 보유 종목 수
    held_count = sum(1 for v in portfolio_data.get("holdings", []) if float(v.get("shares", 0)) > 0)

    # 해외 주식 가치 포함
    total_holdings = total_holdings_value
    if country == "aus" and portfolio_data.get("international_shares"):
        total_holdings += portfolio_data["international_shares"].get("value", 0.0)

    # 현금
    total_cash = float(current_equity) - float(total_holdings)

    # 누적 수익률 및 TopN
    app_settings = get_app_settings(country)
    initial_capital_local = float(app_settings.get("initial_capital", 0)) if app_settings else 0.0
    cum_ret_pct = ((current_equity / initial_capital_local) - 1.0) * 100.0 if initial_capital_local > 0 else 0.0
    portfolio_topn = app_settings.get("portfolio_topn", 0) if app_settings else 0

    # Determine trading-calendar-based label/date via pykrx
    ref_ticker_for_cal = next(iter(data_by_tkr.keys())) if data_by_tkr else None

    def get_next_trading_day(start_date: pd.Timestamp, ref_ticker: str) -> pd.Timestamp:
        """주어진 날짜 또는 그 이후의 가장 가까운 거래일을 효율적으로 찾습니다."""
        if country == "kor":
            if _stock is None or ref_ticker is None:
                return start_date  # pykrx 사용 불가 시, 입력일을 그대로 반환
            try:
                # 앞으로 2주간의 데이터를 한 번에 조회하여 가장 빠른 거래일을 찾습니다.
                from_date_str = start_date.strftime("%Y%m%d")
                to_date_str = (start_date + pd.Timedelta(days=14)).strftime("%Y%m%d")
                df = _stock.get_market_ohlcv_by_date(from_date_str, to_date_str, ref_ticker)
                if not df.empty:
                    return df.index[0]
            except Exception:
                pass
            return start_date  # 조회 실패 시, 입력일을 그대로 반환
        elif country == "aus":
            if yf is None or ref_ticker is None:
                return start_date
            ticker_yf = format_aus_ticker_for_yfinance(ref_ticker)
            # yfinance는 주말/공휴일을 자동으로 건너뛰므로, 하루씩 더해가며 확인
            current_date = start_date
            for _ in range(14):
                df = yf.download(
                    ticker_yf, start=current_date, end=current_date + pd.Timedelta(days=1), progress=False, auto_adjust=True
                )
                if not df.empty:
                    return current_date
                current_date += pd.Timedelta(days=1)
        return start_date

    today_cal = pd.Timestamp.now().normalize()

    # The date for calculation and display is the one from the file
    label_date = base_date

    # Set the label based on whether the portfolio date is today or in the past
    if base_date.date() == today_cal.date():
        # If it's today, check if it's a trading day to decide between "오늘" and "다음 거래일"
        next_trading_day = get_next_trading_day(base_date, ref_ticker_for_cal)
        if next_trading_day.date() == base_date.date():
            day_label = "오늘"
        else:
            day_label = "다음 거래일"
            label_date = next_trading_day
    else:
        day_label = "기준일"

    # 일간 수익률
    prev_snapshot = get_previous_portfolio_snapshot(country, base_date)
    prev_equity = float(prev_snapshot.get("total_equity", 0.0)) if prev_snapshot else None
    day_ret_pct = ((current_equity / prev_equity) - 1.0) * 100.0 if prev_equity and prev_equity > 0 else 0.0
    day_profit_loss = current_equity - prev_equity if prev_equity else 0.0

    # 평가 수익률
    total_acquisition_cost = sum(d['shares'] * d['avg_cost'] for d in data_by_tkr.values() if d['shares'] > 0)
    eval_ret_pct = ((total_holdings_value / total_acquisition_cost) - 1.0) * 100.0 if total_acquisition_cost > 0 else 0.0
    eval_profit_loss = total_holdings_value - total_acquisition_cost

    # 헤더 문자열 생성
    equity_str = money_formatter(current_equity)
    holdings_str = money_formatter(total_holdings)
    cash_str = money_formatter(total_cash)
    day_ret_str = _format_return_for_header("일간", day_ret_pct, day_profit_loss, money_formatter)
    eval_ret_str = _format_return_for_header("평가", eval_ret_pct, eval_profit_loss, money_formatter)
    cum_ret_str = _format_return_for_header("누적", cum_ret_pct, current_equity - initial_capital_local, money_formatter)

    header_line = (
        f"보유종목: {held_count}/{portfolio_topn} | 평가금액: {equity_str} | 보유금액: {holdings_str} | "
        f"현금: {cash_str} | {day_ret_str} | {eval_ret_str} | {cum_ret_str}"
    )

    if portfolio_data.get("is_equity_stale"):
        stale_date = portfolio_data.get("equity_date")
        target_date = portfolio_data.get("date")
        weekday_map = ["월", "화", "수", "목", "금", "토", "일"]
        weekday_str = weekday_map[target_date.weekday()]
        warning_msg = f"<br><span style='color:orange;'>⚠️ {target_date.strftime('%Y년 %m월 %d일')}({weekday_str})의 평가금액이 없습니다. 최근({stale_date.strftime('%Y-%m-%d')}) 평가금액으로 현황을 계산합니다.</span>"
        header_line += warning_msg

    return header_line, label_date, day_label

def generate_status_report(
    country: str = "kor",
    date_str: Optional[str] = None,
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Optional[Tuple[str, List[str], List[List[str]]]]:
    """지정된 전략에 대한 오늘의 현황 데이터를 생성하여 반환합니다."""
    # 1. 데이터 로드 및 지표 계산
    result = _fetch_and_prepare_data(country, date_str, prefetched_data)
    if not result or not result[0]:
        return None

    portfolio_data, data_by_tkr, total_holdings_value, datestamps, pairs, base_date, regime_info, stock_meta = result
    current_equity = float(portfolio_data.get("total_equity", 0.0))
    holdings = {
        item["ticker"]: {
            "name": item.get("name", ""),
            "shares": item.get("shares", 0),
            "avg_cost": item.get("avg_cost", 0.0),
        }
        for item in portfolio_data.get("holdings", []) if item.get("ticker")
    }

    # 2. 헤더 생성
    header_line, label_date, day_label = _build_header_line(country, portfolio_data, current_equity, total_holdings_value, data_by_tkr, base_date)

    # 3. 보유 기간 및 고점 대비 하락률 계산
    held_tickers = [tkr for tkr, v in holdings.items() if float((v or {}).get("shares") or 0.0) > 0]
    consecutive_holding_info = calculate_consecutive_holding_info(held_tickers, country, base_date)
    for tkr, d in data_by_tkr.items():
        if float(d.get("shares", 0.0)) > 0:
            buy_date = consecutive_holding_info.get(tkr, {}).get("buy_date")
            if buy_date:
                df_holding_period = d["df"][buy_date:]
                if not df_holding_period.empty:
                    peak_high = df_holding_period["High"].max()
                    current_price = d["price"]
                    if pd.notna(peak_high) and peak_high > 0 and pd.notna(current_price):
                        d["drawdown_from_peak"] = ((current_price / peak_high) - 1.0) * 100.0

    app_settings = get_app_settings(country)
    if not app_settings or "portfolio_topn" not in app_settings:
        print(f"오류: '{country}' 국가의 최대 보유 종목 수(portfolio_topn)가 설정되지 않았습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요.")
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
    min_pos = 1.0 / denom

    held_count = sum(1 for v in holdings.values() if float((v or {}).get("shares") or 0.0) > 0)
    total_cash = float(current_equity) - float(total_holdings_value)

    # 4. 초기 매매 결정 생성
    decisions = []
    # 국가별 포맷터 설정
    if country == "aus":
        price_formatter = format_aud_price
        money_formatter = format_aud_money
        ma_formatter = format_aud_price
    else: # kor
        money_formatter = format_kr_money
        price_formatter = lambda p: f"{int(round(p)):,}"
        ma_formatter = lambda p: f"{int(round(p)):,}원"

    def format_shares(quantity):
        if country == 'coin':
            # 소수점 8자리까지 표시하되, 불필요한 0은 제거
            return f"{quantity:,.8f}".rstrip('0').rstrip('.')
        else:
            return f"{int(quantity):,d}"

    # 거래일 계산을 위한 참조 티커를 설정합니다.
    ref_ticker_for_cal = next(iter(data_by_tkr.keys())) if data_by_tkr else None

    for tkr, name in pairs:
        d = data_by_tkr.get(tkr)
        if not d:
            continue
        price = d["price"]
        score = d.get("score", 0.0)
        sh = float(d["shares"])
        ac = float(d.get("avg_cost") or 0.0)

        # 자동 계산된 보유종목의 매수일과 보유일
        buy_signal = False

        consecutive_info = consecutive_holding_info.get(tkr)
        buy_date = consecutive_info.get("buy_date") if consecutive_info else None
        holding_days = 0

        if buy_date:
            # label_date는 naive timestamp이므로, buy_date도 naive로 만듭니다.
            if hasattr(buy_date, 'tzinfo') and buy_date.tzinfo is not None:
                buy_date = buy_date.tz_localize(None)
            buy_date = pd.to_datetime(buy_date).normalize()

        if sh > 0 and buy_date and buy_date <= label_date:
            try:
                # 거래일 기준으로 보유일수 계산 (캐시된 함수 사용)
                trading_days_in_period = get_trading_days(
                    buy_date.strftime("%Y-%m-%d"),
                    label_date.strftime("%Y-%m-%d"),
                    country
                )
                holding_days = len(trading_days_in_period)
            except Exception as e:
                print(f"경고: 보유일 계산 중 오류 발생 ({tkr}): {e}. 달력일 기준으로 대체합니다.")
                # 거래일 계산 실패 시, 달력일 기준으로 계산
                holding_days = (label_date - buy_date).days + 1

        state = "HOLD" if sh > 0 else "WAIT"
        phrase = ""
        qty = 0
        notional = 0.0
        # Current holding return
        hold_ret = ((price / ac) - 1.0) * 100.0 if (sh > 0 and ac > 0 and pd.notna(price)) else None
        # TRIM if exceeding cap
        if sh > 0:
            if stop_loss is not None and ac > 0 and hold_ret <= float(stop_loss):
                state = "CUT_STOPLOSS"  # 결정 코드
                qty = sh
                notional = qty * price
                prof = (price - ac) * qty if ac > 0 else 0.0
                phrase = f"가격기반손절 {format_shares(qty)}주 @ {price_formatter(price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"

        # --- 전략별 매수/매도 로직 ---
        if state == "HOLD":  # 아직 매도 결정이 내려지지 않은 경우
            price, ma, period = d["price"], d["s1"], d["s2"]
            if sh > 0 and not pd.isna(price) and not pd.isna(ma) and price < ma:
                state = "SELL_TREND"  # 결정 코드
                qty = sh
                notional = qty * price
                prof = (price - ac) * qty if ac > 0 else 0.0
                tag = "추세이탈(이익)" if hold_ret >= 0 else "추세이탈(손실)"
                phrase = f"{tag} {format_shares(qty)}주 @ {price_formatter(price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"

        elif state == "WAIT":  # 아직 보유하지 않은 경우
            price, ma, period = d["price"], d["s1"], d["s2"]
            buy_signal_days_today = d["filter"]
            if buy_signal_days_today > 0:
                buy_signal = True
                phrase = f"추세진입 ({buy_signal_days_today}일째)"

        amount = sh * price if pd.notna(price) else 0.0
        # 일간 수익률 계산
        prev_close = d.get("prev_close")
        day_ret = 0.0
        day_ret_str = "-"
        if prev_close is not None and prev_close > 0 and pd.notna(price):
            day_ret = ((price / prev_close) - 1.0) * 100.0
            day_ret_str = f"{day_ret:+.1f}%"

        # 테이블 출력용 신호 포맷팅
        s1_str = ma_formatter(d["s1"]) if not pd.isna(d["s1"]) else "-"  # 이평선(값)
        drawdown_val = d.get("drawdown_from_peak")
        s2_str = f"{drawdown_val:.1f}%" if drawdown_val is not None else "-"  # 고점대비
        filter_str = f"{d['filter']}일" if d.get("filter") is not None else "-"

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
            f"{d.get('drawdown_from_peak'):.1f}%" if d.get("drawdown_from_peak") is not None else "-",  # 고점대비
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
        print(f"오류: '{country}' 국가의 전략 파라미터가 DB에 없습니다. 웹 앱의 '설정' 탭에서 값을 저장해주세요.")
        return None
    # 교체 매매 사용 여부 (bool)
    if "replace_weaker_stock" not in app_settings_for_country:
        print(f"오류: '{country}' 국가의 설정에 'replace_weaker_stock'가 없습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요.")
        return None
    try:
        replace_weaker_stock = bool(app_settings_for_country["replace_weaker_stock"])  
    except Exception:
        print(f"오류: '{country}' 국가의 'replace_weaker_stock' 값이 올바르지 않습니다.")
        return None
    # 하루 최대 교체 수 (int)
    if "max_replacements_per_day" not in app_settings_for_country:
        print(f"오류: '{country}' 국가의 설정에 'max_replacements_per_day'가 없습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요.")
        return None
    try:
        max_replacements_per_day = int(app_settings_for_country["max_replacements_per_day"])  
    except Exception:
        print(f"오류: '{country}' 국가의 'max_replacements_per_day' 값이 올바르지 않습니다.")
        return None
    if "replace_threshold" not in app_settings_for_country:
        print(f"오류: '{country}' 국가의 교체 매매 임계값(replace_threshold)이 DB에 없습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요.")
        return None
    try:
        replace_threshold = float(app_settings_for_country["replace_threshold"])  
    except (ValueError, TypeError):
        print(f"오류: '{country}' 국가의 교체 매매 임계값(replace_threshold) 값이 올바르지 않습니다.")
        return None
    slots_to_fill = denom - held_count
    if slots_to_fill > 0:
        # 매수 후보들을 점수 순으로 정렬
        buy_candidates = sorted(
            [a for a in decisions if a.get("buy_signal")],
            key=lambda x: x["score"],
            reverse=True,
        )
        
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
                
                if country == 'coin':
                    req_qty = min_val / price
                else:
                    from math import ceil
                    req_qty = int(ceil(min_val / price))
                buy_notional = req_qty * price

                if req_qty > 0 and buy_notional <= available_cash:
                    # 매수 결정
                    cand["state"] = "BUY"
                    cand["row"][3] = "BUY"
                    buy_phrase = f"매수 {format_shares(req_qty)}주 @ {price_formatter(price)} ({money_formatter(buy_notional)})"
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
            [a for a in decisions if a["state"] == "HOLD"], key=lambda x: x["score"]
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
                weakest_held["row"][3] = "SELL_REPLACE"
                weakest_held["row"][-1] = sell_phrase

                # 2. 새로 편입될 종목(매수)의 상태 업데이트
                best_new["state"] = "BUY_REPLACE"
                best_new["row"][3] = "BUY_REPLACE"

                # 매수 수량 및 금액 계산
                sell_value = weakest_held["weight"] / 100.0 * current_equity
                buy_price = float(data_by_tkr.get(best_new["tkr"], {}).get("price", 0))
                if buy_price > 0:
                    if country == 'coin':
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
    trades_on_base_date = get_trades_on_date(country, base_date)
    executed_buys_today = {trade['ticker'] for trade in trades_on_base_date if trade['action'] == 'BUY'}
    executed_sells_today = {trade['ticker'] for trade in trades_on_base_date if trade['action'] == 'SELL'}
    # 기준일에 실행된 거래가 있다면, 현황 목록에 '완료' 상태를 표시합니다.
    for decision in decisions:
        tkr = decision['tkr']
        
        # 오늘 매수했고, 현재 보유 중인 종목
        if decision['state'] == 'HOLD' and tkr in executed_buys_today:
            # 이 종목이 오늘 신규 매수되었음을 표시
            decision['row'][-1] = "✅ 완료: 신규 매수"
            
        # 오늘 매도했고, 현재는 미보유(WAIT) 상태인 종목
        elif decision['state'] == 'WAIT' and tkr in executed_sells_today:
            # 이 종목이 오늘 매도되었음을 표시. 기존의 '추세진입' 등 메시지를 덮어씁니다.
            decision['state'] = "SOLD" # 정렬 및 표시를 위한 새로운 상태
            decision['row'][3] = "SOLD"
            decision['row'][-1] = "✅ 완료: 매도"

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
        "매수일",
        "보유",
        "현재가",
        "일간수익률",
        "보유수량",
        "금액",
        "누적수익률",
        "비중",
    ]
    headers.extend(["고점대비", "점수", "지속", "문구"])

    return (header_line, headers, rows_sorted)


def main(country: str = "kor", date_str: Optional[str] = None):
    """CLI에서 오늘의 현황을 실행하고 결과를 출력/저장합니다."""
    result = generate_status_report(country, date_str)

    if result:
        header_line, headers, rows_sorted = result

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

            display_rows.append(display_row)

        aligns = [
            "right",  # #
            "right",  # 티커
            "center", # 상태
            "left",   # 매수일
            "right",  # 보유
            "right",  # 현재가
            "right",  # 일간수익률
            "right",  # 보유수량
            "right",  # 금액
            "right",  # 누적수익률
            "right",  # 비중
            "right",  # 고점대비
            "right",  # 점수
            "center", # 지속
            "center", # 국가
            "left",   # 업종
            "left",   # 문구
        ]

        table_lines = render_table_eaw(headers, display_rows, aligns=aligns)

        print("\n" + header_line)
        print("\n".join(table_lines))


if __name__ == "__main__":
    main(country="kor")
