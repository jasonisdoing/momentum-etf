import glob
import importlib
import json
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

import settings as global_settings

try:
    import pytz
except ImportError:
    pytz = None

# New structure imports
from utils.data_loader import (
    fetch_exchange_rate,
    fetch_naver_realtime_price,
    fetch_ohlcv,
    format_aus_ticker_for_yfinance,
    read_tickers_file,
)
from utils.indicators import supertrend_direction
from utils.report import format_aud_money, format_kr_money, render_table_eaw

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

try:
    from pykrx import stock as _stock
except Exception:
    _stock = None

try:
    import yfinance as yf
except ImportError:
    yf = None


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


def load_portfolio_data(
    portfolio_path: Optional[str] = None, data_dir: str = "data"
) -> Optional[Dict]:
    """
    지정된 포트폴리오 스냅샷 파일 또는 최신 파일을 로드합니다.
    파일을 성공적으로 로드하면 'total_equity', 'holdings' 등이 포함된 딕셔너리를 반환합니다.
    """
    filepath_to_load = None
    if portfolio_path:
        if os.path.exists(portfolio_path):
            filepath_to_load = portfolio_path
        else:
            print(f"경고: 지정된 포트폴리오 파일 '{portfolio_path}'을(를) 찾을 수 없습니다.")
            return None
    else:
        # 최신 파일 찾기
        try:
            portfolio_files = glob.glob(os.path.join(data_dir, "portfolio_*.json"))
            if not portfolio_files:
                return None

            latest_date = None
            for f_path in portfolio_files:
                try:
                    fname = os.path.basename(f_path)
                    date_str = fname.replace("portfolio_", "").replace(".json", "")
                    current_date = pd.to_datetime(date_str)
                    if latest_date is None or current_date > latest_date:
                        latest_date = current_date
                        filepath_to_load = f_path
                except ValueError:
                    continue
        except Exception:
            return None

    if not filepath_to_load:
        return None

    try:
        with open(filepath_to_load, "r", encoding="utf-8") as f:
            data = json.load(f)

        holdings_list = data.get("holdings", [])
        holdings_dict = {
            item["ticker"]: {
                "name": item.get("name", ""),
                "shares": item.get("shares", 0),
                "avg_cost": item.get("avg_cost", 0.0),
            }
            for item in holdings_list
            if item.get("ticker")
        }

        result = {
            "date": data.get("date"),
            "total_equity": data.get("total_equity"),
            "holdings": holdings_dict,
            "filepath": filepath_to_load,
        }
        if "international_shares" in data:
            result["international_shares"] = data["international_shares"]
        return result
    except Exception as e:
        print(f"오류: 포트폴리오 파일 '{filepath_to_load}' 로드 중 오류 발생: {e}")
        return None


def calculate_consecutive_holding_info(
    held_tickers: List[str], data_dir: str = "data"
) -> Dict[str, Dict]:
    """
    과거 포트폴리오 파일들을 스캔하여 각 티커의 연속 보유 기간 정보를 계산합니다.
    'buy_date' (연속 보유 시작일)을 포함한 딕셔너리를 반환합니다.
    """
    holding_info = {tkr: {"buy_date": None} for tkr in held_tickers}
    if not held_tickers:
        return holding_info

    try:
        portfolio_files = sorted(
            glob.glob(os.path.join(data_dir, "portfolio_*.json")), reverse=True
        )
        if not portfolio_files:
            return holding_info

        # 날짜별로 정렬된 파일 목록을 생성 (이미 정렬했지만, 날짜 파싱을 위해 필요)
        # glob.glob의 순서는 보장되지 않으므로, 파일명에서 날짜를 추출하여 정렬합니다.
        parsed_files = []
        for f_path in portfolio_files:
            try:
                fname = os.path.basename(f_path)
                date_str = fname.replace("portfolio_", "").replace(".json", "")
                file_date = pd.to_datetime(date_str)
                parsed_files.append((file_date, f_path))
            except ValueError:
                continue

        if not parsed_files:
            return holding_info

        sorted_files = sorted(parsed_files, key=lambda x: x[0], reverse=True)

        for tkr in held_tickers:
            consecutive_buy_date = None
            for file_date, f_path in sorted_files:
                with open(f_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                holdings_in_file = {
                    item.get("ticker")
                    for item in data.get("holdings", [])
                    if int(item.get("shares", 0)) > 0
                }
                if tkr in holdings_in_file:
                    consecutive_buy_date = file_date
                else:
                    break  # 연속 보유가 끊겼으므로 중단
            if consecutive_buy_date:
                holding_info[tkr]["buy_date"] = consecutive_buy_date
    except Exception as e:
        print(f"-> 경고: 보유일 자동 계산 중 오류 발생: {e}")
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


def generate_action_plan(
    strategy_name: str,
    country: str = "kor",
    portfolio_path: Optional[str] = None,
) -> Optional[Tuple[str, List[str], List[List[str]]]]:
    """지정된 전략에 대한 오늘의 액션 플랜 데이터를 생성하여 반환합니다."""
    print(f"'{strategy_name}' 전략을 사용하여 오늘의 액션 플랜을 생성합니다.")

    # 환율 정보 조회
    # 포트폴리오 파일의 날짜를 기준으로 환율을 조회합니다.
    portfolio_data_for_date = load_portfolio_data(portfolio_path, data_dir=f"data/{country}")
    if not portfolio_data_for_date:
        print(
            f"오류: 포트폴리오 파일(portfolio_*.json)을 찾을 수 없습니다. data/{country}/ 폴더를 확인해주세요."
        )
        return

    try:
        date_str_from_filename = os.path.basename(portfolio_data_for_date["filepath"]).replace("portfolio_", "").replace(".json", "")
        base_date = pd.to_datetime(date_str_from_filename).normalize()
    except (ValueError, TypeError):
        print(f"경고: 포트폴리오 파일명에서 날짜를 추출할 수 없습니다. 현재 날짜를 사용합니다.")
        base_date = pd.Timestamp.now().normalize()

    aud_krw_rate = None
    if country == "aus":
        aud_krw_rate = fetch_exchange_rate("AUDKRW=X", as_of_date=base_date.strftime("%Y-%m-%d"))
        if aud_krw_rate:
            print(f"-> {base_date.strftime('%Y-%m-%d')} 기준 AUD/KRW 환율 적용: {aud_krw_rate:.2f}")
        else:
            # 환율 조회 실패 시 기본값 사용 및 경고
            aud_krw_rate = 900.0  # Fallback rate
            print(f"-> 경고: AUD/KRW 환율 조회 실패. 기본값 {aud_krw_rate}을(를) 사용합니다.")

    # 국가별로 다른 포맷터 사용
    header_money_formatter = format_kr_money
    if country == "aus":
        # 호주: 가격은 AUD, 금액은 KRW로 표시
        price_formatter = format_aud_money
        money_formatter = lambda x: format_kr_money(x * aud_krw_rate)
        ma_formatter = format_aud_money  # 이동평균선 포맷터
    else:
        # 원화(KRW) 형식으로 가격을 포맷합니다.
        money_formatter = format_kr_money
        price_formatter = lambda p: f"{int(round(p)):,}"
        ma_formatter = lambda p: f"{int(round(p)):,}원"

    # 실시간 가격 조회는 포트폴리오 기준일이 오늘일 경우에만 시도합니다.
    today_cal = pd.Timestamp.now().normalize()
    market_is_open = is_market_open(country) and base_date.date() == today_cal.date()
    if market_is_open and base_date.date() == today_cal.date():
        if country == "kor":
            print("-> 장중입니다. 네이버 금융에서 실시간 시세를 가져옵니다 (비공식, 지연 가능).")

    # 전략별 설정 로드
    try:
        strategy_settings = importlib.import_module(f"logics.{strategy_name}.settings")
        print(f"-> '{strategy_name}' 전략의 전용 설정을 로드했습니다.")
    except ImportError:
        print(
            f"-> 경고: '{strategy_name}' 전략의 전용 설정 파일(settings.py)을 "
            "찾을 수 없습니다. 전역 설정을 사용합니다."
        )
        strategy_settings = global_settings

    # 교체 매매 관련 설정 로드 (donchian 전략용)
    replace_weaker_stock = False
    replace_threshold = 0.0
    if strategy_name == "donchian":
        replace_weaker_stock = bool(
            getattr(strategy_settings, "DONCHIAN_REPLACE_WEAKER_STOCK", False)
        )
        replace_threshold = float(
            getattr(strategy_settings, "DONCHIAN_REPLACE_SCORE_THRESHOLD", 0.0)
        )
        atr_period = int(
            getattr(strategy_settings, "DONCHIAN_ATR_PERIOD_FOR_NORMALIZATION", 20)
        )
        max_replacements_per_day = int(
            getattr(strategy_settings, "DONCHIAN_MAX_REPLACEMENTS_PER_DAY", 1)
        )

    # Load initial state from portfolio file.
    data_dir = f"data/{country}"
    portfolio_data = portfolio_data_for_date  # 위에서 이미 로드함

    print(
        f"포트폴리오 파일 '{os.path.basename(portfolio_data['filepath'])}'을(를) 기준으로 오늘의 액션을 계산합니다."
    )
    holdings = portfolio_data.get("holdings", {})
    current_equity = float(portfolio_data.get("total_equity", 0.0))
    international_shares_data = None
    if country == "aus":
        international_shares_data = portfolio_data.get("international_shares")
        if international_shares_data:
            print("-> 해외 주식(International Shares) 정보를 포트폴리오에 포함합니다.")

    # 보유 기간 자동 계산
    held_tickers = [tkr for tkr, v in holdings.items() if int((v or {}).get("shares") or 0) > 0]
    consecutive_holding_info = calculate_consecutive_holding_info(held_tickers, data_dir=data_dir)
    print("-> 과거 포트폴리오 파일을 기반으로 보유 기간을 자동 계산했습니다.")

    # 티커 목록 결정
    print(f"\n[고정 유니버스] data/{country}/tickers_etf.txt와 tickers_stock.txt 파일의 종목을 사용합니다.")
    # tickers.txt와 현재 보유 종목을 합쳐서 전체 유니버스 구성
    etf_pairs = read_tickers_file(f"data/{country}/tickers_etf.txt", country=country)
    stock_pairs = read_tickers_file(f"data/{country}/tickers_stock.txt", country=country)
    static_pairs = etf_pairs + stock_pairs
    pairs = build_pairs_with_holdings(static_pairs, holdings)

    if not pairs:
        print("오류: 투자 대상 티커를 찾을 수 없습니다.")
        print("      data/tickers.txt 파일이 비어있거나 존재하지 않을 수 있습니다.")
        return

    # Fetch recent data for signals
    datestamps = []
    total_holdings_value = 0.0
    data_by_tkr = {}

    # --- 전략별 신호 계산 ---
    if strategy_name == "jason":
        for tkr, _ in pairs:
            df = fetch_ohlcv(tkr, country=country, months_range=[1, 0], base_date=base_date)
            if df is None or len(df) < 11:
                continue
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            # 실시간 가격 조회 시도 (장중일 경우)
            realtime_price = None
            if market_is_open:
                realtime_price = fetch_naver_realtime_price(tkr) if country == "kor" else None

            # 실시간 가격이 없으면 pykrx의 최신 종가를 사용
            c0 = realtime_price if realtime_price else close.iloc[-1]
            if pd.isna(c0):
                continue

            c0 = float(c0)

            # 전일 종가 가져오기
            prev_close = 0.0
            if len(close) >= 2:
                prev_close_val = close.iloc[-2]
                if pd.notna(prev_close_val):
                    prev_close = float(prev_close_val)
            i = len(close) - 1

            c5 = float(close.iloc[i - 5]) if i - 5 >= 0 else c0
            c10 = float(close.iloc[i - 10]) if i - 10 >= 0 else (c5 if i - 5 >= 0 else c0)
            p1 = round(((c0 / c5) - 1.0) * 100.0, 1) if c5 > 0 else 0.0
            p2 = round(((c5 / c10) - 1.0) * 100.0, 1) if c10 > 0 else 0.0
            s2 = p1 + p2
            try:
                st_dir = supertrend_direction(
                    df,
                    int(getattr(strategy_settings, "ST_ATR_PERIOD", 14)),
                    float(getattr(strategy_settings, "ST_ATR_MULTIPLIER", 3.0)),
                )
                stv = int(st_dir.iloc[-1]) if len(st_dir) > 0 else 0
            except Exception:
                stv = 0
            sh = int((holdings.get(tkr) or {}).get("shares") or 0)
            ac = float((holdings.get(tkr) or {}).get("avg_cost") or 0.0)
            total_holdings_value += sh * c0
            datestamps.append(df.index[-1])
            data_by_tkr[tkr] = {
                "price": c0,
                "prev_close": prev_close,
                "s1": p1,
                "s2": p2,
                "score": s2,
                "filter": stv,
                "shares": sh,
                "avg_cost": ac,
            }

    elif strategy_name == "seykota":
        fast_ma_period = int(getattr(strategy_settings, "SEYKOTA_FAST_MA", 50))
        slow_ma_period = int(getattr(strategy_settings, "SEYKOTA_SLOW_MA", 150))
        required_months = (slow_ma_period // 22) + 2

        for tkr, _ in pairs:
            df = fetch_ohlcv(
                tkr, country=country, months_range=[required_months, 0], base_date=base_date
            )
            if df is None or len(df) < slow_ma_period:
                continue
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            fast_ma = close.rolling(window=fast_ma_period).mean()
            slow_ma = close.rolling(window=slow_ma_period).mean()

            # 실시간 가격 조회 시도 (장중일 경우)
            realtime_price = None
            if market_is_open:
                realtime_price = fetch_naver_realtime_price(tkr) if country == "kor" else None

            # 실시간 가격이 없으면 pykrx의 최신 종가를 사용
            c0 = realtime_price if realtime_price else close.iloc[-1]
            if pd.isna(c0):
                continue
            c0 = float(c0)

            # 전일 종가 가져오기
            prev_close = 0.0
            if len(close) >= 2:
                prev_close_val = close.iloc[-2]
                if pd.notna(prev_close_val):
                    prev_close = float(prev_close_val)

            fm = fast_ma.iloc[-1]
            sm = slow_ma.iloc[-1]
            ma_score = (fm / sm - 1.0) * 100.0 if sm > 0 and not pd.isna(sm) else 0.0

            sh = int((holdings.get(tkr) or {}).get("shares") or 0)
            ac = float((holdings.get(tkr) or {}).get("avg_cost") or 0.0)
            total_holdings_value += sh * c0
            datestamps.append(df.index[-1])
            data_by_tkr[tkr] = {
                "price": c0,
                "prev_close": prev_close,
                "s1": fm,
                "s2": sm,
                "score": ma_score,
                "filter": None,
                "shares": sh,
                "avg_cost": ac,
            }
    elif strategy_name == "donchian":
        ma_period_etf = int(getattr(strategy_settings, "DONCHIAN_MA_PERIOD_FOR_ETF", 15))
        ma_period_stock = int(getattr(strategy_settings, "DONCHIAN_MA_PERIOD_FOR_STOCK", 75))
        atr_period_norm = int(
            getattr(strategy_settings, "DONCHIAN_ATR_PERIOD_FOR_NORMALIZATION", 20)
        )

        # --- 티커 유형(ETF/주식) 구분 ---
        etf_pairs_status = read_tickers_file(f"data/{country}/tickers_etf.txt", country=country)
        etf_tickers_status = {ticker for ticker, _ in etf_pairs_status}

        max_ma_period = max(ma_period_etf, ma_period_stock)
        required_days = max(max_ma_period, atr_period_norm) + 5  # 버퍼 추가
        required_months = (required_days // 22) + 2

        from utils.indicators import calculate_atr

        for tkr, _ in pairs:
            df = fetch_ohlcv(
                tkr, country=country, months_range=[required_months, 0], base_date=base_date
            )  # 더 많은 데이터 요청

            # 티커 유형에 따라 이동평균 기간 결정
            ma_period = ma_period_etf if tkr in etf_tickers_status else ma_period_stock

            if df is None or len(df) < max(ma_period, atr_period_norm):
                continue
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            ma = close.rolling(window=ma_period).mean()
            atr = calculate_atr(df, period=atr_period_norm)

            # 실시간 가격 조회 시도 (장중일 경우)
            realtime_price = None
            if market_is_open:
                realtime_price = fetch_naver_realtime_price(tkr) if country == "kor" else None

            # 실시간 가격이 없으면 pykrx의 최신 종가를 사용
            c0 = realtime_price if realtime_price else close.iloc[-1]
            if pd.isna(c0):
                continue
            c0 = float(c0)

            # 전일 종가 가져오기
            prev_close = 0.0
            if len(close) >= 2:
                prev_close_val = close.iloc[-2]
                if pd.notna(prev_close_val):
                    prev_close = float(prev_close_val)

            m = ma.iloc[-1]

            a = atr.iloc[-1]
            ma_score = 0.0
            if pd.notna(m) and pd.notna(a) and a > 0:
                ma_score = (c0 - m) / a

            # 이동평균 돌파 후 연속된 일수 계산
            buy_signal_active = close > ma
            buy_signal_days = (
                buy_signal_active.groupby((buy_signal_active != buy_signal_active.shift()).cumsum())
                .cumsum()
                .fillna(0)
                .astype(int)
            )
            buy_signal_days_today = buy_signal_days.iloc[-1] if not buy_signal_days.empty else 0

            sh = int((holdings.get(tkr) or {}).get("shares") or 0)
            ac = float((holdings.get(tkr) or {}).get("avg_cost") or 0.0)
            total_holdings_value += sh * c0
            datestamps.append(df.index[-1])
            data_by_tkr[tkr] = {
                "price": c0,
                "prev_close": prev_close,
                "s1": m,  # 이동평균 (값)
                "s2": ma_period,  # 이동평균 (기간)
                "score": ma_score,  # 정규화 점수
                "filter": buy_signal_days_today,  # 신호 지속일
                "shares": sh,
                "avg_cost": ac,
            }
    else:
        print(f"오류: '{strategy_name}' 전략에 대한 'today' 로직이 구현되지 않았습니다.")
        return

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

    # Decide label and date to display
    portfolio_filename = os.path.basename(portfolio_data["filepath"])
    try:
        date_str_from_filename = portfolio_filename.replace("portfolio_", "").replace(".json", "")
        base_date = pd.to_datetime(date_str_from_filename).normalize()
    except (ValueError, TypeError):
        print(f"경고: 포트폴리오 파일명 '{portfolio_filename}'에서 날짜를 추출할 수 없습니다. 현재 날짜를 사용합니다.")
        base_date = pd.Timestamp.now().normalize()

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
            label_date = next_trading_day  # Also update label_date to show the *actual* next trading day
    else:
        day_label = "기준일"

    # --- 보유 종목 추가 정보 계산 (고점 대비 하락률) ---
    if strategy_name == "donchian":
        for tkr, d in data_by_tkr.items():
            if int(d.get("shares", 0)) > 0:
                consecutive_info = consecutive_holding_info.get(tkr)
                buy_date = consecutive_info.get("buy_date") if consecutive_info else None
                if buy_date:
                    # 보유 시작일부터 현재까지의 데이터 조회
                    df_holding_period = fetch_ohlcv(
                        tkr,
                        country=country,
                        date_range=[buy_date.strftime("%Y-%m-%d"), label_date.strftime("%Y-%m-%d")],
                    )
                    if df_holding_period is not None and not df_holding_period.empty:
                        # yfinance can return multi-level columns, clean them up
                        if isinstance(df_holding_period.columns, pd.MultiIndex):
                            df_holding_period.columns = (
                                df_holding_period.columns.get_level_values(0)
                            )
                            df_holding_period = df_holding_period.loc[
                                :, ~df_holding_period.columns.duplicated()
                            ]
                        peak_high = df_holding_period["High"].max()
                        current_price = d["price"]
                        if pd.notna(peak_high) and peak_high > 0 and pd.notna(current_price):
                            drawdown_from_peak = ((current_price / peak_high) - 1.0) * 100.0
                            d["drawdown_from_peak"] = drawdown_from_peak

    try:
        denom = int(global_settings.PORTFOLIO_TOPN)
        stop_loss = global_settings.HOLDING_STOP_LOSS_PCT
        max_pos = float(global_settings.MAX_POSITION_PCT)
        min_pos = float(global_settings.MIN_POSITION_PCT)
    except AttributeError as e:
        print(f"오류: '{e.name}' 설정이 전역 settings.py 파일에 반드시 정의되어야 합니다.")
        return

    # Count held tickers from holdings snapshot
    held_count = sum(1 for v in holdings.values() if int((v or {}).get("shares") or 0) > 0)

    # 총 보유금액에 해외 주식 가치를 포함합니다.
    total_holdings = total_holdings_value
    if international_shares_data:
        total_holdings += international_shares_data.get("value", 0.0)

    # 원화(KRW)로 변환된 평가금액
    equity_in_krw = current_equity * aud_krw_rate if country == "aus" and aud_krw_rate else current_equity

    # 누적/일간 수익률 계산
    if country == "kor":
        initial_capital_setting = float(global_settings.KOR_INITIAL_CAPITAL)
    else:  # aus
        initial_capital_setting = float(global_settings.AUS_INITIAL_CAPITAL)

    cum_ret_pct = 0.0
    if initial_capital_setting > 0:
        cum_ret_pct = ((equity_in_krw / initial_capital_setting) - 1.0) * 100.0

    # 이전 포트폴리오 파일을 찾아 일간 수익률 계산
    prev_equity = None
    portfolio_files = sorted(
        glob.glob(os.path.join(data_dir, "portfolio_*.json")), reverse=True
    )
    current_file_path = portfolio_data["filepath"]
    try:
        current_file_index = portfolio_files.index(current_file_path)
        if current_file_index + 1 < len(portfolio_files):
            prev_file_path = portfolio_files[current_file_index + 1]
            with open(prev_file_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            prev_equity = float(prev_data.get("total_equity", 0.0))
    except (ValueError, IndexError, KeyError, json.JSONDecodeError):
        prev_equity = None

    day_ret_pct = 0.0
    day_profit_loss = 0.0
    if prev_equity is not None and prev_equity > 0:
        day_ret_pct = ((current_equity / prev_equity) - 1.0) * 100.0
        day_profit_loss = current_equity - prev_equity

    # 헤더 생성 (모든 금액을 KRW로 변환하여 표시)
    total_cash = float(current_equity) - float(total_holdings)
    holdings_in_krw = total_holdings * aud_krw_rate if country == "aus" and aud_krw_rate else total_holdings
    cash_in_krw = equity_in_krw - holdings_in_krw

    # 총 매수금액 계산 (평가 수익률용)
    total_acquisition_cost = 0.0
    for tkr, d in data_by_tkr.items():
        sh = int(d.get("shares", 0))
        if sh > 0:
            ac = float(d.get("avg_cost", 0.0))
            total_acquisition_cost += sh * ac

    # 평가 수익률 계산 (개별 주식/ETF 대상, 해외주식 제외)
    eval_ret_pct = 0.0
    eval_profit_loss = 0.0
    if total_acquisition_cost > 0:
        eval_ret_pct = ((total_holdings_value / total_acquisition_cost) - 1.0) * 100.0
        eval_profit_loss = total_holdings_value - total_acquisition_cost

    # 수익률에 대한 절대 금액 계산 (KRW 기준)
    day_profit_loss_krw = (
        day_profit_loss * aud_krw_rate if country == "aus" and aud_krw_rate else day_profit_loss
    )
    eval_profit_loss_krw = (
        eval_profit_loss * aud_krw_rate if country == "aus" and aud_krw_rate else eval_profit_loss
    )
    cum_profit_loss_krw = (
        equity_in_krw - initial_capital_setting if initial_capital_setting > 0 else 0.0
    )

    header_line = (
        f"보유종목: {held_count} | 평가금액: {header_money_formatter(equity_in_krw)} | 보유금액: {header_money_formatter(holdings_in_krw)} | "
        f"현금: {header_money_formatter(cash_in_krw)} | 일간: {day_ret_pct:+.2f}%({header_money_formatter(day_profit_loss_krw)}) | "
        f"평가: {eval_ret_pct:+.2f}%({header_money_formatter(eval_profit_loss_krw)}) | 누적: {cum_ret_pct:+.2f}%({header_money_formatter(cum_profit_loss_krw)})"
    )

    actions = []  # List of dictionaries
    for tkr, name in pairs:
        d = data_by_tkr.get(tkr)
        if not d:
            continue
        price = d["price"]
        score = d.get("score", 0.0)
        sh = int(d["shares"])
        ac = float(d.get("avg_cost") or 0.0)

        # 자동 계산된 보유종목의 매수일과 보유일
        consecutive_info = consecutive_holding_info.get(tkr)
        buy_date = consecutive_info.get("buy_date") if consecutive_info else None
        holding_days = 0
        if sh > 0 and buy_date:
            try:
                # 거래일 기준으로 보유일수 계산
                if country == "kor":
                    if _stock and ref_ticker_for_cal and buy_date <= label_date:
                        df_days = _stock.get_market_ohlcv_by_date(
                            buy_date.strftime("%Y%m%d"),
                            label_date.strftime("%Y%m%d"),
                            ref_ticker_for_cal,
                        )
                        holding_days = len(df_days)
                    else:
                        holding_days = (label_date - buy_date).days + 1
                elif country == "aus":
                    if yf and ref_ticker_for_cal and buy_date <= label_date:
                        ticker_yf = format_aus_ticker_for_yfinance(ref_ticker_for_cal)
                        df_days = yf.download(
                            ticker_yf,
                            start=buy_date,
                            end=label_date + pd.Timedelta(days=1),
                            progress=False, auto_adjust=True
                        )
                        holding_days = len(df_days)
                    else:
                        holding_days = (label_date - buy_date).days + 1
                else: # Fallback
                    # pykrx 사용 불가 시, 단순 일수 차이로 계산 (1을 더해 보유일수 개념으로)
                    holding_days = (label_date - buy_date).days + 1
            except Exception:
                holding_days = 0  # fallback

        state = "HOLD" if sh > 0 else "WAIT"
        phrase = ""
        qty = 0
        notional = 0.0
        # Current holding return
        hold_ret = ((price / ac) - 1.0) * 100.0 if (sh > 0 and ac > 0 and pd.notna(price)) else None
        # TRIM if exceeding cap
        if sh > 0:
            equity = current_equity
            curr_val = sh * price if pd.notna(price) else 0.0
            cap_val = max_pos * equity
            if curr_val > cap_val and price > 0:
                to_sell_val = curr_val - cap_val
                qty = int(to_sell_val // price)
                if qty > 0:
                    state = "TRIM_REBALANCE"  # 결정 코드
                    notional = qty * price
                    prof = (price - ac) * qty if ac > 0 else 0.0
                    phrase = f"비중조절 {qty}주 @ {price_formatter(price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"
            # CUT stop loss
            elif stop_loss is not None and ac > 0 and hold_ret <= float(stop_loss):
                state = "CUT_STOPLOSS"  # 결정 코드
                qty = sh
                notional = qty * price
                prof = (price - ac) * qty if ac > 0 else 0.0
                phrase = f"가격기반손절 {qty}주 @ {price_formatter(price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"

        # --- 전략별 매수/매도 로직 ---
        if state == "HOLD":  # 아직 매도 결정이 내려지지 않은 경우
            if strategy_name == "jason":
                s2 = d["score"]
                sell_thr = float(getattr(strategy_settings, "SELL_SUM_THRESHOLD", -3.0))
                if sh > 0 and ac > 0 and hold_ret is not None and (s2 + hold_ret) < sell_thr:
                    state = "SELL_MOMENTUM"  # 결정 코드
                    qty = sh
                    notional = qty * price
                    prof = (price - ac) * qty if ac > 0 else 0.0
                    tag = "모멘텀소진(이익)" if hold_ret >= 0 else "모멘텀소진(손실)"
                    phrase = f"{tag} {qty}주 @ {price_formatter(price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"
            elif strategy_name == "seykota":
                fast_ma, slow_ma = d["s1"], d["s2"]
                if sh > 0 and not pd.isna(fast_ma) and not pd.isna(slow_ma) and fast_ma < slow_ma:
                    state = "SELL_TREND"  # 결정 코드
                    qty = sh
                    notional = qty * price
                    prof = (price - ac) * qty if ac > 0 else 0.0
                    tag = "추세이탈(이익)" if hold_ret >= 0 else "추세이탈(손실)"
                    phrase = f"{tag} {qty}주 @ {price_formatter(price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"
            elif strategy_name == "donchian":
                price, ma, period = d["price"], d["s1"], d["s2"]
                if sh > 0 and not pd.isna(price) and not pd.isna(ma) and price < ma:
                    state = "SELL_TREND"  # 결정 코드
                    qty = sh
                    notional = qty * price
                    prof = (price - ac) * qty if ac > 0 else 0.0
                    tag = "추세이탈(이익)" if hold_ret >= 0 else "추세이탈(손실)"
                    phrase = f"{tag} {qty}주 @ {price_formatter(price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"

        elif state == "WAIT":  # 아직 보유하지 않은 경우
            buy_signal = False
            buy_phrase = ""
            if strategy_name == "jason":
                s2 = d["score"]
                stv = d["filter"]
                buy_thr = float(getattr(strategy_settings, "BUY_SUM_THRESHOLD", 3.0))
                if s2 > buy_thr and stv > 0:
                    buy_signal = True
                    buy_phrase = f"모멘텀 점수 {s2:+.1f}"
            elif strategy_name == "seykota":
                fast_ma, slow_ma = d["s1"], d["s2"]
                if not pd.isna(fast_ma) and not pd.isna(slow_ma) and fast_ma > slow_ma:
                    buy_signal = True
                    buy_phrase = f"골든크로스 ({fast_ma:.0f}>{slow_ma:.0f})"
            elif strategy_name == "donchian":
                price, ma, period = d["price"], d["s1"], d["s2"]
                buy_signal_days_today = d["filter"]
                if buy_signal_days_today > 0:
                    buy_signal = True
                    buy_phrase = f"추세진입 ({buy_signal_days_today}일째)"

            if buy_signal:
                reason_suffix = f" ({buy_phrase})"
                if held_count >= denom:
                    phrase = "포트폴리오 가득 참" + reason_suffix
                elif price > 0:
                    equity = current_equity  # 비중 계산은 현지 통화 기준 자본으로 수행
                    min_val = min_pos * equity
                    cap_val = max_pos * equity
                    need = max(0.0, min_val - 0.0)
                    budget_cap = max(0.0, cap_val - 0.0)
                    budget = min(total_cash, budget_cap)
                    from math import ceil

                    req_qty = int(ceil(need / price)) if price > 0 else 0
                    if req_qty > 0 and (req_qty * price) <= budget:
                        qty = req_qty
                        state = "BUY"
                        notional = qty * price
                        phrase = f"매수 {qty}주 @ {price_formatter(price)} ({money_formatter(notional)})" + reason_suffix
                    else:
                        phrase = ("현금 부족" if total_cash < need else "상한 제한") + reason_suffix
                else:
                    phrase = "가격 정보 없음" + reason_suffix

        amount = sh * price if pd.notna(price) else 0.0
        # 일간 수익률 계산
        prev_close = d.get("prev_close")
        day_ret_str = "-"
        if prev_close is not None and prev_close > 0 and pd.notna(price):
            day_ret = ((price / prev_close) - 1.0) * 100.0
            day_ret_str = f"{day_ret:+.1f}%"

        # 테이블 출력용 신호 포맷팅
        if strategy_name == "jason":
            s1_str = f"{d['s1']:+.1f}%"
            s2_str = f"{d['s2']:+.1f}%"
            filter_str = (
                ("+1" if d["filter"] > 0 else ("-1" if d["filter"] < 0 else "0"))
                if d["filter"] is not None
                else "-"
            )
        elif strategy_name == "seykota":
            s1_str = ma_formatter(d["s1"]) if not pd.isna(d["s1"]) else "-"
            s2_str = ma_formatter(d["s2"]) if not pd.isna(d["s2"]) else "-"
            filter_str = "-"
        elif strategy_name == "donchian":
            s1_str = ma_formatter(d["s1"]) if not pd.isna(d["s1"]) else "-"  # 이평선(값)
            drawdown_val = d.get("drawdown_from_peak")
            s2_str = f"{drawdown_val:.1f}%" if drawdown_val is not None else "-"  # 고점대비
            filter_str = f"{d['filter']}일" if d.get("filter") is not None else "-"

        else:
            s1_str, s2_str, filter_str = "-", "-", "-"

        buy_date_display = buy_date.strftime("%Y-%m-%d") if buy_date else "-"
        holding_days_display = str(holding_days) if holding_days > 0 else "-"

        position_weight_pct = (amount / current_equity) * 100.0 if current_equity > 0 else 0.0
        current_row = [
            0,
            tkr,
            name,
            state,
            buy_date_display,
            holding_days_display,
            price_formatter(price),
            day_ret,
            f"{sh:,}",
            money_formatter(amount),
            hold_ret,
            position_weight_pct,
            s1_str,
            s2_str,
            d.get("score"),  # raw score 값으로 변경
            filter_str,
            phrase,
        ]
        actions.append(
            {
                "state": state,
                "weight": position_weight_pct,
                "score": score,
                "tkr": tkr,
                "row": current_row,
                "buy_signal": buy_signal if state == "WAIT" else False,
            }
        )

    # --- 교체 매매 로직 (포트폴리오가 가득 찼을 경우) ---
    if strategy_name == "donchian" and held_count >= denom and replace_weaker_stock:
        buy_candidates = sorted(
            [a for a in actions if a["buy_signal"]],
            key=lambda x: x["score"],
            reverse=True,
        )
        held_stocks = sorted(
            [a for a in actions if a["state"] == "HOLD"], key=lambda x: x["score"]
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
                sell_price = d_weakest.get("price", 0)
                sell_qty = d_weakest.get("shares", 0)
                avg_cost = d_weakest.get("avg_cost", 0)

                hold_ret = 0.0
                prof = 0.0
                if avg_cost > 0 and sell_price > 0:
                    hold_ret = ((sell_price / avg_cost) - 1.0) * 100.0
                    prof = (sell_price - avg_cost) * sell_qty

                sell_phrase = f"교체매도 {sell_qty}주 @ {price_formatter(sell_price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'} ({best_new['tkr']}(으)로 교체)"

                weakest_held["state"] = "SELL_REPLACE"
                weakest_held["row"][3] = "SELL_REPLACE"
                weakest_held["row"][-1] = sell_phrase

                # 2. 새로 편입될 종목(매수)의 상태 업데이트
                best_new["state"] = "BUY_REPLACE"
                best_new["row"][3] = "BUY_REPLACE"

                # 매수 수량 및 금액 계산
                sell_value = weakest_held["weight"] / 100.0 * current_equity
                buy_price = data_by_tkr.get(best_new["tkr"], {}).get("price", 0)
                if buy_price > 0:
                    # 매도 금액으로 살 수 있는 최대 수량
                    buy_qty = int(sell_value // buy_price)
                    buy_notional = buy_qty * buy_price
                    best_new["row"][
                        -1
                    ] = f"매수 {buy_qty}주 @ {price_formatter(buy_price)} ({money_formatter(buy_notional)}) ({weakest_held['tkr']} 대체)"
                else:
                    best_new["row"][-1] = f"{weakest_held['tkr']}(을)를 대체 (가격정보 없음)"
            else:
                # 점수가 정렬되어 있으므로, 더 이상의 교체는 불가능합니다.
                break

        # 3. 교체되지 않은 나머지 매수 후보들의 상태 업데이트
        for cand in buy_candidates:
            if cand["state"] == "WAIT":  # 아직 매수/교체매수 결정이 안된 경우
                cand["row"][-1] = "포트폴리오 가득 참 (교체대상 아님)"

    # 정렬: 매도/보유 종목을 상단에, 매수 후보를 하단에 정렬
    def sort_key(action_dict):
        state = action_dict["state"]
        weight = action_dict["weight"]
        score = action_dict["score"]
        tkr = action_dict["tkr"]

        state_order = {
            "CUT_STOPLOSS": 0,
            "SELL_MOMENTUM": 1,
            "SELL_TREND": 2,
            "SELL_REPLACE": 3,
            "TRIM_REBALANCE": 4,
            "HOLD": 5,
            "BUY_REPLACE": 6,
            "BUY": 7,
            "WAIT": 8,
        }
        order = state_order.get(state, 99)

        # 보유 종목은 비중이 높은 순으로, 매수/대기 종목은 점수가 높은 순으로 정렬
        sort_value = -weight if state in ("HOLD", "TRIM_REBALANCE") else -score

        # 1. 점수가 높은 순서대로 정렬 (NaN 값은 가장 낮은 우선순위로 처리)
        # 2. 점수가 같을 경우 티커 이름으로 정렬
        return (order, sort_value, tkr)

    actions.sort(key=sort_key)

    rows_sorted = []
    for i, action_dict in enumerate(actions, 1):
        row = action_dict["row"]
        row[0] = i
        rows_sorted.append(row)

    # 호주 시장의 경우, international_shares 정보를 테이블의 최상단에 추가합니다.
    if country == "aus" and international_shares_data:
        is_value = international_shares_data.get("value", 0.0)
        is_change_pct = international_shares_data.get("change_pct", 0.0)
        is_weight_pct = (is_value / current_equity) * 100.0 if current_equity > 0 else 0.0

        num_signal_cols = 4  # 신호 관련 컬럼 수
        special_row = [
            0,  # #
            "IS",  # 티커
            "International Shares",  # 이름
            "HOLD",  # 상태
            "-",  # 매수일
            "-",  # 보유일
            price_formatter(is_value),  # 현재가 (AUD)
            0.0,  # 일간수익률
            "1",  # 보유수량
            money_formatter(is_value),  # 금액 (KRW)
            is_change_pct,  # 누적수익률
            is_weight_pct,  # 비중
        ]
        # 나머지 신호 컬럼과 문구 컬럼을 빈 값으로 채웁니다.
        signal_values = ["-"] * num_signal_cols
        if num_signal_cols > 2:
            signal_values[2] = None  # 점수 컬럼은 None으로 설정
        special_row.extend(signal_values)
        special_row.append("")  # 문구

        rows_sorted.insert(0, special_row)
        # 행을 추가했으므로, 순번을 다시 매깁니다.
        for i, row in enumerate(rows_sorted, 1):
            row[0] = i

    # 전략에 따라 동적으로 헤더를 설정합니다.
    if strategy_name == "jason":
        signal_headers = ["1주수익", "2주수익", "모멘텀점수", "ST"]
    elif strategy_name == "seykota":
        signal_headers = ["단기MA", "장기MA", "MA스코어", "필터"]
    elif strategy_name == "donchian":
        signal_headers = ["이평선(값)", "고점대비", "정규화점수", "신호지속일"]
    else:
        signal_headers = ["신호1", "신호2", "점수", "필터"]

    headers = [
        "#",
        "티커",
        "이름",
        "상태",
        "매수일",
        "보유일",
        "현재가",
        "일간수익률",
        "보유수량",
        "금액",
        "누적수익률",
        "비중",
    ]
    headers.extend(signal_headers)
    headers.append("문구")

    return (header_line, headers, rows_sorted)


def main(strategy_name: str, country: str = "kor", portfolio_path: Optional[str] = None):
    """CLI에서 오늘의 액션 플랜을 실행하고 결과를 출력/저장합니다."""
    result = generate_action_plan(strategy_name, country, portfolio_path)

    if result:
        header_line, headers, rows_sorted = result

        # --- 콘솔 출력용 포맷팅 ---
        # 웹앱은 raw data (rows_sorted)를 사용하고, 콘솔은 포맷된 데이터를 사용합니다.

        # 컬럼 인덱스 찾기
        col_indices = {}
        try:
            score_header_candidates = ["정규화점수", "모멘텀점수", "MA스코어"]
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
                    if strategy_name == "jason":
                        display_row[idx] = f"{val:+.1f}%"
                    elif strategy_name == "seykota":
                        display_row[idx] = f"{val:+.2f}%"
                    elif strategy_name == "donchian":
                        display_row[idx] = f"{val:+.2f}"
                    else:
                        display_row[idx] = str(val)
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

        table_lines = render_table_eaw(headers, display_rows, aligns=None)

        print("\n" + header_line)
        print("\n".join(table_lines))


if __name__ == "__main__":
    main()
