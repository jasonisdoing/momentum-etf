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
from utils.data_loader import fetch_naver_realtime_price, fetch_ohlcv, read_tickers_file
from utils.indicators import supertrend_direction
from utils.report import format_kr_money, render_table_eaw

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

try:
    from pykrx import stock as _stock
except Exception:
    _stock = None


def is_market_open() -> bool:
    """
    한국 주식 시장이 현재 개장 시간인지 확인합니다. (KST, 09:00-15:30, 주말 제외)
    정확한 공휴일은 반영하지 않으며, 시간과 요일만으로 판단합니다.
    """
    if not pytz:
        return False  # pytz 없으면 안전하게 False 반환

    try:
        kst = pytz.timezone("Asia/Seoul")
        now_kst = datetime.now(kst)

        # 주말(토, 일) 확인
        if now_kst.weekday() >= 5:
            return False

        # 개장 시간 확인 (09:00 ~ 15:30)
        market_open_time = now_kst.replace(hour=9, minute=0, second=0, microsecond=0).time()
        market_close_time = now_kst.replace(hour=15, minute=30, second=0, microsecond=0).time()

        return market_open_time <= now_kst.time() <= market_close_time
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
            print(f"경고: 지정된 포트폴리오 파일 '{portfolio_path}'를 찾을 수 없습니다.")
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

        return {
            "date": data.get("date"),
            "total_equity": data.get("total_equity"),
            "holdings": holdings_dict,
            "filepath": filepath_to_load,
        }
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
        portfolio_files = glob.glob(os.path.join(data_dir, "portfolio_*.json"))
        if not portfolio_files:
            return holding_info

        sorted_files = []
        for f_path in portfolio_files:
            try:
                fname = os.path.basename(f_path)
                date_str = fname.replace("portfolio_", "").replace(".json", "")
                file_date = pd.to_datetime(date_str)
                sorted_files.append((file_date, f_path))
            except ValueError:
                continue

        if not sorted_files:
            return holding_info

        sorted_files.sort(key=lambda x: x[0], reverse=True)

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


def main(strategy_name: str, portfolio_path: Optional[str] = None):
    print(f"'{strategy_name}' 전략을 사용하여 오늘의 액션 플랜을 생성합니다.")

    market_is_open = is_market_open()
    if market_is_open:
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

    # Load initial state from portfolio file.
    portfolio_data = load_portfolio_data(portfolio_path)

    if not portfolio_data:
        print(
            "오류: 포트폴리오 파일(portfolio_*.json)을 찾을 수 없습니다. "
            "--portfolio 옵션으로 파일을 지정하거나 data/ 폴더에 파일을 위치시켜주세요."
        )
        print("웹 UI(web_app.py)를 실행하여 새 포트폴리오를 생성할 수 있습니다.")
        return

    print(
        f"포트폴리오 파일 '{os.path.basename(portfolio_data['filepath'])}'을(를) 기준으로 오늘의 액션을 계산합니다."
    )
    holdings = portfolio_data.get("holdings", {})
    init_cap = float(portfolio_data.get("total_equity", 0.0))

    # 보유 기간 자동 계산
    held_tickers = [tkr for tkr, v in holdings.items() if int((v or {}).get("shares") or 0) > 0]
    consecutive_holding_info = calculate_consecutive_holding_info(held_tickers)
    print("-> 과거 포트폴리오 파일을 기반으로 보유 기간을 자동 계산했습니다.")

    # 티커 목록 결정
    print("\n[고정 유니버스] data/tickers.txt 파일의 종목을 사용합니다.")
    # tickers.txt와 현재 보유 종목을 합쳐서 전체 유니버스 구성
    static_pairs = read_tickers_file("data/tickers.txt")
    pairs = build_pairs_with_holdings(static_pairs, holdings)

    if not pairs:
        print("오류: 투자 대상 티커를 찾을 수 없습니다.")
        print("      data/tickers.txt 파일이 비어있거나 존재하지 않을 수 있습니다.")
        return

    # Fetch recent data for signals
    total_holdings = 0.0
    datestamps = []
    data_by_tkr = {}

    # --- 전략별 신호 계산 ---
    if strategy_name == "jason":
        for tkr, _ in pairs:
            df = fetch_ohlcv(tkr, months_range=[1, 0])
            if df is None or len(df) < 11:
                continue
            close = df["Close"]

            # 실시간 가격 조회 시도 (장중일 경우)
            realtime_price = None
            if market_is_open:
                realtime_price = fetch_naver_realtime_price(tkr)

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
            total_holdings += sh * c0
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
            df = fetch_ohlcv(tkr, months_range=[required_months, 0])
            if df is None or len(df) < slow_ma_period:
                continue
            close = df["Close"]
            fast_ma = close.rolling(window=fast_ma_period).mean()
            slow_ma = close.rolling(window=slow_ma_period).mean()

            # 실시간 가격 조회 시도 (장중일 경우)
            realtime_price = None
            if market_is_open:
                realtime_price = fetch_naver_realtime_price(tkr)

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
            total_holdings += sh * c0
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
        ma_period = int(getattr(strategy_settings, "DONCHIAN_MA_PERIOD", 20))
        entry_delay_days = int(getattr(strategy_settings, "DONCHIAN_ENTRY_DELAY_DAYS", 0))
        required_days = ma_period + entry_delay_days + 5  # 버퍼 추가
        required_months = (required_days // 22) + 2

        for tkr, _ in pairs:
            df = fetch_ohlcv(tkr, months_range=[required_months, 0])  # 더 많은 데이터 요청
            if df is None or len(df) < ma_period:
                continue
            close = df["Close"]
            ma = close.rolling(window=ma_period).mean()

            # 실시간 가격 조회 시도 (장중일 경우)
            realtime_price = None
            if market_is_open:
                realtime_price = fetch_naver_realtime_price(tkr)

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

            ma_score = (c0 / m - 1.0) * 100.0 if m > 0 and not pd.isna(m) else 0.0

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
            total_holdings += sh * c0
            datestamps.append(df.index[-1])
            data_by_tkr[tkr] = {
                "price": c0,
                "prev_close": prev_close,
                "s1": m,  # 이동평균 (값)
                "s2": ma_period,  # 이동평균 (기간)
                "score": ma_score,  # 이격도
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

    # Decide label and date to display
    today_cal = pd.Timestamp.now().normalize()
    next_trading_day = get_next_trading_day(today_cal, ref_ticker_for_cal)

    if next_trading_day.date() == today_cal.date():
        day_label = "오늘"
        label_date = next_trading_day
    else:
        day_label = "다음 거래일"
        label_date = next_trading_day

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

    label_date_str = pd.to_datetime(label_date).strftime("%Y-%m-%d")
    total_cash = float(init_cap) - float(total_holdings)
    total_value = total_holdings + max(0.0, total_cash)
    header_line = (
        f"{day_label} {label_date_str} - 보유종목 {held_count} "
        f"잔액(보유+현금): {format_kr_money(total_value)} (보유 {format_kr_money(total_holdings)} + 현금 {format_kr_money(total_cash)})"
    )

    actions = []  # (state, weight, score, tkr, row)
    for tkr, name in pairs:
        d = data_by_tkr.get(tkr)
        if not d:
            continue
        price = d["price"]
        score = d.get("score", 0.0)
        sh = int(d["shares"])
        ac = float(d["avg_cost"] or 0.0)

        # 자동 계산된 보유종목의 매수일과 보유일
        consecutive_info = consecutive_holding_info.get(tkr)
        buy_date = consecutive_info.get("buy_date") if consecutive_info else None
        holding_days = 0
        if sh > 0 and buy_date:
            try:
                # 거래일 기준으로 보유일수 계산
                if _stock and ref_ticker_for_cal and buy_date <= label_date:
                    df_days = _stock.get_market_ohlcv_by_date(
                        buy_date.strftime("%Y%m%d"),
                        label_date.strftime("%Y%m%d"),
                        ref_ticker_for_cal,
                    )
                    holding_days = len(df_days)
                else:
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
            equity = total_value
            curr_val = sh * price if pd.notna(price) else 0.0
            cap_val = max_pos * equity
            if curr_val > cap_val and price > 0:
                to_sell_val = curr_val - cap_val
                qty = int(to_sell_val // price)
                if qty > 0:
                    state = "TRIM_REBALANCE"  # 결정 코드
                    notional = qty * price
                    prof = (price - ac) * qty if ac > 0 else 0.0
                    phrase = f"비중조절 {qty}주 @ {int(round(price)):,} 수익 {format_kr_money(prof)} 손익률 {f'{hold_ret:+.1f}%'}"
            # CUT stop loss
            elif stop_loss is not None and ac > 0 and hold_ret <= float(stop_loss):
                state = "CUT_STOPLOSS"  # 결정 코드
                qty = sh
                notional = qty * price
                prof = (price - ac) * qty if ac > 0 else 0.0
                phrase = f"가격기반손절 {qty}주 @ {int(round(price)):,} 수익 {format_kr_money(prof)} 손익률 {f'{hold_ret:+.1f}%'}"

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
                    phrase = f"{tag} {qty}주 @ {int(round(price)):,} 수익 {format_kr_money(prof)} 손익률 {f'{hold_ret:+.1f}%'}"
            elif strategy_name == "seykota":
                fast_ma, slow_ma = d["s1"], d["s2"]
                if sh > 0 and not pd.isna(fast_ma) and not pd.isna(slow_ma) and fast_ma < slow_ma:
                    state = "SELL_TREND"  # 결정 코드
                    qty = sh
                    notional = qty * price
                    prof = (price - ac) * qty if ac > 0 else 0.0
                    tag = "추세이탈(이익)" if hold_ret >= 0 else "추세이탈(손실)"
                    phrase = f"{tag} {qty}주 @ {int(round(price)):,} 수익 {format_kr_money(prof)} 손익률 {f'{hold_ret:+.1f}%'}"
            elif strategy_name == "donchian":
                price, ma, period = d["price"], d["s1"], d["s2"]
                if sh > 0 and not pd.isna(price) and not pd.isna(ma) and price < ma:
                    state = "SELL_TREND"  # 결정 코드
                    qty = sh
                    notional = qty * price
                    prof = (price - ac) * qty if ac > 0 else 0.0
                    tag = "추세이탈(이익)" if hold_ret >= 0 else "추세이탈(손실)"
                    phrase = f"{tag} {qty}주 @ {int(round(price)):,} 수익 {format_kr_money(prof)} 손익률 {f'{hold_ret:+.1f}%'}"

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
                entry_delay_days = int(getattr(strategy_settings, "DONCHIAN_ENTRY_DELAY_DAYS", 0))
                if buy_signal_days_today > entry_delay_days:
                    buy_signal = True
                    buy_phrase = f"추세진입 ({buy_signal_days_today}일째, MA={period})"

            if buy_signal:
                reason_suffix = f" ({buy_phrase})"
                if held_count >= denom:
                    phrase = "포트폴리오 가득 참" + reason_suffix
                elif price > 0:
                    equity = total_value
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
                        phrase = (
                            f"매수 {qty}주 @ {int(round(price)):,} ({format_kr_money(notional)})"
                            + reason_suffix
                        )
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
            score_str = f"{d['score']:+.1f}%"
            filter_str = (
                ("+1" if d["filter"] > 0 else ("-1" if d["filter"] < 0 else "0"))
                if d["filter"] is not None
                else "-"
            )
        elif strategy_name == "seykota":
            s1_str = f"{d['s1']:.0f}" if not pd.isna(d["s1"]) else "-"
            s2_str = f"{d['s2']:.0f}" if not pd.isna(d["s2"]) else "-"
            score_str = f"{d['score']:+.2f}%"
            filter_str = "-"
        elif strategy_name == "donchian":
            s1_str = f"{d['s1']:.0f}" if not pd.isna(d["s1"]) else "-"  # 이평선(값)
            s2_str = f"{int(d['s2'])}" if not pd.isna(d["s2"]) else "-"  # 이평선(기간)
            score_str = f"{d['score']:+.2f}%" if not pd.isna(d["score"]) else "-"  # 이격도
            filter_str = f"{d['filter']}일" if d.get("filter") is not None else "-"

        else:
            s1_str, s2_str, score_str, filter_str = "-", "-", "-", "-"

        buy_date_display = buy_date.strftime("%Y-%m-%d") if buy_date else "-"
        holding_days_display = str(holding_days) if holding_days > 0 else "-"

        position_weight_pct = (amount / total_value) * 100.0 if total_value > 0 else 0.0
        current_row = [
            0,
            tkr,
            name,
            state,
            buy_date_display,
            holding_days_display,
            f"{int(round(price)):,}",
            day_ret_str,
            f"{sh:,}",
            format_kr_money(amount),
            f"{hold_ret:+.1f}%" if hold_ret is not None else "-",
            f"{position_weight_pct:.0f}%",
            s1_str,
            s2_str,
            score_str,
            filter_str,
            phrase,
        ]
        actions.append((state, position_weight_pct, score, tkr, current_row))

    # 정렬: 전략별 점수(우선순위)가 높은 순으로 정렬
    def sort_key(action_tuple):
        _, _, score, tkr, _ = action_tuple

        # 1. 점수가 높은 순서대로 정렬 (NaN 값은 가장 낮은 우선순위로 처리)
        sort_score = -score if pd.notna(score) else float("inf")

        # 2. 점수가 같을 경우 티커 이름으로 정렬
        return (sort_score, tkr)

    actions.sort(key=sort_key)

    rows_sorted = []
    for i, (_, _, _, _, row) in enumerate(actions, 1):
        row[0] = i
        rows_sorted.append(row)

    # 전략에 따라 동적으로 헤더를 설정합니다.
    if strategy_name == "jason":
        signal_headers = ["1주수익", "2주수익", "모멘텀점수", "ST"]
    elif strategy_name == "seykota":
        signal_headers = ["단기MA", "장기MA", "MA스코어", "필터"]
    elif strategy_name == "donchian":
        signal_headers = ["이평선(값)", "이평선(기간)", "이격도", "신호지속일"]
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

    aligns = [
        "right",
        "right",
        "left",
        "center",
        "left",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "center",
        "left",
    ]

    # Render table for both console and log file
    table_lines = render_table_eaw(headers, rows_sorted, aligns)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"today_{strategy_name}.log")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(header_line + "\n\n")
            f.write("\n".join(table_lines) + "\n")
    except Exception:
        pass
    print(header_line)
    print("\n".join(table_lines))


if __name__ == "__main__":
    main()
