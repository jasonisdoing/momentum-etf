import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from typing import Dict, Tuple, List, Optional
import json
import os
import glob
import importlib
import re
import pandas as pd

import settings as global_settings
# New structure imports
from utils.data_loader import read_tickers_file, fetch_ohlcv
from utils.indicators import supertrend_direction
from utils.report import render_table_eaw, format_kr_money
try:
    from pykrx import stock as _stock
except Exception:
    _stock = None


def load_portfolio_data(portfolio_path: Optional[str] = None, data_dir: str = 'data') -> Optional[Dict]:
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
            portfolio_files = glob.glob(os.path.join(data_dir, 'portfolio_*.json'))
            if not portfolio_files:
                return None
            
            latest_date = None
            for f_path in portfolio_files:
                try:
                    fname = os.path.basename(f_path)
                    date_str = fname.replace('portfolio_', '').replace('.json', '')
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
        with open(filepath_to_load, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        holdings_list = data.get('holdings', [])
        holdings_dict = {
            item['ticker']: {
                'name': item.get('name', ''),
                'shares': item.get('shares', 0),
                'avg_cost': item.get('avg_cost', 0.0)
            } for item in holdings_list if item.get('ticker')
        }

        return {
            'date': data.get('date'),
            'total_equity': data.get('total_equity'),
            'holdings': holdings_dict,
            'filepath': filepath_to_load
        }
    except Exception as e:
        print(f"오류: 포트폴리오 파일 '{filepath_to_load}' 로드 중 오류 발생: {e}")
        return None


def build_pairs_with_holdings(pairs: List[Tuple[str, str]], holdings: dict) -> List[Tuple[str, str]]:
    name_map = {t: n for t, n in pairs if n}
    out_map = {t: n for t, n in pairs}
    # If holdings has tickers not in pairs, add with blank name
    for tkr in holdings.keys():
        if tkr not in out_map:
            out_map[tkr] = name_map.get(tkr, '')
    return [(t, out_map.get(t, '')) for t in out_map.keys()]


def main(strategy_name: str, portfolio_path: Optional[str] = None):
    print(f"'{strategy_name}' 전략을 사용하여 오늘의 액션 플랜을 생성합니다.")

    # 전략별 설정 로드
    try:
        strategy_settings = importlib.import_module(f"logics.{strategy_name}.settings")
        print(f"-> '{strategy_name}' 전략의 전용 설정을 로드했습니다.")
    except ImportError:
        print(f"-> 경고: '{strategy_name}' 전략의 전용 설정 파일(settings.py)을 찾을 수 없습니다. 전역 설정을 사용합니다.")
        strategy_settings = global_settings

    # Load initial state from portfolio file.
    portfolio_data = load_portfolio_data(portfolio_path)

    if not portfolio_data:
        print("오류: 포트폴리오 파일(portfolio_*.json)을 찾을 수 없습니다. --portfolio 옵션으로 파일을 지정하거나 data/ 폴더에 파일을 위치시켜주세요.")
        print("웹 UI(web_app.py)를 실행하여 새 포트폴리오를 생성할 수 있습니다.")
        return

    print(f"포트폴리오 파일 '{os.path.basename(portfolio_data['filepath'])}'을(를) 기준으로 오늘의 액션을 계산합니다.")
    holdings = portfolio_data.get('holdings', {})
    init_cap = float(portfolio_data.get('total_equity', 0.0))

    # 티커 목록 결정
    print(f"\n[고정 유니버스] data/tickers.txt 파일의 종목을 사용합니다.")
    # tickers.txt와 현재 보유 종목을 합쳐서 전체 유니버스 구성
    static_pairs = read_tickers_file('data/tickers.txt')
    pairs = build_pairs_with_holdings(static_pairs, holdings)

    if not pairs:
        print('오류: 투자 대상 티커를 찾을 수 없습니다.')
        print('      data/tickers.txt 파일이 비어있거나 존재하지 않을 수 있습니다.')
        return

    # Fetch recent data for signals
    rows = []
    total_holdings = 0.0
    datestamps = []
    data_by_tkr = {}

    # --- 전략별 신호 계산 ---
    if strategy_name == 'jason':
        for tkr, _ in pairs:
            df = fetch_ohlcv(tkr, months_range=[1,0])
            if df is None or len(df) < 11: continue
            close = df['Close']; i = len(close) - 1
            c0 = float(close.iloc[i]); c5 = float(close.iloc[i-5]) if i-5 >= 0 else c0; c10 = float(close.iloc[i-10]) if i-10 >= 0 else (c5 if i-5>=0 else c0)
            p1 = round(((c0/c5)-1.0)*100.0, 1) if c5>0 else 0.0
            p2 = round(((c5/c10)-1.0)*100.0, 1) if c10>0 else 0.0
            s2 = p1 + p2
            try:
                st_dir = supertrend_direction(df, int(getattr(strategy_settings,'ST_ATR_PERIOD',14)), float(getattr(strategy_settings,'ST_ATR_MULTIPLIER',3.0)))
                stv = int(st_dir.iloc[-1]) if len(st_dir)>0 else 0
            except Exception: stv = 0
            sh = int((holdings.get(tkr) or {}).get('shares') or 0); ac = float((holdings.get(tkr) or {}).get('avg_cost') or 0.0)
            total_holdings += sh * c0; datestamps.append(df.index[-1])
            data_by_tkr[tkr] = {'price': c0, 's1': p1, 's2': p2, 'score': s2, 'filter': stv, 'shares': sh, 'avg_cost': ac}
    
    elif strategy_name == 'seykota':
        fast_ma_period = int(getattr(strategy_settings, 'SEYKOTA_FAST_MA', 50))
        slow_ma_period = int(getattr(strategy_settings, 'SEYKOTA_SLOW_MA', 150))
        required_months = (slow_ma_period // 22) + 2

        for tkr, _ in pairs:
            df = fetch_ohlcv(tkr, months_range=[required_months, 0])
            if df is None or len(df) < slow_ma_period: continue
            close = df['Close']
            fast_ma = close.rolling(window=fast_ma_period).mean()
            slow_ma = close.rolling(window=slow_ma_period).mean()
            
            c0 = close.iloc[-1]; fm = fast_ma.iloc[-1]; sm = slow_ma.iloc[-1]
            ma_score = (fm / sm - 1.0) * 100.0 if sm > 0 and not pd.isna(sm) else 0.0
            
            sh = int((holdings.get(tkr) or {}).get('shares') or 0); ac = float((holdings.get(tkr) or {}).get('avg_cost') or 0.0)
            total_holdings += sh * c0; datestamps.append(df.index[-1])
            data_by_tkr[tkr] = {'price': c0, 's1': fm, 's2': sm, 'score': ma_score, 'filter': None, 'shares': sh, 'avg_cost': ac}
    else:
        print(f"오류: '{strategy_name}' 전략에 대한 'today' 로직이 구현되지 않았습니다.")
        return

    dt = max(datestamps) if datestamps else pd.Timestamp.now()
    # Determine trading-calendar-based label/date via pykrx
    ref_ticker_for_cal = next(iter(data_by_tkr.keys())) if data_by_tkr else None

    def get_next_trading_day(start_date: pd.Timestamp, ref_ticker: str) -> pd.Timestamp:
        """주어진 날짜 또는 그 이후의 가장 가까운 거래일을 효율적으로 찾습니다."""
        if _stock is None or ref_ticker is None:
            return start_date # pykrx 사용 불가 시, 입력일을 그대로 반환
        try:
            # 앞으로 2주간의 데이터를 한 번에 조회하여 가장 빠른 거래일을 찾습니다.
            from_date_str = start_date.strftime('%Y%m%d')
            to_date_str = (start_date + pd.Timedelta(days=14)).strftime('%Y%m%d')
            df = _stock.get_market_ohlcv_by_date(from_date_str, to_date_str, ref_ticker)
            if not df.empty:
                return df.index[0]
        except Exception:
            pass
        return start_date # 조회 실패 시, 입력일을 그대로 반환

    # Decide label and date to display
    today_cal = pd.Timestamp.now().normalize()
    next_trading_day = get_next_trading_day(today_cal, ref_ticker_for_cal)

    if next_trading_day.date() == today_cal.date():
        day_label = '오늘'
        label_date = next_trading_day
    else:
        day_label = '다음 거래일'
        label_date = next_trading_day

    denom = int(getattr(global_settings, 'PORTFOLIO_TOPN', 10))
    # Count held tickers from holdings snapshot
    held_count = sum(1 for v in holdings.values() if int((v or {}).get('shares') or 0) > 0)

    label_date_str = pd.to_datetime(label_date).strftime('%Y-%m-%d')
    total_cash = float(init_cap) - float(total_holdings)
    total_value = total_holdings + max(0.0, total_cash)
    header_line = (
        f"{day_label} {label_date_str} - 보유종목 {held_count} "
        f"잔액(보유+현금): {format_kr_money(total_value)} (보유 {format_kr_money(total_holdings)} + 현금 {format_kr_money(total_cash)})"
    )

    # Decide next action per ticker
    stop_loss = getattr(global_settings, 'HOLDING_STOP_LOSS_PCT', None)
    max_pos = float(getattr(global_settings, 'MAX_POSITION_PCT', 0.20))
    min_pos = float(getattr(global_settings, 'MIN_POSITION_PCT', 0.10))

    actions = []  # (notional, score, row)
    for tkr, name in pairs:
        d = data_by_tkr.get(tkr)
        if not d:
            continue
        price = d['price']
        score = d.get('score', 0.0)
        sh = int(d['shares'])
        ac = float(d['avg_cost'] or 0.0)
        state = 'HOLD' if sh>0 else 'WAIT'
        phrase = ''
        qty = 0
        notional = 0.0
        # Current holding return
        hold_ret = ((price/ac)-1.0)*100.0 if (sh>0 and ac>0) else None
        # TRIM if exceeding cap
        if sh > 0:
            equity = total_value
            curr_val = sh * price
            cap_val = max_pos * equity
            if curr_val > cap_val and price > 0:
                to_sell_val = curr_val - cap_val
                qty = int(to_sell_val // price)
                if qty > 0:
                    state = 'TRIM'
                    notional = qty * price
                    phrase = f"부분매도 {qty}주 @ {int(round(price)):,}"
            # CUT stop loss
            elif stop_loss is not None and ac > 0 and hold_ret <= float(stop_loss):
                state = 'CUT'
                qty = sh
                notional = qty * price
                phrase = f"손절 {qty}주 @ {int(round(price)):,}"

        # --- 전략별 매수/매도 로직 ---
        if state == 'HOLD': # 아직 매도 결정이 내려지지 않은 경우
            if strategy_name == 'jason':
                s2 = d['score']
                sell_thr = float(getattr(strategy_settings, 'SELL_SUM_THRESHOLD', -3.0))
                if sh > 0 and ac > 0 and hold_ret is not None and (s2 + hold_ret) < sell_thr:
                    state = 'SELL'
                    qty = sh; notional = qty * price; phrase = f"이익실현 {qty}주 @ {int(round(price)):,}"
            elif strategy_name == 'seykota':
                fast_ma, slow_ma = d['s1'], d['s2']
                if sh > 0 and not pd.isna(fast_ma) and not pd.isna(slow_ma) and fast_ma < slow_ma:
                    state = 'SELL'
                    qty = sh; notional = qty * price; phrase = f"데드크로스 ({fast_ma:.0f}<{slow_ma:.0f})"

        elif state == 'WAIT': # 아직 보유하지 않은 경우
            buy_signal = False
            buy_phrase = ""
            if strategy_name == 'jason':
                s2 = d['score']; stv = d['filter']
                buy_thr = float(getattr(strategy_settings, 'BUY_SUM_THRESHOLD', 3.0))
                if s2 > buy_thr and stv > 0:
                    buy_signal = True; buy_phrase = f"모멘텀 점수 {s2:+.1f}"
            elif strategy_name == 'seykota':
                fast_ma, slow_ma = d['s1'], d['s2']
                if not pd.isna(fast_ma) and not pd.isna(slow_ma) and fast_ma > slow_ma:
                    buy_signal = True; buy_phrase = f"골든크로스 ({fast_ma:.0f}>{slow_ma:.0f})"

            if buy_signal and price > 0:
                equity = total_value
                min_val = min_pos * equity; cap_val = max_pos * equity
                need = max(0.0, min_val - 0.0); budget_cap = max(0.0, cap_val - 0.0)
                budget = min(total_cash, budget_cap)
                from math import ceil
                req_qty = int(ceil(need / price)) if price > 0 else 0
                if req_qty > 0 and (req_qty * price) <= budget:
                    qty = req_qty
                    state = 'BUY'
                    notional = qty * price
                    phrase = f"매수 {qty}주 @ {int(round(price)):,}"
                else:
                    phrase = '현금 부족' if total_cash < need else ('상한 제한' if budget_cap < need else '')

        amount = sh * price
        day_ret = 0.0  # not computed in light mode consistently per ticker

        # 테이블 출력용 신호 포맷팅
        if strategy_name == 'jason':
            s1_str = f"{d['s1']:+.1f}%"; s2_str = f"{d['s2']:+.1f}%"; score_str = f"{d['score']:+.1f}%"
            filter_str = ('+1' if d['filter'] > 0 else ('-1' if d['filter'] < 0 else '0')) if d['filter'] is not None else '-'
        elif strategy_name == 'seykota':
            s1_str = f"{d['s1']:.0f}" if not pd.isna(d['s1']) else '-'; s2_str = f"{d['s2']:.0f}" if not pd.isna(d['s2']) else '-'
            score_str = f"{d['score']:+.2f}%"; filter_str = '-'
        else:
            s1_str, s2_str, score_str, filter_str = '-', '-', '-', '-'

        rows.append([
            0,
            tkr,
            name,
            state,
            '-',
            '0',
            f"{int(round(price)):,}",
            f"{day_ret:+.1f}%",
            f"{sh:,}",
            format_kr_money(amount),
            (f"{hold_ret:+.1f}%" if hold_ret is not None else '-'),
            f"{(amount/total_value*100.0) if total_value>0 else 0.0:.0f}%",
            s1_str,
            s2_str,
            score_str,
            filter_str,
            phrase,
        ])
        actions.append((notional, score, rows[-1]))

    # Sort by score for seykota, or by notional for others
    if strategy_name in ['jason', 'seykota']:
        # jason, seykota 전략은 점수가 높은 순으로 정렬
        actions.sort(key=lambda x: x[1], reverse=True)
    else:
        # 기본 정렬: 거래금액(notional)이 높은 순
        actions.sort(key=lambda x: x[0], reverse=True)

    rows_sorted = []
    for i, (_, _, row) in enumerate(actions, 1):
        row[0] = i
        rows_sorted.append(row)

    headers = ['#','티커','이름','상태','매수일','보유일','현재가','일간수익률','보유수량','금액','누적수익률','비중','신호1','신호2','점수','필터','문구']
    aligns = ['right','right','left','center','left','right','right','right','right','right','right','right','right','right','right','center','left']
    
    # Render table for both console and log file
    table_lines = render_table_eaw(headers, rows_sorted, aligns)

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'today_{strategy_name}.log')
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(header_line + "\n\n")
            f.write("\n".join(table_lines) + "\n")
    except Exception:
        pass
    print(header_line)
    print("\n".join(table_lines))


if __name__ == '__main__':
    main()
