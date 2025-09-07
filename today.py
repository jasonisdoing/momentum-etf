import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from typing import Dict, Tuple, List, Optional
import json
import os
import glob
import pandas as pd

import settings
# New structure imports
from utils.data_loader import read_tickers_file, read_holdings_file, fetch_ohlcv
from utils.indicators import supertrend_direction
from utils.report import render_table_eaw
try:
    from pykrx import stock as _stock
except Exception:
    _stock = None

def _fmt_money_kr(v: float) -> str:
    """금액을 '억'과 '만원' 단위의 한글 문자열로 포맷합니다."""
    if v is None:
        return "-"
    man = int(round(v / 10_000))
    if man >= 10_000:
        uk = man // 10_000
        rem = man % 10_000
        return f"{uk}억 {rem:,}만원" if rem > 0 else f"{uk}억"
    return f"{man:,}만원"


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


def main(portfolio_path: Optional[str] = None):
    # Load initial state from portfolio file.
    portfolio_data = load_portfolio_data(portfolio_path)

    if not portfolio_data:
        print("오류: 포트폴리오 파일(portfolio_*.json)을 찾을 수 없습니다. --portfolio 옵션으로 파일을 지정하거나 data/ 폴더에 파일을 위치시켜주세요.")
        print("웹 UI(web_app.py)를 실행하여 새 포트폴리오를 생성할 수 있습니다.")
        return

    print(f"포트폴리오 파일 '{os.path.basename(portfolio_data['filepath'])}'을(를) 기준으로 오늘의 액션을 계산합니다.")
    holdings = portfolio_data.get('holdings', {})
    init_cap = float(portfolio_data.get('total_equity', 0.0))
    # Build a complete list of tickers from tickers.txt and current holdings
    pairs = build_pairs_with_holdings(read_tickers_file('data/tickers.txt'), holdings)
    if not pairs:
        print('티커를 찾을 수 없습니다. data/tickers.txt 파일을 채워주세요.')
        return

    # Fetch recent data for signals
    rows = []
    total_holdings = 0.0
    datestamps = []
    data_by_tkr = {}
    for tkr, _ in pairs:
        df = fetch_ohlcv(tkr, months_range=[1,0])
        if df is None or len(df) < 11:
            continue
        close = df['Close']
        i = len(close) - 1
        c0 = float(close.iloc[i])
        c5 = float(close.iloc[i-5]) if i-5 >= 0 else c0
        c10 = float(close.iloc[i-10]) if i-10 >= 0 else (c5 if i-5>=0 else c0)
        p1 = round(((c0/c5)-1.0)*100.0, 1) if c5>0 else 0.0
        p2 = round(((c5/c10)-1.0)*100.0, 1) if c10>0 else 0.0
        s2 = p1 + p2
        # ST direction
        try:
            st_dir = supertrend_direction(df, int(getattr(settings,'ST_ATR_PERIOD',14)), float(getattr(settings,'ST_ATR_MULTIPLIER',3.0)))
            stv = int(st_dir.iloc[-1]) if len(st_dir)>0 else 0
        except Exception:
            stv = 0
        sh = int((holdings.get(tkr) or {}).get('shares') or 0)
        ac = float((holdings.get(tkr) or {}).get('avg_cost') or 0.0)
        total_holdings += sh * c0
        datestamps.append(df.index[-1])
        data_by_tkr[tkr] = {
            'price': c0,
            'p1': p1,
            'p2': p2,
            's2': s2,
            'st': stv,
            'shares': sh,
            'avg_cost': ac,
        }

    dt = max(datestamps) if datestamps else pd.Timestamp.now()
    # Determine trading-calendar-based label/date via pykrx
    ref_ticker = next(iter(data_by_tkr.keys())) if data_by_tkr else None
    def is_trading_day(d: pd.Timestamp) -> bool:
        if _stock is None or ref_ticker is None:
            # Fallback: assume trading day only if equals last data date
            return d.date() == pd.to_datetime(dt).date()
        try:
            s = d.strftime('%Y%m%d')
            df = _stock.get_market_ohlcv_by_date(s, s, ref_ticker)
            return df is not None and len(df) > 0
        except Exception:
            return False
    # Decide label and date to display
    today_cal = pd.Timestamp.now().normalize()
    if is_trading_day(today_cal):
        day_label = '오늘'
        label_date = today_cal
    else:
        # find next trading day within next 14 calendar days
        nd = today_cal + pd.Timedelta(days=1)
        for _ in range(14):
            if is_trading_day(nd):
                label_date = nd
                break
            nd += pd.Timedelta(days=1)
        else:
            # fallback to last data date
            label_date = pd.to_datetime(dt)
        day_label = '다음 거래일'

    denom = int(getattr(settings, 'PORTFOLIO_TOPN', 10))
    # Count held tickers from holdings snapshot
    held_count = sum(1 for v in holdings.values() if int((v or {}).get('shares') or 0) > 0)

    label_date_str = pd.to_datetime(label_date).strftime('%Y-%m-%d')
    total_cash = float(init_cap) - float(total_holdings)
    total_value = total_holdings + max(0.0, total_cash)
    header_line = (
        f"{day_label} {label_date_str} - 보유종목 {held_count} "
        f"잔액(보유+현금): {_fmt_money_kr(total_value)} (보유 {_fmt_money_kr(total_holdings)} + 현금 {_fmt_money_kr(total_cash)})"
    )

    # Decide next action per ticker
    sell_thr = float(getattr(settings, 'SELL_SUM_THRESHOLD', -3.0))
    buy_thr = float(getattr(settings, 'BUY_SUM_THRESHOLD', 3.0))
    stop_loss = getattr(settings, 'HOLDING_STOP_LOSS_PCT', None)
    max_pos = float(getattr(settings, 'MAX_POSITION_PCT', 1.0))
    min_pos = float(getattr(settings, 'MIN_POSITION_PCT', 0.0))

    actions = []  # (notional, row)
    for tkr, name in pairs:
        d = data_by_tkr.get(tkr)
        if not d:
            continue
        price = d['price']
        sh = int(d['shares'])
        ac = float(d['avg_cost'] or 0.0)
        p1 = d['p1']; p2 = d['p2']; s2 = d['s2']; stv = int(d['st'])
        state = 'HOLD' if sh>0 else 'WAIT'
        phrase = ''
        qty = 0
        notional = 0.0
        # Current holding return
        hold_ret = ((price/ac)-1.0)*100.0 if (sh>0 and ac>0) else None
        # TRIM if exceeding cap
        equity = total_value
        curr_val = sh * price
        cap_val = max_pos * equity
        if sh>0 and curr_val > cap_val and price>0:
            to_sell_val = curr_val - cap_val
            qty = int(to_sell_val // price)
            if qty>0:
                state = 'TRIM'
                notional = qty * price
                phrase = f"부분매도 {qty}주 @ {int(round(price)):,}"
        # CUT stop loss
        elif sh>0 and stop_loss is not None and ac>0:
            curr_hold = ((price/ac)-1.0)*100.0
            if curr_hold <= float(stop_loss):
                state = 'CUT'
                qty = sh
                notional = qty * price
                phrase = f"손절 {qty}주 @ {int(round(price)):,}"
        # SELL if s2 + hold_ret < thr
        elif sh>0 and ac>0 and hold_ret is not None and (s2 + hold_ret) < sell_thr:
            state = 'SELL'
            qty = sh
            notional = qty * price
            phrase = f"이익실현 {qty}주 @ {int(round(price)):,}"
        # BUY if eligible and cash allows reaching MIN position
        elif sh==0 and s2>buy_thr and stv>0 and price>0:
            equity = total_value
            min_val = min_pos * equity
            cap_val = max_pos * equity
            need = max(0.0, min_val - 0.0)
            budget_cap = max(0.0, cap_val - 0.0)
            budget = min(total_cash, budget_cap)
            from math import ceil
            req_qty = int(ceil(need / price)) if price>0 else 0
            if req_qty>0 and (req_qty*price) <= budget:
                qty = req_qty
                state = 'BUY'
                notional = qty * price
                phrase = f"매수 {qty}주 @ {int(round(price)):,}"
            else:
                phrase = '현금 부족' if total_cash < need else ('상한 제한' if budget_cap < need else '')

        amount = sh * price
        day_ret = 0.0  # not computed in light mode consistently per ticker
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
            _fmt_money_kr(amount),
            (f"{hold_ret:+.1f}%" if hold_ret is not None else '-'),
            f"{(amount/total_value*100.0) if total_value>0 else 0.0:.0f}%",
            f"{p1:+.1f}%",
            f"{p2:+.1f}%",
            f"{s2:+.1f}%",
            ('+1' if stv>0 else ('-1' if stv<0 else '0')),
            phrase,
        ])
        actions.append((notional, rows[-1]))

    # Sort by action notional desc, then by amount desc
    actions.sort(key=lambda x: x[0], reverse=True)
    rows_sorted = []
    for i, (_, row) in enumerate(actions, 1):
        row[0] = i
        rows_sorted.append(row)

    headers = ['#','티커','이름','상태','매수일','보유일','현재가','일간수익률','보유수량','금액','누적수익률','비중','1주','2주','합계','ST','문구']
    aligns = ['right','right','left','center','left','right','right','right','right','right','right','right','right','right','right','center','left']
    
    # Render table for both console and log file
    amb_wide_console = bool(getattr(settings, 'EAW_AMBIGUOUS_AS_WIDE', True))
    table_lines = render_table_eaw(headers, rows_sorted, aligns, amb_wide=amb_wide_console)

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'today.log')
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
