import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import os
import importlib
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd

import settings
from utils.data_loader import read_tickers_file, read_holdings_file
from utils.report import render_table_eaw


def main(strategy_name: str = 'jason'):
    pairs = read_tickers_file('data/tickers.txt')
    if not pairs:
        print('data/tickers.txt 파일이 비어있거나 없습니다.')
        return

    initial_capital = float(getattr(settings, 'INITIAL_CAPITAL', 1_000_000.0))
    months_range = getattr(settings, 'MONTHS_RANGE', [12, 0])

    # Determine core start date label (without warmup)
    try:
        from datetime import datetime as _dt
        now = pd.to_datetime(_dt.now().strftime('%Y%m%d'))
        core_start_dt = now - pd.DateOffset(months=int(months_range[0]))
        period_label = f"{int(months_range[0])}개월~{'현재' if int(months_range[1])==0 else str(int(months_range[1]))+'개월'}"
    except Exception:
        core_start_dt = None
        period_label = str(months_range)

    # Setup tee to also write console output into logs/test.log
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'test.log')
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    log_f = open(log_path, 'w', encoding='utf-8')
    # Header in log
    log_f.write(f"# 시작 {datetime.now().isoformat()} | MONTHS_RANGE={months_range} | 초기자본={int(initial_capital):,}\n")
    log_f.flush()
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(sys.stdout, log_f)

    # 전략 모듈 로드
    try:
        strategy_module = importlib.import_module(f"logics.{strategy_name}")
        portfolio_topn_series = getattr(strategy_module, 'portfolio_topn_series')
        simple_daily_series = getattr(strategy_module, 'simple_daily_series')
        print(f"[{strategy_name} 전략]을 사용하여 백테스트를 실행합니다.")
    except (ImportError, AttributeError):
        print(f"오류: '{strategy_name}' 전략을 찾을 수 없거나, logics/{strategy_name}.py 파일에 portfolio_topn_series 또는 simple_daily_series 함수가 없습니다.")
        log_f.close()
        sys.stdout = _orig_stdout
        return

    print(f"[settings] INITIAL_CAPITAL={int(initial_capital):,}원, MONTHS_RANGE={months_range} | {period_label}")

    # Fetch and simulate per ticker or portfolio
    per_ticker_ts: Dict[str, pd.DataFrame] = {}
    name_by_ticker: Dict[str, str] = {t: n for t, n in pairs}
    portfolio_topn = int(getattr(settings, 'PORTFOLIO_TOPN', 0) or 0)
    if portfolio_topn > 0:
        # Optional: load initial holdings if provided
        try:
            init_positions = read_holdings_file('data/holdings.csv')
        except Exception:
            init_positions = None
        per_ticker_ts = portfolio_topn_series(
            pairs,
            months_range=months_range,
            initial_capital=initial_capital,
            core_start_date=core_start_dt,
            top_n=portfolio_topn,
            initial_positions=init_positions,
        ) or {}
        # Add display name for CASH row
        if 'CASH' in per_ticker_ts:
            name_by_ticker['CASH'] = '현금'
    else:
        for ticker, name in pairs:
            ts = simple_daily_series(
                ticker,
                df=None,
                months_back=None,
                months_range=months_range,
                initial_capital=initial_capital,
                core_start_date=core_start_dt,
            )
            if not ts.empty:
                per_ticker_ts[ticker] = ts

    if not per_ticker_ts:
        print('시뮬레이션할 유효한 데이터가 없습니다.')
        return

    # Align on common dates across all tickers (intersection)
    common_index = None
    for tkr, ts in per_ticker_ts.items():
        common_index = ts.index if common_index is None else common_index.intersection(ts.index)
    if common_index is None or len(common_index) == 0:
        print('종목들 간에 공통된 거래일이 없습니다.')
        return

    # Iterate day by day and print block:
    # ____________________________________________________________________________
    # 날짜 - 보유종목 X/Y 평가금액 1234만원 수익률 +0.8% 현금 400만원 비중 CASH 35%
    # 1. 449450\tPLUS K방산                   | 수익률   3.0%
    # ...
    # ____________________________________________________________________________
    prev_total_pv = None
    prev_dt: Optional[pd.Timestamp] = None
    # Trackers for buy date and holding days per ticker
    buy_date_by_ticker: Dict[str, Optional[pd.Timestamp]] = {}
    hold_days_by_ticker: Dict[str, int] = {}
    total_cnt = len(per_ticker_ts)
    portfolio_topn = int(getattr(settings, 'PORTFOLIO_TOPN', 0) or 0)
    # Portfolio initial capital across all tickers (for cumulative return)
    try:
        if portfolio_topn > 0:
            total_init = float(initial_capital)
        else:
            total_init = float(initial_capital) * float(total_cnt)
    except Exception:
        total_init = 0.0
    try:
        for dt in common_index:
            # Aggregate
            total_value = 0.0
            total_holdings = 0.0
            held_count = 0
            for tkr, ts in per_ticker_ts.items():
                row = ts.loc[dt]
                total_value += float(row.get('pv', 0.0))
                if tkr != 'CASH':
                    sh = int(row.get('shares', 0))
                    price = float(row.get('price') or 0.0)
                    total_holdings += price * sh
                    if sh > 0:
                        held_count += 1

            total_cash = total_value - total_holdings

            if total_value <= 0:
                continue

            # Daily portfolio return
            if prev_total_pv is None:
                day_ret_pct = 0.0
            else:
                day_ret_pct = ((total_value / prev_total_pv) - 1.0) * 100.0 if prev_total_pv > 0 else 0.0
            prev_total_pv = total_value

            # Cumulative portfolio return from initial capital
            cum_ret_pct = ((total_value / total_init) - 1.0) * 100.0 if total_init and total_init > 0 else 0.0

            # Header line
            def _fmt_mw(v: float) -> str:
                man = int(round(v / 10_000))
                if man >= 10_000:
                    uk = man // 10_000
                    rem = man % 10_000
                    return f"{uk}억 {rem:,}만원" if rem else f"{uk}억"
                return f"{man:,}만원"
            # In portfolio mode, show denominator as TOPN slots
            denom = portfolio_topn if portfolio_topn > 0 else total_cnt
            date_str = pd.to_datetime(dt).strftime('%Y-%m-%d')
            print(
                f"{date_str} - 보유종목 {held_count}/{denom} 잔액(보유+현금): {_fmt_mw(total_value)} (보유 {_fmt_mw(total_holdings)} + 현금 {_fmt_mw(total_cash)}) "
                f"금일 수익률 {day_ret_pct:+.1f}%, 누적 수익률 {cum_ret_pct:+.1f}%"
            )

            # 포지션 표: 모든 티커 표시 (BUY/HOLD/SELL/WAIT), 정렬은 HOLD 먼저, 비중 내림차순
            rows = []
            # Add 현재가/일간수익률/보유수량/금액/누적수익률 and diagnostics (1주/2주/합계/ST), plus 매수일/보유일
            # Place Profit | P/L at the far right
            headers = [
                "#", "티커", "이름", "상태", "매수일", "보유일",
                "현재가", "일간수익률", "보유수량", "금액", "누적수익률",
                "비중", "1주", "2주", "합계", "ST", "문구"
            ]
            def _status_rank(s: str) -> int:
                s = (s or '').upper()
                return 0 if s == 'HOLD' else 1

            # Build raw entries
            for tkr, ts in per_ticker_ts.items():
                row = ts.loc[dt]
                name = name_by_ticker.get(tkr, '')
                decision = str(row.get('decision', '')).upper()
                price_today = float(row['price']) if 'price' in row else 0.0
                shares = int(row['shares'])
                # Default display values
                disp_price = price_today
                disp_shares = shares
                # trade_amount from logic: use for BUY/SELL; for HOLD/WAIT show holding value
                trade_amount = float(row.get('trade_amount', 0.0))
                amount = trade_amount if decision in ('BUY','SELL','CUT','TRIM','SELL_TRIM') else (shares * price_today)
                # per-ticker day return
                try:
                    price_prev = float(ts.loc[prev_dt]['price']) if (prev_dt is not None and prev_dt in ts.index) else None
                except Exception:
                    price_prev = None
                if price_today and price_prev and price_prev > 0:
                    tkr_day_ret = (price_today / price_prev - 1.0) * 100.0
                else:
                    tkr_day_ret = 0.0
                # weight based on total portfolio pv
                # Weight calculation: in portfolio mode, use holdings value; else pv/total
                if portfolio_topn > 0:
                    w_val = (shares * price_today)
                else:
                    w_val = float(row['pv'])
                w = (w_val / total_value * 100.0) if total_value > 0 else 0.0
                # Special handling for CASH pseudo-row in portfolio mode: show actual cash weight and amount
                if portfolio_topn > 0 and tkr == 'CASH':
                    disp_price = 1
                    disp_shares = 1
                    amount = total_cash
                    w = (total_cash / total_value * 100.0) if total_value > 0 else 0.0
                # realized profit for SELL/CUT/TRIM
                prof = float(row.get('trade_profit', 0.0)) if decision in ('SELL','CUT','TRIM','SELL_TRIM') else 0.0
                plpct = float(row.get('trade_pl_pct', 0.0)) if decision in ('SELL','CUT','TRIM','SELL_TRIM') else 0.0
                # holding cumulative return (누적수익률) based on avg_cost
                avg_cost = float(row.get('avg_cost', 0.0)) if row.get('avg_cost', None) is not None else 0.0
                if shares > 0 and avg_cost > 0:
                    hold_ret_pct = (price_today / avg_cost - 1.0) * 100.0
                    hold_ret_str = f"{hold_ret_pct:+.1f}%"
                else:
                    hold_ret_str = "-"
                # diagnostics
                p1 = row.get('p1', None)
                p2 = row.get('p2', None)
                ssum = row.get('s2_sum', None)
                stv = int(row.get('st_dir', 0)) if row.get('st_dir', None) is not None else 0
                def _fmt_pct(x):
                    try:
                        return f"{float(x):+,.1f}%"
                    except Exception:
                        return "-"
                # Show ST direction as +1 / -1 (0 if neutral/unavailable)
                if stv > 0:
                    st_str = '+1'
                elif stv < 0:
                    st_str = '-1'
                else:
                    st_str = '0'
                # Status display in English (WAIT/BUY/HOLD/SELL/TRIM)
                display_status = 'TRIM' if decision == 'SELL_TRIM' else decision
                # Phrase with trade summary when events occur; quantities are integers
                phrase = ''
                if decision in ('BUY', 'SELL', 'CUT', 'TRIM', 'SELL_TRIM') and amount and price_today:
                    qty_calc = int(float(amount) // float(price_today)) if float(price_today) > 0 else 0
                    if decision == 'BUY':
                        phrase = f"매수 {qty_calc}주 @ {int(round(price_today)):,}"
                    else:
                        tag = '이익실현' if decision == 'SELL' else ('손절' if decision == 'CUT' else '부분매도')
                        prof_str = _fmt_mw(prof)
                        pl_str = f"{plpct:+.1f}%"
                        phrase = f"{tag} {qty_calc}주 @ {int(round(price_today)):,} 수익 {prof_str} 손익률 {pl_str}"
                elif decision in ('WAIT','HOLD'):
                    phrase = str(row.get('note', '') or '')
                # Buy date and holding days tracking
                if tkr not in buy_date_by_ticker:
                    buy_date_by_ticker[tkr] = None
                    hold_days_by_ticker[tkr] = 0
                if decision == 'BUY' and shares > 0:
                    buy_date_by_ticker[tkr] = dt
                    hold_days_by_ticker[tkr] = 1  # BUY 당일을 1일로 간주
                elif shares > 0:
                    if buy_date_by_ticker.get(tkr) is None:
                        buy_date_by_ticker[tkr] = dt
                        hold_days_by_ticker[tkr] = 1  # 최초 보유 시작일도 1일로 시작
                    else:
                        hold_days_by_ticker[tkr] = hold_days_by_ticker.get(tkr, 0) + 1
                else:
                    buy_date_by_ticker[tkr] = None
                    hold_days_by_ticker[tkr] = 0
                bd = buy_date_by_ticker.get(tkr)
                bd_str = pd.to_datetime(bd).strftime('%Y-%m-%d') if bd is not None else '-'
                hd = hold_days_by_ticker.get(tkr, 0)
                rows.append([
                    0,  # placeholder for index, set later
                    tkr,
                    name,
                    display_status,
                    bd_str,
                    f"{hd}",
                    f"{int(round(disp_price)):,}",  # 현재가(현금은 1)
                    f"{tkr_day_ret:+.1f}%",         # 일간수익률
                    f"{disp_shares:,}",              # 보유수량(현금은 1)
                    (_fmt_mw(amount) if amount is not None else "-"),  # 금액
                    hold_ret_str,                     # 누적수익률
                    f"{w:.0f}%",
                    (_fmt_pct(p1) if p1 is not None else "-"),
                    (_fmt_pct(p2) if p2 is not None else "-"),
                    (_fmt_pct(ssum) if ssum is not None else "-"),
                    st_str,
                    phrase,
                _status_rank(decision), -w])

            # Sort: HOLD first, then by weight desc; stable within groups by ticker
            rows.sort(key=lambda r: (r[-2], r[-1], r[1]))
            # Renumber and drop sort keys
            for i, r in enumerate(rows, 1):
                r[0] = i
                del r[-1]
                del r[-1]

            aligns = [
                'right',  # #
                'right',  # 티커
                'left',   # 이름
                'center', # 상태
                'left',   # 매수일
                'right',  # 보유일
                'right',  # 현재가
                'right',  # 일간수익률
                'right',  # 보유수량
                'right',  # 금액
                'right',  # 누적수익률
                'right',  # 비중
                'right',  # 1주
                'right',  # 2주
                'right',  # 합계
                'center', # ST
                'left',   # 문구
            ]
            str_rows = [[str(c) for c in row] for row in rows]

            # Render tables for console and log with different width settings
            # Console: Use setting for ambiguous width
            # Log file: Treat ambiguous as narrow for better compatibility
            amb_wide_console = bool(getattr(settings, 'EAW_AMBIGUOUS_AS_WIDE', True))
            lines_console = render_table_eaw(headers, str_rows, aligns, amb_wide=amb_wide_console)
            lines_log = render_table_eaw(headers, str_rows, aligns, amb_wide=False)

            # Bypass Tee to write distinct renderings
            for ln in lines_console:
                try:
                    _orig_stdout.write(ln + "\n")
                except Exception:
                    pass
            for ln in lines_log:
                try:
                    log_f.write(ln + "\n")
                except Exception:
                    pass
            prev_dt = dt
            print() # Add a blank line between daily reports
    finally:
        # Restore stdout and close file
        sys.stdout = _orig_stdout
        try:
            log_f.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
