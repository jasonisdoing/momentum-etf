import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import os
import importlib
import glob
import json
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd

import settings
from utils.data_loader import read_tickers_file, read_holdings_file
from utils.report import render_table_eaw


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


def main(strategy_name: str = 'jason', portfolio_path: Optional[str] = None, quiet: bool = False):
    # --test 실행 시 포트폴리오 파일을 사용하지 않고 settings.py의 설정을 사용합니다.
    # portfolio_path 인자는 무시됩니다.
    initial_capital = float(getattr(settings, 'INITIAL_CAPITAL', 100_000_000))
    init_positions = {}  # 백테스트는 보유 종목 없이 시작

    # 기간 설정 로직
    test_date_range = getattr(settings, 'TEST_DATE_RANGE', None)
    months_range = None
    core_start_dt = None
    period_label = ""

    if test_date_range and len(test_date_range) == 2:
        try:
            core_start_dt = pd.to_datetime(test_date_range[0])
            core_end_dt = pd.to_datetime(test_date_range[1])
            period_label = f"{core_start_dt.strftime('%Y-%m-%d')}~{core_end_dt.strftime('%Y-%m-%d')}"
        except (ValueError, TypeError):
            print("오류: settings.py의 TEST_DATE_RANGE 형식이 잘못되었습니다. ['YYYY-MM-DD', 'YYYY-MM-DD'] 형식으로 지정해주세요.")
            test_date_range = None # Invalidate
    
    if not test_date_range:
        months_range = getattr(settings, 'MONTHS_RANGE', [12, 0]) # Fallback
        core_start_dt = (pd.Timestamp.now() - pd.DateOffset(months=int(months_range[0]))).normalize()
        period_label = f"{int(months_range[0])}개월~{'현재' if int(months_range[1])==0 else str(int(months_range[1]))+'개월'}"

    # 티커 목록 결정
    if not quiet:
        print(f"\n[고정 유니버스] data/tickers.txt 파일의 종목을 사용합니다.")
    pairs = read_tickers_file('data/tickers.txt')

    if not pairs:
        print('오류: 백테스트에 사용할 티커를 찾을 수 없습니다.')
        print('      data/tickers.txt 파일이 비어있거나 존재하지 않을 수 있습니다.')
        return
    
    _orig_stdout = sys.stdout
    log_f = None
    return_value = None

    try:
        if not quiet:
            # Setup tee to also write console output into logs/test.log
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'test.log')
            class _Tee:
                def __init__(self, *streams): self.streams = streams
                def write(self, data):
                    for s in self.streams:
                        try: s.write(data)
                        except Exception: pass
                def flush(self):
                    for s in self.streams:
                        try: s.flush()
                        except Exception: pass
                def close(self):
                    pass
            log_f = open(log_path, 'w', encoding='utf-8')
            sys.stdout = _Tee(sys.stdout, log_f)
        else:
            sys.stdout = open(os.devnull, 'w')

        # Header in log
        print(f"백테스트를 `settings.py` 설정으로 실행합니다.")
        if log_f:
            log_f.write(f"# 시작 {datetime.now().isoformat()} | 기간={period_label} | 초기자본={int(initial_capital):,}\n")

        # 전략 모듈 로드
        try:
            # 패키지 기반 전략(e.g., logics/jason/strategy.py)을 먼저 시도
            strategy_module = importlib.import_module(f"logics.{strategy_name}.strategy")
            try:
                strategy_settings_module = importlib.import_module(f"logics.{strategy_name}.settings")
            except ImportError:
                strategy_settings_module = None
        except ImportError:
            try:
                # 단일 파일 기반 전략(e.g., logics/my_strategy.py)으로 폴백
                strategy_module = importlib.import_module(f"logics.{strategy_name}")
                strategy_settings_module = None
            except ImportError:
                print(f"오류: '{strategy_name}' 전략을 찾을 수 없습니다. logics/{strategy_name}/strategy.py 또는 logics/{strategy_name}.py 파일을 확인해주세요.")
                return

        try:
            run_portfolio_backtest = getattr(strategy_module, 'run_portfolio_backtest')
            run_single_ticker_backtest = getattr(strategy_module, 'run_single_ticker_backtest')
            print(f"[{strategy_name} 전략]을 사용하여 백테스트를 실행합니다.")
        except AttributeError:
            print(f"오류: '{strategy_name}' 전략 모듈에 run_portfolio_backtest 또는 run_single_ticker_backtest 함수가 없습니다.")
            return
        
        print(f"[settings] INITIAL_CAPITAL={int(initial_capital):,}원 | 기간={period_label}")
        
        # 1. 모든 티커의 원본 데이터를 로드합니다.
        from utils.data_loader import fetch_ohlcv
        raw_data_by_ticker: Dict[str, pd.DataFrame] = {}
        for ticker, name in pairs:
            df = fetch_ohlcv(ticker, months_range=months_range, date_range=test_date_range)
            if df is not None and not df.empty:
                raw_data_by_ticker[ticker] = df
        
        # 2. 모든 종목에 대한 공통 거래일을 계산합니다.
        common_index = None
        for df in raw_data_by_ticker.values():
            common_index = df.index if common_index is None else common_index.intersection(df.index)

        if common_index is None or common_index.empty:
            if not quiet: print("오류: 모든 종목에 걸쳐 공통된 거래일이 없습니다.")
            return None

        # 3. 공통 시작일이 요청된 시작일을 만족하는지 확인합니다.
        actual_start_date = common_index[0]
        if core_start_dt and actual_start_date > core_start_dt:
            if not quiet:
                late_tickers = {tkr: df.index[0] for tkr, df in raw_data_by_ticker.items() if not df.index.empty and df.index[0] > core_start_dt}
                print(f"\n오류: '{strategy_name}' 전략의 백테스트를 중단합니다.")
                print(f"      요청된 시작일 '{core_start_dt.strftime('%Y-%m-%d')}'보다 실제 가능한 시작일 '{actual_start_date.strftime('%Y-%m-%d')}'이 늦습니다.")
                if late_tickers:
                    print(f"      원인으로 추정되는 종목:")
                    for tkr, s_date in sorted(late_tickers.items(), key=lambda item: item[1]):
                         print(f"      - {tkr} (데이터 시작일: {s_date.strftime('%Y-%m-%d')})")
            return None

        # 4. 미리 로드한 데이터를 사용하여 시뮬레이션을 실행합니다.
        per_ticker_ts: Dict[str, pd.DataFrame] = {}
        name_by_ticker: Dict[str, str] = {t: n for t, n in pairs}
        portfolio_topn = int(getattr(settings, 'PORTFOLIO_TOPN', 0) or 0)
        if portfolio_topn > 0:
            per_ticker_ts = run_portfolio_backtest(
                pairs, months_range=months_range, initial_capital=initial_capital,
                core_start_date=core_start_dt, top_n=portfolio_topn,
                date_range=test_date_range, prefetched_data=raw_data_by_ticker
            ) or {}
            if 'CASH' in per_ticker_ts:
                name_by_ticker['CASH'] = '현금'
        else:
            # 종목별 고정 자본 방식: 전체 자본을 종목 수로 나눔
            capital_per_ticker = initial_capital / len(pairs) if pairs else 0
            for ticker, _ in pairs:
                df_ticker = raw_data_by_ticker.get(ticker)
                if df_ticker is None:
                    continue
                ts = run_single_ticker_backtest(
                    ticker, df=df_ticker, months_range=months_range,
                    initial_capital=capital_per_ticker, core_start_date=core_start_dt, date_range=test_date_range
                )
                if not ts.empty:
                    per_ticker_ts[ticker] = ts

        if not per_ticker_ts:
            if not quiet: print('시뮬레이션할 유효한 데이터가 없습니다.')
            return

        # Align on common dates across all tickers (intersection)
        common_index = None
        for tkr, ts in per_ticker_ts.items():
            common_index = ts.index if common_index is None else common_index.intersection(ts.index)
        if common_index is None or len(common_index) == 0:
            print('종목들 간에 공통된 거래일이 없습니다.')
            return

        portfolio_values = []
        portfolio_dates = []
        prev_total_pv = None
        prev_dt: Optional[pd.Timestamp] = None
        buy_date_by_ticker: Dict[str, Optional[pd.Timestamp]] = {}
        hold_days_by_ticker: Dict[str, int] = {}
        total_cnt = len(per_ticker_ts)

        total_init = float(initial_capital)

        for dt in common_index:
            portfolio_dates.append(dt)

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
            portfolio_values.append(total_value)

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

            if not quiet:
                # Header line
                denom = portfolio_topn if portfolio_topn > 0 else total_cnt
                date_str = pd.to_datetime(dt).strftime('%Y-%m-%d')
                print(
                    f"{date_str} - 보유종목 {held_count}/{denom} 잔액(보유+현금): {_fmt_money_kr(total_value)} (보유 {_fmt_money_kr(total_holdings)} + 현금 {_fmt_money_kr(total_cash)}) "
                    f"금일 수익률 {day_ret_pct:+.1f}%, 누적 수익률 {cum_ret_pct:+.1f}%"
                )

                rows = []
                headers = [
                    "#", "티커", "이름", "상태", "매수일", "보유일", "현재가", "일간수익률",
                    "보유수량", "금액", "누적수익률", "비중", "1주", "2주", "합계", "ST", "문구"
                ]
                def _status_rank(s: str) -> int:
                    s = (s or '').upper()
                    return 0 if s == 'HOLD' else 1

                for tkr, ts in per_ticker_ts.items():
                    row = ts.loc[dt]
                    name = name_by_ticker.get(tkr, '')
                    decision = str(row.get('decision', '')).upper()
                    price_today = float(row['price']) if 'price' in row else 0.0
                    shares = int(row['shares'])
                    disp_price = price_today
                    disp_shares = shares
                    trade_amount = float(row.get('trade_amount', 0.0))
                    amount = trade_amount if decision in ('BUY','SELL','CUT','TRIM') else (shares * price_today)
                    try:
                        price_prev = float(ts.loc[prev_dt]['price']) if (prev_dt is not None and prev_dt in ts.index) else None
                    except Exception:
                        price_prev = None
                    tkr_day_ret = (price_today / price_prev - 1.0) * 100.0 if price_today and price_prev and price_prev > 0 else 0.0
                    w_val = (shares * price_today) if portfolio_topn > 0 else float(row['pv'])
                    w = (w_val / total_value * 100.0) if total_value > 0 else 0.0
                    if portfolio_topn > 0 and tkr == 'CASH':
                        disp_price, disp_shares, amount = 1, 1, total_cash
                        w = (total_cash / total_value * 100.0) if total_value > 0 else 0.0
                    prof = float(row.get('trade_profit', 0.0)) if decision in ('SELL','CUT','TRIM') else 0.0
                    plpct = float(row.get('trade_pl_pct', 0.0)) if decision in ('SELL','CUT','TRIM') else 0.0
                    avg_cost = float(row.get('avg_cost', 0.0)) if row.get('avg_cost', None) is not None else 0.0
                    hold_ret_str = f"{(price_today / avg_cost - 1.0) * 100.0:+.1f}%" if shares > 0 and avg_cost > 0 else "-"
                    p1, p2, ssum = row.get('p1'), row.get('p2'), row.get('s2_sum')
                    stv = int(row.get('st_dir', 0)) if row.get('st_dir', None) is not None else 0
                    def _fmt_pct(x):
                        try: return f"{float(x):+,.1f}%"
                        except Exception: return "-"
                    st_str = '+1' if stv > 0 else ('-1' if stv < 0 else '0')
                    display_status = decision
                    phrase = ''
                    if decision in ('BUY', 'SELL', 'CUT', 'TRIM') and amount and price_today:
                        qty_calc = int(float(amount) // float(price_today)) if float(price_today) > 0 else 0
                        if decision == 'BUY':
                            phrase = f"매수 {qty_calc}주 @ {int(round(price_today)):,}"
                        else:
                            tag = '이익실현' if decision == 'SELL' else ('손절' if decision == 'CUT' else '부분매도')
                            phrase = f"{tag} {qty_calc}주 @ {int(round(price_today)):,} 수익 {_fmt_money_kr(prof)} 손익률 {f'{plpct:+.1f}%'}"
                    elif decision in ('WAIT','HOLD'):
                        phrase = str(row.get('note', '') or '')
                    if tkr not in buy_date_by_ticker:
                        buy_date_by_ticker[tkr], hold_days_by_ticker[tkr] = None, 0
                    if decision == 'BUY' and shares > 0:
                        buy_date_by_ticker[tkr], hold_days_by_ticker[tkr] = dt, 1
                    elif shares > 0:
                        if buy_date_by_ticker.get(tkr) is None:
                            buy_date_by_ticker[tkr], hold_days_by_ticker[tkr] = dt, 1
                        else:
                            hold_days_by_ticker[tkr] = hold_days_by_ticker.get(tkr, 0) + 1
                    else:
                        buy_date_by_ticker[tkr], hold_days_by_ticker[tkr] = None, 0
                    bd = buy_date_by_ticker.get(tkr)
                    bd_str = pd.to_datetime(bd).strftime('%Y-%m-%d') if bd is not None else '-'
                    hd = hold_days_by_ticker.get(tkr, 0)
                    rows.append([
                        0, tkr, name, display_status, bd_str, f"{hd}", f"{int(round(disp_price)):,}",
                        f"{tkr_day_ret:+.1f}%", f"{disp_shares:,}", _fmt_money_kr(amount),
                        hold_ret_str, f"{w:.0f}%", (_fmt_pct(p1) if p1 is not None else "-"),
                        (_fmt_pct(p2) if p2 is not None else "-"), (_fmt_pct(ssum) if ssum is not None else "-"),
                        st_str, phrase, _status_rank(decision), -w
                    ])

                rows.sort(key=lambda r: (r[-2], r[-1], r[1]))
                for idx, r in enumerate(rows, 1):
                    r[0] = idx
                    del r[-1]; del r[-1]

                aligns = ['right','right','left','center','left','right','right','right','right','right','right','right','right','right','right','center','left']
                str_rows = [[str(c) for c in row] for row in rows]

                table_lines = render_table_eaw(headers, str_rows, aligns)

                for ln in table_lines:
                    try: _orig_stdout.write(ln + "\n")
                    except Exception: pass
                if log_f:
                    for ln in table_lines:
                        try: log_f.write(ln + "\n")
                        except Exception: pass
                print()

            prev_dt = dt

        if not portfolio_values:
            print('시뮬레이션 결과가 없습니다.')
        else:
            final_value = portfolio_values[-1]
            peak = -1
            max_drawdown = 0
            for value in portfolio_values:
                if value > peak: peak = value
                if peak > 0:
                    drawdown = (peak - value) / peak
                    if drawdown > max_drawdown: max_drawdown = drawdown
            
            start_date = portfolio_dates[0]
            end_date = portfolio_dates[-1]
            years = (end_date - start_date).days / 365.25
            cagr = 0
            if years > 0 and initial_capital > 0:
                cagr = ((final_value / initial_capital) ** (1 / years)) - 1

            summary = {
                "strategy": strategy_name, "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'), "initial_capital": initial_capital,
                "final_value": final_value, "cagr_pct": cagr * 100, "mdd_pct": max_drawdown * 100,
                "cumulative_return_pct": (final_value / initial_capital - 1) * 100 if initial_capital > 0 else 0
            }

            # 월별/연간 수익률 계산
            if portfolio_values:
                pv_series = pd.Series(portfolio_values, index=pd.to_datetime(portfolio_dates))
                # 수익률 계산을 위해 시작점에 초기 자본을 추가
                start_row = pd.Series([initial_capital], index=[start_date - pd.Timedelta(days=1)])
                pv_series_with_start = pd.concat([start_row, pv_series])

                # 월별 수익률
                monthly_returns = pv_series_with_start.resample('ME').last().pct_change().dropna()
                summary['monthly_returns'] = monthly_returns

                # 월별 누적 수익률
                eom_pv = pv_series.resample('ME').last()
                monthly_cum_returns = (eom_pv / initial_capital - 1).ffill()
                summary['monthly_cum_returns'] = monthly_cum_returns

                # 연간 수익률
                yearly_returns = pv_series_with_start.resample('YE').last().pct_change().dropna()
                summary['yearly_returns'] = yearly_returns

            return_value = summary

            # 종목별 성과 계산
            ticker_summaries = []
            for tkr, ts in per_ticker_ts.items():
                if tkr == 'CASH':
                    continue

                trades = ts[ts['decision'].isin(['SELL', 'CUT', 'TRIM'])]
                total_trades = len(trades)

                # 거래가 있는 종목만 요약에 포함
                if total_trades > 0:
                    winning_trades = len(trades[trades['trade_profit'] > 0])
                    win_rate = (winning_trades / total_trades) * 100.0
                    total_profit = trades['trade_profit'].sum()
                    avg_profit = total_profit / total_trades
                    ticker_summaries.append({
                        'ticker': tkr, 'name': name_by_ticker.get(tkr, ''),
                        'total_trades': total_trades, 'win_rate': win_rate,
                        'total_profit': total_profit, 'avg_profit': avg_profit,
                    })

            if not quiet:
                sys.stdout = _orig_stdout # Print summary to original stdout
                print("\n" + "="*30 + f"\n 백테스트 결과 요약 ({strategy_name}) ".center(30, "=") + "\n" + "="*30)
                print(f"| 기간: {summary['start_date']} ~ {summary['end_date']} ({years:.2f} 년)")
                print(f"| 초기 자본: {_fmt_money_kr(summary['initial_capital'])}")
                print(f"| 최종 자산: {_fmt_money_kr(summary['final_value'])}")
                print(f"| 누적 수익률: {summary['cumulative_return_pct']:+.2f}%")
                print(f"| CAGR (연간 복리 성장률): {summary['cagr_pct']:+.2f}%")
                print(f"| MDD (최대 낙폭): {-summary['mdd_pct']:.2f}%")
                print("="*30)

                # 월별 성과 요약 테이블 출력
                if 'monthly_returns' in summary and not summary['monthly_returns'].empty:
                    print("\n" + "="*30 + f"\n 월별 성과 요약 ".center(30, "=") + "\n" + "="*30)
                    print("(괄호 안은 누적 수익률)")
                    
                    monthly_returns = summary['monthly_returns']
                    yearly_returns = summary['yearly_returns']
                    monthly_cum_returns = summary.get('monthly_cum_returns')

                    pivot_df = monthly_returns.mul(100).to_frame('return').pivot_table(
                        index=monthly_returns.index.year,
                        columns=monthly_returns.index.month,
                        values='return'
                    )

                    if not yearly_returns.empty:
                        yearly_series = yearly_returns.mul(100)
                        yearly_series.index = yearly_series.index.year
                        pivot_df['연간'] = yearly_series
                    # If yearly_returns is empty, the '연간' column will not be added, and .get() will return None later.

                    cum_pivot_df = None
                    if monthly_cum_returns is not None and not monthly_cum_returns.empty:
                        cum_pivot_df = monthly_cum_returns.mul(100).to_frame('cum_return').pivot_table(
                            index=monthly_cum_returns.index.year,
                            columns=monthly_cum_returns.index.month,
                            values='cum_return'
                        )
                    
                    headers = ["연도"] + [f'{m}월' for m in range(1, 13)] + ["연간"]
                    rows_data = []
                    for year, row in pivot_df.iterrows():
                        row_data = [str(year)]
                        for month in range(1, 13):
                            val = row.get(month)
                            if pd.notna(val):
                                cell_str = f"{val:+.2f}%"
                                if cum_pivot_df is not None and year in cum_pivot_df.index and month in cum_pivot_df.columns and pd.notna(cum_pivot_df.loc[year, month]):
                                    cum_val = cum_pivot_df.loc[year, month]
                                    cell_str += f" ({cum_val:+.2f}%)"
                                row_data.append(cell_str)
                            else:
                                row_data.append("-")
                        yearly_val = row.get('연간')
                        row_data.append(f"{yearly_val:+.2f}%" if pd.notna(yearly_val) else "-")
                        rows_data.append(row_data)
                    aligns = ['left'] + ['right'] * (len(headers) - 1)
                    print("\n".join(render_table_eaw(headers, rows_data, aligns)))

                # 종목별 성과 요약 테이블 출력
                if ticker_summaries:
                    print("\n" + "="*30 + f"\n 종목별 성과 요약 ".center(30, "=") + "\n" + "="*30)
                    headers = ["티커", "종목명", "거래횟수", "승률", "총손익", "평균손익"]
                    
                    sorted_summaries = sorted(ticker_summaries, key=lambda x: x['total_profit'], reverse=True)
                    
                    rows = [[s['ticker'], s['name'], f"{s['total_trades']}회", f"{s['win_rate']:.1f}%", _fmt_money_kr(s['total_profit']), _fmt_money_kr(s['avg_profit'])] for s in sorted_summaries]
                    
                    aligns = ['right', 'left', 'right', 'right', 'right', 'right']
                    table_lines = render_table_eaw(headers, rows, aligns)
                    print("\n".join(table_lines))

    finally:
        # Restore stdout and close file
        if sys.stdout != _orig_stdout:
            sys.stdout.close()
        sys.stdout = _orig_stdout
        if log_f:
            try: log_f.close()
            except Exception: pass
    
    return return_value


if __name__ == '__main__':
    main(quiet=False)
