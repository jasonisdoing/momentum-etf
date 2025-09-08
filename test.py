import importlib
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

import settings
from utils.data_loader import read_tickers_file
from utils.report import format_kr_money, render_table_eaw

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


def main(
    strategy_name: str = "jason",
    portfolio_path: Optional[str] = None,
    quiet: bool = False,
):
    # --test 실행 시 포트폴리오 파일을 사용하지 않고 settings.py의 설정을 사용합니다.
    # portfolio_path 인자는 무시됩니다.
    initial_capital = float(getattr(settings, "INITIAL_CAPITAL", 100_000_000))

    # 기간 설정 로직
    test_date_range = getattr(settings, "TEST_DATE_RANGE", None)
    months_range = None
    core_start_dt = None
    period_label = ""

    if test_date_range and len(test_date_range) == 2:
        try:
            core_start_dt = pd.to_datetime(test_date_range[0])
            core_end_dt = pd.to_datetime(test_date_range[1])
            period_label = (
                f"{core_start_dt.strftime('%Y-%m-%d')}~{core_end_dt.strftime('%Y-%m-%d')}"
            )
        except (ValueError, TypeError):
            print(
                "오류: settings.py의 TEST_DATE_RANGE 형식이 잘못되었습니다. "
                "['YYYY-MM-DD', 'YYYY-MM-DD'] 형식으로 지정해주세요."
            )
            test_date_range = None  # Invalidate

    if not test_date_range:
        months_range = getattr(settings, "MONTHS_RANGE", [12, 0])  # Fallback
        core_start_dt = pd.Timestamp.now() - pd.DateOffset(months=int(months_range[0]))
        end_label = "현재" if int(months_range[1]) == 0 else f"{int(months_range[1])}개월"
        period_label = f"{int(months_range[0])}개월~{end_label}"

    # 티커 목록 결정
    if not quiet:
        print("\n[고정 유니버스] data/tickers.txt 파일의 종목을 사용합니다.")
    pairs = read_tickers_file("data/tickers.txt")

    if not pairs:
        print("오류: 백테스트에 사용할 티커를 찾을 수 없습니다.")
        print("      data/tickers.txt 파일이 비어있거나 존재하지 않을 수 있습니다.")
        return

    _orig_stdout = sys.stdout
    log_f = None
    return_value = None

    try:
        if not quiet:
            # Setup tee to also write console output into logs/test.log
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"test_{strategy_name}.log")

            class ConsoleAndFileLogger:
                """콘솔과 파일에 동시에 로그를 기록하는 클래스."""

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

                def close(self):
                    # sys.stdout은 닫지 않도록 처리
                    pass

            log_f = open(log_path, "w", encoding="utf-8")
            sys.stdout = ConsoleAndFileLogger(sys.stdout, log_f)
        else:
            sys.stdout = open(os.devnull, "w")

        # Header in log
        print("백테스트를 `settings.py` 설정으로 실행합니다.")
        if log_f:
            log_f.write(
                f"# 시작 {datetime.now().isoformat()} | 기간={period_label} | 초기자본={int(initial_capital):,}\n"
            )

        # 전략 모듈 로드
        try:
            # 패키지 기반 전략(e.g., logics/jason/strategy.py)을 먼저 시도
            strategy_module = importlib.import_module(f"logics.{strategy_name}.strategy")
        except ImportError:
            try:
                # 단일 파일 기반 전략(e.g., logics/my_strategy.py)으로 폴백
                strategy_module = importlib.import_module(f"logics.{strategy_name}")
            except ImportError:
                print(
                    f"오류: '{strategy_name}' 전략을 찾을 수 없습니다. logics/{strategy_name}/strategy.py "
                    f"또는 logics/{strategy_name}.py 파일을 확인해주세요."
                )
                return

        # 전략 모듈에서 백테스트 함수를 가져옵니다.
        try:
            run_portfolio_backtest = getattr(strategy_module, "run_portfolio_backtest")
            run_single_ticker_backtest = getattr(strategy_module, "run_single_ticker_backtest")
        except AttributeError:
            print(
                f"오류: '{strategy_name}' 전략 모듈에 run_portfolio_backtest 또는 "
                "run_single_ticker_backtest 함수가 정의되지 않았습니다."
            )
            return

        # 시뮬레이션 실행
        per_ticker_ts: Dict[str, pd.DataFrame] = {}
        name_by_ticker: Dict[str, str] = {t: n for t, n in pairs}
        # 전역 설정에서 PORTFOLIO_TOPN을 가져옵니다.
        portfolio_topn = int(getattr(settings, "PORTFOLIO_TOPN", 0) or 0)
        if portfolio_topn > 0:
            per_ticker_ts = (
                run_portfolio_backtest(
                    pairs,
                    months_range=months_range,
                    initial_capital=initial_capital,
                    core_start_date=core_start_dt,
                    top_n=portfolio_topn,
                    date_range=test_date_range,
                )
                or {}
            )
            if "CASH" in per_ticker_ts:
                name_by_ticker["CASH"] = "현금"
        else:
            # 개별 종목 백테스트는 여전히 데이터를 미리 로드해야 합니다.
            from utils.data_loader import fetch_ohlcv

            raw_data_by_ticker: Dict[str, pd.DataFrame] = {}
            for ticker, name in pairs:
                df = fetch_ohlcv(ticker, months_range=months_range, date_range=test_date_range)
                if df is not None and not df.empty:
                    raw_data_by_ticker[ticker] = df

            # 종목별 고정 자본 방식: 전체 자본을 종목 수로 나눔
            capital_per_ticker = initial_capital / len(pairs) if pairs else 0
            for ticker, _ in pairs:
                df_ticker = raw_data_by_ticker.get(ticker)
                if df_ticker is None:
                    continue
                ts = run_single_ticker_backtest(
                    ticker,
                    df=df_ticker,
                    months_range=months_range,
                    initial_capital=capital_per_ticker,
                    core_start_date=core_start_dt,
                    date_range=test_date_range,
                )
                if not ts.empty:
                    per_ticker_ts[ticker] = ts

        if not per_ticker_ts:
            if not quiet:
                print("시뮬레이션할 유효한 데이터가 없습니다.")
            return

        # Align on common dates across all tickers (intersection)
        common_index = None
        for tkr, ts in per_ticker_ts.items():
            common_index = ts.index if common_index is None else common_index.intersection(ts.index)
        if common_index is None or len(common_index) == 0:
            print("종목들 간에 공통된 거래일이 없습니다.")
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

                # NaN 값을 안전하게 처리하여 total_value를 계산합니다.
                pv_val = row.get("pv")
                total_value += float(pv_val) if pd.notna(pv_val) else 0.0

                if tkr != "CASH":
                    sh = int(row.get("shares", 0))

                    # NaN 값을 안전하게 처리하여 price를 가져옵니다.
                    price_val = row.get("price")
                    price = float(price_val) if pd.notna(price_val) else 0.0

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
                day_ret_pct = (
                    ((total_value / prev_total_pv) - 1.0) * 100.0 if prev_total_pv > 0 else 0.0
                )
            prev_total_pv = total_value

            # Cumulative portfolio return from initial capital
            cum_ret_pct = (
                ((total_value / total_init) - 1.0) * 100.0 if total_init and total_init > 0 else 0.0
            )

            if not quiet:
                # Header line
                denom = portfolio_topn if portfolio_topn > 0 else total_cnt
                date_str = pd.to_datetime(dt).strftime("%Y-%m-%d")
                print(
                    (
                        f"{date_str} - 보유종목 {held_count}/{denom} "
                        f"잔액(보유+현금): {format_kr_money(total_value)} "
                        f"(보유 {format_kr_money(total_holdings)} + 현금 {format_kr_money(total_cash)}) "
                        f"금일 수익률 {day_ret_pct:+.1f}%, 누적 수익률 {cum_ret_pct:+.1f}%"
                    )
                )

                # 전략에 따라 동적으로 헤더를 설정합니다.
                if strategy_name == "jason":
                    signal_headers = ["1주수익", "2주수익", "모멘텀점수", "ST"]
                elif strategy_name == "seykota":
                    signal_headers = ["단기MA", "장기MA", "MA스코어", "필터"]
                elif strategy_name == "donchian":
                    signal_headers = ["이평선(값)", "이평선(기간)", "이격도", "신호지속일"]
                else:
                    signal_headers = ["신호1", "신호2", "점수", "필터"]

                rows = []
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

                actions = []
                for tkr, ts in per_ticker_ts.items():
                    row = ts.loc[dt]
                    name = name_by_ticker.get(tkr, "")
                    decision = str(row.get("decision", "")).upper()

                    # NaN 값에 대한 안정성 강화: 모든 숫자 변수를 사용 전에 확인하고 처리합니다.
                    price_val = row.get("price")
                    price_today = float(price_val) if pd.notna(price_val) else 0.0

                    shares_val = row.get("shares")
                    shares = int(shares_val) if pd.notna(shares_val) else 0

                    trade_amount_val = row.get("trade_amount", 0.0)
                    trade_amount = float(trade_amount_val) if pd.notna(trade_amount_val) else 0.0

                    disp_price = price_today
                    disp_shares = shares
                    is_trade_decision = decision.startswith(("BUY", "SELL", "CUT", "TRIM"))
                    amount = trade_amount if is_trade_decision else (shares * price_today)
                    try:
                        price_prev_val = (
                            ts.loc[prev_dt]["price"]
                            if (prev_dt is not None and prev_dt in ts.index)
                            else None
                        )
                        price_prev = float(price_prev_val) if pd.notna(price_prev_val) else None
                    except Exception:
                        price_prev = None
                    tkr_day_ret = (
                        (price_today / price_prev - 1.0) * 100.0
                        if price_today and price_prev and price_prev > 0
                        else 0.0
                    )
                    w_val = (shares * price_today) if portfolio_topn > 0 else float(row["pv"])
                    w = (w_val / total_value * 100.0) if total_value > 0 else 0.0
                    if portfolio_topn > 0 and tkr == "CASH":
                        disp_price, disp_shares, amount = 1, 1, total_cash
                        w = (total_cash / total_value * 100.0) if total_value > 0 else 0.0
                    prof = float(row.get("trade_profit", 0.0)) if is_trade_decision else 0.0
                    plpct = float(row.get("trade_pl_pct", 0.0)) if is_trade_decision else 0.0

                    avg_cost_val = row.get("avg_cost", 0.0)
                    avg_cost = float(avg_cost_val) if pd.notna(avg_cost_val) else 0.0

                    hold_ret_str = (
                        f"{(price_today / avg_cost - 1.0) * 100.0:+.1f}%"
                        if shares > 0 and avg_cost > 0
                        else "-"
                    )

                    s1, s2, score, filter_val = (
                        row.get("signal1"),
                        row.get("signal2"),
                        row.get("score"),
                        row.get("filter"),
                    )

                    # 전략에 따라 신호 값의 포맷을 다르게 지정합니다.
                    if strategy_name == "jason":
                        s1_str = f"{float(s1):+,.1f}%" if pd.notna(s1) else "-"
                        s2_str = f"{float(s2):+,.1f}%" if pd.notna(s2) else "-"
                        score_str = f"{float(score):+,.1f}%" if pd.notna(score) else "-"
                        filter_str = (
                            "+1"
                            if pd.notna(filter_val) and filter_val > 0
                            else ("-1" if pd.notna(filter_val) and filter_val < 0 else "0")
                        )
                    elif strategy_name == "seykota":
                        s1_str = f"{int(s1):,}" if pd.notna(s1) else "-"
                        s2_str = f"{int(s2):,}" if pd.notna(s2) else "-"
                        score_str = f"{float(score):+,.2f}%" if pd.notna(score) else "-"
                        filter_str = "-"  # seykota는 필터를 사용하지 않음
                    elif strategy_name == "donchian":
                        s1_str = f"{int(s1):,}" if pd.notna(s1) else "-"  # 이평선(값)
                        s2_str = f"{int(s2):,}" if pd.notna(s2) else "-"  # 이평선(기간)
                        score_str = f"{float(score):+,.2f}%" if pd.notna(score) else "-"  # 이격도
                        filter_str = f"{int(filter_val)}일" if pd.notna(filter_val) else "-"

                    else:  # 일반적인 폴백
                        s1_str, s2_str, score_str, filter_str = "-", "-", "-", "-"

                    display_status = decision
                    phrase = ""
                    if is_trade_decision and amount and price_today:
                        qty_calc = (
                            int(float(amount) // float(price_today))
                            if float(price_today) > 0
                            else 0
                        )
                        if decision == "BUY":
                            phrase = f"매수 {qty_calc}주 @ {int(round(price_today)):,} ({format_kr_money(amount)})"
                        else:
                            # 결정 코드에 따라 상세한 사유를 생성합니다.
                            if decision == "SELL_MOMENTUM":
                                tag = "모멘텀소진(이익)" if prof >= 0 else "모멘텀소진(손실)"
                            elif decision == "SELL_TREND":
                                tag = "추세이탈(이익)" if prof >= 0 else "추세이탈(손실)"
                            elif decision == "CUT_STOPLOSS":
                                tag = "가격기반손절"
                            elif decision == "TRIM_REBALANCE":
                                tag = "비중조절"
                            else:  # 이전 버전 호환용
                                tag = "매도"
                            phrase = f"{tag} {qty_calc}주 @ {int(round(price_today)):,} 수익 {format_kr_money(prof)} 손익률 {f'{plpct:+.1f}%'}"
                    elif decision in ("WAIT", "HOLD"):
                        phrase = str(row.get("note", "") or "")
                    if tkr not in buy_date_by_ticker:
                        buy_date_by_ticker[tkr], hold_days_by_ticker[tkr] = None, 0
                    if decision == "BUY" and shares > 0:
                        buy_date_by_ticker[tkr], hold_days_by_ticker[tkr] = dt, 1
                    elif shares > 0:
                        if buy_date_by_ticker.get(tkr) is None:
                            buy_date_by_ticker[tkr], hold_days_by_ticker[tkr] = dt, 1
                        else:
                            hold_days_by_ticker[tkr] += 1
                    else:
                        buy_date_by_ticker[tkr], hold_days_by_ticker[tkr] = None, 0
                    bd = buy_date_by_ticker.get(tkr)
                    bd_str = pd.to_datetime(bd).strftime("%Y-%m-%d") if bd is not None else "-"
                    hd = hold_days_by_ticker.get(tkr, 0)
                    current_row = [
                        0,
                        tkr,
                        name,
                        display_status,
                        bd_str,
                        f"{hd}",
                        f"{int(round(disp_price)):,}",
                        f"{tkr_day_ret:+.1f}%",
                        f"{disp_shares:,}",
                        format_kr_money(amount),
                        hold_ret_str,
                        f"{w:.0f}%",
                        s1_str,
                        s2_str,
                        score_str,
                        filter_str,
                        phrase,
                    ]
                    actions.append((decision, w, score, tkr, current_row))

                def sort_key(action_tuple):
                    state, weight, score, tkr, _ = action_tuple
                    is_hold = 1 if state == "HOLD" else 2
                    is_wait = 1 if state == "WAIT" else 0
                    sort_value = -score if pd.notna(score) and state == "WAIT" else -weight
                    return (is_hold, is_wait, sort_value, tkr)

                actions.sort(key=sort_key)

                rows_sorted = []
                for idx, (_, _, _, _, row) in enumerate(actions, 1):
                    row[0] = idx
                    rows_sorted.append(row)

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
                str_rows = [[str(c) for c in row] for row in rows_sorted]

                # 일별 상세 테이블을 콘솔과 로그 파일에 모두 출력합니다.
                print("\n".join(render_table_eaw(headers, str_rows, aligns)))
                print()

            prev_dt = dt

        if not portfolio_values:
            print("시뮬레이션 결과가 없습니다.")
        else:
            final_value = portfolio_values[-1]
            peak = -1
            max_drawdown = 0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                if peak > 0:
                    drawdown = (peak - value) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

            start_date = portfolio_dates[0]
            end_date = portfolio_dates[-1]
            years = (end_date - start_date).days / 365.25
            cagr = 0
            if years > 0 and initial_capital > 0:
                cagr = ((final_value / initial_capital) ** (1 / years)) - 1

            # Sharpe Ratio 계산
            pv_series = pd.Series(portfolio_values, index=pd.to_datetime(portfolio_dates))
            daily_returns = pv_series.pct_change().dropna()

            sharpe_ratio = 0
            # 일일 수익률이 있을 경우에만 계산
            if not daily_returns.empty and daily_returns.std() > 0:
                # 연간 252 거래일로 가정
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252**0.5)

            # Sortino Ratio 계산 (하락 위험만 고려)
            downside_returns = daily_returns[daily_returns < 0]
            sortino_ratio = 0
            if not downside_returns.empty:
                downside_deviation = downside_returns.std()
                if downside_deviation > 0:
                    sortino_ratio = (daily_returns.mean() / downside_deviation) * (252**0.5)

            # Calmar Ratio 계산 (CAGR / MDD)
            calmar_ratio = (cagr * 100) / (max_drawdown * 100) if max_drawdown > 0 else 0

            summary = {
                "strategy": strategy_name,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "initial_capital": initial_capital,
                "final_value": final_value,
                "cagr_pct": cagr * 100,
                "mdd_pct": max_drawdown * 100,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "cumulative_return_pct": (
                    (final_value / initial_capital - 1) * 100 if initial_capital > 0 else 0
                ),
            }

            # 월별/연간 수익률 계산
            if portfolio_values:
                # 수익률 계산을 위해 시작점에 초기 자본을 추가
                start_row = pd.Series([initial_capital], index=[start_date - pd.Timedelta(days=1)])
                pv_series_with_start = pd.concat([start_row, pv_series])

                # 월별 수익률
                monthly_returns = pv_series_with_start.resample("ME").last().pct_change().dropna()
                summary["monthly_returns"] = monthly_returns

                # 월별 누적 수익률
                eom_pv = pv_series.resample("ME").last()
                monthly_cum_returns = (eom_pv / initial_capital - 1).ffill()
                summary["monthly_cum_returns"] = monthly_cum_returns

                # 연간 수익률
                yearly_returns = pv_series_with_start.resample("YE").last().pct_change().dropna()
                summary["yearly_returns"] = yearly_returns

            return_value = summary

            # 종목별 성과 계산
            ticker_summaries = []
            for tkr, ts in per_ticker_ts.items():
                if tkr == "CASH":
                    continue

                trades = ts[ts["decision"].isin(["SELL", "CUT", "TRIM"])]
                total_trades = len(trades)

                # 거래가 있는 종목만 요약에 포함
                if total_trades > 0:
                    winning_trades = len(trades[trades["trade_profit"] > 0])
                    win_rate = (winning_trades / total_trades) * 100.0
                    total_profit = trades["trade_profit"].sum()
                    avg_profit = total_profit / total_trades
                    ticker_summaries.append(
                        {
                            "ticker": tkr,
                            "name": name_by_ticker.get(tkr, ""),
                            "total_trades": total_trades,
                            "win_rate": win_rate,
                            "total_profit": total_profit,
                            "avg_profit": avg_profit,
                        }
                    )

            if not quiet:
                print(
                    "\n"
                    + "=" * 30
                    + f"\n 백테스트 결과 요약 ({strategy_name}) ".center(30, "=")
                    + "\n"
                    + "=" * 30
                )
                print(f"| 기간: {summary['start_date']} ~ {summary['end_date']} ({years:.2f} 년)")
                print(f"| 초기 자본: {format_kr_money(summary['initial_capital'])}")
                print(f"| 최종 자산: {format_kr_money(summary['final_value'])}")
                print(f"| 누적 수익률: {summary['cumulative_return_pct']:+.2f}%")
                print(f"| CAGR (연간 복리 성장률): {summary['cagr_pct']:+.2f}%")
                print(f"| MDD (최대 낙폭): {-summary['mdd_pct']:.2f}%")
                print(f"| Sharpe Ratio: {summary.get('sharpe_ratio', 0.0):.2f}")
                print(f"| Sortino Ratio: {summary.get('sortino_ratio', 0.0):.2f}")
                print(f"| Calmar Ratio: {summary.get('calmar_ratio', 0.0):.2f}")
                print("=" * 30)
                print("\n[지표 설명]")
                print(
                    "  - Sharpe Ratio (샤프 지수): 위험(변동성) 대비 수익률. 높을수록 좋음 (기준: >1 양호, >2 우수)."
                )
                print(
                    "  - Sortino Ratio (소티노 지수): 하락 위험 대비 수익률. 높을수록 좋음 (기준: >2 양호, >3 우수)."
                )
                print(
                    "  - Calmar Ratio (칼마 지수): 최대 낙폭 대비 연간 수익률. 높을수록 좋음 (기준: >1 양호, >3 우수)."
                )

                # 월별 성과 요약 테이블 출력
                if "monthly_returns" in summary and not summary["monthly_returns"].empty:
                    print("\n" + "=" * 30 + "\n 월별 성과 요약 ".center(30, "=") + "\n" + "=" * 30)

                    monthly_returns = summary["monthly_returns"]
                    yearly_returns = summary["yearly_returns"]
                    monthly_cum_returns = summary.get("monthly_cum_returns")

                    pivot_df = (
                        monthly_returns.mul(100)
                        .to_frame("return")
                        .pivot_table(
                            index=monthly_returns.index.year,
                            columns=monthly_returns.index.month,
                            values="return",
                        )
                    )

                    if not yearly_returns.empty:
                        yearly_series = yearly_returns.mul(100)
                        yearly_series.index = yearly_series.index.year
                        pivot_df["연간"] = yearly_series
                    # If yearly_returns is empty, the '연간' column will not be added, and .get() will return None later.

                    cum_pivot_df = None
                    if monthly_cum_returns is not None and not monthly_cum_returns.empty:
                        cum_pivot_df = (
                            monthly_cum_returns.mul(100)
                            .to_frame("cum_return")
                            .pivot_table(
                                index=monthly_cum_returns.index.year,
                                columns=monthly_cum_returns.index.month,
                                values="cum_return",
                            )
                        )

                    headers = ["연도"] + [f"{m}월" for m in range(1, 13)] + ["연간"]
                    rows_data = []
                    for year, row in pivot_df.iterrows():
                        # 월간 수익률 행
                        monthly_row_data = [str(year)]
                        for month in range(1, 13):
                            val = row.get(month)
                            monthly_row_data.append(f"{val:+.2f}%" if pd.notna(val) else "-")

                        yearly_val = row.get("연간")
                        monthly_row_data.append(
                            f"{yearly_val:+.2f}%" if pd.notna(yearly_val) else "-"
                        )
                        rows_data.append(monthly_row_data)

                        # 누적 수익률 행
                        if cum_pivot_df is not None and year in cum_pivot_df.index:
                            cum_row = cum_pivot_df.loc[year]
                            cum_row_data = ["  (누적)"]
                            for month in range(1, 13):
                                cum_val = cum_row.get(month)
                                cum_row_data.append(
                                    f"{cum_val:+.2f}%" if pd.notna(cum_val) else "-"
                                )

                            # 연말 누적 수익률을 찾습니다.
                            last_valid_month_index = cum_row.last_valid_index()
                            if last_valid_month_index is not None:
                                cum_annual_val = cum_row[last_valid_month_index]
                                cum_row_data.append(f"{cum_annual_val:+.2f}%")
                            else:
                                cum_row_data.append("-")
                            rows_data.append(cum_row_data)

                    aligns = ["left"] + ["right"] * (len(headers) - 1)
                    print("\n".join(render_table_eaw(headers, rows_data, aligns)))

                # 종목별 성과 요약 테이블 출력
                if ticker_summaries:
                    print(
                        "\n" + "=" * 30 + "\n 종목별 성과 요약 ".center(30, "=") + "\n" + "=" * 30
                    )
                    headers = [
                        "티커",
                        "종목명",
                        "거래횟수",
                        "승률",
                        "총손익",
                        "평균손익",
                    ]

                    sorted_summaries = sorted(
                        ticker_summaries, key=lambda x: x["total_profit"], reverse=True
                    )

                    rows = [
                        [
                            s["ticker"],
                            s["name"],
                            f"{s['total_trades']}회",
                            f"{s['win_rate']:.1f}%",
                            format_kr_money(s["total_profit"]),
                            format_kr_money(s["avg_profit"]),
                        ]
                        for s in sorted_summaries
                    ]

                    aligns = ["right", "left", "right", "right", "right", "right"]
                    table_lines = render_table_eaw(headers, rows, aligns)
                    print("\n".join(table_lines))

    finally:
        # Restore stdout and close file
        if sys.stdout != _orig_stdout:
            sys.stdout.close()
        sys.stdout = _orig_stdout
        if log_f:
            try:
                log_f.close()
            except Exception:
                pass

    return return_value


if __name__ == "__main__":
    main(quiet=False)
