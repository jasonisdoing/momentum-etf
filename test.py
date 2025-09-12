import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logic import settings
from logic import strategy as strategy_module
from utils.report import format_aud_money, format_kr_money, render_table_eaw, format_aud_price
from utils.db_manager import get_stocks, get_app_settings, get_common_settings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# 이 파일에서는 매매 전략에 사용되는 고유 파라미터를 정의합니다.
INITIAL_CAPITAL = 100000000
# 백테스트를 진행할 최근 개월 수 (예: 12 -> 최근 12개월 데이터로 테스트)
TEST_MONTHS_RANGE = 60

def main(
    country: str = "kor",
    quiet: bool = False,
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
):
    """
    지정된 전략에 대한 백테스트를 실행하고 결과를 요약합니다.
    `quiet=True` 모드에서는 로그를 출력하지 않고 최종 요약만 반환합니다.
    """
    try:
        initial_capital = INITIAL_CAPITAL
    except AttributeError:
        print("오류: INITIAL_CAPITAL 설정이 logic/settings.py 에 정의되어야 합니다.")
        return

    # DB에서 앱 설정을 불러와 logic.settings에 동적으로 설정합니다.
    app_settings = get_app_settings(country)
    if (not app_settings 
        or "ma_period_etf" not in app_settings 
        or "ma_period_stock" not in app_settings
        or "portfolio_topn" not in app_settings):
        print(f"오류: '{country}' 국가의 설정(TopN, MA 기간)이 DB에 없습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요.")
        return
    
    # 필수 설정값이 모두 있는지 검증 (fallback 금지)
    if "replace_threshold" not in app_settings:
        print(f"오류: '{country}' 국가의 설정에 'replace_threshold'가 없습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요.")
        return
    # 추가 필수값: replace_weaker_stock, max_replacements_per_day
    if "replace_weaker_stock" not in app_settings:
        print(f"오류: '{country}' 국가의 설정에 'replace_weaker_stock'가 없습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요.")
        return
    if "max_replacements_per_day" not in app_settings:
        print(f"오류: '{country}' 국가의 설정에 'max_replacements_per_day'가 없습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요.")
        return

    try:
        settings.MA_PERIOD_FOR_ETF = int(app_settings["ma_period_etf"])
        settings.MA_PERIOD_FOR_STOCK = int(app_settings["ma_period_stock"])
        portfolio_topn = int(app_settings["portfolio_topn"])
        # 교체 매매 파라미터 (백테스트용, 필수)
        settings.REPLACE_SCORE_THRESHOLD = float(app_settings["replace_threshold"])  
        settings.REPLACE_WEAKER_STOCK = bool(app_settings["replace_weaker_stock"])  
        settings.MAX_REPLACEMENTS_PER_DAY = int(app_settings["max_replacements_per_day"])  
    except (ValueError, TypeError):
        print(f"오류: '{country}' 국가의 DB 설정값이 올바르지 않습니다.")
        return

    # 공통(전역) 설정 로드 및 주입 (필수)
    common = get_common_settings()
    if not common:
        print("오류: 공통 설정이 DB에 없습니다. 웹 앱의 '설정' 탭에서 먼저 값을 저장해주세요.")
        return
    required_common_keys = [
        "HOLDING_STOP_LOSS_PCT",
        "COOLDOWN_DAYS",
        "ATR_PERIOD_FOR_NORMALIZATION",
        "MARKET_REGIME_FILTER_ENABLED",
        "MARKET_REGIME_FILTER_TICKER",
        "MARKET_REGIME_FILTER_MA_PERIOD",
    ]
    missing = [k for k in required_common_keys if k not in common]
    if missing:
        print(f"오류: 공통 설정에 다음 값이 없습니다: {', '.join(missing)}")
        return
    try:
        # Interpret positive input as a negative threshold (e.g., 10 -> -10)
        settings.HOLDING_STOP_LOSS_PCT = -abs(float(common["HOLDING_STOP_LOSS_PCT"]))
        settings.COOLDOWN_DAYS = int(common["COOLDOWN_DAYS"])
        settings.ATR_PERIOD_FOR_NORMALIZATION = int(common["ATR_PERIOD_FOR_NORMALIZATION"])
        settings.MARKET_REGIME_FILTER_ENABLED = bool(common["MARKET_REGIME_FILTER_ENABLED"])
        settings.MARKET_REGIME_FILTER_TICKER = str(common["MARKET_REGIME_FILTER_TICKER"])
        settings.MARKET_REGIME_FILTER_MA_PERIOD = int(common["MARKET_REGIME_FILTER_MA_PERIOD"])
    except (ValueError, TypeError):
        print("오류: 공통 설정 값의 형식이 올바르지 않습니다.")
        return

    # 국가별로 다른 포맷터 사용
    if country == "aus":
        money_formatter = format_aud_money
        price_formatter = format_aud_price
        ma_formatter = format_aud_price
    else:
        # 원화(KRW) 형식으로 가격을 포맷합니다.
        money_formatter = format_kr_money
        price_formatter = lambda p: f"{int(round(p)):,}"
        ma_formatter = lambda p: f"{int(round(p)):,}원"

    # 기간 설정 로직 (필수 설정)
    try:
        test_months_range = TEST_MONTHS_RANGE
    except AttributeError:
        print("오류: TEST_MONTHS_RANGE 설정이 logic/settings.py 에 정의되어야 합니다.")
        return
    core_end_dt = pd.Timestamp.now()
    core_start_dt = core_end_dt - pd.DateOffset(months=test_months_range)
    test_date_range = [core_start_dt.strftime('%Y-%m-%d'), core_end_dt.strftime('%Y-%m-%d')]
    period_label = f"최근 {test_months_range}개월 ({core_start_dt.strftime('%Y-%m-%d')}~{core_end_dt.strftime('%Y-%m-%d')})"
    months_range = None  # 이 변수는 더 이상 사용되지 않습니다.

    # 티커 목록 결정
    if not quiet:
        print(f"\nDB의 '{country}_stocks' 컬렉션에서 종목을 가져와 백테스트를 실행합니다.")
    stocks_from_db = get_stocks(country)
    if not stocks_from_db:
        print(f"오류: '{country}_stocks' 컬렉션에서 백테스트에 사용할 종목을 찾을 수 없습니다.")
        return

    logger = logging.getLogger("backtester")
    logger.propagate = False  # 중복 로깅 방지
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    if not quiet:
        # 파일 핸들러 설정
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"test_{country}.log")
        file_handler = logging.FileHandler(log_path, "w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)
    else:
        # quiet 모드에서는 모든 출력을 비활성화
        logger.addHandler(logging.NullHandler())

    return_value = None

    try:
        # 로그 파일 헤더
        logger.info("백테스트를 `settings.py` 설정으로 실행합니다.")
        logger.info(
            f"# 시작 {datetime.now().isoformat()} | 기간={period_label} | 초기자본={int(initial_capital):,}\n"
        )

        # 전략 모듈에서 백테스트 함수를 가져옵니다.
        try:
            run_portfolio_backtest = getattr(strategy_module, "run_portfolio_backtest")
            run_single_ticker_backtest = getattr(strategy_module, "run_single_ticker_backtest")
        except AttributeError:
            print(
                "오류: 'logic.strategy' 모듈에 run_portfolio_backtest 또는 "
                "run_single_ticker_backtest 함수가 정의되지 않았습니다."
            )
            return

        # 시뮬레이션 실행
        time_series_by_ticker: Dict[str, pd.DataFrame] = {}
        name_by_ticker: Dict[str, str] = {s['ticker']: s['name'] for s in stocks_from_db}
        if portfolio_topn > 0:
            time_series_by_ticker = (
                run_portfolio_backtest(
                    stocks=stocks_from_db,
                    months_range=months_range,
                    initial_capital=initial_capital,
                    core_start_date=core_start_dt,
                    top_n=portfolio_topn,
                    date_range=test_date_range,
                    country=country,
                    prefetched_data=prefetched_data,
                )
                or {}
            )
            if "CASH" in time_series_by_ticker:
                name_by_ticker = {s['ticker']: s['name'] for s in stocks_from_db}
                name_by_ticker["CASH"] = "현금"
        else:
            # 개별 종목 백테스트는 여전히 데이터를 미리 로드해야 합니다.
            from utils.data_loader import fetch_ohlcv

            raw_data_by_ticker: Dict[str, pd.DataFrame] = {}
            for stock in stocks_from_db:
                ticker = stock['ticker']
                df = fetch_ohlcv(
                    ticker, country=country, months_range=months_range, date_range=test_date_range
                )
                if df is not None and not df.empty:
                    raw_data_by_ticker[ticker] = df

            # 종목별 고정 자본 방식: 전체 자본을 종목 수로 나눔
            capital_per_ticker = initial_capital / len(stocks_from_db) if stocks_from_db else 0
            for stock in stocks_from_db:
                ticker = stock['ticker']
                df_ticker = raw_data_by_ticker.get(ticker)
                if df_ticker is None:
                    continue
                ts = run_single_ticker_backtest(
                    ticker,
                    stock_type=stock.get('type', 'stock'),
                    df=df_ticker,
                    months_range=months_range,
                    initial_capital=capital_per_ticker,
                    core_start_date=core_start_dt,
                    date_range=test_date_range,
                    country=country,
                )
                if not ts.empty:
                    time_series_by_ticker[ticker] = ts

        if not time_series_by_ticker:
            if not quiet:
                logger.info("시뮬레이션할 유효한 데이터가 없습니다.")
            return

        # 모든 티커에 걸쳐 공통된 날짜로 정렬 (교집합)
        common_index = None
        for tkr, ts in time_series_by_ticker.items():
            common_index = ts.index if common_index is None else common_index.intersection(ts.index)
        if common_index is None or len(common_index) == 0:
            logger.info("종목들 간에 공통된 거래일이 없습니다.")
            return

        portfolio_values = []
        portfolio_dates = []
        prev_total_pv = float(initial_capital)
        prev_dt: Optional[pd.Timestamp] = None
        buy_date_by_ticker: Dict[str, Optional[pd.Timestamp]] = {}
        holding_days_by_ticker: Dict[str, int] = {}
        total_cnt = len(time_series_by_ticker)

        total_init = float(initial_capital)

        for dt in common_index:
            portfolio_dates.append(dt)

            # 일별 자산 집계
            total_value = 0.0
            total_holdings = 0.0
            held_count = 0
            for tkr, ts in time_series_by_ticker.items():
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

            # 일일 포트폴리오 수익률
            if prev_total_pv is not None and prev_total_pv > 0:
                day_ret_pct = (
                    ((total_value / prev_total_pv) - 1.0) * 100.0 if prev_total_pv > 0 else 0.0
                )
            prev_total_pv = total_value

            # 초기 자본 대비 누적 포트폴리오 수익률
            cum_ret_pct = (
                ((total_value / total_init) - 1.0) * 100.0 if total_init and total_init > 0 else 0.0
            )

            if not quiet:
                # Header line
                denom = portfolio_topn if portfolio_topn > 0 else total_cnt
                date_str = pd.to_datetime(dt).strftime("%Y-%m-%d")
                logger.info(
                    (  # noqa: T201
                        f"{date_str} - 보유종목 {held_count}/{denom} "
                        f"잔액(보유+현금): {money_formatter(total_value)} "
                        f"(보유 {money_formatter(total_holdings)} + 현금 {money_formatter(total_cash)}) "
                        f"금일 수익률 {day_ret_pct:+.1f}%, 누적 수익률 {cum_ret_pct:+.1f}%"
                    )
                )

                # 전략에 따라 동적으로 헤더를 설정합니다.
                # signal_headers = ["이평선(값)", "고점대비", "점수", "신호지속일"]
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
                headers.extend(["이평선(값)", "고점대비", "점수", "신호지속일"])
                headers.append("문구")

                decisions_list = []
                for tkr, ts in time_series_by_ticker.items():
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
                    is_trade_decision = decision.startswith(("BUY", "SELL", "CUT"))
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
                    s1_str = ma_formatter(s1) if pd.notna(s1) else "-"
                    s2_str = f"{float(s2):.1f}%" if pd.notna(s2) else "-"  # 고점대비
                    score_str = f"{float(score):+,.2f}" if pd.notna(score) else "-"  # 점수
                    filter_str = f"{int(filter_val)}일" if pd.notna(filter_val) else "-"

                    display_status = decision
                    phrase = ""
                    note_from_strategy = str(row.get("note", "") or "")
                    if is_trade_decision and amount > 0 and price_today > 0:
                        qty_calc = (
                            int(float(amount) // float(price_today))
                            if float(price_today) > 0
                            else 0
                        )
                        if decision.startswith("BUY"):
                            tag = "매수"
                            if decision == "BUY_REPLACE":
                                tag = "교체매수"
                            phrase = f"{tag} {qty_calc}주 @ {price_formatter(price_today)} ({money_formatter(amount)})"
                            if note_from_strategy:
                                phrase += f" ({note_from_strategy})"
                        else:
                            # 결정 코드에 따라 상세한 사유를 생성합니다.
                            if decision == "SELL_MOMENTUM":
                                tag = "모멘텀소진(이익)" if prof >= 0 else "모멘텀소진(손실)"
                            elif decision == "SELL_TREND":
                                tag = "추세이탈(이익)" if prof >= 0 else "추세이탈(손실)"
                            elif decision == "CUT_STOPLOSS":
                                tag = "가격기반손절"
                            elif decision == "SELL_REPLACE":
                                tag = "교체매도"
                            elif decision == "SELL_REGIME_FILTER":
                                tag = "시장위험회피"
                            else:  # 이전 버전 호환용 (e.g. "SELL")
                                tag = "매도"
                            phrase = f"{tag} {qty_calc}주 @ {price_formatter(price_today)} 수익 {money_formatter(prof)} 손익률 {f'{plpct:+.1f}%'}"
                            if note_from_strategy:
                                phrase += f" ({note_from_strategy})"
                    elif decision in ("WAIT", "HOLD"):
                        phrase = note_from_strategy
                    if tkr not in buy_date_by_ticker:
                        buy_date_by_ticker[tkr], holding_days_by_ticker[tkr] = None, 0
                    if decision == "BUY" and shares > 0:
                        buy_date_by_ticker[tkr], holding_days_by_ticker[tkr] = dt, 1
                    elif shares > 0:
                        if buy_date_by_ticker.get(tkr) is None:
                            buy_date_by_ticker[tkr], holding_days_by_ticker[tkr] = dt, 1
                        else:
                            holding_days_by_ticker[tkr] += 1
                    else:
                        buy_date_by_ticker[tkr], holding_days_by_ticker[tkr] = None, 0
                    bd = buy_date_by_ticker.get(tkr)
                    bd_str = pd.to_datetime(bd).strftime("%Y-%m-%d") if bd is not None else "-"
                    hd = holding_days_by_ticker.get(tkr, 0)
                    current_row = [
                        0,
                        tkr,
                        name,
                        display_status,
                        bd_str,
                        f"{hd}",
                        price_formatter(disp_price),
                        f"{tkr_day_ret:+.1f}%",
                        f"{disp_shares:,}",
                        money_formatter(amount),
                        hold_ret_str,
                        f"{w:.0f}%",
                        s1_str,
                        s2_str,
                        score_str,
                        filter_str,
                        phrase,
                    ]
                    decisions_list.append((decision, w, score, tkr, current_row))

                def sort_key(decision_tuple):
                    state, weight, score, tkr, _ = decision_tuple
                    is_hold = 1 if state == "HOLD" else 2
                    is_wait = 1 if state == "WAIT" else 0
                    sort_value = -score if pd.notna(score) and state == "WAIT" else -weight
                    return (is_hold, is_wait, sort_value, tkr)

                decisions_list.sort(key=sort_key)

                rows_sorted = []
                for idx, (_, _, _, _, row) in enumerate(decisions_list, 1):
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
                logger.info("\n" + "\n".join(render_table_eaw(headers, str_rows, aligns)))
                logger.info("")

            prev_dt = dt

        if not portfolio_values:
            logger.info("시뮬레이션 결과가 없습니다.")
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

            # 시장 위험 회피(Risk-Off) 기간을 계산합니다.
            risk_off_periods = []
            try:
                market_regime_filter_enabled = bool(settings.MARKET_REGIME_FILTER_ENABLED)
            except AttributeError:
                print("오류: MARKET_REGIME_FILTER_ENABLED 설정이 logic/settings.py 에 정의되어야 합니다.")
                return

            if market_regime_filter_enabled:
                # '시장 위험 회피' 노트가 있는지 확인하여 리스크 오프 기간을 식별합니다.
                # 모든 티커의 노트를 하나의 데이터프레임으로 합칩니다.
                notes_df = pd.DataFrame({
                    tkr: ts['note'] for tkr, ts in time_series_by_ticker.items() if tkr != "CASH"
                }, index=common_index)
                
                # 어느 한 종목이라도 '시장 위험 회피' 노트를 가지면 해당일은 리스크 오프입니다.
                if not notes_df.empty:
                    is_risk_off_series = (notes_df == "시장 위험 회피").any(axis=1)

                    # 연속된 True 블록(리스크 오프 기간)을 찾습니다.
                    in_risk_off_period = False
                    start_of_period = None
                    for dt, is_off in is_risk_off_series.items():
                        if is_off and not in_risk_off_period:
                            in_risk_off_period = True
                            start_of_period = dt
                        elif not is_off and in_risk_off_period:
                            in_risk_off_period = False
                            # 리스크 오프 기간의 마지막 날은 is_off가 False가 되기 바로 전날입니다.
                            end_of_period = is_risk_off_series.index[is_risk_off_series.index.get_loc(dt) - 1]
                            risk_off_periods.append((start_of_period, end_of_period))
                            start_of_period = None
                    
                    # 백테스트가 리스크 오프 기간 중에 끝나는 경우를 처리합니다.
                    if in_risk_off_period and start_of_period:
                        risk_off_periods.append((start_of_period, is_risk_off_series.index[-1]))

            start_date = portfolio_dates[0]
            end_date = portfolio_dates[-1]
            years = (end_date - start_date).days / 365.25
            cagr = 0
            if years > 0 and initial_capital > 0:
                cagr = ((final_value / initial_capital) ** (1 / years)) - 1

            # --- 벤치마크 (S&P 500) 성과 계산 ---
            from utils.data_loader import fetch_ohlcv
            benchmark_ticker = "^GSPC"
            benchmark_df = fetch_ohlcv(
                benchmark_ticker,
                country=country, # country는 지수 티커에 영향을 주지 않음
                date_range=[start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")],
            )
            
            benchmark_cum_ret_pct = 0.0
            benchmark_cagr_pct = 0.0
            if benchmark_df is not None and not benchmark_df.empty:
                # 벤치마크 데이터를 실제 백테스트 날짜와 일치시킵니다.
                benchmark_df = benchmark_df.loc[benchmark_df.index.isin(portfolio_dates)]
                if not benchmark_df.empty:
                    benchmark_start_price = benchmark_df["Close"].iloc[0]
                    benchmark_end_price = benchmark_df["Close"].iloc[-1]
                    if benchmark_start_price > 0:
                        benchmark_cum_ret_pct = ((benchmark_end_price / benchmark_start_price) - 1) * 100
                        if years > 0:
                            benchmark_cagr_pct = ((benchmark_end_price / benchmark_start_price) ** (1 / years) - 1) * 100

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
                "risk_off_periods": risk_off_periods,
                "benchmark_cum_ret_pct": benchmark_cum_ret_pct,
                "benchmark_cagr_pct": benchmark_cagr_pct,
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
                monthly_cum_returns = (eom_pv / initial_capital - 1).ffill() if initial_capital > 0 else pd.Series()
                summary["monthly_cum_returns"] = monthly_cum_returns

                # 연간 수익률
                yearly_returns = pv_series_with_start.resample("YE").last().pct_change().dropna()
                summary["yearly_returns"] = yearly_returns

            return_value = summary

            # 종목별 성과 계산
            ticker_summaries = []
            for tkr, ts in time_series_by_ticker.items():
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
                logger.info(
                    "\n"
                    + "=" * 30
                    + "\n 백테스트 결과 요약 ".center(30, "=")
                    + "\n"
                    + "=" * 30
                )
                try:
                    test_months_range = settings.TEST_MONTHS_RANGE
                except AttributeError:
                    test_months_range = None
                logger.info(
                    f"| 기간: {summary['start_date']} ~ {summary['end_date']} ({test_months_range} 개월)"
                )
                if summary.get("risk_off_periods"):
                    for start, end in summary["risk_off_periods"]:
                        logger.info(
                            f"| 투자 중단: {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}"
                        )
                logger.info(f"| 초기 자본: {money_formatter(summary['initial_capital'])}")
                logger.info(f"| 최종 자산: {money_formatter(summary['final_value'])}")
                logger.info(
                    f"| 누적 수익률: {summary['cumulative_return_pct']:+.2f}% (S&P 500: {summary.get('benchmark_cum_ret_pct', 0.0):+.2f}%)"
                )
                logger.info(
                    f"| CAGR (연간 복리 성장률): {summary['cagr_pct']:+.2f}% (S&P 500: {summary.get('benchmark_cagr_pct', 0.0):+.2f}%)"
                )
                logger.info(f"| MDD (최대 낙폭): {-summary['mdd_pct']:.2f}%")
                logger.info(f"| Sharpe Ratio: {summary.get('sharpe_ratio', 0.0):.2f}")
                logger.info(f"| Sortino Ratio: {summary.get('sortino_ratio', 0.0):.2f}")
                logger.info(f"| Calmar Ratio: {summary.get('calmar_ratio', 0.0):.2f}")
                logger.info("=" * 30)
                logger.info("\n[지표 설명]")
                logger.info(
                    "  - Sharpe Ratio (샤프 지수): 위험(변동성) 대비 수익률. 높을수록 좋음 (기준: >1 양호, >2 우수)."
                )
                logger.info(
                    "  - Sortino Ratio (소티노 지수): 하락 위험 대비 수익률. 높을수록 좋음 (기준: >2 양호, >3 우수)."
                )
                logger.info(
                    "  - Calmar Ratio (칼마 지수): 최대 낙폭 대비 연간 수익률. 높을수록 좋음 (기준: >1 양호, >3 우수)."
                )

                # 월별 성과 요약 테이블 출력
                if "monthly_returns" in summary and not summary["monthly_returns"].empty:
                    logger.info(
                        "\n" + "=" * 30 + "\n 월별 성과 요약 ".center(30, "=") + "\n" + "=" * 30
                    )

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
                    logger.info("\n" + "\n".join(render_table_eaw(headers, rows_data, aligns)))

                # 종목별 성과 요약 테이블 출력
                if ticker_summaries:
                    logger.info(
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
                            money_formatter(s["total_profit"]),
                            money_formatter(s["avg_profit"]),
                        ]
                        for s in sorted_summaries
                    ]

                    aligns = ["right", "left", "right", "right", "right", "right"]
                    table_lines = render_table_eaw(headers, rows, aligns)
                    logger.info("\n" + "\n".join(table_lines))

    finally:
        logging.shutdown()

    return return_value


if __name__ == "__main__":
    main(country="kor", quiet=False)
