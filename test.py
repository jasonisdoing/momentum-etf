import os
import sys
import warnings
from datetime import datetime
from typing import Dict, Optional, List, Any

import pandas as pd

# pkg_resources 워닝 억제
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logic.entry_point as strategy_module
from logic.strategies.maps.shared import SIGNAL_TABLE_HEADERS
from utils.account_registry import (
    get_country_settings,
    get_strategy_rules,
    get_common_file_settings,
)
from utils.tee import Tee
from utils.report import (
    format_kr_money,
    format_aud_money,
    format_usd_money,
    render_table_eaw,
)
from utils.data_loader import get_latest_trading_day
from utils.data_loader import get_aud_to_krw_rate, get_usd_to_krw_rate
from utils.stock_list_io import get_etfs as get_etfs_from_files
from utils.notification import build_summary_line_from_summary_data
from utils.backtest_utils import print_backtest_summary, format_period_return_with_listing_date

# 이 파일에서는 매매 전략에 사용되는 고유 파라미터를 정의합니다.
# 백테스트를 진행할 최근 개월 수 (예: 12 -> 최근 12개월 데이터로 테스트)
TEST_MONTHS_RANGE = 12


# _format_period_return_with_listing_date and _print_backtest_summary functions have been moved to utils/backtest_utils.py


def main(
    account: str,
    quiet: bool = False,
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
    override_settings: Optional[Dict] = None,
):
    """
    지정된 전략에 대한 백테스트를 실행하고 결과를 요약합니다.
    `quiet=True` 모드에서는 로그를 출력하지 않고 최종 요약만 반환합니다.
    """
    if not account:
        raise ValueError("account is required for backtest execution")

    account_info = get_account_info(account)
    if not account_info:
        raise ValueError(f"등록되지 않은 계좌입니다: {account}")
    country = str(account_info.get("country") or "").strip()
    if not country:
        raise ValueError(f"'{account}' 계좌에 국가 정보가 없습니다.")

    # --- 로그 파일 설정 ---
    # quiet 모드가 아닐 때만 파일 로깅을 설정합니다.
    original_stdout = sys.stdout
    log_file = None
    if not quiet:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"test_{country}_{account}.log"
        log_path = os.path.join(log_dir, log_filename)
        try:
            log_file = open(log_path, "w", encoding="utf-8")
            sys.stdout = Tee(original_stdout, log_file)
            print(f"백테스트 로그가 다음 파일에 저장됩니다: {log_path}")
        except Exception as e:
            print(f"경고: 로그 파일을 열 수 없습니다: {e}")
            # 파일 열기 실패 시, 콘솔 출력은 계속 유지됩니다.
            sys.stdout = original_stdout

    from logic import entry_point as settings

    # 파일에서 초기 자본금 및 모든 계좌 설정을 가져옵니다.
    try:
        account_settings = get_account_file_settings(account)
        strategy_rules = get_strategy_rules_for_account(account)

        initial_capital_krw = account_settings["initial_capital_krw"]
        currency = account_info.get("currency", "KRW")

        # 호주 계좌의 경우, KRW로 설정된 초기 자본금을 AUD로 변환합니다.
        if currency == "AUD":
            aud_krw_rate = get_aud_to_krw_rate()
            if aud_krw_rate and aud_krw_rate > 0:
                initial_capital_krw /= aud_krw_rate
            else:
                if not quiet:
                    print("오류: AUD/KRW 환율을 가져올 수 없어 백테스트를 진행할 수 없습니다.")
                if log_file:
                    log_file.close()
                return
        elif currency == "USD":
            usd_krw_rate = get_usd_to_krw_rate()
            if usd_krw_rate and usd_krw_rate > 0:
                initial_capital_krw /= usd_krw_rate
            else:
                if not quiet:
                    print("오류: USD/KRW 환율을 가져올 수 없어 백테스트를 진행할 수 없습니다.")
                if log_file:
                    log_file.close()
                return

        settings.MA_PERIOD = strategy_rules.ma_period
        portfolio_topn = strategy_rules.portfolio_topn
        settings.REPLACE_SCORE_THRESHOLD = strategy_rules.replace_threshold
    except SystemExit as e:
        print(str(e))
        if log_file:
            log_file.close()
        return None

    # Optional overrides from caller (e.g., tuning scripts)
    if override_settings:
        try:
            if "ma_period" in override_settings:
                settings.MA_PERIOD = int(override_settings["ma_period"])
            if "portfolio_topn" in override_settings:
                portfolio_topn = int(override_settings["portfolio_topn"])
            if "replace_threshold" in override_settings:
                settings.REPLACE_SCORE_THRESHOLD = float(override_settings["replace_threshold"])
        except Exception:
            # Silently ignore malformed overrides
            pass

    try:
        # 공통(전역) 설정 로드 및 주입 (필수)
        common = get_common_file_settings()
        # 양수 입력을 음수 임계값으로 해석합니다 (예: 10 -> -10)
        settings.HOLDING_STOP_LOSS_PCT = -abs(float(common["HOLDING_STOP_LOSS_PCT"]))
        settings.COOLDOWN_DAYS = int(account_settings.get("cooldown_days", 0))
        settings.MARKET_REGIME_FILTER_ENABLED = bool(common["MARKET_REGIME_FILTER_ENABLED"])
        settings.MARKET_REGIME_FILTER_TICKER = str(common["MARKET_REGIME_FILTER_TICKER"])
        settings.MARKET_REGIME_FILTER_MA_PERIOD = int(common["MARKET_REGIME_FILTER_MA_PERIOD"])
    except (SystemExit, KeyError, ValueError, TypeError) as e:
        print(f"오류: 공통 설정 파일을 읽는 중 문제가 발생했습니다: {e}")
        if log_file:
            log_file.close()
        return None

    currency = account_info.get("currency", "KRW")
    precision = account_info.get("amt_precision", account_info.get("precision", 0))
    try:
        precision = int(precision)
    except (TypeError, ValueError):
        precision = 0

    from utils.report import (
        format_aud_price,
        format_usd_price,
    )

    # 국가별로 다른 포맷터 사용
    if currency == "AUD":
        money_formatter = format_aud_money
        price_formatter = format_aud_price
    elif currency == "USD":
        money_formatter = format_usd_money
        price_formatter = format_usd_price
    else:
        # 원화(KRW) 형식으로 가격을 포맷합니다.
        money_formatter = format_kr_money

        def _kr_price_formatter(p):
            return f"{int(round(p)):,}"

        price_formatter = _kr_price_formatter

    # 기간 설정 로직 (필수 설정)
    try:
        test_months_range = (
            int(override_settings.get("test_months_range"))
            if override_settings and "test_months_range" in override_settings
            else TEST_MONTHS_RANGE
        )
    except Exception:
        if not quiet:
            print("오류: 테스트 기간(test_months_range) 설정이 올바르지 않습니다.")
        return_value = None
        return

    # 기간 설정 로직: override_settings에 start_date, end_date가 있으면 우선 사용
    if override_settings and "start_date" in override_settings and "end_date" in override_settings:
        core_start_dt = pd.to_datetime(override_settings["start_date"])
        core_end_dt = pd.to_datetime(override_settings["end_date"])
        period_label = (
            f"지정 기간 ({core_start_dt.strftime('%Y-%m-%d')}~{core_end_dt.strftime('%Y-%m-%d')})"
        )
    else:
        core_end_dt = get_latest_trading_day(country)
        core_start_dt = core_end_dt - pd.DateOffset(months=test_months_range)
        period_label = f"최근 {test_months_range}개월 ({core_start_dt.strftime('%Y-%m-%d')}~{core_end_dt.strftime('%Y-%m-%d')})"

    test_date_range = [
        core_start_dt.strftime("%Y-%m-%d"),
        core_end_dt.strftime("%Y-%m-%d"),
    ]

    # 티커 목록 결정
    if not quiet:
        print(f"\n'data/{country}/' 폴더의 'etf.json' 파일에서 종목을 가져와 백테스트를 실행합니다.")
    all_etfs_from_file = get_etfs_from_files(country)
    etfs_from_file = all_etfs_from_file
    if not etfs_from_file:
        if not quiet:
            print(f"오류: 'data/{country}/' 폴더에서 '{country}' 국가의 백테스트에 사용할 종목을 찾을 수 없습니다.")
        return_value = None
        return

    # 티커 오버라이드: 콤마 구분 리스트. coin은 파일 목록과 합집합, 그 외는 교집합.
    if override_settings and override_settings.get("tickers_override"):
        allow = [str(t).upper() for t in override_settings.get("tickers_override")]
        allow_set = set(allow)

        if country == "coin":
            # 합집합: 파일 목록에 override tickers를 추가하고 중복 제거
            existing_set = {str(s.get("ticker") or "").upper() for s in etfs_from_file}
            for t in allow:
                if t not in existing_set:
                    etfs_from_file.append(
                        {"ticker": t, "name": t, "type": "etf", "country": "coin"}
                    )
        else:
            # 교집합: 파일에 있는 티커 중 override 목록에 있는 것만 사용
            filtered = [
                s for s in etfs_from_file if str(s.get("ticker") or "").upper() in allow_set
            ]
            if not filtered:
                if not quiet:
                    print("오류: override tickers 가 DB 목록과 일치하지 않습니다.")
                return_value = None
                return None
            etfs_from_file = filtered

    return_value = None

    try:
        if not quiet:
            header_suffix = f" (account={account})" if account else ""
            print(f"백테스트를 `settings.py` 설정으로 실행합니다{header_suffix}.")
            print(
                f"# 시작 {datetime.now().isoformat()} | 기간={period_label} | 초기자본={int(initial_capital_krw):,}\n"
            )
        # 전략 모듈에서 백테스트 함수를 가져옵니다.
        try:
            run_portfolio_backtest = getattr(strategy_module, "run_portfolio_backtest")
            run_single_ticker_backtest = getattr(strategy_module, "run_single_ticker_backtest")
        except AttributeError:
            module_name = getattr(strategy_module, "__name__", "알 수 없는 모듈")
            print(
                f"오류: '{module_name}' 모듈에 run_portfolio_backtest 또는 "
                "run_single_ticker_backtest 함수가 정의되지 않았습니다."
            )
            if log_file:
                log_file.close()
            return None

        # 시뮬레이션 실행
        time_series_by_ticker: Dict[str, pd.DataFrame] = {}
        name_by_ticker: Dict[str, str] = {s["ticker"]: s.get("name", "") for s in etfs_from_file}
        category_by_ticker: Dict[str, str] = {
            s["ticker"]: str(s.get("category") or "") for s in etfs_from_file
        }
        if portfolio_topn > 0:
            time_series_by_ticker = (
                run_portfolio_backtest(
                    stocks=etfs_from_file,
                    initial_capital=initial_capital_krw,
                    core_start_date=core_start_dt,
                    top_n=portfolio_topn,
                    date_range=test_date_range,
                    country=country,
                    prefetched_data=prefetched_data,
                    ma_period=settings.MA_PERIOD,
                    replace_threshold=settings.REPLACE_SCORE_THRESHOLD,
                    regime_filter_enabled=settings.MARKET_REGIME_FILTER_ENABLED,
                    regime_filter_ticker=settings.MARKET_REGIME_FILTER_TICKER,
                    regime_filter_ma_period=settings.MARKET_REGIME_FILTER_MA_PERIOD,
                    stop_loss_pct=settings.HOLDING_STOP_LOSS_PCT,
                    cooldown_days=settings.COOLDOWN_DAYS,
                )
                or {}
            )
            if "CASH" in time_series_by_ticker:
                name_by_ticker = {s["ticker"]: s.get("name", "") for s in etfs_from_file}
                name_by_ticker["CASH"] = "현금"
                category_by_ticker["CASH"] = "-"
        else:
            # 종목별 고정 자본 방식: 전체 자본을 종목 수로 나눔
            capital_per_ticker = initial_capital_krw / len(etfs_from_file) if etfs_from_file else 0
            for etf in etfs_from_file:
                ticker = etf["ticker"]
                # prefetched_data가 있으면 사용하고, 없으면 None을 전달하여 함수 내부에서 조회하도록 합니다.
                df_ticker = prefetched_data.get(ticker) if prefetched_data else None
                if df_ticker is None:
                    continue
                ts = run_single_ticker_backtest(
                    ticker,
                    df=df_ticker,
                    initial_capital=capital_per_ticker,
                    core_start_date=core_start_dt,
                    date_range=test_date_range,
                    country=country,
                    ma_period=settings.MA_PERIOD,
                    stop_loss_pct=settings.HOLDING_STOP_LOSS_PCT,
                    cooldown_days=settings.COOLDOWN_DAYS,
                )
                if not ts.empty:
                    time_series_by_ticker[ticker] = ts

        if not time_series_by_ticker:
            if not quiet:
                print("시뮬레이션할 유효한 데이터가 없습니다.")
            if log_file:
                log_file.close()
            return None

        # 원본 시계열 데이터를 보관합니다 (기간 수익률 계산용)
        original_time_series_by_ticker = {
            tkr: ts.copy() for tkr, ts in time_series_by_ticker.items()
        }

        # 모든 티커에 걸쳐 공통된 날짜로 정렬 (교집합)
        common_index = None
        for tkr, ts in time_series_by_ticker.items():
            common_index = ts.index if common_index is None else common_index.intersection(ts.index)
        if common_index is None or len(common_index) == 0:
            if log_file:
                log_file.close()
            if not quiet:
                print("종목들 간에 공통된 거래일이 없습니다.")
            return

        portfolio_values = []
        portfolio_dates = []
        prev_total_pv = float(initial_capital_krw)
        prev_dt: Optional[pd.Timestamp] = None
        buy_date_by_ticker: Dict[str, Optional[pd.Timestamp]] = {}
        holding_days_by_ticker: Dict[str, int] = {}
        total_cnt = len(time_series_by_ticker)

        total_init = float(initial_capital_krw)

        for dt in common_index:
            portfolio_dates.append(dt)

            # 일별 자산 집계
            total_value = 0.0
            total_holdings = 0.0
            total_acquisition_cost = 0.0
            held_count = 0
            for tkr, ts in time_series_by_ticker.items():
                row = ts.loc[dt]

                # NaN 값을 안전하게 처리하여 total_value를 계산합니다.
                pv_val = row.get("pv")
                total_value += float(pv_val) if pd.notna(pv_val) else 0.0

                if tkr != "CASH":
                    # 호주는 소수점 수량을 허용 (최대 4자리)
                    sh_raw = row.get("shares", 0)
                    sh = float(sh_raw) if pd.notna(sh_raw) else 0.0

                    # NaN 값을 안전하게 처리하여 price를 가져옵니다.
                    price_val = row.get("price")
                    price = float(price_val) if pd.notna(price_val) else 0.0

                    avg_cost_val = row.get("avg_cost", 0.0)
                    avg_cost = float(avg_cost_val) if pd.notna(avg_cost_val) else 0.0

                    total_holdings += price * sh
                    if sh > 0:
                        held_count += 1
                        total_acquisition_cost += avg_cost * sh

            total_cash = total_value - total_holdings
            portfolio_values.append(total_value)

            # 일일 포트폴리오 수익률
            prev_equity = prev_total_pv
            if prev_total_pv is not None and prev_total_pv > 0:
                day_ret_pct = (
                    ((total_value / prev_total_pv) - 1.0) * 100.0 if prev_total_pv > 0 else 0.0
                )
            day_profit_loss = total_value - prev_equity if prev_equity is not None else 0.0
            prev_total_pv = total_value

            # 초기 자본 대비 누적 포트폴리오 수익률
            cum_ret_pct = (
                ((total_value / total_init) - 1.0) * 100.0 if total_init and total_init > 0 else 0.0
            )

            if not quiet:
                # 헤더 라인 출력
                denom = portfolio_topn if portfolio_topn > 0 else total_cnt
                date_kor = f"{pd.to_datetime(dt).year}년 {pd.to_datetime(dt).month}월 {pd.to_datetime(dt).day}일"

                if total_acquisition_cost > 0:
                    eval_profit_loss = total_holdings - total_acquisition_cost
                    eval_return_pct = (total_holdings / total_acquisition_cost - 1.0) * 100.0
                else:
                    eval_profit_loss = 0.0
                    eval_return_pct = 0.0

                summary_data = {
                    "principal": float(initial_capital_krw),
                    "total_equity": float(total_value),
                    "total_holdings_value": float(total_holdings),
                    "total_cash": float(total_cash),
                    "daily_profit_loss": float(day_profit_loss),
                    "daily_return_pct": float(day_ret_pct),
                    "eval_profit_loss": float(eval_profit_loss),
                    "eval_return_pct": float(eval_return_pct),
                    "cum_profit_loss": float(total_value - total_init),
                    "cum_return_pct": float(cum_ret_pct),
                    "held_count": int(held_count),
                    "portfolio_topn": int(denom),
                }

                summary_line = build_summary_line_from_summary_data(
                    summary_data,
                    money_formatter,
                    use_html=False,
                    prefix=f"{date_kor} |",
                )
                print(summary_line)

                # 전략에 따라 동적으로 헤더를 설정합니다.
                # signal_headers = ["이평선(값)", "고점대비", "점수", "신호지속일"] (참고용 예시)
                headers = list(SIGNAL_TABLE_HEADERS)

                decisions_list = []
                for tkr, ts in time_series_by_ticker.items():
                    row = ts.loc[dt]
                    name = name_by_ticker.get(tkr, "")
                    category_raw = category_by_ticker.get(tkr, "")
                    category_display = (
                        category_raw if category_raw and category_raw.upper() != "TBD" else "-"
                    )
                    decision = str(row.get("decision", "")).upper()

                    # NaN 값에 대한 안정성 강화: 모든 숫자 변수를 사용 전에 확인하고 처리합니다.
                    price_val = row.get("price")
                    price_today = float(price_val) if pd.notna(price_val) else 0.0

                    shares_val = row.get("shares")
                    shares = float(shares_val) if pd.notna(shares_val) else 0.0

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

                    _, s2, score, filter_val = (
                        row.get("signal1"),
                        row.get("signal2"),
                        row.get("score"),
                        row.get("filter"),
                    )

                    # 전략에 따라 신호 값의 포맷을 다르게 지정합니다.
                    # s1_str = ma_formatter(s1) if pd.notna(s1) else "-"
                    s2_str = f"{float(s2):.1f}%" if pd.notna(s2) else "-"  # 고점대비
                    score_str = f"{float(score):.1f}" if pd.notna(score) else "-"  # 점수
                    filter_str = f"{int(filter_val)}일" if pd.notna(filter_val) else "-"

                    display_status = decision
                    phrase = ""
                    note_from_strategy = str(row.get("note", "") or "")
                    if is_trade_decision and amount > 0 and price_today > 0:
                        # 거래 문구의 수량 포맷: coin/aus는 소수점 4자리, kor는 정수
                        if country in ("coin", "aus"):
                            qty_calc = round(float(amount) / float(price_today), 4)
                        else:
                            qty_calc = int(float(amount) // float(price_today))
                        if decision.startswith("BUY"):
                            tag = "매수"
                            if decision == "BUY_REPLACE":
                                tag = "교체매수"
                            # 보유수량 표시 포맷팅
                            if country in ("coin", "aus"):
                                qty_str = f"{qty_calc:,.4f}".rstrip("0").rstrip(".")
                            else:
                                qty_str = f"{int(qty_calc):,d}"
                            phrase = f"{tag} {qty_str}주 @ {price_formatter(price_today)} ({money_formatter(amount)})"
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
                            else:  # 이전 버전 호환용 (예: "SELL")
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
                    # 보유수량 표시 포맷: coin/aus는 4자리 소수, 그 외 정수
                    if country in ("coin", "aus"):
                        disp_shares_str = f"{disp_shares:,.4f}".rstrip("0").rstrip(".")
                    else:
                        disp_shares_str = f"{int(disp_shares):,d}"

                    current_row = [
                        0,
                        tkr,
                        name or tkr,
                        category_display,
                        display_status,
                        bd_str,
                        f"{hd}",
                        price_formatter(disp_price),
                        f"{tkr_day_ret:+.1f}%",
                        disp_shares_str,
                        money_formatter(amount),
                        hold_ret_str,
                        f"{w:.0f}%",
                        s2_str,
                        score_str,
                        filter_str,
                        phrase,
                    ]
                    decisions_list.append((decision, w, score, tkr, current_row))

                def sort_key(decision_tuple):
                    state, weight, score, tkr, _ = decision_tuple
                    order = strategy_module.DECISION_CONFIG.get(state, {}).get("order", 99)
                    try:
                        score_val = float(score)
                    except (TypeError, ValueError):
                        score_val = float("-inf")
                    sort_value = -score_val
                    return (order, sort_value, tkr)

                decisions_list.sort(key=sort_key)

                rows_sorted = []
                for idx, (_, _, _, _, row) in enumerate(decisions_list, 1):
                    row[0] = idx
                    rows_sorted.append(row)

                aligns = [
                    "right",
                    "right",
                    "left",
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
                    "center",
                    "left",
                ]
                str_rows = [[str(c) for c in row] for row in rows_sorted]

                # 일별 상세 테이블을 콘솔에 출력합니다.
                print("\n" + "\n".join(render_table_eaw(headers, str_rows, aligns)))
                print("")

            prev_dt = dt

        if not portfolio_values:
            if not quiet:
                print("시뮬레이션 결과가 없습니다.")
        else:
            final_value = portfolio_values[-1]
            if not quiet:
                print(f"\n백테스트 최종 자산: {money_formatter(final_value)}")
            peak = -1
            max_drawdown = 0
            drawdowns_pct = []  # Ulcer Index 계산을 위한 일별 낙폭(%) 리스트
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = 0.0
                if peak > 0:
                    drawdown = (peak - value) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                drawdowns_pct.append(drawdown * 100.0)

            # 시장 위험 회피(Risk-Off) 기간을 계산합니다.
            risk_off_periods = []
            try:
                market_regime_filter_enabled = bool(settings.MARKET_REGIME_FILTER_ENABLED)
            except AttributeError:
                print("오류: MARKET_REGIME_FILTER_ENABLED 설정이 logic/settings.py 에 정의되어야 합니다.")
                if log_file:
                    log_file.close()
                return None

            if market_regime_filter_enabled:
                # '시장 위험 회피' 노트가 있는지 확인하여 리스크 오프 기간을 식별합니다.
                # 모든 티커의 노트를 하나의 데이터프레임으로 합칩니다.
                notes_df = pd.DataFrame(
                    {tkr: ts["note"] for tkr, ts in time_series_by_ticker.items() if tkr != "CASH"},
                    index=common_index,
                )

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
                            end_of_period = is_risk_off_series.index[
                                is_risk_off_series.index.get_loc(dt) - 1
                            ]
                            risk_off_periods.append((start_of_period, end_of_period))
                            start_of_period = None

                    # 백테스트가 리스크 오프 기간 중에 끝나는 경우를 처리합니다.
                    if in_risk_off_period and start_of_period:
                        risk_off_periods.append((start_of_period, is_risk_off_series.index[-1]))

            start_date = portfolio_dates[0]
            end_date = portfolio_dates[-1]
            years = (end_date - start_date).days / 365.25
            cagr = 0
            if years > 0 and initial_capital_krw > 0:
                cagr = ((final_value / initial_capital_krw) ** (1 / years)) - 1

            # --- 벤치마크 (S&P 500) 성과 계산 ---
            from utils.data_loader import fetch_ohlcv

            # 국가에 따라 벤치마크 티커와 이름을 설정합니다.
            if country == "coin":
                benchmark_ticker = "BTC"
                benchmark_country = "coin"
            else:
                benchmark_ticker = "^GSPC"
                benchmark_country = country  # country는 지수 티커에 영향을 주지 않음

            benchmark_df = fetch_ohlcv(
                benchmark_ticker,
                country=benchmark_country,
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
                        benchmark_cum_ret_pct = (
                            (benchmark_end_price / benchmark_start_price) - 1
                        ) * 100
                        if years > 0:
                            benchmark_cagr_pct = (
                                (benchmark_end_price / benchmark_start_price) ** (1 / years) - 1
                            ) * 100

            # 샤프 지수(Sharpe Ratio) 계산
            pv_series = pd.Series(portfolio_values, index=pd.to_datetime(portfolio_dates))
            daily_returns = pv_series.pct_change().dropna()

            sharpe_ratio = 0
            # 일일 수익률이 있을 경우에만 계산
            if not daily_returns.empty and daily_returns.std() > 0:
                # 연간 252 거래일로 가정
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252**0.5)

            # 소르티노 지수(Sortino Ratio) 계산 (하락 위험만 고려)
            downside_returns = daily_returns[daily_returns < 0]
            sortino_ratio = 0
            if not downside_returns.empty:
                downside_deviation = downside_returns.std()
                if downside_deviation > 0:
                    sortino_ratio = (daily_returns.mean() / downside_deviation) * (252**0.5)

            # 칼마 지수(Calmar Ratio) 계산 (CAGR / MDD)
            calmar_ratio = (cagr * 100) / (max_drawdown * 100) if max_drawdown > 0 else 0

            # 얼서 지수(Ulcer Index) 계산
            ulcer_index = 0.0
            if drawdowns_pct:
                drawdowns_squared = [d**2 for d in drawdowns_pct]
                ulcer_index = (sum(drawdowns_squared) / len(drawdowns_squared)) ** 0.5

            # CUI (Calmar / Ulcer Index) 계산
            cui = calmar_ratio / ulcer_index if ulcer_index > 0 else 0.0

            summary = {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "initial_capital_krw": initial_capital_krw,
                "final_value": final_value,
                "cagr_pct": cagr * 100,
                "mdd_pct": max_drawdown * 100,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "ulcer_index": ulcer_index,
                "cui": cui,
                "cumulative_return_pct": (
                    (final_value / initial_capital_krw - 1) * 100 if initial_capital_krw > 0 else 0
                ),
                "risk_off_periods": risk_off_periods,
                "benchmark_cum_ret_pct": benchmark_cum_ret_pct,
                "benchmark_cagr_pct": benchmark_cagr_pct,
            }

            # 월별/연간 수익률 계산
            if portfolio_values:
                # 수익률 계산을 위해 시작점에 초기 자본을 추가
                start_row = pd.Series(
                    [initial_capital_krw], index=[start_date - pd.Timedelta(days=1)]
                )
                pv_series_with_start = pd.concat([start_row, pv_series])

                # 월별 수익률
                monthly_returns = pv_series_with_start.resample("ME").last().pct_change().dropna()
                summary["monthly_returns"] = monthly_returns

                # 월별 누적 수익률
                eom_pv = pv_series.resample("ME").last()
                monthly_cum_returns = (
                    (eom_pv / initial_capital_krw - 1).ffill()
                    if initial_capital_krw > 0
                    else pd.Series()
                )
                summary["monthly_cum_returns"] = monthly_cum_returns

                # 연간 수익률
                yearly_returns = pv_series_with_start.resample("YE").last().pct_change().dropna()
                summary["yearly_returns"] = yearly_returns

            return_value = summary

            # 종목별 성과 계산
            ticker_summaries = []
            # 원본 시계열 데이터를 사용하여 전체 기간에 대한 성과를 계산합니다.
            for tkr, ts_original in original_time_series_by_ticker.items():
                if tkr == "CASH":
                    continue

                # 1. 실현 손익 계산 (모든 매도 유형 포함)
                sell_decisions = [
                    "SELL_MOMENTUM",
                    "SELL_TREND",
                    "CUT_STOPLOSS",
                    "SELL_REPLACE",
                    "SELL_REGIME_FILTER",
                ]
                trades = ts_original[ts_original["decision"].isin(sell_decisions)]
                realized_profit = trades["trade_profit"].sum()
                total_trades = len(trades)
                winning_trades = len(trades[trades["trade_profit"] > 0])

                # 2. 미실현 손익 계산 (백테스트 종료 시점 기준)
                last_row = ts_original.iloc[-1]
                final_shares = float(last_row.get("shares", 0.0))
                unrealized_profit = 0.0
                if final_shares > 0:
                    final_price = float(last_row.get("price", 0.0))
                    avg_cost = float(last_row.get("avg_cost", 0.0))
                    if avg_cost > 0:
                        unrealized_profit = (final_price - avg_cost) * final_shares

                # 3. 총 기여도 (실현 + 미실현)
                total_contribution = realized_profit + unrealized_profit

                # 4. 기간 수익률 계산 (테스트 시작일과 상장일 중 더 늦은 날짜부터 계산)
                period_return_pct = 0.0
                listing_date = None
                if not ts_original.empty and "price" in ts_original.columns:
                    # 실제로 거래가 발생한 기간만을 고려하여 계산
                    # price가 0보다 큰 값들만 필터링
                    valid_price_mask = ts_original["price"] > 0
                    valid_data = ts_original[valid_price_mask]

                    if not valid_data.empty:
                        first_valid_price = valid_data["price"].iloc[0]
                        last_valid_price = valid_data["price"].iloc[-1]
                        listing_date = valid_data.index[0].strftime("%Y-%m-%d")

                        if (
                            pd.notna(first_valid_price)
                            and pd.notna(last_valid_price)
                            and first_valid_price > 0
                        ):
                            period_return_pct = ((last_valid_price / first_valid_price) - 1) * 100.0

                # 거래가 있거나, 최종 보유 수량이 있는 종목만 요약에 포함
                if total_trades > 0 or final_shares > 0:
                    win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
                    ticker_summaries.append(
                        {
                            "ticker": tkr,
                            "name": name_by_ticker.get(tkr, ""),
                            "total_trades": total_trades,
                            "win_rate": win_rate,
                            "realized_profit": realized_profit,
                            "unrealized_profit": unrealized_profit,
                            "total_contribution": total_contribution,
                            "period_return_pct": period_return_pct,
                            "listing_date": listing_date,
                        }
                    )

    finally:
        # 원래의 stdout으로 복원하고 로그 파일을 닫습니다.
        if not quiet and log_file:
            sys.stdout = original_stdout
            log_file.close()
            print(f"\n백테스트가 완료되었습니다. 상세 내용은 {log_path} 파일을 확인하세요.")

    # 백테스트 결과 출력
    if not quiet:
        print(
            f"Backtest result for {country} from {test_months_range[0]} to {test_months_range[1]}"
        )
        print(f"Initial capital: {initial_capital:,}")
        print(f"Portfolio topn: {portfolio_topn}")
        print(f"Core start date: {core_start_dt}")
        print(f"Summary: {result.summary}")
        print(f"Ticker summaries: {result.ticker_summaries}")

    return return_value


if __name__ == "__main__":
    main(country="kor", quiet=False)
