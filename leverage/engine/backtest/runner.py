"""백테스트 핵심 로직."""

import numpy as np
import pandas as pd
import yfinance as yf

from leverage.constants import INITIAL_CAPITAL_KRW
from leverage.data_adapter import (
    _extract_field,
    compute_bounds,
    current_trading_day,
    download_fx,
    download_opens,
    download_prices,
)
from leverage.engine.backtest.signals import compute_signals, pick_target
from leverage.report import format_kr_money, render_table_eaw


def run_backtest(
    settings: dict,
    *,
    pre_prices: pd.DataFrame | None = None,
    pre_opens: pd.DataFrame | None = None,
    pre_fx: pd.Series | None = None,
    pre_bench: pd.DataFrame | None = None,
    start_bound_override: pd.Timestamp | None = None,
    drop_today: bool = False,
) -> dict[str, object]:
    start_bound_base, warmup_start, end_bound = compute_bounds(settings)
    start_bound = start_bound_override or start_bound_base

    prices_full = pre_prices.copy() if pre_prices is not None else download_prices(settings, warmup_start)
    opens_full = pre_opens.copy() if pre_opens is not None else download_opens(settings, warmup_start)

    offense = settings["offense_ticker"]
    defense = settings["defense_ticker"]
    assets = list({offense, defense})

    signal_df_full = compute_signals(prices_full[settings["signal_ticker"]], settings)
    returns_full = opens_full[assets].pct_change(fill_method=None)

    common_index = (
        signal_df_full.index.intersection(prices_full.index)
        .intersection(opens_full.index)
        .intersection(returns_full.index)
    )
    common_index = common_index[common_index >= start_bound]

    # 장중(미완성 봉) 제외: 신호는 "마지막으로 닫힌 거래일 종가"로만 확정한다.
    # drop_today=True 이면 오늘(현지 기준) 세션을 신호 계산에서 제외한다.
    if drop_today and len(common_index) > 0:
        cutoff = current_trading_day(settings.get("market", "kor"))
        common_index = common_index[common_index < cutoff]
        if len(common_index) == 0:
            raise ValueError("오늘 세션을 제외하면 사용할 수 있는 거래일이 없습니다.")

    opens = opens_full.loc[common_index]
    # 종가 신호는 다음 거래일 시초가에만 매매할 수 있으므로 실행 타깃은 하루 늦춘다.
    signal_targets = []
    prev_signal_target = settings["offense_ticker"]
    for _, row in signal_df_full.iterrows():
        signal_target = pick_target(row, prev_signal_target, settings)
        signal_targets.append(signal_target)
        prev_signal_target = signal_target
    signal_target_series = pd.Series(signal_targets, index=signal_df_full.index, name="signal_target")
    execution_target_series = signal_target_series.shift(1)

    common_index = common_index[execution_target_series.reindex(common_index).notna()]
    if len(common_index) == 0:
        raise ValueError("전일 종가 신호를 적용할 수 있는 실행 데이터가 없습니다.")

    signal_df = signal_df_full.loc[common_index].copy()
    signal_df["signal_target"] = signal_target_series.loc[common_index]
    signal_df["target"] = execution_target_series.loc[common_index]
    returns = returns_full.loc[common_index]
    if returns.dropna().empty:
        raise ValueError("수익률 데이터가 비어 있습니다. 가격/기간 설정을 확인하세요.")

    # 환율 데이터(원/달러) - 한국 시장은 1.0 고정
    fx = pre_fx.copy() if pre_fx is not None else download_fx(settings, start_bound)
    fx = fx.reindex(common_index, method="ffill").dropna()

    # 초기 자본: 원화 -> 달러
    first_date = common_index[0]
    init_fx = fx.loc[first_date]
    cash_usd = INITIAL_CAPITAL_KRW / init_fx
    initial_capital_usd = cash_usd
    qty = {s: 0 for s in assets}
    prev_value = cash_usd

    equity = []
    daily_rets = []
    daily_log: list[str] = []
    segment_lines: list[str] = []
    seg_target = None
    seg_start_date = None
    seg_start_value = None
    seg_days = 0
    seg_qty = 0
    last_total_value = prev_value

    weekday_map = ["월", "화", "수", "목", "금", "토", "일"]

    def _fmt_date(dt: pd.Timestamp) -> str:
        return f"{dt.date()}({weekday_map[dt.weekday()]})"

    # 시장에 따른 금액 포맷
    market = settings.get("market", "us")

    def _fmt_money(val: float) -> str:
        """시장에 따라 금액을 원화 또는 달러로 포맷."""
        if market == "kor":
            return format_kr_money(val)
        else:
            return f"${val:,.2f}"

    # 티커-이름 매핑 (구간 요약에서 사용)
    _ticker_names = {
        settings["offense_ticker"]: settings.get("offense_name", settings["offense_ticker"]),
        settings["defense_ticker"]: settings.get("defense_name", settings["defense_ticker"]),
        settings["signal_ticker"]: settings.get("signal_name", settings["signal_ticker"]),
        "CASH": "현금",
    }

    def _get_display_name(ticker: str) -> str:
        """이름(티커) 형식으로 표시."""
        name = _ticker_names.get(ticker, ticker)
        if name != ticker:
            return f"{name}({ticker})"
        return ticker

    def _add_segment(start_dt, end_dt, days, tgt, qty_val, pnl_val, pct_val):
        if start_dt is None or end_dt is None or tgt is None:
            return
        display_name = _get_display_name(tgt)
        segment_lines.append(f"[{_fmt_date(start_dt)} ~ {_fmt_date(end_dt)}] {display_name}: {days} 거래일")
        if tgt != "CASH":
            segment_lines.append(f" - 보유수량: {qty_val:,}")
            segment_lines.append(f" - 손익: {_fmt_money(pnl_val)}")
            segment_lines.append(f" - 손익(%): {pct_val * 100:+.4f}%")

    hold_days = {s: 0 for s in assets}
    hold_start_dates = {s: None for s in assets}
    pre_switch_hold_days = {s: 0 for s in assets}
    pre_switch_hold_start_dates = {s: None for s in assets}
    pre_switch_prev_target = None
    asset_pnl = {s: 0.0 for s in assets}
    prev_pos_value = {s: 0.0 for s in assets}
    asset_exposure_days = {s: 0 for s in assets}
    trade_counts = {s: 0 for s in assets}
    win_days = {s: 0 for s in assets}
    trade_days = {s: 0 for s in assets}
    prev_target = None
    slip_raw = settings["slippage"]
    slip = slip_raw / 100 if slip_raw > 1 else slip_raw / 100
    buy_slip = slip
    sell_slip = slip

    for date in common_index:
        # 매 반복 시작 전 상태 스냅샷 (경고 모드에서 전환 전 상태 복원용)
        pre_switch_hold_days = dict(hold_days)
        pre_switch_hold_start_dates = dict(hold_start_dates)
        pre_switch_prev_target = prev_target
        start_value_today = last_total_value
        target = signal_df.at[date, "target"]
        trade_cf = {s: 0.0 for s in assets}  # 자산별 현금흐름(매수+: 자금투입, 매도-: 인출)
        end_val_for_seg = None
        changed = seg_target is not None and target != seg_target

        # 초기 세그먼트 설정
        if seg_target is None:
            seg_target = target
            seg_start_date = date
            seg_start_value = None  # 거래 후 값으로 설정
            seg_days = 0
            seg_qty = qty[target] if target != "CASH" else 0

        # 오늘 시초가
        prices_today = {s: opens.at[date, s] for s in assets}

        # 포지션 변경 시 청산
        if prev_target and prev_target != target and prev_target != "CASH":
            sell_price_raw = prices_today[prev_target]
            if not pd.isna(sell_price_raw):
                sell_price = sell_price_raw * (1 - sell_slip)
                cash_usd += qty[prev_target] * sell_price
                trade_cf[prev_target] -= qty[prev_target] * sell_price
                qty[prev_target] = 0
            end_val_for_seg = cash_usd + sum(
                (qty[s] * prices_today[s] if not pd.isna(prices_today[s]) else 0.0) for s in assets
            )

        # 세그먼트 전환 처리: 청산 후 평가액을 사용해 종료
        if changed:
            end_value = end_val_for_seg if end_val_for_seg is not None else start_value_today
            pnl_seg = end_value - seg_start_value
            pct_seg = (end_value / seg_start_value - 1) if seg_start_value != 0 else 0.0
            _add_segment(
                seg_start_date,
                date,  # 전환일 포함
                seg_days + 0,  # 전환일까지 포함한 보유일(오늘까지 이미 카운트됨)
                seg_target,
                seg_qty,
                pnl_seg,
                pct_seg,
            )
            seg_target = target
            seg_start_date = date
            seg_start_value = None  # 거래 후 값으로 설정
            seg_days = 0
            seg_qty = qty[target] if target != "CASH" else 0

        # 매수
        if target != "CASH":
            buy_price_raw = prices_today[target]
            if not pd.isna(buy_price_raw):
                buy_price = buy_price_raw * (1 + buy_slip)
                if buy_price <= 0:
                    raise ValueError(f"유효하지 않은 매수가입니다: {target} @ {date.date()} = {buy_price}")
                purch_qty = int(cash_usd / buy_price)
                if purch_qty > 0 and (prev_target != target):
                    cash_usd -= purch_qty * buy_price
                    trade_cf[target] += purch_qty * buy_price
                    qty[target] += purch_qty

        # 평가
        position_value = {s: (qty[s] * prices_today[s] if not pd.isna(prices_today[s]) else 0.0) for s in assets}
        total_value = cash_usd + sum(position_value.values())
        daily_ret = (total_value - prev_value) / prev_value if prev_value != 0 else 0.0
        pnl = total_value - prev_value
        prev_value = total_value

        # 세그먼트 시작값을 거래 후 기준으로 설정
        if seg_start_value is None:
            seg_start_value = total_value
            seg_qty = qty[target] if target != "CASH" else 0

        equity.append(total_value)
        daily_rets.append(daily_ret)
        seg_days += 1
        seg_qty = qty[target] if target != "CASH" else 0
        last_total_value = total_value

        # 보유일/노출일 및 티커별 기여도
        for sym in assets:
            # 포지션 가치 증감에서 거래 현금흐름을 제외해 가격 변동만 귀속
            delta = (position_value[sym] - prev_pos_value[sym]) - trade_cf[sym]
            asset_pnl[sym] += delta
            prev_pos_value[sym] = position_value[sym]

            if sym == target and (qty[sym] > 0 or sym == "CASH"):
                hold_days[sym] += 1
                asset_exposure_days[sym] += 1
                trade_days[sym] += 1
                if daily_ret > 0:
                    win_days[sym] += 1
                if prev_target != sym:
                    trade_counts[sym] += 1
                if prev_target != sym or hold_start_dates[sym] is None:
                    hold_start_dates[sym] = date.strftime("%Y-%m-%d")
            else:
                hold_days[sym] = 0
                hold_start_dates[sym] = None

        if target == "CASH":
            weights = {s: 0.0 for s in assets}
            cash_value = cash_usd
            total_value = cash_usd
        else:
            position_value = {s: qty[s] * prices_today[s] for s in assets}
            total_pos = sum(position_value.values())
            total_value = cash_usd + total_pos
            weights = {s: (position_value[s] / total_value if total_value > 0 else 0.0) for s in assets}
            cash_value = cash_usd
        fx_today = fx.loc[date]
        krw_value = total_value * fx_today

        # 테이블 데이터 준비
        # 테이블 데이터 준비
        headers = [
            "#",
            "티커",
            "상태",
            "보유일",
            "현재가",
            "일간(%)",
            "수량",
            "금액",
            "평가손익",
            "평가(%)",
            "비중",
            "문구",
        ]
        aligns = [
            "center",
            "center",
            "center",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "left",
        ]
        rows = []

        # 현금 행
        cash_val_fmt = format_kr_money(cash_value) if market == "kor" else f"{cash_value:,.2f}"
        rows.append(
            [
                "1",
                "CASH",
                "HOLD" if cash_value > 0 else "WAIT",
                str(hold_days.get("CASH", 0)),
                "1",
                "+0.0%",
                f"{cash_value:,.0f}" if market == "kor" else f"{cash_value:,.2f}",
                cash_val_fmt,
                "0원" if market == "kor" else "0",
                "+0.0%",
                f"{cash_value / total_value:0.1%}",
                "",
            ]
        )

        # 자산 행들
        row_idx = 2
        for sym in assets:
            price = prices_today[sym]
            ret = returns.at[date, sym] if sym in returns.columns else 0.0
            weight = weights[sym]
            position_value = qty[sym] * price
            qty_disp = qty[sym]

            if sym == target and prev_target != sym:
                state = "BUY"
            elif sym == target and prev_target == sym:
                state = "HOLD"
            elif sym != target and prev_target == sym:
                state = "SELL"
            else:
                state = "WAIT"

            eval_pnl = pnl if sym == target else 0.0
            eval_pct = ret if weight > 0 else 0.0

            note = ""
            if sym == target:
                note = "타깃"
            elif sym == settings["offense_ticker"] and state in ["WAIT", "SELL"]:
                # Trade Ticker가 선택되지 않은 경우 드로다운 정보 표시
                current_dd = signal_df.at[date, "drawdown"]

                # 현재 상태에 따라 적용되는 컷오프가 다름
                # 하지만 "왜 안 샀냐"를 설명할 때는 "매수 기준"을 보여주는 것이 적절함
                # 또는 현재 보유 중이 아니므로 매수 기준(buy_cutoff)을 만족해야 함

                buy_cut_raw = settings["drawdown_buy_cutoff"]
                buy_cut = buy_cut_raw / 100 if buy_cut_raw > 1 else buy_cut_raw
                threshold = -buy_cut

                # 드로다운이 임계값보다 낮아서(더 많이 떨어져서) 못 사는 경우
                if current_dd <= threshold:
                    needed = threshold - current_dd
                    note = f"DD {current_dd * 100:.2f}% (매수컷 {threshold * 100:.2f}%, 필요 {needed * 100:+.2f}%)"

            rows.append(
                [
                    str(row_idx),
                    _get_display_name(sym),
                    state,
                    str(hold_days[sym]),
                    f"{price:,.0f}" if market == "kor" else f"{price:,.2f}",
                    f"{ret:+.2%}",
                    f"{qty_disp:,.0f}",
                    format_kr_money(position_value) if market == "kor" else f"{position_value:,.2f}",
                    format_kr_money(eval_pnl) if market == "kor" else f"{eval_pnl:,.2f}",
                    f"{eval_pct * 100:+.2f}%",
                    f"{weight:0.0%}",
                    note,
                ]
            )
            row_idx += 1

        table_lines = render_table_eaw(headers, rows, aligns)

        if market == "kor":
            header_line = (
                f"{date.date()} | 목표: {target} | 총자산: {format_kr_money(krw_value)} | "
                f"일간수익률: {daily_ret:+.2%} | 누적수익률: {(total_value / initial_capital_usd - 1):+.2%}"
            )
        else:
            header_line = (
                f"{date.date()} | 목표: {target} | 총자산: ${total_value:,.2f} "
                f"({format_kr_money(krw_value)}) | 일간수익률: {daily_ret:+.2%} | "
                f"누적수익률: {(total_value / initial_capital_usd - 1):+.2%}"
            )

        daily_log.append(header_line)
        daily_log.extend(table_lines)
        daily_log.append("")  # 빈 줄

        # 다음 일자를 위한 상태 업데이트
        prev_target = target

    equity_series = pd.Series(equity, index=returns.index)
    daily_rets_series = pd.Series(daily_rets, index=returns.index)

    # KRW 기준 성과(초기/최종 자산도 KRW로 표기하므로 일관)
    krw_series = equity_series * fx.loc[equity_series.index]
    cagr = (krw_series.iloc[-1] / INITIAL_CAPITAL_KRW) ** (252 / len(krw_series)) - 1
    vol = daily_rets_series.std() * np.sqrt(252)
    sharpe = cagr / vol if vol != 0 else np.nan
    running_max = krw_series.cummax()
    drawdown = krw_series / running_max - 1
    max_dd = drawdown.min()
    win_rate = (daily_rets_series > 0).mean()
    last_fx = fx.iloc[-1]
    final_krw = krw_series.iloc[-1]
    period_return = krw_series.iloc[-1] / INITIAL_CAPITAL_KRW - 1
    start_date = equity_series.index[0].date()
    end_date = equity_series.index[-1].date()
    total_pnl = krw_series.iloc[-1] - INITIAL_CAPITAL_KRW

    # 벤치마크(VOO, QQQ)의 기간 수익률 및 CAGR
    bench_raw_entries = settings["benchmarks"]
    bench_info: list[dict[str, str]] = []
    for b in bench_raw_entries:
        if isinstance(b, dict):
            ticker = b.get("ticker")
            name = b.get("name") or ticker
        else:
            ticker = str(b)
            name = str(b)
        if not ticker:
            continue
        bench_info.append({"ticker": ticker, "name": name})
    bench_tickers = [b["ticker"] for b in bench_info]
    bench_lines = []
    bench_status: list[str] = []
    try:
        if pre_bench is not None:
            bench_prices = pre_bench.copy()
        else:
            # 시장에 따라 데이터 소스 분기
            market = settings.get("market", "us")
            if market == "kor":
                # 한국: pykrx 사용
                try:
                    from pykrx import stock as pykrx_stock
                except ImportError:
                    raise ImportError("pykrx 패키지가 설치되어 있지 않습니다.")
                start_str = start_bound.strftime("%Y%m%d")
                end_str = pd.Timestamp.today().strftime("%Y%m%d")
                bench_dfs = {}
                for ticker in bench_tickers:
                    df = pykrx_stock.get_market_ohlcv_by_date(start_str, end_str, ticker)
                    if df is not None and not df.empty:
                        bench_dfs[ticker] = df["종가"]
                if bench_dfs:
                    bench_prices = pd.DataFrame(bench_dfs)
                    bench_prices.index = pd.to_datetime(bench_prices.index)
                else:
                    bench_prices = pd.DataFrame()
            else:
                # 미국: yfinance 사용
                bench_raw = yf.download(bench_tickers, start=start_bound, auto_adjust=True, progress=False)
                bench_prices = _extract_field(bench_raw, "Close", bench_tickers)
        bench_prices = bench_prices.reindex(equity_series.index, method="ffill")
        for b in bench_info:
            t = b["ticker"]
            if t not in bench_prices.columns:
                bench_status.append(f"{t}: 다운로드 실패 또는 컬럼 없음")
                continue
            series_full = bench_prices[t]
            series = series_full.reindex(returns.index).dropna()
            if series.empty:
                first_valid = series_full.first_valid_index()
                if first_valid is None:
                    bench_status.append(f"{t}: 데이터 없음(상장일 이후 기간 부족?)")
                else:
                    bench_status.append(f"{t}: 데이터 없음(가용 시작 {first_valid.date()} 이후, 기간 부족)")
                continue
            br = series.iloc[-1] / series.iloc[0] - 1
            cagr_b = (1 + br) ** (252 / len(series)) - 1
            eq_norm = series / series.iloc[0]
            dd_b = (eq_norm / eq_norm.cummax() - 1).min()
            bench_lines.append((t, b["name"], br, cagr_b, dd_b))
    except Exception as exc:
        bench_status.append(f"벤치마크 다운로드 실패: {exc}")
        bench_lines = []

    # 요약 섹션
    months = max(1, round((end_date - start_date).days / 30.0, 1))
    summary_lines = [
        "8. ========= 백테스트 결과 요약 ==========",
        f"| 기간: {start_date} ~ {end_date} ({months} 개월)",
        f"| 초기 자본: {format_kr_money(INITIAL_CAPITAL_KRW)}",
        f"| 최종 자산: {format_kr_money(final_krw)}",
        f"| 기간수익률(%): {period_return * 100:+.2f}%",
    ]

    if "months_range" in settings:
        requested_months = settings["months_range"]
        if months < requested_months:
            summary_lines.append(f"| 참고: months_range {requested_months} → 실제 {months}개월 (데이터 가용 기간)")
    else:
        # start_date 모드
        summary_lines.append(f"| 설정 시작일: {settings.get('start_date', 'N/A')}")

    bench_table_lines: list[str] = []
    bench_summary_lines: list[str] = []
    bench_error = None

    summary_lines.extend(
        [
            f"| CAGR(%): {cagr * 100:+.2f}%",
            f"| MDD(%): {max_dd * 100:.2f}%",
        ]
    )

    # 전략 vs 벤치마크 비교 테이블
    trade_display = settings["offense_ticker"]
    defense_display = settings["defense_ticker"]
    if settings.get("offense_name") and settings["offense_name"] != settings["offense_ticker"]:
        trade_display = f"{settings['offense_name']}({settings['offense_ticker']})"
    if settings.get("defense_name") and settings["defense_name"] != settings["defense_ticker"]:
        defense_display = f"{settings['defense_name']}({settings['defense_ticker']})"
    strat_label = f"{trade_display}<->{defense_display}"
    perf_rows = [
        [
            "1",
            strat_label,
            f"{period_return * 100:+.2f}%",
            f"{cagr * 100:+.2f}%",
            f"{max_dd * 100:.2f}%",
        ]
    ]
    idx = 2
    if bench_lines:
        for t, name, br, cagr_b, dd_b in bench_lines:
            perf_rows.append(
                [
                    str(idx),
                    f"{name}({t})" if name and name != t else t,
                    f"{br * 100:+.2f}%",
                    f"{cagr_b * 100:+.2f}%",
                    f"{dd_b * 100:.2f}%",
                ]
            )
            idx += 1
    else:
        perf_rows.append(
            [
                str(idx),
                "벤치마크",
                "-",
                bench_error or "-",
                "-",
            ]
        )

    bench_table_lines = render_table_eaw(
        ["#", "티커", "기간 수익률(%)", "CAGR(%)", "MDD(%)"],
        perf_rows,
        ["center", "left", "right", "right", "right"],
    )

    # 마지막 세그먼트 닫기
    if seg_target is not None and seg_start_date is not None:
        end_value = equity_series.iloc[-1]
        pnl_seg = end_value - seg_start_value
        pct_seg = (end_value / seg_start_value - 1) if seg_start_value != 0 else 0.0
        _add_segment(
            seg_start_date,
            common_index[-1],
            seg_days,
            seg_target,
            seg_qty,
            pnl_seg,
            pct_seg,
        )

    # 종목별 성과 요약
    # 티커+이름 매핑
    ticker_names = {
        settings["offense_ticker"]: settings.get("offense_name", settings["offense_ticker"]),
        settings["defense_ticker"]: settings.get("defense_name", settings["defense_ticker"]),
        settings["signal_ticker"]: settings.get("signal_name", settings["signal_ticker"]),
        "CASH": "현금",
    }

    asset_rows = []
    for idx, sym in enumerate(["CASH"] + assets, start=1):
        # 티커+이름 표시
        name = ticker_names.get(sym, sym)
        display_name = f"{name}({sym})" if name != sym else sym

        if sym == "CASH":
            pnl_usd = 0.0
            days = asset_exposure_days.get(sym, 0)
            trades = trade_counts.get(sym, 0)
            w_days = win_days.get(sym, 0)
            t_days = trade_days.get(sym, 0)
            win_rate_sym = w_days / t_days * 100 if t_days > 0 else 0.0
        else:
            pnl_usd = asset_pnl.get(sym, 0.0)
            days = asset_exposure_days.get(sym, 0)
            trades = trade_counts.get(sym, 0)
            w_days = win_days.get(sym, 0)
            t_days = trade_days.get(sym, 0)
            win_rate_sym = w_days / t_days * 100 if t_days > 0 else 0.0
        krw_pnl = pnl_usd * last_fx
        asset_rows.append(
            [
                str(idx),
                display_name,
                f"{pnl_usd:,.2f}",
                f"{pnl_usd:,.2f}",
                format_kr_money(krw_pnl),
                str(days),
                str(trades),
                f"{win_rate_sym:.1f}%",
                ("주요 기여" if (total_pnl != 0 and abs(pnl_usd) >= abs(total_pnl) * 0.1) else ""),
            ]
        )

    asset_headers = [
        "#",
        "티커",
        "기여도(USD)",
        "총손익(USD)",
        "총손익(KRW)",
        "노출일수",
        "거래횟수",
        "승률",
        "비고",
    ]
    asset_aligns = [
        "center",
        "center",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "left",
    ]
    asset_summary_lines = ["7. ========= 종목별 성과 요약 =========="]
    asset_summary_lines.extend(render_table_eaw(asset_headers, asset_rows, asset_aligns))

    # 주별/월별 성과 요약
    weekly_lines: list[str] = ["5. ========= 주별 성과 요약 =========="]
    try:
        weekly = krw_series.resample("W-FRI").last().dropna()
        weekly_ret = weekly.pct_change().fillna(0)
        base_krw = INITIAL_CAPITAL_KRW
        weekly_cum = weekly / base_krw - 1
        w_rows = []
        for d, val in weekly.items():
            w_rows.append(
                [
                    d.date().isoformat(),
                    f"{format_kr_money(val)}",
                    f"{weekly_ret.loc[d] * 100:+.2f}%",
                    f"{weekly_cum.loc[d] * 100:+.2f}%",
                ]
            )
        weekly_lines.extend(
            render_table_eaw(
                ["주차(종료일)", "평가금액", "주간 수익률", "누적 수익률"],
                w_rows,
                ["center", "right", "right", "right"],
            )
        )
    except Exception:
        weekly_lines.append("| 주간 요약 생성 실패")

    monthly_lines: list[str] = ["6. ========= 월별 성과 요약 =========="]
    try:
        # 'M'는 pandas에서 deprecated 예정 → 'ME'(month end)로 변경
        monthly = krw_series.resample("ME").last().dropna()
        monthly_ret = monthly.pct_change().fillna(0)

        years = sorted({d.year for d in monthly.index})
        headers = ["연도"] + [f"{m}월" for m in range(1, 13)] + ["연간"]
        rows = []
        for yr in years:
            monthly_vals = []
            year_vals = monthly[monthly.index.year == yr]
            year_ret = None
            for m in range(1, 13):
                # Find the date for the last day of the month in the current year
                # This handles cases where a month might be missing or incomplete
                month_end_dates = year_vals[year_vals.index.month == m].index
                if len(month_end_dates) == 0:
                    monthly_vals.append("  -    ")
                    continue
                # Use the last available date for the month to get its return
                dt = month_end_dates[-1]
                r = monthly_ret.loc[dt]
                monthly_vals.append(f"{r * 100:+.2f}%")
            if len(year_vals) > 0:
                year_ret = year_vals.iloc[-1] / year_vals.iloc[0] - 1
            rows.append([str(yr)] + monthly_vals + [f"{year_ret * 100:+.2f}%" if year_ret is not None else "  -    "])

        monthly_lines.extend(render_table_eaw(headers, rows, ["center"] + ["right"] * 13 + ["right"]))
    except Exception:
        monthly_lines.append("| 월간 요약 생성 실패")

    used_settings_lines = [
        "3. ========= 사용된 설정값 ==========",
    ]
    if "months_range" in settings:
        used_settings_lines.append(f"| 테스트 기간: 최근 {settings['months_range']}개월 (실제 {months}개월)")
    else:
        used_settings_lines.append(f"| 테스트 기간: {start_date} ~ (실제 {months}개월)")

    used_settings_lines.extend(
        [
            f"| 초기 자본: {format_kr_money(INITIAL_CAPITAL_KRW)}",
            f"| buy_cutoff: {settings['drawdown_buy_cutoff']}%",
            f"| sell_cutoff: {settings['drawdown_sell_cutoff']}%",
            f"| signal_ticker: {settings['signal_ticker']}",
            f"| offense_ticker: {settings['offense_ticker']}",
            f"| defense_ticker: {settings['defense_ticker']}",
            f"| slippage: {settings['slippage']}%",
        ]
    )

    # 추천용 추가 데이터 (마지막 날 정보)
    last_date = signal_df.index[-1]
    last_row = signal_df.iloc[-1]
    last_prices = {sym: prices_full.at[last_date, sym] for sym in assets if sym in prices_full.columns}

    # 전일 종가 및 보유 시작 시점 종가 추출 (수익률 계산용)
    prev_prices = {}
    holding_start_prices = {}
    last_idx = prices_full.index.get_loc(last_date)
    last_target = signal_df["signal_target"].iloc[-1]

    for sym in assets:
        if sym in prices_full.columns:
            # 전일 종가
            if last_idx > 0:
                prev_prices[sym] = prices_full.iloc[last_idx - 1][sym]
            else:
                prev_prices[sym] = last_prices[sym]

            # 보유 시작 시점 종가 (N거래일 보유 중이면 N거래일 전 종가 기준)
            # sym == last_target 인 경우만 계산 (사용자 요청: 미보유 종목은 0%)
            if sym == last_target:
                h_days = hold_days.get(sym, 0)
                # h_days가 6이면, 6일 전 종가(last_idx - 6) 대비 현재가로 수익률 계산
                if h_days > 0 and last_idx >= h_days:
                    holding_start_prices[sym] = prices_full.iloc[last_idx - h_days][sym]
                elif h_days > 0:
                    # If not enough history for h_days, use the earliest available price
                    holding_start_prices[sym] = prices_full.iloc[0][sym]
            else:
                # For non-target assets, cumulative return from holding start is 0
                holding_start_prices[sym] = last_prices[sym]  # Set to current price to make cum_returns 0

    current_drawdown = last_row["drawdown"]
    buy_cutoff = -settings["drawdown_buy_cutoff"] / 100
    needed_recovery = buy_cutoff - current_drawdown if current_drawdown < buy_cutoff else 0

    # 장중 정보용: 오늘(미완성 봉 포함) 실시간 시그널 드로다운
    live_drawdown = float(signal_df_full["drawdown"].iloc[-1])

    recommendation_data = {
        "last_date": last_date.date().isoformat(),
        "last_prices": last_prices,
        "daily_returns": {
            sym: (last_prices[sym] / prev_prices[sym] - 1) if sym in prev_prices else 0.0 for sym in assets
        },
        "cum_returns": {
            sym: (last_prices[sym] / holding_start_prices[sym] - 1) if sym in holding_start_prices else 0.0
            for sym in assets
        },
        "holding_start_prices": holding_start_prices,
        "current_drawdown": current_drawdown,
        "live_drawdown": live_drawdown,
        "buy_cutoff": settings["drawdown_buy_cutoff"],
        "sell_cutoff": settings["drawdown_sell_cutoff"],
        "needed_recovery": needed_recovery * 100,  # 퍼센트로 변환
    }

    # 전환 전 상태 데이터 (경고 모드에서 사용)
    pre_switch_data = {
        "target": pre_switch_prev_target,
        "hold_days": pre_switch_hold_days,
        "holding_start_date": pre_switch_hold_start_dates.get(pre_switch_prev_target),
    }
    # 전환 전 타깃의 누적 수익률 계산
    ps_target = pre_switch_prev_target
    if ps_target and ps_target != last_target and ps_target in prices_full.columns:
        pre_switch_data["last_price"] = float(last_prices[ps_target])
        if ps_target in prev_prices and prev_prices[ps_target] != 0:
            pre_switch_data["daily_return"] = float(last_prices[ps_target] / prev_prices[ps_target] - 1)
        ps_h = pre_switch_hold_days.get(ps_target, 0)
        if ps_h > 0 and last_idx >= ps_h:
            ps_start_price = prices_full.iloc[last_idx - ps_h][ps_target]
            pre_switch_data["cum_return"] = float(last_prices[ps_target] / ps_start_price - 1)
        else:
            pre_switch_data["cum_return"] = 0.0

    # 장중 정보용: 신호 확정과 무관하게 "가장 최근(오늘 포함) 가격"을 함께 반환한다.
    # drop_today=True 로 신호는 어제 종가로 확정했더라도, 표시용 현재가는 오늘 실시간값을 쓰기 위함.
    live_date = prices_full.index[-1]
    live_prices = {sym: float(prices_full.at[live_date, sym]) for sym in assets if sym in prices_full.columns}

    return {
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "live_date": live_date.date().isoformat(),
        "live_prices": live_prices,
        "final_equity_usd": round(equity_series.iloc[-1], 2),
        "final_equity_krw": round(final_krw, 0),
        "period_return": period_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "last_target": last_target,
        "daily_log": daily_log,
        "summary_lines": summary_lines,
        "bench_summary_lines": bench_summary_lines,
        "bench_table_lines": bench_table_lines,
        "asset_summary_lines": asset_summary_lines,
        "weekly_summary_lines": weekly_lines,
        "monthly_summary_lines": monthly_lines,
        "bench_error": bench_error,
        "used_settings_lines": used_settings_lines,
        "segment_lines": segment_lines,
        "recommendation_data": recommendation_data,
        "holding_days": hold_days.get(last_target, 0),
        "holding_start_date": hold_start_dates.get(last_target),
        "pre_switch_data": pre_switch_data,
    }
