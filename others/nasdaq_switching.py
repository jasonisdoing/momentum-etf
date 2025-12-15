"""
Nasdaq Leverage Switching Strategy Recommendation Script (Standalone)
이 파일은 프로젝트의 추천 로직과 튜닝 로직을 단일 파일로 번들링한 것입니다.
실행 시 자동으로 최적 파라미터를 튜닝하고, 그 결과를 바탕으로 추천을 생성합니다.
"""

import itertools
import multiprocessing
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from unicodedata import east_asian_width, normalize

import numpy as np
import pandas as pd
import yfinance as yf

# =============================================================================
# 1. Settings & Config
# =============================================================================

DEFAULT_SETTINGS = {
    "months_range": 12,  # 기본 12개월 (튜닝 시에도 이 기간 사용)
    "signal_ticker": "QQQ",
    "trade_ticker": "TQQQ",
    "slippage": 0.05,
    "backtested_date": datetime.now().strftime("%Y-%m-%d"),
    "defense_ticker": "GLDM",
    "drawdown_buy_cutoff": 0.3,  # 초기값 (튜닝으로 덮어씌워짐)
    "drawdown_sell_cutoff": 0.4,  # 초기값 (튜닝으로 덮어씌워짐)
    "benchmarks": [
        {"ticker": "SPMO", "name": "모멘텀"},
        {"ticker": "VOO", "name": "S&P 500"},
        {"ticker": "QQQ", "name": "Nasdaq 1배"},
        {"ticker": "QLD", "name": "Nasdaq 2배"},
        {"ticker": "TQQQ", "name": "Nasdaq 3배"},
        {"ticker": "GLDM", "name": "SPDR 금 미니 ETF"},
        {"ticker": "GDX", "name": "반에크 금광 ETF"},
    ],
}

# 튜닝 범위 설정 (tune.py와 동일)
TUNING_CONFIG = {
    "drawdown_buy_cutoff": np.round(np.arange(0.1, 3.1, 0.1), 1),
    "drawdown_sell_cutoff": np.round(np.arange(0.1, 3.1, 0.1), 1),
    "defense_ticker": ["SCHD", "SGOV", "SPLV", "GLDM", "GDX"],
}


def load_settings() -> dict:
    """
    기본 설정을 로드합니다.
    """
    settings = DEFAULT_SETTINGS.copy()
    return settings


# =============================================================================
# 2. Data Logic
# =============================================================================


def compute_bounds(settings: dict, end_bound: pd.Timestamp | None = None):
    """백테스트/튜닝/추천 모두 동일한 기간 산정 로직을 사용하도록 범위를 계산."""
    end = end_bound or pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(months=settings["months_range"])
    warmup_bdays = 300  # 고정 웜업 기간
    warmup_start = start - pd.offsets.BDay(warmup_bdays)
    return start, warmup_start, end


def _extract_field(data: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """yfinance 다운로드 결과에서 특정 필드(Open/Close 등)를 안전하게 추출."""
    key = field.lower()
    if isinstance(data.columns, pd.MultiIndex):
        candidates = [key, f"adj {key}"]
        level_idx = None
        field_key = None
        for level in range(data.columns.nlevels):
            level_values = data.columns.get_level_values(level)
            for cand in candidates:
                matches = [v for v in level_values if str(v).lower() == cand]
                if matches:
                    level_idx = level
                    field_key = matches[0]
                    break
            if level_idx is not None:
                break
        if level_idx is None:
            raise ValueError(f"{field} 컬럼을 찾지 못했습니다. 사용 가능 컬럼: {list(data.columns)}")
        out = data.xs(field_key, axis=1, level=level_idx)
    else:
        candidates = [c for c in [field, field.capitalize()] if c in data.columns]
        field_col = candidates[0] if candidates else data.columns[0]
        out = data[[field_col]].rename(columns={field_col: tickers[0]})

    out = out.dropna(how="all")
    return out


def download_prices(settings: dict, start) -> pd.DataFrame:
    # 튜닝 시에는 모든 후보군을 다 받아야 함
    tickers = list(
        {
            settings["trade_ticker"],
            settings["signal_ticker"],
            settings["defense_ticker"],
        }
    )
    # 튜닝 후보군도 포함
    tickers.extend(TUNING_CONFIG["defense_ticker"])
    tickers = list(set(tickers))

    # CASH는 다운로드 대상 아님
    tickers = [t for t in tickers if t != "CASH"]

    if not tickers:
        return pd.DataFrame()

    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise ValueError(f"가격 데이터를 받아오지 못했습니다: {tickers}")
    prices = _extract_field(data, "Close", tickers)

    # 필수 데이터 체크
    needed = [settings["trade_ticker"], settings["signal_ticker"]]
    prices = prices.dropna(subset=needed)

    if prices.empty:
        raise ValueError(f"가격 데이터가 비어 있습니다: {tickers}")
    return prices


# =============================================================================
# 3. Signals Logic
# =============================================================================


def compute_signals(prices: pd.Series, settings: dict) -> pd.DataFrame:
    """가격 시계열로 추세/변동성/드로다운 신호를 계산합니다."""
    df = pd.DataFrame(index=prices.index)
    df["close"] = prices

    # 변동성 필터 제거: 변동성은 0으로 두고 사용하지 않음
    df["vol"] = 0.0

    peak = prices.cummax()
    df["drawdown"] = prices / peak - 1.0
    return df.dropna()


def pick_target(row, prev_target: str, settings: dict) -> str:
    """
    신호 행과 이전 타깃을 받아 매수 대상 티커를 결정합니다 (이중 임계값 적용).
    """
    buy_cut = -settings["drawdown_buy_cutoff"] / 100
    sell_cut = -settings["drawdown_sell_cutoff"] / 100

    offense = settings["trade_ticker"]
    defense = settings["defense_ticker"]

    current_dd = row["drawdown"]

    if prev_target == offense:
        # 공격 자산 보유 중: 매도 기준보다 더 떨어지면 방어 전환
        if current_dd < sell_cut:
            return defense
        return offense
    else:
        # 방어 자산 보유 중: 매수 기준보다 더 오르면 공격 전환
        if current_dd > buy_cut:
            return offense
        return defense


# =============================================================================
# 4. Backtest Engine (Internal)
# =============================================================================


class Backtester:
    def __init__(self, settings: dict, prices: pd.DataFrame, signal_df: pd.DataFrame):
        self.settings = settings
        self.prices = prices
        self.signal_df = signal_df
        self.start_date = signal_df.index.min()
        self.end_date = signal_df.index.max()

    def run(self) -> dict:
        """단일 백테스트 실행"""
        # 초기 자본
        initial_capital = 10_000_000

        # 상태 추적
        prev_target = self.settings["trade_ticker"]

        # 일별 수익률 계산을 위한 데이터 준비
        # 전체 기간에 대해 미리 계산
        assets = [self.settings["trade_ticker"], self.settings["defense_ticker"]]
        daily_rets = self.prices[assets].pct_change().fillna(0)

        # 시뮬레이션
        equity_curve = [initial_capital]

        # 벡터화된 연산을 위해 타깃 시그널 생성
        targets = []
        for idx, row in self.signal_df.iterrows():
            tgt = pick_target(row, prev_target, self.settings)
            targets.append(tgt)
            prev_target = tgt

        # 수익률 적용
        # target[i]는 i일의 종가 기준으로 결정된 포지션 -> i+1일의 수익률에 적용
        # 여기서는 단순화를 위해 당일 종가 매매 가정 (슬리피지 적용)

        # 실제로는 루프를 돌며 자산 가치 변동을 추적해야 정확함 (특히 전환 시점)
        current_equity = initial_capital
        prev_target = self.settings["trade_ticker"]  # 초기 상태

        for date, target in zip(self.signal_df.index, targets):
            # 전일 대비 수익률 적용 (보유 중인 자산)
            # 첫날은 변동 없음
            if date == self.signal_df.index[0]:
                continue

            # 어제 결정한 타깃을 오늘 보유하고 있음
            holding_ticker = prev_target

            if holding_ticker == "CASH":
                ret = 0.0
            else:
                ret = daily_rets.at[date, holding_ticker]

            # 자산 변동
            current_equity *= 1 + ret

            # 교체 비용 (슬리피지)
            if target != prev_target:
                slippage = self.settings["slippage"] / 100
                current_equity *= 1 - slippage

            prev_target = target
            equity_curve.append(current_equity)

        final_equity = current_equity

        # CAGR 계산
        days = (self.end_date - self.start_date).days
        years = days / 365.25
        cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0

        # MDD 계산
        equity_series = pd.Series(equity_curve)
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()

        # Sharpe Ratio (간이)
        returns = pd.Series(equity_curve).pct_change().dropna()
        if returns.std() == 0:
            sharpe = 0
        else:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

        return {
            "cagr": cagr * 100,
            "mdd": max_drawdown * 100,
            "sharpe": sharpe,
            "final_equity": final_equity,
            "settings": self.settings,
        }


# =============================================================================
# 5. Tuning Logic
# =============================================================================


def _worker(args):
    """병렬 처리를 위한 워커 함수"""
    case_settings, prices, signal_df = args
    bt = Backtester(case_settings, prices, signal_df)
    return bt.run()


def run_tuning(base_settings: dict) -> dict:
    """전수 조사 튜닝 실행"""
    print(f"\n[튜닝 시작] 최적 파라미터 탐색 중... (기간: {base_settings['months_range']}개월)")

    # 데이터 준비
    start_bound, warmup_start, end_bound = compute_bounds(base_settings)
    prices_full = download_prices(base_settings, warmup_start)

    # Signal Ticker 데이터 (QQQ)
    signal_prices = prices_full[base_settings["signal_ticker"]]
    signal_df_full = compute_signals(signal_prices, base_settings)

    # 유효 기간 필터링
    valid_index = prices_full.index[prices_full.index >= start_bound]
    prices = prices_full.loc[valid_index]
    signal_df = signal_df_full.loc[valid_index]

    if signal_df.empty:
        raise ValueError("튜닝을 위한 데이터가 부족합니다.")

    # 조합 생성
    keys = list(TUNING_CONFIG.keys())
    values = list(TUNING_CONFIG.values())
    combinations = list(itertools.product(*values))

    total_cases = len(combinations)
    print(f"[튜닝 설정] 총 조합: {total_cases}개")

    tasks = []
    for combo in combinations:
        # 조합을 설정 딕셔너리로 변환
        case_settings = base_settings.copy()
        for k, v in zip(keys, combo):
            case_settings[k] = v

        # 유효성 검사 (buy < sell)
        if case_settings["drawdown_buy_cutoff"] >= case_settings["drawdown_sell_cutoff"]:
            continue

        tasks.append((case_settings, prices, signal_df))

    valid_cases = len(tasks)
    print(f"[튜닝 진행] 유효 조합: {valid_cases}개 (Buy < Sell 조건 적용)")

    results = []
    completed = 0

    # 병렬 처리
    with ProcessPoolExecutor() as executor:
        # 청크 단위로 제출하지 않고 map 사용 시 진행률 표시가 어려우므로 submit 사용
        futures = [executor.submit(_worker, task) for task in tasks]

        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
            except Exception:
                pass

            completed += 1
            if completed % 100 == 0 or completed == valid_cases:
                progress = (completed / valid_cases) * 100
                sys.stdout.write(f"\r[튜닝 진행] {progress:.1f}% ({completed}/{valid_cases})")
                sys.stdout.flush()

    print("\n[튜닝 완료] 결과 정렬 중...")

    # 정렬: CAGR 내림차순
    results.sort(key=lambda x: x["cagr"], reverse=True)

    best_result = results[0]
    best_settings = best_result["settings"]

    print("\n=== 🏆 최적 파라미터 (CAGR 기준) ===")
    print(f"Defense Ticker : {best_settings['defense_ticker']}")
    print(f"Buy Cutoff     : {best_settings['drawdown_buy_cutoff']}%")
    print(f"Sell Cutoff    : {best_settings['drawdown_sell_cutoff']}%")
    print(f"CAGR           : {best_result['cagr']:.2f}%")
    print(f"MDD            : {best_result['mdd']:.2f}%")
    print("====================================\n")

    return best_result


# =============================================================================
# 6. Report Logic
# =============================================================================


def render_table_eaw(headers: list[str], rows: list[list[str]], aligns: list[str]) -> list[str]:
    """
    동아시아 문자 너비를 고려하여 리스트 데이터를 ASCII 테이블 문자열로 렌더링합니다.
    """

    _ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    def _clean(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        s = _ansi_re.sub("", s)
        s = normalize("NFKC", s)
        return s

    def _disp_width_eaw(s: str) -> int:
        """동아시아 문자를 포함한 문자열의 실제 터미널 출력 너비를 계산합니다."""
        s = _clean(s)
        w = 0
        for ch in s:
            # 박스 드로잉 문자는 터미널에서 넓게 렌더링되는 경우가 많습니다.
            if "\u2500" <= ch <= "\u257f":
                w += 2
                continue
            eaw = east_asian_width(ch)
            # 'Ambiguous'(A) 문자를 Wide로 처리하여 대부분의 터미널에서 정렬이 깨지지 않도록 합니다.
            if eaw in ("W", "F", "A"):
                w += 2
            else:
                w += 1
        return w

    def _pad(s: str, width: int, align: str) -> str:
        """주어진 너비와 정렬에 맞게 문자열에 패딩을 추가합니다."""
        s_str = str(s)
        s_clean = _clean(s_str)
        dw = _disp_width_eaw(s_clean)
        if dw >= width:
            return s_str
        pad = width - dw
        if align == "right":
            return " " * pad + s_str
        elif align == "center":
            left = pad // 2
            right = pad - left
            return " " * left + s_str + " " * right
        else:  # 왼쪽 정렬
            return s_str + " " * pad

    widths = [max(_disp_width_eaw(v) for v in [headers[j]] + [r[j] for r in rows]) for j in range(len(headers))]

    def _hline():
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    out = [_hline()]
    header_cells = [
        _pad(headers[j], widths[j], "center" if aligns[j] == "center" else "left") for j in range(len(headers))
    ]
    out.append("| " + " | ".join(header_cells) + " |")
    out.append(_hline())
    for r in rows:
        cells = [_pad(r[j], widths[j], aligns[j]) for j in range(len(headers))]
        out.append("| " + " | ".join(cells) + " |")
    out.append(_hline())
    return out


# =============================================================================
# 7. Recommendation Runner Logic
# =============================================================================


def run_recommend(settings: dict) -> dict[str, object]:
    start_bound, warmup_start, end_bound = compute_bounds(settings)

    prices_full = download_prices(settings, warmup_start)
    signal_df_full = compute_signals(prices_full[settings["signal_ticker"]], settings)
    valid_index = prices_full.index[prices_full.index >= start_bound]
    prices = prices_full.loc[valid_index]
    signal_df = signal_df_full.loc[valid_index]
    if signal_df.empty:
        raise ValueError("시그널 계산에 필요한 데이터가 없습니다.")
    last_date = signal_df.index.max()

    # 상태 기반 로직을 위해 과거 데이터부터 순차적으로 상태 추적
    # (백테스트와 동일하게 초기 상태는 offense로 가정)
    prev_target = settings["trade_ticker"]

    # 마지막 날짜 전까지 상태 진행
    # (실제로는 전체를 다 돌리고 마지막 날의 target을 구하면 됨)
    # 효율성을 위해 전체 루프를 돌림
    targets = []
    for idx, row in signal_df.iterrows():
        tgt = pick_target(row, prev_target, settings)
        targets.append(tgt)
        prev_target = tgt

    signal_df["target"] = targets

    last_row = signal_df.loc[last_date]
    target = last_row["target"]

    # 상태 계산: 타깃을 BUY, 나머지 WAIT
    offense = settings["trade_ticker"]
    defense = settings["defense_ticker"]
    assets = [offense]
    if defense != "CASH":
        assets.append(defense)

    # 테이블에 CASH 행을 항상 포함해 현금 보유 상태를 표시
    table_assets = ["CASH"] + assets if defense == "CASH" else assets

    statuses = {}
    if defense == "CASH":
        statuses["CASH"] = "HOLD" if target == "CASH" else "WAIT"
    for sym in assets:
        statuses[sym] = "BUY" if sym == target else "WAIT"

    # 일간 수익률은 전일 대비 종가 기준
    daily_rets = prices[assets].pct_change()
    last_ret = daily_rets.loc[last_date] if last_date in daily_rets.index else pd.Series(dtype=float)

    def _gap_message(row, price_today):
        # 추천 시점의 '문구'는 보통 "왜 안 샀냐"를 설명하는 용도이므로
        # 매수 기준(buy_cutoff)을 보여주는 것이 적절함
        buy_cut_raw = settings["drawdown_buy_cutoff"]
        buy_cut = buy_cut_raw / 100
        threshold = -buy_cut
        current_dd = row["drawdown"]

        # 드로다운이 임계값보다 낮아서(더 많이 떨어져서) 못 사는 경우
        if current_dd <= threshold:
            needed = threshold - current_dd
            return f"DD {current_dd * 100:.2f}% (매수컷 {threshold * 100:.2f}%, 필요 {needed * 100:+.2f}%)"
        return ""

    # 테이블 대신 세로형 카드 포맷 생성
    table_lines = []
    for idx, sym in enumerate(table_assets, start=1):
        if sym == "CASH":
            price = 1.0
            ret = 0.0
        else:
            price = prices.at[last_date, sym]
            ret = last_ret.get(sym, 0.0) if not last_ret.empty else 0.0

        note = ""
        if sym == target:
            note = "타깃"
        elif sym == offense:
            note = _gap_message(last_row, price if sym != "CASH" else 1.0)
        elif sym == defense and defense != "CASH":
            note = "방어"

        st = statuses.get(sym, "WAIT")
        st_emoji = "✅️" if st in ["BUY", "HOLD"] else "⏳️"

        # 세로형 출력 생성
        table_lines.append(f"📌 {sym}")
        table_lines.append(f"  상태: {st} {st_emoji}")
        table_lines.append(f"  일간: {ret * 100:+.2f}%")
        table_lines.append(f"  현재가: ${price:,.2f}")
        if note:
            table_lines.append(f"  비고: {note}")
        table_lines.append("")  # 공백 라인 추가

    return {
        "as_of": last_date.date().isoformat(),
        "target": target,
        "table_lines": table_lines,
        "raw_data": {
            "statuses": statuses,
            "prices": {sym: prices.at[last_date, sym] for sym in assets if sym in prices.columns},
            "drawdown": last_row["drawdown"],
            "drawdown_buy_cutoff": settings["drawdown_buy_cutoff"],
            "drawdown_sell_cutoff": settings["drawdown_sell_cutoff"],
        },
    }


# =============================================================================
# 8. Public Interface
# =============================================================================


def get_result() -> dict:
    """
    외부에서 호출 가능한 함수.
    자동으로 튜닝을 수행하고 최적의 파라미터로 추천 결과와 튜닝 결과를 반환합니다.

    Returns:
        Dict: 추천 결과 리포트 (target, as_of, table_lines, tuning_result 등 포함)
    """
    # 1. 설정 로드 (기본값)
    settings = load_settings()

    # 2. 자동 튜닝 수행
    tuning_result = run_tuning(settings)
    best_settings = tuning_result["settings"]

    # 3. 최적 설정 적용
    settings.update(best_settings)

    # 4. 추천 실행
    report = run_recommend(settings)

    # 5. 튜닝 결과 포함
    report["tuning_result"] = {
        "cagr": tuning_result["cagr"],
        "mdd": tuning_result["mdd"],
        "sharpe": tuning_result["sharpe"],
        "defense_ticker": best_settings["defense_ticker"],
        "drawdown_buy_cutoff": best_settings["drawdown_buy_cutoff"],
        "drawdown_sell_cutoff": best_settings["drawdown_sell_cutoff"],
    }

    return report


# =============================================================================
# 9. Main Entry Point
# =============================================================================


def main():
    """스크립트 직접 실행 시 진입점"""
    # Windows/macOS 멀티프로세싱 지원을 위해 freeze_support 호출
    multiprocessing.freeze_support()

    try:
        # 1. 설정 로드 (기본값)
        settings = load_settings()

        # 2. 자동 튜닝 수행
        tuning_result = run_tuning(settings)
        best_settings = tuning_result["settings"]

        # 3. 최적 설정 적용
        settings.update(best_settings)

        # 4. 추천 실행
        report = run_recommend(settings)

        print("\n=== 추천 목록 ===")
        for line in report["table_lines"]:
            print(line)

        print(f"\n[INFO] 기준일: {report['as_of']}")
        print(f"[INFO] 최종 타깃: {report['target']}")
        print(
            f"[INFO] 적용 파라미터: {settings['defense_ticker']} / "
            f"Buy {settings['drawdown_buy_cutoff']}% / "
            f"Sell {settings['drawdown_sell_cutoff']}%"
        )

    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
