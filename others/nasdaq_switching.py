"""
Nasdaq Leverage Switching Strategy Recommendation Script
이 파일은 프로젝트의 추천 로직을 단일 파일로 번들링한 것입니다.
다른 프로젝트에서 import하여 사용하거나 직접 실행할 수 있습니다.
"""

import json
import re
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from unicodedata import east_asian_width, normalize

import numpy as np
import pandas as pd
import yfinance as yf


# =============================================================================
# 1. Settings Logic
# =============================================================================

DEFAULT_SETTINGS = {
    "months_range": 12,
    "signal_ticker": "QQQ",
    "trade_ticker": "TQQQ",
    "slippage": 0.05,
    "backtested_date": "2025-12-02",
    "defense_ticker": "GDX",
    "drawdown_buy_cutoff": 0.3,
    "drawdown_sell_cutoff": 0.4,
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


def load_settings(path: Path | str = "settings.json") -> Dict:
    """
    설정을 로드합니다.
    1. path에 지정된 파일이 존재하면 해당 파일을 로드하여 DEFAULT_SETTINGS를 덮어씁니다.
    2. 파일이 없으면 DEFAULT_SETTINGS를 그대로 반환합니다.
    """
    settings = DEFAULT_SETTINGS.copy()

    p = Path(path)
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                file_settings = json.load(f)
            # 프로젝트 루트 경로 추가
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            settings.update(file_settings)
            # print(f"[INFO] 설정 파일 로드됨: {p.absolute()}")
        except Exception as e:
            print(f"[WARNING] 설정 파일 로드 실패 ({e}). 기본 설정을 사용합니다.")
    else:
        # 파일이 없어도 조용히 기본값 사용 (단일 파일 모드 지원)
        pass

    return settings


# =============================================================================
# 2. Data Logic
# =============================================================================


def compute_bounds(settings: Dict, end_bound: pd.Timestamp | None = None):
    """백테스트/튜닝/추천 모두 동일한 기간 산정 로직을 사용하도록 범위를 계산."""
    end = end_bound or pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(months=settings["months_range"])
    warmup_bdays = 300  # 고정 웜업 기간
    warmup_start = start - pd.offsets.BDay(warmup_bdays)
    return start, warmup_start, end


def _extract_field(data: pd.DataFrame, field: str, tickers: List[str]) -> pd.DataFrame:
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


def download_prices(settings: Dict, start) -> pd.DataFrame:
    tickers = list({settings["trade_ticker"], settings["signal_ticker"], settings["defense_ticker"]})
    # CASH는 다운로드 대상 아님
    tickers = [t for t in tickers if t != "CASH"]

    if not tickers:
        return pd.DataFrame()

    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise ValueError(f"가격 데이터를 받아오지 못했습니다: {tickers}")
    prices = _extract_field(data, "Close", tickers)

    needed = [t for t in [settings["trade_ticker"], settings["signal_ticker"], settings["defense_ticker"]] if t != "CASH"]
    prices = prices.dropna(subset=needed)
    if prices.empty:
        raise ValueError(f"가격 데이터가 비어 있습니다: {tickers}")
    return prices


# =============================================================================
# 3. Signals Logic
# =============================================================================


def compute_signals(prices: pd.Series, settings: Dict) -> pd.DataFrame:
    """가격 시계열로 추세/변동성/드로다운 신호를 계산합니다."""
    df = pd.DataFrame(index=prices.index)
    df["close"] = prices

    # 변동성 필터 제거: 변동성은 0으로 두고 사용하지 않음
    df["vol"] = 0.0

    peak = prices.cummax()
    df["drawdown"] = prices / peak - 1.0
    return df.dropna()


def pick_target(row, prev_target: str, settings: Dict) -> str:
    """
    신호 행과 이전 타깃을 받아 매수 대상 티커를 결정합니다 (이중 임계값 적용).

    - drawdown_buy_cutoff (예: 1.0 -> -1.0%): 이보다 높으면(회복되면) 공격 자산 매수
    - drawdown_sell_cutoff (예: 2.0 -> -2.0%): 이보다 낮으면(악화되면) 공격 자산 매도
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
# 4. Report Logic
# =============================================================================


def render_table_eaw(headers: List[str], rows: List[List[str]], aligns: List[str]) -> List[str]:
    """
    동아시아 문자 너비를 고려하여 리스트 데이터를 ASCII 테이블 문자열로 렌더링합니다.
    """

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def _clean(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        s = _ANSI_RE.sub("", s)
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
    header_cells = [_pad(headers[j], widths[j], "center" if aligns[j] == "center" else "left") for j in range(len(headers))]
    out.append("| " + " | ".join(header_cells) + " |")
    out.append(_hline())
    for r in rows:
        cells = [_pad(r[j], widths[j], aligns[j]) for j in range(len(headers))]
        out.append("| " + " | ".join(cells) + " |")
    out.append(_hline())
    return out


# =============================================================================
# 5. Recommendation Runner Logic
# =============================================================================


def run_recommend(settings: Dict) -> Dict[str, object]:
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
            return f"DD {current_dd*100:.2f}% (매수컷 {threshold*100:.2f}%, 필요 {needed*100:+.2f}%)"
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
        table_lines.append(f"  일간: {ret*100:+.2f}%")
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
# 6. Public Interface
# =============================================================================


def get_recommendation(settings_path: str = "settings.json") -> Dict:
    """
    외부에서 호출 가능한 추천 함수.

    Returns:
        Dict: 추천 결과 리포트 (target, as_of, table_lines 등 포함)
    """
    settings = load_settings(settings_path)
    return run_recommend(settings)


def main():
    """스크립트 직접 실행 시 진입점"""
    try:
        report = get_recommendation()

        print("\n=== 추천 목록 ===")
        for line in report["table_lines"]:
            print(line)

        print(f"\n[INFO] 기준일: {report['as_of']}")
        print(f"[INFO] 최종 타깃: {report['target']}")

    except Exception as e:
        print(f"[ERROR] 추천 실행 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
