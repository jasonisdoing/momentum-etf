"""Steady Momentum 선정 결과를 콘솔 표로 출력한다.

전략 규칙·계산은 `utils/steady_momentum_service.py` 가 단일 소스이며,
이 스크립트는 화면(/strategy-sm)과 같은 월 확정 포트폴리오를 콘솔에서 보는
얇은 래퍼다. 설정은 화면에서 저장한 값(system_config.steady_momentum_settings)을
쓰고, 저장 전이면 기본값으로 동작한다.

사용법
------
    python scripts/find_steady_momentum_stocks.py            # 저장 설정으로 선정
    python scripts/find_steady_momentum_stocks.py --top 20   # 종목 수만 재정의
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.env import load_env_if_present
from utils.report import render_table_eaw
from utils.steady_momentum_service import compute_picks, load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Steady Momentum 선정 출력")
    parser.add_argument("--top", type=int, default=None, help="종목 수 (기본: 저장 설정)")
    args = parser.parse_args()

    load_env_if_present()

    settings, is_saved = load_settings()
    if args.top is not None:
        settings = {**settings, "top_n": args.top}
    print(f"설정({'저장됨' if is_saved else '기본값'}): {settings}")

    result = compute_picks(settings)
    lookback = settings["lookback_months"]

    headers = ["순위", "연속", "티커", "종목명", f"{lookback}개월(%)", f"상대{lookback}개월(%)", "월승률", "R²", "점수"]
    aligns = ["r", "c", "l", "l", "r", "r", "c", "r", "r"]
    rows = []
    for row in result["rows"]:
        streak = row["streak_months"]
        streak_label = "-" if streak is None else ("신규" if streak <= 1 else f"{streak}개월")
        rows.append(
            [
                str(row["rank"]) + ("*" if row["is_reserve"] else ""),
                streak_label,
                row["ticker"],
                row["name"][:16],
                f"{row['return_lookback_pct']:+.1f}",
                f"{row['rel_return_pct']:+.1f}",
                row["win_label"],
                f"{row['r_squared']:.3f}",
                f"{row['momentum_score']:.1f}",
            ]
        )

    print()
    print(
        f"Steady Momentum {result['portfolio_month']} 포트폴리오 · 교체 {result['rebalance_date']} "
        f"(판정 {result['signal_date']}) · 유니버스 {result['universe_count']} → 후보 {result['candidate_count']}"
    )
    print("점수 = 연율화 상대기울기 × R² (시장 대비 꾸준한 모멘텀) · 순위* = 차순위 후보")
    for line in render_table_eaw(headers, rows, aligns):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
