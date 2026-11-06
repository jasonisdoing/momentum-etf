"""모멘텀 전략 선정 결과를 콘솔 표로 출력한다.

전략 규칙·계산은 `utils/momentum_service.py` 가 단일 소스이며,
이 스크립트는 화면(/strategy-momentum)과 같은 월 확정 포트폴리오를 콘솔에서 보는
얇은 래퍼다. 설정은 화면에서 저장한 값(pool_settings 의 풀 문서)을
쓴다. 저장된 설정이 없으면 기본값으로 넘어가지 않고 에러로 중단한다.

사용법
------
    python scripts/find_momentum_stocks.py            # 저장 설정으로 선정
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.env import load_env_if_present
from utils.momentum_backtest import current_positions
from utils.momentum_service import load_settings
from utils.report import render_table_eaw


def main() -> int:
    load_env_if_present()

    settings = load_settings()
    print(f"설정: {settings}")

    result = current_positions(settings)

    headers = ["상태", "티커", "종목명", "업종", "단기 여유(%)", "장기 여유(%)"]
    aligns = ["c", "l", "l", "l", "r", "r"]

    def _pct(value: float | None) -> str:
        return f"{value:+.1f}" if value is not None else "-"

    def _row(status: str, row: dict) -> list[str]:
        return [
            status,
            row["ticker"],
            str(row.get("name") or "")[:16],
            str(row.get("industry") or "")[:10],
            _pct(row.get("short_gap_pct")),
            _pct(row.get("long_gap_pct")),
        ]

    rows = [
        _row("매도 예정" if row.get("status") == "sell" else ("진입" if row.get("is_new") else f"{row['days']}일"), row)
        for row in result["holdings"]
    ]
    rows += [_row("매수 예정", row) for row in result["planned_entries"]]
    rows += [_row("후보", row) for row in result["candidates"]]

    print()
    print(
        f"모멘텀 전략 {result['as_of']} 기준 · 다음 체결 {result['next_session']} · "
        f"보유 {len(result['holdings'])}/{result['top_n']} · 유니버스 {result['universe_count']}"
    )
    print("장기 이평선 이격(%)이 큰 순으로 담는다 · 단기·장기 여유가 0 이하가 되면 다음 거래일 시가에 판다")
    for line in render_table_eaw(headers, rows, aligns):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
