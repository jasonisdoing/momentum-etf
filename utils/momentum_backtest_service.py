"""모멘텀 백테스트 실행(큐) + 상태/결과 조회 서비스 (UI/API 용).

레버리지 튜닝과 동일하게 공유 배치 큐로 실행하고, 진행도/결과는
`backtest/results/<prefix>-backtest_<날짜>.log` 파일을 폴링해 보여준다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_JOB_NAME = "momentum_backtest"
_SCRIPT = "scripts/momentum_backtest.py"
_RESULTS_DIR = Path(__file__).resolve().parents[1] / "backtest" / "results"


def trigger_momentum_backtest() -> dict[str, Any]:
    """백테스트 작업을 배치 큐에 추가한다. 이미 대기/실행 중이면 무시."""
    from utils.batch_queue import enqueue

    result = enqueue(_JOB_NAME, _SCRIPT, triggered_by="manual")
    return {"enqueued": bool(result.get("enqueued")), "reason": result.get("reason")}


def _parse_backtest_log(text: str) -> dict[str, Any]:
    """백테스트 로그에서 진행률/완료 여부를 파싱한다."""
    done = "종료 시각" in text
    progress_pct: float | None = None
    completed = total = None
    m = re.search(r"진행:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)", text)
    if m:
        completed, total = int(m.group(1)), int(m.group(2))
        progress_pct = float(m.group(3))
    if done:
        progress_pct = 100.0
    return {"done": done, "progress_pct": progress_pct, "completed": completed, "total": total}


def list_backtest_result_files() -> list[str]:
    """메인 결과 로그 파일명을 최신(수정시각)순으로 반환한다.

    엔진은 결과(`<prefix>-backtest_<date>.log`)와 상세(`<prefix>-backtest_details_<date>.log`)를
    함께 쓰는데, 진행률·상위표가 있는 **메인 결과 파일만** 노출한다(상세는 제외).
    """
    if not _RESULTS_DIR.exists():
        return []
    files = [p for p in _RESULTS_DIR.glob("*-backtest_*.log") if "-backtest_details_" not in p.name]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files]


def _read_result_file(file: str | None) -> dict[str, Any]:
    """선택 파일(없으면 최신) 결과 로그 내용·진행률을 반환한다(화이트리스트로 경로 조작 차단)."""
    files = list_backtest_result_files()
    empty = {"log_text": "", "log_file": None, "selected_file": None, "done": False, "progress_pct": None, "completed": None, "total": None}
    if not files:
        return empty

    selected = file if (file and file in files) else files[0]  # 목록에 없는 입력은 무시 → 최신
    path = _RESULTS_DIR / selected
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    return {"log_text": text, "log_file": selected, "selected_file": selected, **_parse_backtest_log(text)}


def momentum_backtest_status(file: str | None = None) -> dict[str, Any]:
    """백테스트 실행 상태 + 선택(없으면 최신) 결과 로그 + 파일 목록을 반환한다."""
    from utils.batch_queue import get_latest_item

    item = get_latest_item(_JOB_NAME)
    queue_status = item.get("status") if item else None  # pending/running/done/failed/None

    log = _read_result_file(file)

    def _iso(value: Any) -> str | None:
        return value.isoformat() if hasattr(value, "isoformat") else value

    return {
        "queue_status": queue_status,
        "running": queue_status in ("pending", "running"),
        "exit_code": item.get("exit_code") if item else None,
        "error": item.get("error") if item else None,
        "triggered_at": _iso(item.get("triggered_at")) if item else None,
        "started_at": _iso(item.get("started_at")) if item else None,
        "ended_at": _iso(item.get("ended_at")) if item else None,
        "files": list_backtest_result_files(),
        **log,
    }
