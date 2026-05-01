"""S&P500, NASDAQ100 구성종목 JSON 파일 로더."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent.parent / "data"

SUPPORTED_INDICES = {"SP500", "NDX100"}
_INDEX_FILE_MAP = {
    "SP500": _DATA_DIR / "sp500_tickers.json",
    "NDX100": _DATA_DIR / "ndx100_tickers.json",
}


def load_index_constituents(index: str) -> list[dict[str, Any]]:
    """지정한 인덱스의 구성종목 목록을 반환한다. ticker, name, sector, industry 포함."""
    key = str(index or "").strip().upper()
    if key not in SUPPORTED_INDICES:
        raise ValueError(f"지원하지 않는 인덱스입니다: {index} (지원: {', '.join(sorted(SUPPORTED_INDICES))})")

    path = _INDEX_FILE_MAP[key]
    if not path.exists():
        raise FileNotFoundError(
            f"{key} 구성종목 파일이 없습니다: {path}\n"
            "scripts/fetch_index_constituents.py 를 실행해 파일을 생성하세요."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("tickers") or [])


def load_index_meta(index: str) -> dict[str, Any]:
    """updated_at, source, count 등 메타 정보를 반환한다."""
    key = str(index or "").strip().upper()
    if key not in SUPPORTED_INDICES:
        raise ValueError(f"지원하지 않는 인덱스입니다: {index}")

    path = _INDEX_FILE_MAP[key]
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in payload.items() if k != "tickers"}
