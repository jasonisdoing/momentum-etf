from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pandas.io.formats.style import Styler


_BASE_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _BASE_DIR / "data" / "results"

_DISPLAY_COLUMNS = [
    "#",
    "티커",
    "종목명",
    "카테고리",
    "상태",
    "보유일",
    "현재가",
    "일간(%)",
    "점수",
    "지속",
    "문구",
]


@lru_cache(maxsize=None)
def load_recommendations(country: str) -> list[dict[str, Any]]:
    """지정한 국가의 추천 종목 JSON을 로드합니다."""

    normalized_country = country.strip().lower()
    path = _DATA_DIR / f"{normalized_country}.json"

    if not path.exists():
        raise FileNotFoundError(f"추천 종목 파일을 찾을 수 없습니다: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - 단순 예외 메시지 강화
        raise ValueError(f"추천 종목 JSON 파싱 실패: {path.name}") from exc

    if not isinstance(raw, list):
        raise ValueError(f"추천 종목 JSON은 리스트 형태여야 합니다: {path.name}")

    normalized: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, dict):
            normalized.append(entry.copy())

    normalized.sort(key=lambda row: row.get("rank", 0))
    return normalized


def _resolve_phrase(row: dict[str, Any]) -> str:
    phrase = row.get("phrase")
    if phrase is None:
        return ""
    return str(phrase)


def _format_currency(value: Any, country: str) -> str:
    if value is None:
        return "-"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)

    if country == "kor":
        return f"{int(round(amount)):,}원"
    if country == "aus":
        return f"A${amount:,.2f}"
    return f"{amount:,.2f}"


def _format_percent(value: Any) -> str:
    if value is None:
        return "-"
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{pct:+.2f}%"


def _format_score(value: Any) -> str:
    if value is None:
        return "-"
    try:
        score = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{score:.1f}"


def _format_days(value: Any) -> str:
    if value in (None, "", "-"):
        return "-"
    try:
        days = int(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{days}일"


def recommendations_to_dataframe(country: str, rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """추천 종목 데이터를 표 렌더링에 적합한 DataFrame으로 변환합니다."""

    display_rows: list[dict[str, Any]] = []
    for row in rows:
        rank = row.get("rank")
        ticker = row.get("ticker", "-")
        name = row.get("name", "-")
        category = row.get("category", "-")
        state = row.get("state", "-").upper()
        holding_days = _format_days(row.get("holding_days"))
        price = _format_currency(row.get("price"), country)
        daily_pct = _format_percent(row.get("daily_pct"))
        score = _format_score(row.get("score"))
        streak = _format_days(row.get("streak"))
        phrase = _resolve_phrase(row)
        display_rows.append(
            {
                "#": rank if rank is not None else "-",
                "티커": ticker,
                "종목명": name,
                "카테고리": category,
                "상태": state,
                "보유일": holding_days,
                "현재가": price,
                "일간(%)": daily_pct,
                "점수": score,
                "지속": streak,
                "문구": phrase or row.get("phrase", ""),
            }
        )

    df = pd.DataFrame(display_rows, columns=_DISPLAY_COLUMNS)
    return df


def _state_style(value: Any) -> str:
    text = str(value).upper()
    if text == "BUY":
        return "color:#d32f2f;font-weight:600"
    if text == "WAIT":
        return "color:#1565c0;font-weight:600"
    return ""


def _pct_style(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("+"):
        return "color:#d32f2f;font-weight:600"
    if text.startswith("-"):
        return "color:#1565c0;font-weight:600"
    return ""


def _score_style(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return ""
    if score >= 15:
        return "font-weight:600;color:#d81b60"
    if score >= 10:
        return "font-weight:600;color:#6a1b9a"
    return ""


def style_recommendations_dataframe(country: str, df: pd.DataFrame) -> Styler:
    styled = df.style
    styled = styled.set_table_styles(
        [
            {
                "selector": "th",
                "props": "text-align:center;font-weight:700;background-color:#0f1116;color:white",
            },
            {
                "selector": "td",
                "props": "text-align:center;font-family:'Noto Sans KR', 'Pretendard', sans-serif;font-size:0.95rem",
            },
        ]
    )
    styled = styled.set_properties(subset=["종목명"], **{"text-align": "left"})
    styled = styled.applymap(_state_style, subset=["상태"])
    styled = styled.applymap(_pct_style, subset=["일간(%)"])
    styled = styled.applymap(_score_style, subset=["점수"])
    return styled


def get_recommendations_dataframe(country: str) -> pd.DataFrame:
    """로딩과 포맷팅을 한 번에 수행하는 헬퍼."""

    rows = load_recommendations(country)
    return recommendations_to_dataframe(country, rows)


def get_recommendations_styler(country: str) -> tuple[pd.DataFrame, Styler]:
    df = get_recommendations_dataframe(country)
    return df, style_recommendations_dataframe(country, df)
