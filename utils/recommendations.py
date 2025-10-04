from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pandas.io.formats.style import Styler

from strategies.maps.constants import DECISION_CONFIG
from utils.logger import get_app_logger


_BASE_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _BASE_DIR / "data" / "results"
logger = get_app_logger()

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


def load_recommendations(country: str) -> list[dict[str, Any]]:
    """지정한 국가의 추천 종목 JSON을 로드합니다."""

    # 캐시 무효화를 위해 파일의 마지막 수정 시간을 확인
    normalized_country = country.strip().lower()
    path = _DATA_DIR / f"recommendation_{normalized_country}.json"

    # 파일이 존재하지 않으면 빈 리스트 반환
    if not path.exists():
        raise FileNotFoundError(f"추천 종목 파일을 찾을 수 없습니다: {path}")

    # 파일의 마지막 수정 시간을 기반으로 캐시 키 생성
    file_mtime = path.stat().st_mtime
    cache_key = f"{normalized_country}_{file_mtime}"

    # 캐시된 결과가 있으면 반환
    if hasattr(load_recommendations, "_cache") and cache_key in load_recommendations._cache:
        return load_recommendations._cache[cache_key]

    # 파일에서 데이터 로드
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"추천 종목 JSON 파싱 실패: {path.name}") from exc

    if not isinstance(raw, list):
        raise ValueError(f"추천 종목 JSON은 리스트 형태여야 합니다: {path.name}")

    normalized: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, dict):
            normalized.append(entry.copy())

    normalized.sort(key=lambda row: row.get("rank", 0))

    # 결과를 캐시에 저장 (함수 객체에 캐시 딕셔너리 추가)
    if not hasattr(load_recommendations, "_cache"):
        load_recommendations._cache = {}
    load_recommendations._cache[cache_key] = normalized

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

    country_code = (country or "").strip().lower()

    if country_code == "kor":
        return f"{int(round(amount)):,}원"
    if country_code == "aus":
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


_STATE_BACKGROUND_MAP = {
    key.upper(): cfg.get("background")
    for key, cfg in DECISION_CONFIG.items()
    if isinstance(cfg, dict)
}


def _row_background_styles(row: pd.Series) -> pd.Series:
    state = str(row.get("상태", "")).upper()
    color = _STATE_BACKGROUND_MAP.get(state)
    if not color:
        return pd.Series(["" for _ in row], index=row.index)
    return pd.Series([f"background-color:{color}" for _ in row], index=row.index)


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
    styled = styled.apply(_row_background_styles, axis=1)
    return styled


def get_recommendations_dataframe(country: str, *, source_key: str | None = None) -> pd.DataFrame:
    """로딩과 포맷팅을 한 번에 수행하는 헬퍼.

    Args:
        country: 표시/포맷팅에 사용할 시장 코드
        source_key: 결과 JSON 파일을 선택할 때 사용할 키 (기본값: country)

    Returns:
        포맷팅된 추천 종목 데이터프레임
    """
    # load_recommendations는 파일 수정 시각을 캐시 키에 반영하므로, 별도 초기화 없이도 최신 데이터를 사용한다.
    try:
        data_key = (source_key or country).strip().lower()
        rows = load_recommendations(data_key)
        return recommendations_to_dataframe(country, rows)
    except Exception as e:
        logger.error("추천 데이터를 불러오지 못했습니다 (%s): %s", country, e)
        # 오류 발생 시 빈 데이터프레임 반환
        return pd.DataFrame(columns=_DISPLAY_COLUMNS)


def get_recommendations_styler(country: str) -> tuple[pd.DataFrame, Styler]:
    df = get_recommendations_dataframe(country)
    return df, style_recommendations_dataframe(country, df)
