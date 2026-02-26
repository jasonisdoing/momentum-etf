from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
from pandas.io.formats.style import Styler

from strategies.maps.constants import BACKTEST_STATUS_LIST, RECOMMEND_STATUS_LIST
from utils.formatters import format_trading_days
from utils.logger import get_app_logger
from utils.recommendation_storage import fetch_latest_recommendations

logger = get_app_logger()

_BASE_DISPLAY_COLUMNS = [
    "#",
    "버킷",
    "티커",
    "종목명",
    "상태",
    "일간(%)",
    "평가(%)",
    "보유일",
    "현재가",
    "1주(%)",
    "2주(%)",
    "1달(%)",
    "3달(%)",
    "6달(%)",
    "12달(%)",
    "고점대비",
    "추세(3달)",
    "점수",
    "지속",
    "문구",
]


def load_recommendations(account_key: str) -> list[dict[str, Any]]:
    """지정한 계정/국가의 추천 종목 스냅샷을 로드합니다."""

    normalized_key = (account_key or "").strip().lower()
    if not normalized_key:
        raise ValueError("account_key is required to load recommendations")

    doc = fetch_latest_recommendations(normalized_key)
    if doc is None:
        raise FileNotFoundError(f"추천 스냅샷을 찾을 수 없습니다: {normalized_key}")

    raw = doc.get("recommendations") or []
    if not isinstance(raw, list):
        raise ValueError("추천 스냅샷 데이터가 리스트 형태가 아닙니다.")

    normalized: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, dict):
            normalized.append(entry.copy())

    normalized.sort(key=lambda row: row.get("rank_order") or row.get("rank") or 0)
    return normalized


def _resolve_phrase(row: dict[str, Any]) -> str:
    phrase = row.get("phrase")
    if phrase is None:
        return ""
    return str(phrase)


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


def _format_days(value: Any) -> int | None:
    """지속일을 숫자로 반환합니다 (NumberColumn 사용을 위해)."""
    if value is None:
        return None
    try:
        days = int(value)
        return days
    except (TypeError, ValueError):
        return None


def _trend_series(row: dict[str, Any]) -> list[float]:
    raw_series = row.get("trend_prices")
    if raw_series is None:
        raw_series = row.get("trend_returns")
    if isinstance(raw_series, (list, tuple)):
        values: list[float] = []
        for raw in raw_series:
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        return values
    return []


def recommendations_to_dataframe(country: str, rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """추천 종목 데이터를 표 렌더링에 적합한 DataFrame으로 변환합니다."""

    country_lower = (country or "").strip().lower()
    price_label = "현재가"
    nav_mode = country_lower in {"kr", "kor"}
    show_deviation = country_lower in {"kr", "kor"}

    display_rows: list[dict[str, Any]] = []
    for row in rows:
        rank = row.get("rank")
        ticker = row.get("ticker", "-")
        name = row.get("name", "-")

        state = row.get("state", "-").upper()
        holding_days = _format_days(row.get("holding_days"))
        price_display = row.get("price")
        daily_pct = row.get("daily_pct")
        evaluation_pct = row.get("evaluation_pct", 0.0)
        price_deviation = row.get("price_deviation") if show_deviation else None
        return_1w = row.get("return_1w", 0.0)
        return_2w = row.get("return_2w", 0.0)
        return_1m = row.get("return_1m", 0.0)
        return_3m = row.get("return_3m", 0.0)
        return_6m = row.get("return_6m", 0.0)
        return_12m = row.get("return_12m", 0.0)
        drawdown_from_high = row.get("drawdown_from_high", 0.0)
        score = row.get("score")
        streak = _format_days(row.get("streak"))
        phrase = _resolve_phrase(row)
        bucket_names = {
            1: "1. 모멘텀",
            2: "2. 혁신기술",
            3: "3. 시장지수",
            4: "4. 배당방어",
            5: "5. 대체헷지",
        }
        bucket_id = row.get("bucket", 1)
        bucket_name = bucket_names.get(bucket_id, f"{bucket_id}. 기타")

        display_rows.append(
            {
                "#": rank if rank is not None else "-",
                "버킷": bucket_name,
                "티커": ticker,
                "종목명": name,
                "상태": state,
                "보유일": format_trading_days(holding_days),
                "일간(%)": daily_pct,
                "평가(%)": evaluation_pct,
                price_label: price_display,
                **({"Nav": row.get("nav_price")} if nav_mode else {}),
                **({"괴리율": price_deviation} if show_deviation else {}),
                "1주(%)": return_1w,
                "2주(%)": return_2w,
                "1달(%)": return_1m,
                "3달(%)": return_3m,
                "6달(%)": return_6m,
                "12달(%)": return_12m,
                "고점대비": drawdown_from_high,
                "추세(3달)": _trend_series(row),
                "점수": score,
                "지속": streak,
                "문구": phrase or row.get("phrase", ""),
                "bucket": row.get("bucket", 1),
            }
        )

    columns = list(_BASE_DISPLAY_COLUMNS)
    if "현재가" in columns:
        idx = columns.index("현재가")
        columns[idx] = price_label

    # [User Request] 현재가 - 괴리율 - Nav 순서로 변경
    if show_deviation and "괴리율" not in columns:
        insert_pos = columns.index(price_label) + 1
        columns.insert(insert_pos, "괴리율")

    if nav_mode and "Nav" not in columns:
        # 괴리율이 있으면 그 다음(+2), 없으면 현재가 다음(+1)
        insert_pos = columns.index(price_label) + (2 if show_deviation else 1)
        columns.insert(insert_pos, "Nav")

    # [Fix] 버킷 정보가 누락되지 않도록 컬럼 추가
    if "bucket" not in columns:
        columns.append("bucket")

    df = pd.DataFrame(display_rows, columns=columns)
    return df


def _state_style(value: Any) -> str:
    text = str(value).upper()
    if text in ("BUY", "BUY_REPLACE"):
        return "color:#d32f2f;font-weight:600"
    if text == "WAIT":
        return "color:#1565c0;font-weight:600"
    if text in ("SELL", "SELL_REPLACE"):
        return "color:#1565c0;font-weight:600"
    return ""


_STATE_BACKGROUND_MAP = {
    key.upper(): cfg.get("background")
    for key, cfg in {**BACKTEST_STATUS_LIST, **RECOMMEND_STATUS_LIST}.items()
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


def _deviation_style(value: Any) -> str:
    if value is None:
        return ""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""

    if val == 0:
        return ""

    style = "color:#d32f2f" if val > 0 else "color:#1565c0"
    if abs(val) >= 2.0:
        style += ";font-weight:700"
    return style


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


def style_recommendations_dataframe(df: pd.DataFrame) -> Styler:
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

    # 괴리율 스타일 적용 (컬럼이 있는 경우에만)
    if "괴리율" in df.columns:
        styled = styled.applymap(_deviation_style, subset=["괴리율"])

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
        columns = [col for col in _BASE_DISPLAY_COLUMNS if col != "괴리율" or country.lower() in {"kr", "kor"}]
        if "현재가" in columns:
            idx = columns.index("현재가")
            columns[idx] = "현재가"
        if country.lower() in {"kr", "kor"}:
            insert_pos = columns.index("현재가") + 1
            columns.insert(insert_pos, "Nav")
        if country.lower() not in {"kr", "kor"} and "괴리율" in columns:
            columns.remove("괴리율")
        return pd.DataFrame(columns=columns)


def get_recommendations_styler(country: str) -> tuple[pd.DataFrame, Styler]:
    df = get_recommendations_dataframe(country)
    return df, style_recommendations_dataframe(df)
