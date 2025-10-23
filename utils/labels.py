"""표시용 라벨을 관리하는 헬퍼."""

from __future__ import annotations

from config import ETF_PRICE_SOURCE

_NAVER_PRICE_SOURCE = (ETF_PRICE_SOURCE or "").strip().lower()


def get_price_column_label(country_code: str) -> str:
    """국가 코드에 따라 현재가 컬럼 라벨을 반환한다."""

    normalized = (country_code or "").strip().lower()
    if normalized in {"kr", "kor"} and _NAVER_PRICE_SOURCE == "nav":
        return "현재가(Nav)"
    return "현재가"
