"""종목풀 업종 맵 — 업종 컬럼이 있는 모든 화면의 단일 소스.

시장마다 분류 체계가 다르고 수집 경로도 다르다.
- 한국·호주: 종목 문서(`stock_meta.industry`). 한국 개별주는 네이버 분류(한국어 원본)를
  메타 배치가 채운다.
- 미국: 지수 구성종목(SP500/NDX100)의 yfinance 분류. 미국 종목 문서에는 이 필드를
  채우지 않으므로 구성종목에서 가져와야 한다.

종목풀이 국가별로 나뉘어 있어 한 풀 안에서는 항상 한 체계다.
순위(`/pools-rank`)·전략 SM(`/strategy-sm`)·신고점(`/strategy-new-high`)이 같은 값을
보도록 여기서만 정의한다.
"""

from __future__ import annotations

import warnings

from utils.logger import get_app_logger

logger = get_app_logger()

# 미국 업종을 가져올 지수 구성종목. 앞선 지수의 값을 우선한다(중복 종목은 먼저 만난 값 유지).
_US_INDEX_SOURCES = ("SP500", "NDX100")


def industry_map(pool: str) -> dict[str, str]:
    """티커 → 업종. 업종 상한(`max_per_industry`) 그룹핑과 화면 표시에 함께 쓴다.

    분류가 없는 종목은 맵에 넣지 않는다 — 업종 상한이 적용되지 않을 뿐,
    임의 값으로 묶지 않는다.

    구성종목이 아직 없으면 표시용 정보 하나 때문에 화면 전체가 막히지 않도록
    빈 맵으로 두되, 없다는 사실 자체는 경고로 남긴다.
    """
    from utils.settings_loader import get_ticker_type_settings

    settings = get_ticker_type_settings(pool) or {}
    country = str(settings.get("country_code") or "").strip().lower()

    if country != "us":
        from utils.stock_list_io import _load_ticker_type_stocks_raw

        result: dict[str, str] = {}
        for item in _load_ticker_type_stocks_raw(pool):
            ticker = str(item.get("ticker") or "").strip()
            industry = str(item.get("industry") or "").strip()
            if ticker and industry:
                result[ticker] = industry
        return result

    return us_industry_map()


def us_industry_map() -> dict[str, str]:
    """미국 티커 → yfinance 업종 (지수 구성종목 기준). 풀과 무관하게 같은 값이다."""
    from utils.index_constituents_loader import load_index_constituents

    result: dict[str, str] = {}
    for index_name in _US_INDEX_SOURCES:
        try:
            constituents = load_index_constituents(index_name)
        except (FileNotFoundError, LookupError) as error:
            warnings.warn(f"{index_name} 구성종목이 없어 업종을 채우지 못했습니다: {error}", stacklevel=2)
            logger.warning("%s 구성종목이 없어 업종을 채우지 못했습니다: %s", index_name, error)
            continue
        for item in constituents:
            ticker = str(item.get("ticker") or "").strip().upper()
            industry = str(item.get("industry") or "").strip()
            if ticker and industry and ticker not in result:
                result[ticker] = industry
    return result
