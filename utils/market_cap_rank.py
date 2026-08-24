"""시총 순위 — 국가별 시장 전체에서의 시가총액 순위를 메타 캐시에 적어 둔다.

화면(순위·모멘텀·신고가)이 매번 시총 목록을 받아 순위를 매기면 한국은 네이버 전체
페이지(약 2,600종목) 조회가 붙어 느려진다. 그래서 **배치 B 가 하루 한 번** 국가별 순위표를
만들어 개별주 풀 종목의 ``stock_cache_meta.meta_cache.market_cap_rank`` 에 써 두고, 화면은
이미 읽는 메타 캐시에서 꺼내기만 한다(런타임 비용 0). 순위 기준 시각은 배치 시각이다.

순위의 분모(국가별 "시장"):
  - kor: 네이버 시총 목록 KOSPI + KOSDAQ 전체 (ETF·ETN 제외 — 개별주만)
  - us : ``index_constituents`` 의 SP500 ∪ NDX100 (시총은 배치가 채운 yfinance 값)
  - au : ``index_constituents`` 의 ASX200
목록에 없는 종목은 순위 없음(None) — 임의 보정 없이 화면은 '-'.
"""

from __future__ import annotations

import math
from typing import Any

from utils.logger import get_app_logger

META_FIELD = "market_cap_rank"


def _rank_by_cap(caps: dict[str, float]) -> dict[str, int]:
    ordered = sorted(((t, c) for t, c in caps.items() if c is not None and c > 0), key=lambda item: (-item[1], item[0]))
    return {ticker: idx for idx, (ticker, _) in enumerate(ordered, start=1)}


def _kor_caps() -> dict[str, float]:
    """KOSPI·KOSDAQ 전체를 네이버 시총 목록에서 페이지 단위로 받는다 (배치 전용 — 화면에서 부르지 않는다)."""
    from utils.kor_stock_market_service import (
        _fetch_market_value_page,
        _parse_number,
        is_individual_stock_item,
    )

    caps: dict[str, float] = {}
    page_size = 100
    for market in ("KOSPI", "KOSDAQ"):
        first = _fetch_market_value_page(market, page=1, page_size=page_size)
        total_count = int(first.get("totalCount") or 0)
        total_pages = max(1, math.ceil(total_count / page_size)) if total_count > 0 else 1
        payload = first
        for page in range(1, total_pages + 1):
            for item in payload.get("stocks") or []:
                # ETF·ETN 은 분모에서 뺀다 — 개별주 순위인데 섞이면 그만큼 번호가 밀린다
                # (KOSPI 상위 100 중 10건이 ETF 다).
                if not is_individual_stock_item(item):
                    continue
                ticker = str(item.get("itemCode") or "").strip()
                cap = _parse_number(item.get("marketValue"))
                if ticker and cap is not None:
                    caps[ticker] = float(cap)
            if page < total_pages:
                payload = _fetch_market_value_page(market, page=page + 1, page_size=page_size)
    return caps


def _index_caps(indexes: tuple[str, ...]) -> dict[str, float]:
    from utils.index_constituents_loader import load_index_constituents

    caps: dict[str, float] = {}
    for index in indexes:
        for item in load_index_constituents(index):
            ticker = str(item.get("ticker") or "").strip().upper()
            cap = item.get("market_cap")
            if ticker and cap is not None and float(cap) > 0:
                caps[ticker] = max(float(cap), caps.get(ticker, 0.0))
    return caps


def load_market_cap_rank_map(country_code: str) -> dict[str, int]:
    """국가별 시장 전체의 티커 → 시총 순위."""
    country = str(country_code or "").strip().lower()
    if country == "kor":
        return _rank_by_cap(_kor_caps())
    if country == "us":
        return _rank_by_cap(_index_caps(("SP500", "NDX100")))
    if country == "au":
        return _rank_by_cap(_index_caps(("ASX200",)))
    raise ValueError(f"시총 순위를 지원하지 않는 국가입니다: {country_code}")


def update_market_cap_ranks(ticker_types: list[str] | None = None) -> dict[str, int]:
    """개별주 풀(pool_kind=stock)의 종목에 시총 순위를 써 넣는다. 반환: 풀별 갱신 건수.

    ``meta_cache`` 전체를 덮는 배치 B 의 종목별 갱신 **뒤에** 돌아야 한다(앞에 돌면 지워진다).
    """
    from utils.settings_loader import _load_pool_configs
    from utils.stock_cache_meta_io import set_stock_cache_meta_field
    from utils.stock_list_io import get_etfs

    logger = get_app_logger()
    wanted = {str(t).strip().lower() for t in ticker_types} if ticker_types else None
    rank_by_country: dict[str, dict[str, int]] = {}
    updated: dict[str, int] = {}
    for config in _load_pool_configs():
        pool = str(config.get("ticker_type") or "").strip().lower()
        if not pool or (wanted is not None and pool not in wanted):
            continue
        if str(config.get("pool_kind") or "").strip().lower() != "stock":
            continue  # ETF 풀은 시총 순위 개념이 없다
        country = str(config.get("country_code") or "").strip().lower()
        if country not in rank_by_country:
            try:
                rank_by_country[country] = load_market_cap_rank_map(country)
            except Exception as exc:
                logger.error("[배치 B] 시총 순위표 생성 실패 (%s): %s", country, exc)
                rank_by_country[country] = {}
        ranks = rank_by_country[country]
        if not ranks:
            continue
        # 반복 횟수가 아니라 **실제로 기록된 건수**를 센다 — 예전에는 한 건도 안 써져도
        # "201종목 기록" 으로 찍혀서 화면의 시총 컬럼이 비어 있는 걸 로그로 못 잡았다.
        written = 0
        missing = 0
        for stock in get_etfs(pool) or []:
            ticker = str(stock.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            rank = ranks.get(ticker)
            if rank is None:
                missing += 1
            if set_stock_cache_meta_field(pool, ticker, META_FIELD, rank):
                written += 1
        updated[pool] = written
        logger.info(
            "[배치 B] 시총 순위 기록: %s %d건 (국가 %s, 순위표 %d종목, 순위 없음 %d종목)",
            pool,
            written,
            country,
            len(ranks),
            missing,
        )
    return updated


def market_cap_rank_of(meta_cache: dict[str, Any] | None) -> int | None:
    """메타 캐시 문서에서 시총 순위를 꺼낸다 (화면 공용)."""
    value = (meta_cache or {}).get(META_FIELD)
    return int(value) if isinstance(value, (int, float)) else None
