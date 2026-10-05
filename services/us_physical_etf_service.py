"""미국 실물 보유 ETF의 공식 보유 목록 안내 수집."""

from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup

from utils.http_session import shared_session

# 공식 상품 구조와 실물 보유 목록을 확인한 상품만 등록한다.
_PRODUCTS = {
    "GLD": {
        "asset_name": "금",
        "page_url": "https://www.spdrgoldshares.com/usa/gld/",
        "url": "https://api.spdrgoldshares.com/api/v1/barlist?underlying=gld",
    },
    "SLV": {
        "asset_name": "은",
        "page_url": "https://www.ishares.com/us/products/239855/ishares-silver-trust-fund",
        "url": "https://emea-markets.jpmorgan.com/metalicsWebAppJanus/publicUnauthenticated/BONY_SLV.pdf",
    },
}


def fetch_us_physical_etf_holdings(ticker: str) -> dict[str, Any] | None:
    """실물 상품은 주식 구성종목 대신 공식 공시 링크를 반환한다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("실물 ETF 조회에 티커가 필요합니다.")
    product = _PRODUCTS.get(ticker_norm)
    if product is None:
        return None
    response = shared_session.get(product["page_url"], timeout=25)
    response.raise_for_status()
    links = {link.get("href") for link in BeautifulSoup(response.text, "html.parser").select("a[href]")}
    if product["url"] not in links:
        raise ValueError(f"{ticker_norm} 공식 상품 페이지에서 실물 보유 목록 링크를 확인할 수 없습니다.")
    return {
        "source": "physical_asset_disclosure",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": None,
        "holdings_count": 0,
        "holdings": [],
        "disclosure": {"asset_name": product["asset_name"], "url": product["url"]},
    }
