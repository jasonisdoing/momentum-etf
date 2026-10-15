"""Global X 미국 ETF 공식 상품 명단과 전체 구성종목 CSV 수집."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup

from config import CACHE_TTL_SLOW
from services.component_price_service import infer_yahoo_symbol_currency
from utils.asx_ticker import ensure_asx_prefix
from utils.http_session import shared_session
from utils.normalization import resolve_bloomberg_listing
from utils.ttl_cache import TtlCache
from utils.us_etf_market_service import load_us_security_exchange_map

_BASE_URL = "https://www.globalxetfs.com"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="globalx_us_products", max_entries=1)
_FOOTER = (
    "The information contained herein may not be reproduced, redistributed or used to create any derivative works."
)


def _fetch_product_map() -> dict[str, str]:
    response = shared_session.get(f"{_BASE_URL}/explore", timeout=25)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    tables = [
        props
        for value in _page_records(soup).values()
        for props in _page_props(value)
        if "fundPathPrefix" in props and "rows" in props
    ]
    if len(tables) != 1 or tables[0]["fundPathPrefix"] != "funds":
        raise ValueError("Global X 공식 전체 상품 명단을 확인할 수 없습니다.")
    products = {}
    for row in tables[0]["rows"]:
        ticker = row["ticker"]
        if re.fullmatch(r"[A-Z0-9]+", ticker) is None or ticker in products:
            raise ValueError("Global X 공식 상품 명단의 티커가 잘못됐거나 중복됐습니다.")
        products[ticker] = f"{_BASE_URL}/funds/{ticker.lower()}"
    if not products:
        raise ValueError("Global X 공식 상품 명단에 ETF가 없습니다.")
    return products


def _page_records(soup: BeautifulSoup) -> dict[str, Any]:
    """공식 페이지가 전달한 다운로드 설정을 JSON으로 읽는다."""
    chunks = []
    for script in soup.select("script"):
        match = re.fullmatch(r"self\.__next_f\.push\((.*)\)", script.get_text(), re.DOTALL)
        if match:
            entry = json.loads(match[1])
            if entry[0] == 1:
                chunks.append(entry[1])
    records = {}
    for line in "".join(chunks).splitlines():
        match = re.fullmatch(r"([a-f0-9]+):([\[{].*)", line)
        if match:
            records[match[1]] = json.loads(match[2])
    return records


def _page_props(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _page_props(child)
    elif isinstance(value, list):
        for child in value:
            yield from _page_props(child)


def _find_holdings_download(html: str, ticker: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.select_one("h1")
    if title is None or title.get_text(strip=True) != ticker:
        raise ValueError(f"Global X {ticker} 공식 상품 페이지의 티커가 다릅니다.")
    records = _page_records(soup)
    settings = [
        props
        for value in records.values()
        for props in _page_props(value)
        if props.get("data-document-type") == "Holdings CSV" and "getCsv" in props
    ]
    if len(settings) != 1 or settings[0].get("ticker") != ticker:
        raise ValueError(f"Global X {ticker} 전체 구성종목 다운로드 설정이 없거나 중복됐습니다.")
    setting = settings[0]
    match = re.fullmatch(rf"{re.escape(ticker)}-holdings-([A-Za-z]{{3}}-\d{{1,2}}-\d{{4}})\.csv", setting["fileName"])
    reference = re.fullmatch(r"\$h([a-f0-9]+)", setting["getCsv"])
    action = records.get(reference[1]) if reference else None
    if match is None or not isinstance(action, dict) or action.get("bound") is not None:
        raise ValueError(f"Global X {ticker} 공식 CSV의 상품명 또는 다운로드 요청이 잘못됐습니다.")
    action_id = action.get("id", "")
    if re.fullmatch(r"[a-f0-9]{42}", action_id) is None:
        raise ValueError(f"Global X {ticker} 공식 CSV 다운로드 식별자가 잘못됐습니다.")
    return action_id, datetime.strptime(match[1], "%b-%d-%Y").date().isoformat()


def _download_csv(url: str, ticker: str, action_id: str) -> str:
    """공식 CSV 버튼과 같은 공개 요청을 보내고 길이 지정 문자열을 읽는다."""
    response = shared_session.post(
        url,
        headers={"Next-Action": action_id, "Content-Type": "text/plain;charset=UTF-8", "Accept": "text/x-component"},
        data=json.dumps([ticker]),
        timeout=25,
    )
    response.raise_for_status()
    content = response.content
    records = {}
    while content:
        match = re.match(rb"([a-f0-9]+):", content)
        if match is None:
            raise ValueError(f"Global X {ticker} CSV 응답 형식이 잘못됐습니다.")
        key = match[1].decode()
        content = content[match.end() :]
        text_header = re.match(rb"T([a-f0-9]+),", content)
        if text_header:
            length = int(text_header[1], 16)
            start = text_header.end()
            if len(content) < start + length:
                raise ValueError(f"Global X {ticker} CSV 응답이 잘렸습니다.")
            records[key] = content[start : start + length].decode("utf-8")
            content = content[start + length :]
        else:
            line, separator, content = content.partition(b"\n")
            if not separator:
                raise ValueError(f"Global X {ticker} CSV 응답이 완전하지 않습니다.")
            records[key] = json.loads(line)
    root = records.get("0")
    if not isinstance(root, dict) or root.get("a") != "$@1":
        raise ValueError(f"Global X {ticker} CSV 다운로드 결과가 없습니다.")
    result = records.get("1")
    if isinstance(result, str) and re.fullmatch(r"\$[a-f0-9]+", result):
        result = records.get(result[1:])
    if not isinstance(result, str) or not result:
        raise ValueError(f"Global X {ticker} 전체 구성종목 CSV가 비었습니다.")
    return result


def _parse_holdings(text: str, ticker: str) -> list[dict[str, str]]:
    reader = csv.reader(io.StringIO(text))
    headers = next(reader, [])
    required = {"Net Assets (%)", "Ticker", "Name", "SEDOL", "Market Price ($)", "Shares Held", "Market Value ($)"}
    if len(headers) != len(required) or set(headers) != required:
        raise ValueError(f"Global X {ticker} 구성종목 CSV 필수 열이 잘못됐습니다.")
    rows = []
    footer_seen = False
    for values in reader:
        if not values:
            continue
        if values == [_FOOTER] and not footer_seen:
            footer_seen = True
            continue
        if footer_seen or len(values) != len(headers):
            raise ValueError(f"Global X {ticker} 구성종목 CSV 행 구조가 잘못됐습니다.")
        rows.append(dict(zip(headers, values, strict=True)))
    if not rows or not footer_seen:
        raise ValueError(f"Global X {ticker} 전체 구성종목 CSV가 비었거나 완전하지 않습니다.")
    return rows


def _normalize_holding(raw: dict[str, str], us_listings: dict[str, str]) -> dict[str, Any]:
    label = raw["Ticker"].strip()
    name = raw["Name"].strip()
    identifier = raw["SEDOL"].strip()
    weight = float(raw["Net Assets (%)"].strip().removesuffix("%"))
    if not name or not math.isfinite(weight):
        raise ValueError("Global X 구성종목 이름 또는 비중이 잘못됐습니다.")
    symbol, exchange = (
        resolve_bloomberg_listing(label, currency=None, us_listings=us_listings) if label else (None, None)
    )
    ticker = symbol or label or identifier or None
    if symbol:
        ticker = ensure_asx_prefix(symbol[:-3]) if symbol.endswith(".AX") else symbol
        if symbol.endswith((".KS", ".KQ")):
            ticker = symbol[:-3]
    return {
        "ticker": ticker,
        "name": f"현금 · {name}" if name == "CASH" else name,
        "weight": weight,
        "raw_code": identifier or None,
        "yahoo_symbol": symbol,
        "listing_currency": infer_yahoo_symbol_currency(symbol) if symbol else None,
        "listing_exchange": exchange,
        "asset_class": "Cash" if name == "CASH" else None,
        "price_lookup_supported": symbol is not None,
    }


def fetch_globalx_us_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 명단에 없는 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("Global X 구성종목 조회에 티커가 필요합니다.")
    url = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map).get(ticker_norm)
    if url is None:
        return None
    response = shared_session.get(url, timeout=25)
    response.raise_for_status()
    action_id, reference_date = _find_holdings_download(response.text, ticker_norm)
    rows = _parse_holdings(_download_csv(url, ticker_norm, action_id), ticker_norm)
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(raw, us_listings) for raw in rows]
    return {
        "source": "globalx_us_csv",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
