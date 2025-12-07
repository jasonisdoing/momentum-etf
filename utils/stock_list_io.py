import json
import os
from collections.abc import Iterable
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()


def _get_data_dir():
    """Helper to get the absolute path to the 'zsettings' directory."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "zsettings")


_COUNTRY_RAW_CACHE: dict[str, list[dict]] = {}
_LISTING_CACHE: dict[tuple[str, str], str | None] = {}


def _load_country_raw(country: str) -> list[dict]:
    country_norm = (country or "").strip().lower()
    if country_norm in _COUNTRY_RAW_CACHE:
        return _COUNTRY_RAW_CACHE[country_norm]

    file_path = os.path.join(_get_data_dir(), "stocks", f"{country_norm}.json")
    if not os.path.exists(file_path):
        _COUNTRY_RAW_CACHE[country_norm] = []
        return []

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                _COUNTRY_RAW_CACHE[country_norm] = data
                return data
            logger.warning("'%s' 파일의 루트는 리스트여야 합니다.", file_path)
    except json.JSONDecodeError as exc:
        logger.error("'%s' JSON 파싱 실패: %s", file_path, exc)
    except Exception as exc:
        logger.warning("'%s' 파일 읽기 실패: %s", file_path, exc)

    _COUNTRY_RAW_CACHE[country_norm] = []
    return []


def get_etfs(country: str, include_extra_tickers: Iterable[str] | None = None) -> list[dict[str, str]]:
    """
    'zsettings/stocks/{country}.json' 파일에서 종목 목록을 반환합니다.
    """
    all_etfs: list[dict[str, Any]] = []
    seen_tickers = set()

    data = _load_country_raw(country)
    by_ticker: dict[str, dict[str, Any]] = {}
    for category_block in data:
        if not isinstance(category_block, dict) or "tickers" not in category_block:
            continue

        category_name = category_block.get("category", "Uncategorized")
        tickers_list = category_block.get("tickers", [])
        if not isinstance(tickers_list, list):
            continue

        for item in tickers_list:
            if not isinstance(item, dict) or not item.get("ticker"):
                continue

            ticker = item["ticker"]
            ticker_norm = str(ticker).strip()
            if not ticker_norm or ticker_norm in seen_tickers:
                continue

            seen_tickers.add(ticker_norm)

            new_item = dict(item)
            new_item["ticker"] = ticker_norm
            new_item["type"] = "etf"
            new_item["category"] = category_name
            if item.get("listing_date"):
                new_item["listing_date"] = item["listing_date"]
            all_etfs.append(new_item)
            by_ticker[ticker_norm.upper()] = new_item

    if not all_etfs:
        return all_etfs

    filtered = all_etfs

    logger.info(
        "[%s] 전체 ETF 유니버스 로딩: %d개 종목",
        (country or "").upper(),
        len(filtered),
    )

    if include_extra_tickers:
        existing = {item["ticker"].upper() for item in filtered}
        for ticker in include_extra_tickers:
            norm = str(ticker or "").strip().upper()
            if not norm or norm in existing:
                continue
            src = by_ticker.get(norm)
            if src:
                filtered.append(src)
                existing.add(norm)

    return filtered


def get_all_etfs(country: str) -> list[dict[str, Any]]:
    """Return every ETF entry defined in zsettings/stocks/{country}.json without filtering."""

    raw_data = _load_country_raw(country)
    if not raw_data:
        return []

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for category_block in raw_data:
        if not isinstance(category_block, dict):
            continue
        raw_category = category_block.get("category", "")
        if isinstance(raw_category, (list, set, tuple)):
            raw_category = next(iter(raw_category), "") if raw_category else ""
        category_name = str(raw_category or "").strip()
        if not category_name:
            raise ValueError(
                "카테고리 블록에 카테고리 이름이 없습니다. 모든 카테고리 블록은 'category' 필드가 있어야 합니다."
            )
        tickers_list = category_block.get("tickers", [])
        if not isinstance(tickers_list, list):
            continue
        for item in tickers_list:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker") or "").strip()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            entry = dict(item)
            entry["ticker"] = ticker
            entry.setdefault("type", "etf")
            entry.setdefault("category", category_name)
            results.append(entry)
    return results


def save_etfs(country: str, data: list[dict]):
    """
    주어진 데이터를 'zsettings/stocks/{country}.json' 파일에 저장합니다.
    """
    stocks_data_dir = os.path.join(_get_data_dir(), "stocks")
    os.makedirs(stocks_data_dir, exist_ok=True)
    file_path = os.path.join(stocks_data_dir, f"{country}.json")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info("%d개 카테고리의 종목 정보가 '%s'에 저장되었습니다.", len(data), file_path)
        country_norm = (country or "").strip().lower()
        _COUNTRY_RAW_CACHE[country_norm] = data
    except Exception as e:
        logger.error("'%s' 파일 저장 실패: %s", file_path, e)
        raise


def get_etf_categories(country: str) -> list[str]:
    """
    지정된 국가의 모든 ETF 카테고리 목록을 반환합니다.
    """
    categories = set()

    file_path = os.path.join(_get_data_dir(), "stocks", f"{country}.json")
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for category_block in data:
                    if isinstance(category_block, dict) and "category" in category_block:
                        categories.add(category_block["category"])
    except Exception as e:
        logger.warning("'%s' 파일에서 카테고리 읽기 실패: %s", file_path, e)

    return sorted(list(categories))


def get_listing_date(country: str, ticker: str) -> str | None:
    country_norm = (country or "").strip().lower()
    ticker_norm = str(ticker or "").strip()
    cache_key = (country_norm, ticker_norm)
    if cache_key in _LISTING_CACHE:
        return _LISTING_CACHE[cache_key]

    data = _load_country_raw(country_norm)
    for category_block in data:
        tickers_list = category_block.get("tickers") if isinstance(category_block, dict) else None
        if not isinstance(tickers_list, list):
            continue
        for item in tickers_list:
            if not isinstance(item, dict):
                continue
            item_ticker = str(item.get("ticker") or "").strip()
            if item_ticker != ticker_norm:
                continue
            listing_date = item.get("listing_date")
            _LISTING_CACHE[cache_key] = listing_date
            return listing_date

    _LISTING_CACHE[cache_key] = None
    return None


def set_listing_date(country: str, ticker: str, listing_date: str) -> None:
    country_norm = (country or "").strip().lower()
    ticker_norm = str(ticker or "").strip()
    if not ticker_norm:
        return

    data = _load_country_raw(country_norm)
    updated = False
    changed = False
    for category_block in data:
        tickers_list = category_block.get("tickers") if isinstance(category_block, dict) else None
        if not isinstance(tickers_list, list):
            continue
        for item in tickers_list:
            if not isinstance(item, dict):
                continue
            item_ticker = str(item.get("ticker") or "").strip()
            if item_ticker != ticker_norm:
                continue
            current = item.get("listing_date")
            if current == listing_date:
                updated = True
                break
            item["listing_date"] = listing_date
            updated = True
            changed = True
            break
        if updated:
            break

    if updated and changed:
        save_etfs(country_norm, data)
        _LISTING_CACHE[(country_norm, ticker_norm)] = listing_date
    elif updated:
        _LISTING_CACHE[(country_norm, ticker_norm)] = listing_date
    else:
        _LISTING_CACHE[(country_norm, ticker_norm)] = listing_date
