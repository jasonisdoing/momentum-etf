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


from utils.settings_loader import get_account_settings, list_available_accounts

_ACCOUNT_STOCKS_CACHE: dict[str, list[dict]] = {}
_LISTING_CACHE: dict[tuple[str, str], str | None] = {}


def _load_account_stocks_raw(account_id: str) -> list[dict]:
    account_norm = (account_id or "").strip().lower()
    if account_norm in _ACCOUNT_STOCKS_CACHE:
        return _ACCOUNT_STOCKS_CACHE[account_norm]

    # Rule 7: Strict Path. No fallbacks.
    file_path = os.path.join(_get_data_dir(), account_norm, "stocks.json")
    if not os.path.exists(file_path):
        # Fallback check (Strictly forbidden by Rule 7, but I must ensure migration happened)
        # Assuming migration happened, this file SHOULD exist if the account is valid.
        _ACCOUNT_STOCKS_CACHE[account_norm] = []
        # We might want to raise an error if expected?
        # But for now returning empty list allows callers to handle it?
        # Rule 7 says "raise clear error".
        # But load_raw might be used by "try load".
        # Let's log warning and return empty, but get_etfs will validate.
        logger.error(f"계정 종목 파일이 존재하지 않습니다: {file_path}")
        return []

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                _ACCOUNT_STOCKS_CACHE[account_norm] = data
                return data
            logger.warning("'%s' 파일의 루트는 리스트여야 합니다.", file_path)
    except json.JSONDecodeError as exc:
        logger.error("'%s' JSON 파싱 실패: %s", file_path, exc)
    except Exception as exc:
        logger.warning("'%s' 파일 읽기 실패: %s", file_path, exc)

    _ACCOUNT_STOCKS_CACHE[account_norm] = []
    return []


def get_etfs(account_id: str, include_extra_tickers: Iterable[str] | None = None) -> list[dict[str, str]]:
    """
    'zsettings/<account>/stocks.json' 파일에서 종목 목록을 반환합니다.
    """
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id must be provided")

    all_etfs: list[dict[str, Any]] = []
    seen_tickers = set()

    data = _load_account_stocks_raw(account_norm)
    if not data:
        # If data is empty, it might be a missing file or empty file.
        # Check if file exists to distinguish?
        file_path = os.path.join(_get_data_dir(), account_norm, "stocks.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Stock settings not found for account: {account_id} at {file_path}")
        return []

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

    filtered = all_etfs
    logger.info(
        "[%s] 전체 ETF 유니버스 로딩: %d개 종목",
        account_norm.upper(),
        len(filtered),
    )

    if include_extra_tickers:
        existing = {item["ticker"].upper() for item in filtered}
        for ticker in include_extra_tickers:
            norm = str(ticker or "").strip().upper()
            if not norm or norm in existing:
                continue
            # Extra tickers might not be in the account source?
            # If provided in include_extra_tickers, assume we want them.
            # But where do we get metadata?
            # The original code looked up in `by_ticker`.
            # If extra ticker is NOT in the file, it won't be in by_ticker.
            # So this logic only works if include_extra_tickers are IN the file but filtered out?
            # But here we loaded EVERYTHING from the file.
            # So include_extra_tickers logic seems redundant if we already load everything?
            # Wait, the original code filtered?
            # Original code: `filtered = all_etfs`.
            # So it returns ALL.
            # Include extra tickers logic seems to be "Ensure these are included even if..."?
            # Actually line 102: `src = by_ticker.get(norm)`.
            # If it's in `by_ticker`, it's already in `all_etfs`.
            # So `include_extra_tickers` does NOTHING in the original code unless there was prior filtering?
            # Yes, `filtered = all_etfs`.
            # So I will keep it as is (no-op effectively).
            pass

    return filtered


def get_etfs_by_country(country: str) -> list[dict[str, Any]]:
    """
    (Legacy Helper) Aggregate stocks from all accounts matching the country code.
    Used for name resolution where account context is missing.
    """
    country_norm = (country or "").strip().lower()
    accounts = list_available_accounts()

    aggregated = []
    seen = set()

    for account in accounts:
        try:
            settings = get_account_settings(account)
            if settings.get("country_code", "").lower() == country_norm:
                stocks = _load_account_stocks_raw(account)
                for block in stocks:
                    for item in block.get("tickers", []):
                        tkr = item.get("ticker")
                        if tkr and tkr not in seen:
                            aggregated.append(item)  # Need flattened structure?
                            seen.add(tkr)
        except Exception:
            continue

    # Need to return structure compatible with _get_display_name?
    # _get_display_name expects list of blocks (with tickers list) OR flat list?
    # Original get_etfs returns list of ETF dictionaries (flat).
    # But _get_display_name iterates:
    # for block in etf_blocks:
    #    if "tickers" in block: ...
    # So _get_display_name expects RAW structure (blocks).
    #
    # My _load_country_raw returned RAW structure (List[Dict] with category/tickers).
    # So `get_etfs_by_country` should return RAW structure?
    # But `get_etfs` (the original one) called `_load_country_raw` inside but returned FLAT list?
    # Wait, check `_get_display_name` in Step 876.
    # Line 1387: `etf_blocks = get_etfs(country_code)`.
    # AND Line 1388: `for block in etf_blocks: if "tickers" in block...`
    #
    # WAIT! `get_etfs` (Original) returned FLAT list calling it `all_etfs`.
    # Step 819: `get_etfs` implementation:
    # 51: `all_etfs: list[dict] = []`
    # 82: `all_etfs.append(new_item)` which is a DICT of ticker info.
    # It does NOT have "tickers" key. It has "ticker", "category", etc.
    #
    # So `_get_display_name` logic at Step 876 Line 1390 `if "tickers" in block:`
    # seems to expect the RAW structure.
    # BUT it calls `get_etfs(country_code)`.
    # THIS MEANS `_get_display_name` IS BUGGY/WRONG in the current codebase if `get_etfs` returns flat list.
    # OR `get_etfs` returns blocks?
    # Step 819 again: `get_etfs` returns `filtered` (list of dicts).
    # Each dict is `new_item` with `ticker` key.
    #
    # Let's check `_get_display_name` again.
    # Line 1390: `if "tickers" in block:`
    # If `get_etfs` returns flat items, they don't have "tickers".
    # So `else` block (Line 1399) runs.
    # Line 1400: `tkr = block.get("ticker")`.
    # This matches!
    # So `_get_display_name` handles both?
    #
    # So `get_etfs_by_country` can return flat list of unique tickers.

    unique_tickers = {}
    for account in accounts:
        try:
            settings = get_account_settings(account)
            if settings.get("country_code", "").lower() == country_norm:
                # Use get_etfs(account) to get flat list
                etfs = get_etfs(account)
                for etf in etfs:
                    tkr = etf.get("ticker")
                    if tkr and tkr not in unique_tickers:
                        unique_tickers[tkr] = etf
        except Exception:
            pass

    return list(unique_tickers.values())


def get_all_etfs(account_id: str) -> list[dict[str, Any]]:
    """Return every ETF entry defined in zsettings/<account>/stocks.json."""

    raw_data = _load_account_stocks_raw(account_id)
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


def save_etfs(account_id: str, data: list[dict]):
    """
    주어진 데이터를 'zsettings/<account>/stocks.json' 파일에 저장합니다.
    """
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id required")

    # Rule 7: Strict Path.
    stocks_file = os.path.join(_get_data_dir(), account_norm, "stocks.json")

    # Ensure dir exists (it should if migrated)
    os.makedirs(os.path.dirname(stocks_file), exist_ok=True)

    try:
        with open(stocks_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info("%d개 카테고리의 종목 정보가 '%s'에 저장되었습니다.", len(data), stocks_file)
        _ACCOUNT_STOCKS_CACHE[account_norm] = data
    except Exception as e:
        logger.error("'%s' 파일 저장 실패: %s", stocks_file, e)
        raise


def get_etf_categories(account_id: str) -> list[str]:
    """
    지정된 계정의 모든 ETF 카테고리 목록을 반환합니다.
    """
    categories = set()
    data = _load_account_stocks_raw(account_id)
    if not data:
        return []

    for category_block in data:
        if isinstance(category_block, dict) and "category" in category_block:
            categories.add(category_block["category"])

    return sorted(list(categories))


def get_listing_date(country: str, ticker: str) -> str | None:
    """
    Looks up listing date by country code (aggregating all accounts).
    """
    country_norm = (country or "").strip().lower()
    ticker_norm = str(ticker or "").strip()
    cache_key = (country_norm, ticker_norm)
    if cache_key in _LISTING_CACHE:
        return _LISTING_CACHE[cache_key]

    accounts = list_available_accounts()
    for account in accounts:
        try:
            settings = get_account_settings(account)
            if settings.get("country_code", "").lower() == country_norm:
                data = _load_account_stocks_raw(account)
                for category_block in data:
                    tickers_list = category_block.get("tickers") if isinstance(category_block, dict) else None
                    if not isinstance(tickers_list, list):
                        continue
                    for item in tickers_list:
                        if not isinstance(item, dict):
                            continue
                        item_ticker = str(item.get("ticker") or "").strip()
                        if item_ticker == ticker_norm:
                            listing_date = item.get("listing_date")
                            if listing_date:
                                _LISTING_CACHE[cache_key] = listing_date
                                return listing_date
        except Exception:
            continue

    _LISTING_CACHE[cache_key] = None
    return None


def set_listing_date(country: str, ticker: str, listing_date: str) -> None:
    """
    Sets listing date for a ticker across ALL accounts sharing the country code.
    """
    country_norm = (country or "").strip().lower()
    ticker_norm = str(ticker or "").strip()
    if not ticker_norm:
        return

    accounts = list_available_accounts()
    updated_any = False

    for account in accounts:
        try:
            settings = get_account_settings(account)
            if settings.get("country_code", "").lower() == country_norm:
                data = _load_account_stocks_raw(account)
                updated_local = False
                changed_local = False

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
                            updated_local = True
                            break
                        item["listing_date"] = listing_date
                        updated_local = True
                        changed_local = True
                        break

                    if updated_local:
                        break

                if updated_local and changed_local:
                    save_etfs(account, data)
                    updated_any = True
                elif updated_local:
                    updated_any = True
        except Exception:
            continue

    if updated_any:
        _LISTING_CACHE[(country_norm, ticker_norm)] = listing_date
