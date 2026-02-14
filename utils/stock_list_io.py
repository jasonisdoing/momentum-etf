import json
import os
from collections.abc import Iterable
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()


def _get_data_dir():
    """Helper to get the absolute path to the 'zaccounts' directory."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "zaccounts")


from utils.settings_loader import get_account_settings, list_available_accounts

_ACCOUNT_STOCKS_CACHE: dict[str, list[dict]] = {}
_LISTING_CACHE: dict[tuple[str, str], str | None] = {}


def _load_account_stocks_raw(account_id: str) -> list[dict]:
    account_norm = (account_id or "").strip().lower()
    if account_norm in _ACCOUNT_STOCKS_CACHE:
        return _ACCOUNT_STOCKS_CACHE[account_norm]

    file_path = os.path.join(_get_data_dir(), account_norm, "stocks.json")
    if not os.path.exists(file_path):
        _ACCOUNT_STOCKS_CACHE[account_norm] = []
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
    'zaccounts/<account>/stocks.json' 파일에서 종목 목록을 반환합니다.
    플랫 리스트 형태: [{ticker, name, listing_date, ...}, ...]
    """
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id must be provided")

    all_etfs: list[dict[str, Any]] = []
    seen_tickers = set()

    data = _load_account_stocks_raw(account_norm)
    if not data:
        file_path = os.path.join(_get_data_dir(), account_norm, "stocks.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Stock settings not found for account: {account_id} at {file_path}")
        return []

    for item in data:
        if not isinstance(item, dict) or not item.get("ticker"):
            continue

        ticker = str(item["ticker"]).strip()
        if not ticker or ticker in seen_tickers:
            continue

        seen_tickers.add(ticker)

        new_item = dict(item)
        new_item["ticker"] = ticker
        new_item["type"] = "etf"
        all_etfs.append(new_item)

    logger.info(
        "[%s] 전체 ETF 유니버스 로딩: %d개 종목",
        account_norm.upper(),
        len(all_etfs),
    )

    return all_etfs


def get_etfs_by_country(country: str) -> list[dict[str, Any]]:
    """
    (Legacy Helper) Aggregate stocks from all accounts matching the country code.
    Used for name resolution where account context is missing.
    """
    country_norm = (country or "").strip().lower()
    accounts = list_available_accounts()

    unique_tickers = {}
    for account in accounts:
        try:
            settings = get_account_settings(account)
            if settings.get("country_code", "").lower() == country_norm:
                etfs = get_etfs(account)
                for etf in etfs:
                    tkr = etf.get("ticker")
                    if tkr and tkr not in unique_tickers:
                        unique_tickers[tkr] = etf
        except Exception:
            pass

    return list(unique_tickers.values())


def get_all_etfs(account_id: str) -> list[dict[str, Any]]:
    """Return every ETF entry defined in zaccounts/<account>/stocks.json."""

    raw_data = _load_account_stocks_raw(account_id)
    if not raw_data:
        return []

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_data:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        entry = dict(item)
        entry["ticker"] = ticker
        entry.setdefault("type", "etf")
        results.append(entry)
    return results


def save_etfs(account_id: str, data: list[dict]):
    """
    주어진 데이터를 'zaccounts/<account>/stocks.json' 파일에 저장합니다.
    """
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id required")

    stocks_file = os.path.join(_get_data_dir(), account_norm, "stocks.json")

    os.makedirs(os.path.dirname(stocks_file), exist_ok=True)

    try:
        with open(stocks_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info("%d개 종목 정보가 '%s'에 저장되었습니다.", len(data), stocks_file)
        _ACCOUNT_STOCKS_CACHE[account_norm] = data
    except Exception as e:
        logger.error("'%s' 파일 저장 실패: %s", stocks_file, e)
        raise


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
                for item in data:
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

                for item in data:
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

                if updated_local and changed_local:
                    save_etfs(account, data)
                    updated_any = True
                elif updated_local:
                    updated_any = True
        except Exception:
            continue

    if updated_any:
        _LISTING_CACHE[(country_norm, ticker_norm)] = listing_date
