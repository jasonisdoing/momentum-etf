"""Vanguard Australia 공식 API — 호주 ETF 구성종목·운용보수 수집.

yfinance 는 호주 ETF 의 운용보수를 제공하지 않고, 일부 종목(VVLU·VTS 등)은
`quoteType=EQUITY` 로 잘못 내려주어 구성종목조차 받을 수 없다. Vanguard 공식
API 는 구성종목에 `countryCode` 까지 붙여 주고 운용보수(MER)도 제공한다.

API 는 ASX 티커가 아니라 내부 식별자(portId)로 조회한다. 티커→portId 목록을
주는 엔드포인트가 없어, portId 구간을 한 번 훑어 매핑을 만들고 DB 에 캐시한다.
"""

from __future__ import annotations

import concurrent.futures
import json
import urllib.request
from datetime import datetime
from typing import Any

from utils.asx_ticker import normalize_ticker, strip_asx_prefix
from utils.logger import get_app_logger

logger = get_app_logger()

_API_BASE = "https://www.vanguard.com.au/adviser/api"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
_TIMEOUT_SECONDS = 20

# holdings API 는 limit 이 크면 500 을 돌려준다. 확인된 안전값.
_HOLDINGS_LIMIT = 100
# 운용보수(MER)에 해당하는 Vanguard 비용 코드.
_MER_EXPENSE_CODE = "ADJEXPRTPC"

# portId 스캔 구간. Vanguard AU 상장 ETF 는 이 구간에 모여 있다.
_PORT_ID_SCAN_START = 8100
_PORT_ID_SCAN_END = 8400
_PORT_ID_SCAN_WORKERS = 12

_CONFIG_COLLECTION = "system_config"
_PORT_ID_MAP_KEY = "vanguard_au_portid_map"

# 프로세스 수명 동안 유지되는 티커→portId 매핑 (DB 캐시를 한 번만 읽는다).
_PORT_ID_MAP_CACHE: dict[str, int] | None = None
# 스캔은 수백 회 요청이라 프로세스당 1회로 제한한다. Vanguard 상품이 아닌 티커
# (BetaShares·iShares 등)는 매핑에 영원히 없으므로 종목마다 재스캔하면 안 된다.
_PORT_ID_SCAN_DONE = False


def _get_json(url: str) -> dict[str, Any] | None:
    try:
        request = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8", errors="ignore"))
    except Exception as exc:
        logger.debug(f"[Vanguard] 요청 실패 {url}: {exc}")
        return None


def _fetch_profile(port_id: int) -> tuple[str, int] | None:
    """portId 의 ASX 티커를 조회한다. ETF 가 아니거나 없으면 None."""
    payload = _get_json(f"{_API_BASE}/data/products/fund-profile-type/{port_id}")
    items = (payload or {}).get("data") or []
    if not items:
        return None
    profile = items[0]
    if str(profile.get("productType") or "").strip().lower() != "etf":
        return None
    ticker = normalize_ticker((profile.get("tradingSymbols") or {}).get("TICKER"))
    if not ticker:
        return None
    return ticker, port_id


def _load_port_id_map_from_db() -> dict[str, int]:
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        return {}
    doc = db[_CONFIG_COLLECTION].find_one({"_id": _PORT_ID_MAP_KEY}) or {}
    stored = doc.get("map") or {}
    return {str(k): int(v) for k, v in stored.items() if str(v).isdigit()}


def _save_port_id_map_to_db(port_id_map: dict[str, int]) -> None:
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        return
    db[_CONFIG_COLLECTION].update_one(
        {"_id": _PORT_ID_MAP_KEY},
        {"$set": {"map": port_id_map, "updated_at": datetime.now().isoformat()}},
        upsert=True,
    )


def _scan_port_id_map() -> dict[str, int]:
    """portId 구간을 훑어 티커→portId 매핑을 만든다 (수백 회 요청이라 병렬 처리)."""
    logger.info(f"[Vanguard] portId 스캔 시작 ({_PORT_ID_SCAN_START}~{_PORT_ID_SCAN_END})")
    port_id_map: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=_PORT_ID_SCAN_WORKERS) as executor:
        for result in executor.map(_fetch_profile, range(_PORT_ID_SCAN_START, _PORT_ID_SCAN_END)):
            if result:
                ticker, port_id = result
                port_id_map[ticker] = port_id
    logger.info(f"[Vanguard] portId 스캔 완료 — ETF {len(port_id_map)}개")
    return port_id_map


def resolve_port_id(ticker: str, *, allow_rescan: bool = True) -> int | None:
    """ASX 티커에 대응하는 Vanguard portId. 매핑에 없으면 None.

    매핑이 아직 없을 때만 스캔하며, 스캔은 프로세스당 1회다. Vanguard 상품이 아닌
    티커는 매핑에 들어올 일이 없으므로 여기서 재스캔하면 종목 수만큼 스캔이 반복된다.
    신규 Vanguard ETF 를 반영하려면 `system_config` 의 매핑 문서를 지우고 다시 실행한다.
    """
    global _PORT_ID_MAP_CACHE, _PORT_ID_SCAN_DONE

    ticker_norm = strip_asx_prefix(ticker)
    if not ticker_norm:
        return None

    if _PORT_ID_MAP_CACHE is None:
        _PORT_ID_MAP_CACHE = _load_port_id_map_from_db()

    port_id = _PORT_ID_MAP_CACHE.get(ticker_norm)
    if port_id is not None:
        return port_id

    # 매핑이 비어 있을 때만(=아직 한 번도 스캔한 적 없음) 채운다.
    if not allow_rescan or _PORT_ID_SCAN_DONE or _PORT_ID_MAP_CACHE:
        return None

    _PORT_ID_SCAN_DONE = True
    scanned = _scan_port_id_map()
    if not scanned:
        return None
    _PORT_ID_MAP_CACHE = scanned
    _save_port_id_map_to_db(scanned)
    return scanned.get(ticker_norm)


def _to_component_ticker(raw_ticker: str, country_code: str) -> str:
    """구성종목 티커 표준화 — 호주 상장이면 시스템 표준 표기(ASX:NAB)."""
    from utils.asx_ticker import ensure_asx_prefix

    normalized = normalize_ticker(raw_ticker)
    if str(country_code or "").strip().upper() == "AU":
        return ensure_asx_prefix(normalized)
    return normalized


def fetch_vanguard_au_holdings(ticker: str) -> dict[str, Any] | None:
    """Vanguard 공식 API 에서 구성종목을 가져온다.

    반환 형식은 `fetch_betashares_holdings` / `fetch_yfinance_holdings` 와 동일하다.
    """
    port_id = resolve_port_id(ticker)
    if port_id is None:
        logger.debug(f"[Vanguard] portId 매핑 없음: {ticker}")
        return None

    payload = _get_json(f"{_API_BASE}/data/products/holdings/{port_id}?limit={_HOLDINGS_LIMIT}")
    raw_items = ((payload or {}).get("data") or {}).get("items") or []
    if not raw_items:
        return None

    items: list[dict[str, Any]] = []
    reference_date: str | None = None
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        raw_ticker = normalize_ticker(raw.get("ticker"))
        if not raw_ticker:
            continue
        try:
            weight = float(raw.get("marketValPercent") or 0.0)
        except (TypeError, ValueError):
            continue
        country_code = str(raw.get("countryCode") or "").strip().upper()
        item: dict[str, Any] = {
            "ticker": _to_component_ticker(raw_ticker, country_code),
            "name": str(raw.get("longName") or raw.get("name") or "").strip() or raw_ticker,
            "weight": weight,
        }
        if country_code:
            item["listing_country_code"] = country_code
        items.append(item)
        if not reference_date and raw.get("effectiveDate"):
            reference_date = str(raw.get("effectiveDate")).strip()

    if not items:
        return None

    items.sort(key=lambda x: x.get("weight") or 0.0, reverse=True)
    now = datetime.now()
    return {
        "source": "vanguard_au_api",
        "fetched_at": now.isoformat(),
        "as_of_date": reference_date or now.strftime("%Y-%m-%d"),
        "holdings_count": len(items),
        "holdings": items,
    }


def fetch_vanguard_au_expense_ratio_pct(ticker: str) -> float | None:
    """운용보수(MER)를 퍼센트 단위로 반환한다. 없으면 None."""
    port_id = resolve_port_id(ticker)
    if port_id is None:
        return None

    payload = _get_json(f"{_API_BASE}/products/adviser/fund/{port_id}/detail?limit=-1")
    entries = (payload or {}).get("data") or []
    if not entries:
        return None

    for fee_group in entries[0].get("fundFees") or []:
        for expense in (fee_group or {}).get("expenseType") or []:
            if str((expense or {}).get("code") or "").strip().upper() != _MER_EXPENSE_CODE:
                continue
            try:
                return float(expense.get("value"))
            except (TypeError, ValueError):
                return None
    return None
