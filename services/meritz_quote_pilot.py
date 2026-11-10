"""메리츠 시세 병행 검증. 수집 결과는 데이터 소스 화면의 진단에만 사용한다."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo

from utils.market_session import AFTERMARKET, CLOSED, DAYMARKET, PREMARKET, REGULAR, market_session

WS_URL = "wss://openapi.imeritz.com:29443/websocket"
MARKETS = {"kor": {"005930": "TJ", "069500": "J"}, "us": {"NVDA": "OQ", "SPMO": "AX"}}
_US_STREAMS = {
    DAYMARKET: "ohhts_exst_cntd",
    PREMARKET: "ohhts_exst_cntg",  # 프리마켓 수신 여부를 확인할 후보.
    REGULAR: "ohhts_exst_cntg",
    AFTERMARKET: "ohhts_exst_cnta",
}
_US_REST_SUFFIX = {DAYMARKET: "/daytime", AFTERMARKET: "/after", PREMARKET: "", REGULAR: ""}
_STATE_ID = "meritz_quote_pilot"
_KST = ZoneInfo("Asia/Seoul")


def _state():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("메리츠 시세 검증 기록에 연결할 수 없습니다.")
    return db.system_config


def load_quote_pilot() -> dict[str, Any]:
    doc = _state().find_one({"_id": _STATE_ID}) or {}
    return {
        country: {
            "latest": doc.get("latest", {}).get(country),
            "verified": doc.get("verified", {}).get(country, {}),
        }
        for country in MARKETS
    }


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if isfinite(result) else None


def _quote(price: Any, *, traded_at: str | None, reference: Any, change: Any) -> dict[str, Any] | None:
    number = _number(price)
    if number is None or number <= 0:
        return None
    return {
        "price": number,
        "traded_at": traded_at,
        "reference_price": _number(reference),
        "change_pct": _number(change),
        "received_at": datetime.now(_KST).isoformat(timespec="seconds"),
    }


def _rest_quotes(account_id: str, country: str, session: str, targets: list[str]) -> tuple[dict, list[str]]:
    from services.meritz_api_service import BrokerApiError, _request

    rows, issues = {}, []
    for ticker in targets:
        market = MARKETS[country][ticker]
        if country == "kor":
            path = "/market/v1/prices"
            params = {"mrkt_div_code": market, "iscd": ticker}
            fields = ("stck_prpr", "sdpr", "prdy_ctrt")
        else:
            path = "/market/v1/overseas/prices" + _US_REST_SUFFIX[session]
            params = {"mrkt_div_code": "OV", "mrkt_cls_code": market, "iscd": ticker, "dely_rltm_cls_code": "0"}
            fields = {
                DAYMARKET: ("dytm_prpr", None, "dytm_prdy_ctrt"),
                AFTERMARKET: ("mkt_aft_prpr", None, "mkt_aft_prdy_ctrt"),
            }.get(session, ("prpr", "prdy_clpr", "prdy_ctrt"))
        try:
            data = _request(account_id, "GET", path, success=frozenset({"0000"}), params=params)["data"]
            price, reference, change = (data.get(field) if field else None for field in fields)
            quote = _quote(price, traded_at=None, reference=reference, change=change)
            if quote:
                rows[ticker] = quote
        except (BrokerApiError, KeyError, AttributeError) as exc:
            issues.append(f"REST {ticker}: {type(exc).__name__}")
    return rows, issues


def _stream_targets(country: str, session: str, tickers: list[str]) -> dict[tuple[str, str], str]:
    if country == "us":
        return {(_US_STREAMS[session], f"{MARKETS[country][ticker]}@{ticker}"): ticker for ticker in tickers}
    return {
        ("ohhts_unt_cntg" if MARKETS[country][ticker] == "TJ" else "ohhts_stck_cntg", ticker): ticker
        for ticker in tickers
    }


def _trade_time(country: str, body: dict, received_at: datetime) -> datetime | None:
    if country == "kor":
        date, clock = received_at.strftime("%Y%m%d"), body.get("bsop_hour")
    else:
        date, clock = body.get("korea_date"), body.get("korea_hour")
    if not date or not clock:
        return None
    try:
        return datetime.strptime(f"{date}{clock}", "%Y%m%d%H%M%S").replace(tzinfo=_KST)
    except ValueError:
        return None


def _stream_quotes(token: str, country: str, session: str, tickers: list[str], seconds: int) -> tuple[dict, list[str]]:
    from websockets.exceptions import WebSocketException
    from websockets.sync.client import connect

    targets = _stream_targets(country, session, tickers)
    rows, issues = {}, []
    started = datetime.now(_KST)
    deadline = time.monotonic() + seconds
    for attempt in range(2):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            with connect(WS_URL, open_timeout=min(remaining, 8), close_timeout=2) as socket:
                for tr, key in targets:
                    socket.send(
                        json.dumps(
                            {
                                "header": {"token": f"Bearer {token}", "tr_type": "1"},
                                "body": {"tr_cd": tr, "tr_key": key},
                            }
                        )
                    )
                    time.sleep(0.4)
                while (remaining := deadline - time.monotonic()) > 0:
                    try:
                        frame = json.loads(socket.recv(timeout=remaining))
                    except TimeoutError:
                        break
                    header, body = frame.get("header") or {}, frame.get("body") or {}
                    code = header.get("rsp_cd")
                    if code:
                        if code != "00000":
                            issues.append(f"웹소켓 구독 오류 {code}")
                        continue
                    ticker = targets.get((header.get("tr_cd"), header.get("tr_key")))
                    if ticker is None or not isinstance(body, dict):
                        continue
                    now = datetime.now(_KST)
                    traded = _trade_time(country, body, now)
                    delayed = {"0": False, "1": True}.get(body.get("dely_yn"))
                    change_field = "prdy_ctrt" if country == "kor" else "rsfl_rt"
                    quote = _quote(
                        body.get("prpr"),
                        traded_at=traded.isoformat() if traded else None,
                        reference=None,
                        change=body.get(change_field),
                    )
                    if not quote:
                        continue
                    verified = (
                        traded is not None
                        and started - timedelta(seconds=2) <= traded <= now + timedelta(seconds=2)
                        and market_session(country, traded)["session"] == session
                        and (country == "kor" or delayed is False)
                    )
                    quote.update(
                        delayed=delayed,
                        verified=verified,
                        venue=body.get("exch_code"),
                        session_code=body.get("mkt_cls"),
                        market_date=body.get("acpl_date"),
                        business_date=body.get("acpl_bsop_date"),
                    )
                    if verified or not rows.get(ticker, {}).get("verified"):
                        rows[ticker] = quote
                    if all(rows.get(symbol, {}).get("verified") for symbol in tickers):
                        break
            break
        except (OSError, WebSocketException, ValueError) as exc:
            issues.append(f"웹소켓 연결 {attempt + 1}차: {type(exc).__name__}")
    return rows, list(dict.fromkeys(issues))


def _baseline_quotes(country: str, targets: list[str]) -> tuple[dict, list[str]]:
    from services.price_service import get_realtime_snapshot

    rows = {}
    for ticker, data in get_realtime_snapshot(country, targets).items():
        traded = data.get("localTradedAt") or data.get("tradeDateTime")
        quote = _quote(
            data.get("nowVal"),
            traded_at=str(traded) if traded else None,
            reference=data.get("prevClose"),
            change=data.get("changeRate"),
        )
        if quote:
            quote["last_regular_close"] = _number(data.get("lastRegularClose"))
            rows[ticker] = quote
    return rows, []


def collect_quote_pilot(account_id: str, country: str, seconds: int) -> dict[str, Any]:
    from services.meritz_api_service import _access_token
    from utils.market_session import is_instrument_trading_time

    now = datetime.now(_KST)
    session = market_session(country, now)["session"]
    if session == CLOSED:
        raise RuntimeError(f"{country} 시장이 닫혀 있어 체결 검증을 시작할 수 없습니다.")
    targets = [
        ticker
        for ticker in MARKETS[country]
        if country == "us" or is_instrument_trading_time("kor", "etf" if ticker == "069500" else "stock", now=now)
    ]
    if not targets:
        raise RuntimeError("현재 거래 가능한 검증 대상이 없습니다.")
    token = _access_token(account_id)
    report: dict[str, Any] = {"session": session, "targets": targets, "issues": []}
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {
            "websocket": pool.submit(_stream_quotes, token, country, session, targets, seconds),
            "rest": pool.submit(_rest_quotes, account_id, country, session, targets),
            "baseline": pool.submit(_baseline_quotes, country, targets),
        }
        for source, job in jobs.items():
            try:
                report[source], issues = job.result()
                report["issues"].extend(issues)
            except Exception as exc:
                report[source] = {}
                report["issues"].append(f"{source}: {type(exc).__name__}")
    report["checked_at"] = datetime.now(_KST).isoformat(timespec="seconds")
    update: dict[str, Any] = {"$set": {f"latest.{country}": report}}
    verified = {
        f"verified.{country}.{session}.{ticker}": quote["traded_at"]
        for ticker, quote in report["websocket"].items()
        if quote["verified"]
    }
    if verified:
        update["$max"] = verified
    _state().update_one({"_id": _STATE_ID}, update, upsert=True)
    return report
