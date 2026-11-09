"""메리츠증권 Open API — 계좌 잔고 조회(국내·미국 계좌)와 국내주식 호가·지정가 주문·취소·주문내역(사고팔기).

키는 증권사 계좌 하나에 묶여 있다. 그래서 키 이름이 우리 계좌 ID 를 따르고(`{계좌ID}_API_KEY/SECRET`,
`broker_api_service.require_env_keys`), 요청에 계좌번호를 넣지 않는다. 모의투자 서버는 없다(운영만).

- 오류는 종류와 무관하게 HTTP 500 으로 온다 — 성공·실패는 본문 `rsp_cd` 로만 판정한다. 성공 코드는 API 계열마다
  다르다(잔고·주문내역 5766, 호가 0000, 주문·취소 0001).
- 접근토큰(유효 12시간)은 계좌별로 DB(`system_config`) 문서 하나에 공유한다 — 로컬·서버 워커가
  각자 발급하지 않게 한다(나무증권과 같은 이유).
- 모든 요청에 `mac_address`(호출 단말 MAC, 12자리) 헤더가 필수다. 환경변수 `MERITZ_MAC_ADDRESS` 로 정한다 —
  프로세스에서 읽는 MAC 은 macOS 사설 주소·컨테이너 재생성으로 실행마다 달라져 단말을 대표하지 못한다.
- 초당 호출 한도는 문서에 숫자가 없다 — 나무증권과 같이 호출 사이를 1초 띄운다.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any

import requests

from services.broker_api_service import MERITZ_PROVIDER, BrokerApiError, OrderUncertainError, require_env_keys
from utils.logger import get_app_logger

logger = get_app_logger()

BASE_URL = "https://openapi.imeritz.com:9443"
TOKEN_PATH = "/oauth2/token"
VALUATION_PATH = "/accounts/v1/me/valuation"
HOLDINGS_PATH = "/accounts/v1/me/holdings"
OVERSEAS_HOLDINGS_PATH = "/accounts/v1/me/overseas/holdings"
CURRENCIES_PATH = "/accounts/v1/me/overseas/currencies"
ORDERBOOK_PATH = "/market/v1/orderbook"
ORDER_PATH = {"buy": "/trading/v1/orders/buy", "sell": "/trading/v1/orders/sell"}
CANCEL_PATH = "/trading/v1/orders/cancel"
ORDER_HISTORY_PATH = "/trading/v1/orders/history"

# 조회 계열 성공 — 완료·연속조회 더 있음·자료 없음. 이 밖의 rsp_cd 는 전부 오류다.
_INQUIRY_CODES = frozenset({"5766", "5762", "5820"})
_QUOTE_CODES = frozenset({"0000"})
_ORDER_CODES = frozenset({"0001"})
# 토큰 만료·무효 — 받으면 한 번만 새로 발급해 다시 보낸다.
_TOKEN_ERROR_CODES = frozenset({"EGW00121", "EGW00123"})
# 잔고 행의 처리대상구분 — 국내주식("1")·해외주식("J")이 한 목록에 섞여 온다.
_DOMESTIC_ROW = "1"
# 해외 잔고구분 — "1" 일반, "2" 소수, "3" 대출.
_US_GENERAL_ROW = "1"

# 주문은 KRX 정규장 지정가만 낸다 — 사고팔기는 KRX 정규장 규칙이다(나무 `ORDER_MARKET` 과 같다).
_EXCHANGE_KRX = "01"
_ORDERBOOK_KRX = "J"
_LIMIT_ORDER = "01"
_NO_CONDITION = "0"
# 주문 응답의 경고구분 — "0" 정상 접수, 아래 '보류'는 접수 여부를 단정할 수 없다. 그 밖은 경고(미접수, 사람이 확인).
_WARN_OK = "0"
_WARN_HOLD = frozenset({"2", "5", "8", "b", "e", "h"})
# 주문내역 행 — 정정취소구분(1 신규·2 정정·3 취소), 매매구분(1 매도·2 매수), 처리상태(3 거부).
_ROW_CANCEL = "3"
_SIDE_LABELS = {"1": "매도", "2": "매수"}
_STATE_REJECTED = "3"

_MIN_INTERVAL_SECONDS = 1.0
_REQUEST_TIMEOUT_SECONDS = 10
_MAX_PAGES = 20
# 만료 직전 토큰은 쓰지 않는다 — 요청 도중 만료되지 않게 여유를 둔다.
_TOKEN_MARGIN_SECONDS = 60

_call_lock = threading.Lock()
_last_call_at = 0.0


MAC_ENV = "MERITZ_MAC_ADDRESS"
_MAC_RE = re.compile(r"[0-9A-F]{12}")


def _mac_address() -> str:
    """요청 헤더 `mac_address` — 환경변수의 MAC(대문자 16진수 12자리). 없거나 형식이 다르면 오류다."""
    value = os.environ.get(MAC_ENV, "").strip()
    if not _MAC_RE.fullmatch(value):
        raise BrokerApiError(f".env 에 {MAC_ENV}(대문자 16진수 12자리, 예: A1B2C3D4E5F6)가 필요합니다.")
    return value


# ── 토큰(DB 공유) ─────────────────────────────────────────────────────────


def _token_collection():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise BrokerApiError("메리츠증권 토큰 공유 DB 연결에 실패했습니다.")
    return db["system_config"]


def _token_doc_id(account_id: str) -> str:
    return f"meritz_token:{account_id}"


def _issue_token(account_id: str) -> str:
    """새 토큰을 발급해 DB 에 올린다."""
    key, secret = require_env_keys(MERITZ_PROVIDER, account_id)
    response = requests.post(
        f"{BASE_URL}{TOKEN_PATH}",
        data={"grant_type": "client_credentials", "client_id": key, "client_secret": secret, "scope": "login"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    data = _json_of(response)
    token = data.get("access_token")
    expires_in = data.get("expires_in")
    if not response.ok or not token or not expires_in:
        raise BrokerApiError(f"메리츠증권 토큰 발급 실패 ({account_id}): {_message_of(data, response)}")
    exp = time.time() + float(expires_in)
    _token_collection().update_one(
        {"_id": _token_doc_id(account_id)}, {"$set": {"token": token, "exp": exp}}, upsert=True
    )
    logger.info("[MERITZ] %s 토큰 발급 (만료 %s)", account_id, time.strftime("%m-%d %H:%M", time.localtime(exp)))
    return str(token)


def _access_token(account_id: str) -> str:
    """DB 에 공유된 유효 토큰 — 없거나 곧 만료면 새로 발급한다."""
    doc = _token_collection().find_one({"_id": _token_doc_id(account_id)}) or {}
    if doc.get("token") and float(doc.get("exp") or 0) > time.time() + _TOKEN_MARGIN_SECONDS:
        return str(doc["token"])
    return _issue_token(account_id)


# ── 호출 ─────────────────────────────────────────────────────────────────


def _json_of(response: requests.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _message_of(data: dict[str, Any], response: requests.Response) -> str:
    parts = [str(data[key]) for key in ("rsp_msg", "rsp_sub_msg", "error_description") if data.get(key)]
    return " · ".join(parts) or f"HTTP {response.status_code}"


def _request(
    account_id: str,
    method: str,
    path: str,
    *,
    success: frozenset[str],
    params: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """요청 한 건 → 본문. 토큰 오류면 한 번만 재발급해 다시 보낸다. ``success`` 밖의 rsp_cd 는 BrokerApiError.

    주문(POST)이 서버에 닿았는지 모르는 네트워크 오류는 `OrderUncertainError` 다 — 들어갔을 수도 있다.
    """
    global _last_call_at
    token = _access_token(account_id)
    mac = _mac_address()
    with _call_lock:
        for attempt in range(2):
            wait = _MIN_INTERVAL_SECONDS - (time.monotonic() - _last_call_at)
            if wait > 0:
                time.sleep(wait)
            headers = {"authorization": f"Bearer {token}", "mac_address": mac}
            try:
                if method == "GET":
                    response = requests.get(
                        f"{BASE_URL}{path}", headers=headers, params=params, timeout=_REQUEST_TIMEOUT_SECONDS
                    )
                else:
                    response = requests.post(
                        f"{BASE_URL}{path}",
                        headers={**headers, "content-type": "application/json; charset=utf-8"},
                        json=body,
                        timeout=_REQUEST_TIMEOUT_SECONDS,
                    )
            except requests.RequestException as exc:
                error = OrderUncertainError if method == "POST" else BrokerApiError
                raise error(f"메리츠증권 서버 응답을 받지 못했습니다: {exc}") from exc
            finally:
                _last_call_at = time.monotonic()
            data = _json_of(response)
            code = str(data.get("rsp_cd") or "")
            if (response.status_code == 401 or code in _TOKEN_ERROR_CODES) and attempt == 0:
                token = _issue_token(account_id)
                continue
            if code in success:
                return data
            raise BrokerApiError(f"메리츠증권 API 오류: {_message_of(data, response)} (코드 {code or '-'})")
    raise AssertionError("unreachable")


def _get(account_id: str, path: str, params: dict[str, str]) -> dict[str, Any]:
    """조회 계열 GET 한 건 → 본문(완료·연속·자료 없음)."""
    return _request(account_id, "GET", path, success=_INQUIRY_CODES, params=params)


def _get_paged(account_id: str, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
    """연속조회 — `data` 배열을 전 페이지 이어 붙인다. 자료 없음(5820)은 빈 목록이다."""
    rows: list[dict[str, Any]] = []
    page_params = dict(params)
    for _ in range(_MAX_PAGES):
        data = _get(account_id, path, page_params)
        rows.extend(data.get("data") or [])
        if str(data.get("tr_cont") or "0") != "1":
            return rows
        next_key = data.get("tr_cont_key")
        if not next_key:
            raise BrokerApiError("메리츠증권 연속조회 키가 없습니다 — 응답을 확인하세요.")
        page_params = {**params, "tr_cont": "1", "tr_cont_key": str(next_key)}
    raise BrokerApiError(f"메리츠증권 연속조회가 {_MAX_PAGES}페이지를 넘었습니다 — 응답을 확인하세요.")


# ── 잔고 ─────────────────────────────────────────────────────────────────


def _domestic_ticker(iscd: str) -> str:
    """국내 종목코드 "A005930" → "005930". 형식이 다르면 오류다."""
    if len(iscd) != 7 or not iscd.startswith("A"):
        raise BrokerApiError(f"메리츠증권 국내 종목코드 형식이 예상과 다릅니다: {iscd!r}")
    return iscd[1:]


def _require_unique(ticker: str, name: Any, seen: set[str]) -> None:
    """같은 종목이 여러 행(일반·신용·소수 등)이면 평단을 합쳐야 한다 — 임의로 합치지 않고 알린다."""
    if ticker in seen:
        raise BrokerApiError(f"같은 종목의 잔고가 여러 행입니다(일반·신용 등): {name}({ticker})")
    seen.add(ticker)


def _domestic_holdings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """국내 잔고 행(통합 잔고 목록) → 표준 보유 목록. 해외 행이 섞여 있으면 막는다(계좌 국가 하나)."""
    holdings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        iscd = str(row.get("iscd") or "").strip()
        if str(row.get("pros_trgt_cls_code") or "") != _DOMESTIC_ROW:
            raise BrokerApiError(
                f"국내 계좌에 해외 잔고가 있습니다: {row.get('isnm')}({iscd}) — 계좌 국가 설정을 확인하세요."
            )
        quantity = int(float(row.get("rmnd_qty") or 0))
        if quantity <= 0:
            continue
        ticker = _domestic_ticker(iscd)
        _require_unique(ticker, row.get("isnm"), seen)
        holdings.append(
            {
                "ticker": ticker,
                "name": str(row.get("isnm") or "").strip(),
                "quantity": quantity,
                "average_buy_price": float(row.get("byng_unpr") or 0),
                "current_price": float(row.get("psnt_prc") or 0) or None,
                "value": float(row.get("vltn_amt") or 0) or None,
            }
        )
    return holdings


def _us_holdings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """해외 잔고 행 → 표준 보유 목록(단가는 USD). 일반 잔고·USD·정수 수량만 받는다.

    티커는 거래소 접미사를 뗀 심볼(`ovrs_stck_symb_iscd`, "BRK.B")을 공통 시세 심볼("BRK-B")로 바꾼다.
    소수점 잔고는 자산 계산이 수량을 정수로 내려 평가가 빠지므로 막는다.
    """
    from utils.normalization import normalize_exchange_symbol

    holdings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        symbol = str(row.get("ovrs_stck_symb_iscd") or "").strip()
        name = str(row.get("isnm") or "").strip()
        if str(row.get("rmnd_clcd") or "") != _US_GENERAL_ROW:
            raise BrokerApiError(f"일반 잔고가 아닌 해외 잔고({row.get('prod_cls_name')})가 있습니다: {name}({symbol})")
        if str(row.get("crcd") or "") != "USD":
            raise BrokerApiError(f"USD 가 아닌 해외 잔고({row.get('crcd')})가 있습니다: {name}({symbol})")
        quantity = float(row.get("rmnd_dqty") or 0)
        if quantity != int(quantity):
            raise BrokerApiError(f"소수점 수량 잔고({quantity})는 동기화하지 않습니다: {name}({symbol})")
        if quantity <= 0:
            continue
        if not symbol:
            raise BrokerApiError(f"해외 잔고의 심볼이 비어 있습니다: {name}({row.get('iscd')})")
        ticker = normalize_exchange_symbol(symbol, suffix="")
        _require_unique(ticker, name, seen)
        holdings.append(
            {
                "ticker": ticker,
                "name": name,
                "quantity": int(quantity),
                "average_buy_price": float(row.get("avrg_unpr") or 0),
                "current_price": float(row.get("psnt_prc") or 0) or None,
                "value": float(row.get("frcr_vltn_amt") or 0) or None,
            }
        )
    return holdings


def _foreign_cash(account_id: str) -> dict[str, float]:
    """통화별 외화 예수금 — 추정 외화 예수금(`estm_frcr_amt`, 미결제 반영)이다. 원화 D+2 와 같은 기준."""
    rows = _get(account_id, CURRENCIES_PATH, {}).get("data") or []
    cash: dict[str, float] = {}
    for row in rows:
        code = str(row.get("crcd") or "").strip().upper()
        if not code or code in cash:
            raise BrokerApiError(f"메리츠증권 외화 잔고의 통화가 비었거나 겹칩니다: {code!r}")
        cash[code] = float(row.get("estm_frcr_amt") or 0)
    return cash


def fetch_balance(account_id: str, country_code: str) -> dict[str, Any]:
    """계좌 잔고 → 표준형(통화별 현금·총자산·보유). 계좌 국가(kor·us)의 잔고만 받는다.

    - 원화 현금은 D+2 예수금(`d2_aft_tfam`, 미결제 반영 — 나무 `nxt2_dd_dca` 와 같은 기준),
      외화는 추정 외화 예수금(`estm_frcr_amt`). 외화 예수금이 있을 때만 외화 잔고를 조회한다.
    - 잔고 목록은 국내·해외가 한 목록에 섞여 오므로, 계좌 국가와 다른 행이 있으면 막는다.
      미국 계좌의 종목 상세(평균단가 USD 등)는 해외주식 잔고 조회에서 받는다.
    """
    valuation = _get(account_id, VALUATION_PATH, {}).get("data") or {}
    if not valuation:
        raise BrokerApiError("메리츠증권 자산평가 응답이 비어 있습니다.")
    cash_map = {"KRW": float(valuation["d2_aft_tfam"])}
    if float(valuation.get("frcr_tfam") or 0) != 0:
        foreign = _foreign_cash(account_id)
        if "KRW" in foreign:
            raise BrokerApiError("메리츠증권 외화 잔고에 KRW 가 있습니다 — 응답을 확인하세요.")
        cash_map.update(foreign)

    combined = _get_paged(account_id, HOLDINGS_PATH, {"nthn_rmnd_incu_yn": "N"})
    if country_code == "kor":
        holdings = _domestic_holdings(combined)
    elif country_code == "us":
        domestic = [row for row in combined if str(row.get("pros_trgt_cls_code") or "") == _DOMESTIC_ROW]
        if domestic:
            names = ", ".join(f"{row.get('isnm')}({row.get('iscd')})" for row in domestic)
            raise BrokerApiError(f"미국 계좌에 국내 잔고가 있습니다: {names} — 계좌 국가 설정을 확인하세요.")
        holdings = _us_holdings(_get_paged(account_id, OVERSEAS_HOLDINGS_PATH, {}))
    else:
        raise BrokerApiError(f"메리츠증권 잔고 동기화는 국내(kor)·미국(us) 계좌만 지원합니다: {country_code}")
    return {
        "cash_map": cash_map,
        "total_asset": float(valuation["aset_vltn_amt"]),
        "holdings": holdings,
    }


# ── 국내주식 주문(사고팔기) ─────────────────────────────────────────────────


def fetch_best_quote(account_id: str, ticker: str) -> dict[str, Any]:
    """KRX 1호가 — 매도·매수 1호가와 호가 시각. 호가가 비어 있으면 주문 가격을 못 정해 오류다(나무와 같은 표준형)."""
    data = (
        _request(
            account_id,
            "GET",
            ORDERBOOK_PATH,
            success=_QUOTE_CODES,
            params={"mrkt_div_code": _ORDERBOOK_KRX, "iscd": ticker},
        ).get("data")
        or {}
    )
    # 없는 종목코드도 정상 코드로 온다 — 응답의 단축종목코드로 확인한다.
    if str(data.get("shrn_iscd") or "").strip() != ticker:
        raise BrokerApiError(f"메리츠증권 호가 응답의 종목이 다릅니다: {ticker} → {data.get('shrn_iscd')!r}")
    result = {
        "ask": int(data["askp"]),
        "bid": int(data["bidp"]),
        "at": str(data["prpr_hour"]),
    }
    if result["ask"] <= 0 or result["bid"] <= 0:
        raise BrokerApiError(
            f"{ticker} 의 매도·매수 1호가가 없어 주문 가격을 정할 수 없습니다 (매도 {result['ask']}, 매수 {result['bid']})."
        )
    return result


def _accepted_order_no(data: dict[str, Any], what: str) -> str:
    """주문·취소 응답에서 주문번호를 꺼낸다 — 정상 접수(경고구분 "0")이고 번호가 있어야 접수된 것이다.

    경고는 증권사가 사람의 확인을 요구한 것이라 자동으로 재전송하지 않는다(미접수). 보류는 접수 여부를 모른다.
    """
    output = data.get("data") or {}
    warn = str(output.get("warn_cls_code") or "")
    message = str(output.get("warn_msg") or data.get("rsp_msg") or "").strip()
    if warn in _WARN_HOLD:
        raise OrderUncertainError(
            f"{what}이 보류됐습니다 — 증권사 주문내역에서 확인하세요: {message} (경고구분 {warn})"
        )
    if warn != _WARN_OK:
        raise BrokerApiError(f"{what}이 접수되지 않았습니다(증권사 경고): {message} (경고구분 {warn or '-'})")
    order_no = output.get("oder_no")
    if not isinstance(order_no, (int, float)) or int(order_no) <= 0:
        raise OrderUncertainError(f"{what} 응답에 주문번호가 없습니다 — 증권사 주문내역에서 확인하세요: {message}")
    return str(int(order_no))


def place_limit_order(account_id: str, side: str, ticker: str, quantity: int, price: int) -> dict[str, str]:
    """KRX 지정가 주문 한 건 — 접수되면 주문번호를 돌려준다. side 는 buy·sell."""
    if side not in ORDER_PATH:
        raise BrokerApiError(f"알 수 없는 주문 구분입니다: {side}")
    if quantity <= 0 or price <= 0:
        raise BrokerApiError("수량과 가격은 0보다 커야 합니다.")
    data = _request(
        account_id,
        "POST",
        ORDER_PATH[side],
        success=_ORDER_CODES,
        body={
            "iscd": f"A{ticker}",
            "oder_unpr": price,
            "odqt": quantity,
            "oder_cls_code": _LIMIT_ORDER,
            "oder_cond_cls_code": _NO_CONDITION,
            "orgl_oder_no": 0,
            "whol_rctf_cncl_yn": "N",
            "warn_cnfr_yn": "N",
            "exch_kind_code": _EXCHANGE_KRX,
        },
    )
    return {"order_no": _accepted_order_no(data, "주문"), "message": str(data.get("rsp_msg") or "")}


def cancel_order(account_id: str, order_no: str, ticker: str) -> dict[str, str]:
    """주문을 전량 취소한다 — 돌려주는 번호는 취소 요청 자체의 주문번호다."""
    data = _request(
        account_id,
        "POST",
        CANCEL_PATH,
        success=_ORDER_CODES,
        body={
            "iscd": f"A{ticker}",
            "orgl_oder_no": int(order_no),
            "whol_rctf_cncl_yn": "Y",
            "warn_cnfr_yn": "N",
            "exch_kind_code": _EXCHANGE_KRX,
        },
    )
    return {"cancel_order_no": _accepted_order_no(data, "취소"), "message": str(data.get("rsp_msg") or "")}


def _order_status(row: dict[str, Any]) -> str:
    """주문내역 한 줄 → 주문 상태(pending·partial·filled·canceled·rejected) — 나무 `order_status` 와 같은 판정."""
    quantity = float(row.get("odqt") or 0)
    filled = float(row.get("total_cntg_qty") or 0)
    if str(row.get("oder_pros_stat_cls_code") or "") == _STATE_REJECTED:
        return "rejected"
    if quantity > 0 and filled >= quantity:
        return "filled"
    if float(row.get("cncl_qty") or 0) > 0:
        return "canceled"
    if filled > 0:
        return "partial"
    return "pending"


def _day_order_row(row: dict[str, Any]) -> dict[str, Any]:
    """주문내역 한 줄 → 표준 주문 행(나무 `_day_order_row` 와 같은 모양). `origin_order_no` 는 정정 행만 0 이 아니다."""
    side = str(row.get("deal_cls_code") or "")
    if side not in _SIDE_LABELS:
        raise BrokerApiError(f"메리츠증권 주문내역의 매매구분을 알 수 없습니다: {side!r}")
    return {
        "order_no": str(int(row.get("oder_no") or 0)),
        "origin_order_no": str(int(row.get("orgl_oder_no") or 0)),
        "ticker": _domestic_ticker(str(row.get("iscd") or "").strip()),
        "name": str(row.get("kor_issu_abwr_name") or "").strip(),
        "side_label": _SIDE_LABELS[side],
        "quantity": int(float(row.get("odqt") or 0)),
        "price": float(row.get("oder_unpr") or 0),
        "filled_quantity": int(float(row.get("total_cntg_qty") or 0)),
        "unfilled_quantity": int(float(row.get("ncnt_qty") or 0)),
        "canceled_quantity": int(float(row.get("cncl_qty") or 0)),
        "average_fill_price": float(row.get("cntg_unpr") or 0),
        "status": _order_status(row),
    }


def day_order_rows(account_id: str, yyyymmdd: str) -> list[dict[str, Any]]:
    """그날의 주문 행(주문내역 조회) — 표준 주문 행이고 정정 행은 합치지 않았다(`broker_trade_service.merge_amendments`).

    취소 요청 행은 뺀다 — 취소 결과(취소수량)는 취소된 주문 행 자신에 남는다.
    """
    rows = _get_paged(
        account_id,
        ORDER_HISTORY_PATH,
        {"oder_date": yyyymmdd, "deal_cls_code": "0", "cntg_cls_code": "0", "sort_stnd_code": "2"},
    )
    return [_day_order_row(row) for row in rows if str(row.get("rctf_cncl_cls_code") or "") != _ROW_CANCEL]
