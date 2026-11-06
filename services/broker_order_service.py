"""나무증권 주문 — 계좌 설정의 증권사 연동 계좌로 **직접 요청**하는 주문·취소·조회.

SDK(`nhplug`)는 호출 대상 서버를 프로세스 전역 환경변수(`NHPLUG_BASE_URL`)로 읽는다. 요청 처리 중에 그 값을 바꾸면
같은 프로세스에서 도는 운영 조회(잔고 동기화 등)가 다른 서버로 새므로, 이 모듈은 SDK 를 거치지 않고 `requests` 로
계좌의 서버(모의 `MOCK_BASE_URL`·운영 `LIVE_BASE_URL`)에 직접 보낸다. 접근토큰은 SDK 호출과 같은 DB 공유 토큰이다
(`broker_api_service.namu_access_token`) — 키당 유효 토큰이 1개라 따로 발급하면 서로 죽인다.

안전 장치: 모든 호출은 `account_env` 를 먼저 통과한다 — 계좌 설정에 연동된 증권사 계좌이고, 그 계좌의 구분(운영·모의)이
증권사의 계좌구분(`03` 모의투자)과 맞아야 한다. 서버는 이 구분으로만 정한다(요청이 서버를 고르지 못한다).
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import requests

from services.broker_api_service import (
    BALANCE_PATH,
    MOCK_ACCOUNT_TYPE,
    BrokerApiError,
    account_types,
    balance_request,
    is_business_success,
    namu_access_token,
    namu_app_credentials,
    namu_invalidate_token,
    parse_balance,
)
from utils.logger import get_app_logger
from utils.ttl_cache import TtlCache

logger = get_app_logger()

MOCK_BASE_URL = "https://moapi.nhplug.com:8443"
LIVE_BASE_URL = "https://api.nhplug.com:8443"
#: 주문을 보내는 시장 — KRX 로 고정한다. SOR 는 KRX·NXT 로 쪼개져 주문번호가 둘이 되고, 사고팔기는 KRX 정규장 규칙이다.
ORDER_MARKET = "KRX"
_PRICE_TYPE_LIMIT = "01"  # 호가유형 — 보통가(지정가)
_NO_CONDITION = "00"  # 주문조건 없음 / 공매도 구분 정상

ORDER_PATH = {"buy": "/krstock/order/v1/cashBuy", "sell": "/krstock/order/v1/cashSell"}
CANCEL_PATH = "/krstock/order/v1/cancel"
# 체결 조회에 그날 주문이 하나도 없을 때 오는 응답 코드(데이터가 존재하지 않습니다) — 오류가 아니라 빈 목록이다.
NO_DATA_CODE = "11512"
DAY_ORDERS_PATH = "/krstock/inquiry/v1/dailyOrderExecution"
# 서킷브레이커(CB) 발동 중 주문 거부 코드 — 거래소 제도상 매매가 20분 중단된 뒤 10분 단일가 매매로 재개된다.
CIRCUIT_BREAKER_CODE = "21824"
_CIRCUIT_BREAKER_RESUME = timedelta(minutes=30)
_KST = ZoneInfo("Asia/Seoul")

# 모든 호출을 최소 1초 간격으로 직렬화한다 — 운영 쪽(`_CALLS_PER_SECOND`)과 같은 보수적 간격이다.
_MIN_INTERVAL_SECONDS = 1.0
_MAX_PAGES = 20
_REQUEST_TIMEOUT_SECONDS = 10

_call_lock = threading.Lock()
_last_call_at = 0.0
_ACCOUNT_TYPE_CACHE = TtlCache(300, name="broker-account-types")


class OrderUncertainError(BrokerApiError):
    """요청이 서버에 닿았는지 알 수 없는 네트워크 오류 — 주문이 들어갔을 수도 있어 실패로 단정하면 안 된다."""


def linked_account_doc(account_no: str) -> dict[str, Any]:
    """증권사 계좌가 연동된 계좌 설정 문서 — 연동되지 않은 증권사 계좌면 막는다."""
    from utils.account_settings_store import load_account_docs

    doc = next((d for d in load_account_docs() if (d.get("broker_api") or {}).get("account_no") == account_no), None)
    if doc is None:
        raise BrokerApiError("계좌 설정에 연동된 증권사 계좌가 아닙니다.")
    return doc


def account_env(account_no: str) -> str:
    """증권사 계좌가 연동된 계좌 설정의 구분(live·mock) — 이 모듈의 모든 호출이 가장 먼저 통과하는 관문이다.

    계좌 설정에 연동되지 않은 증권사 계좌이거나, 구분이 증권사 계좌구분과 어긋나면 막는다.
    """
    from utils.account_settings_store import ACCOUNT_ENV_MOCK

    doc = linked_account_doc(account_no)
    key = _ACCOUNT_TYPE_CACHE.make_key("types")
    types = _ACCOUNT_TYPE_CACHE.get(key)
    if types is None:
        types = account_types()
        _ACCOUNT_TYPE_CACHE.set(key, types)
    if (doc["env"] == ACCOUNT_ENV_MOCK) != (types.get(account_no) == MOCK_ACCOUNT_TYPE):
        raise BrokerApiError("계좌 설정의 구분과 증권사 계좌구분(03=모의투자)이 맞지 않습니다.")
    return str(doc["env"])


def _base_url(account_no: str) -> str:
    from utils.account_settings_store import ACCOUNT_ENV_MOCK

    return MOCK_BASE_URL if account_env(account_no) == ACCOUNT_ENV_MOCK else LIVE_BASE_URL


def _message_of(data: dict[str, Any]) -> str:
    """오류 본문에서 사람이 읽을 메시지 — 업무 계층(rsp_msg)과 게이트웨이(error_description) 둘 다 훑는다."""
    for key in ("rsp_msg", "error_description", "message"):
        if data.get(key):
            return str(data[key])
    error = data.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return ""


def _post(base_url: str, path: str, payload: dict[str, Any], *, cts: str | None = None, cts_flag: str | None = None):
    """계좌의 서버로 한 건 보낸다 → (본문, 응답 헤더). HTTP 오류는 BrokerApiError 로 올린다."""
    global _last_call_at
    app_key, app_secret = namu_app_credentials()
    token = namu_access_token()
    body = json.dumps({"Input_0": payload})
    with _call_lock:
        response = None
        for attempt in range(2):
            headers = {
                "x-client-id": app_key,
                "x-client-secret": app_secret,
                "authorization": f"Bearer {token}",
                "content-type": "application/json; charset=UTF-8",
            }
            if cts:
                headers["cts"] = cts
            if cts_flag:
                headers["cts_flag"] = cts_flag
            wait = _MIN_INTERVAL_SECONDS - (time.monotonic() - _last_call_at)
            if wait > 0:
                time.sleep(wait)
            try:
                response = requests.post(
                    f"{base_url}{path}", headers=headers, data=body, timeout=_REQUEST_TIMEOUT_SECONDS
                )
            except requests.RequestException as exc:
                raise OrderUncertainError(f"증권사 서버 응답을 받지 못했습니다: {exc}") from exc
            finally:
                _last_call_at = time.monotonic()
            # 토큰이 무효(401)면 새로 받아 한 번만 다시 보낸다 — 다른 오류에는 재발급하지 않는다.
            if response.status_code == 401 and attempt == 0:
                token = namu_invalidate_token()
                continue
            break
    try:
        data = response.json()
    except ValueError:
        data = {}
    if not response.ok:
        raise BrokerApiError(f"나무증권 API 오류: {_message_of(data) or f'HTTP {response.status_code}'}")
    return data, response.headers


def _post_paged(
    base_url: str, path: str, payload: dict[str, Any], list_key: str, *, no_data_code: str | None
) -> dict[str, Any]:
    """연속조회 — 목록(list_key)을 전 페이지 이어 붙인다. 마지막 페이지만 업무 성공을 판정한다.

    `no_data_code` 는 "조회할 데이터가 없다"는 응답 코드다 — 오류가 아니라 빈 목록으로 본다(없으면 None).
    """
    merged: dict[str, Any] = {}
    items: list[Any] = []
    cts = cts_flag = None
    seen: set[str] = set()
    for page in range(1, _MAX_PAGES + 1):
        data, headers = _post(base_url, path, payload, cts=cts, cts_flag=cts_flag)
        next_cts = (headers.get("cts") or "").strip()
        has_next = (headers.get("cts_flag") or "").strip() == "Y" and bool(next_cts) and next_cts not in seen
        code = str(data["rsp_cd"]) if data.get("rsp_cd") is not None else None
        no_data = no_data_code is not None and code == no_data_code
        if not has_next and not no_data and not is_business_success(code, data.get("rsp_msg")):
            raise BrokerApiError(f"나무증권 API 오류: {_message_of(data) or '업무 오류'} (코드 {data.get('rsp_cd')})")
        if not merged:
            merged = {key: value for key, value in data.items() if key != list_key}
        items.extend(data.get(list_key) or [])
        if not has_next:
            merged[list_key] = items
            return merged
        seen.add(next_cts)
        cts, cts_flag = next_cts, "Y"
    raise BrokerApiError(f"나무증권 API 연속조회가 {_MAX_PAGES}페이지를 넘었습니다 — 응답을 확인하세요.")


def fetch_order_balance(account_no: str) -> dict[str, Any]:
    """계좌 잔고 — 계좌 설정의 잔고 동기화와 같은 표준형(예수금 D+2·총자산·보유)이다."""
    return parse_balance(
        _post_paged(_base_url(account_no), BALANCE_PATH, balance_request(account_no), "Output_1", no_data_code=None)
    )


def _circuit_breaker_message(what: str, data: dict[str, Any]) -> str:
    """CB 발동 중 거부 안내 — 발동 시각을 모르므로 지금(주문 시도 시각)부터 중단 20분 + 단일가 10분을 더한 시각을 준다.

    발동은 시도보다 앞서므로 실제 재개는 이 시각보다 빠르거나 같다.
    """
    resume = (datetime.now(_KST) + _CIRCUIT_BREAKER_RESUME).strftime("%H:%M")
    return (
        f"{what}이 접수되지 않았습니다: 서킷브레이커(CB) 발동 중이라 주문할 수 없습니다. "
        f"매매 중단 20분 뒤 10분 단일가 매매로 재개되니 대략 {resume} 이후에 다시 시도하세요 "
        f"(증권사 응답: {_message_of(data)}, 코드 {data.get('rsp_cd')})"
    )


def _accepted_order_no(data: dict[str, Any], what: str) -> str:
    """주문·취소 응답에서 시장주문번호를 꺼낸다 — **번호가 있어야 접수된 것이다**.

    응답 코드는 매수 00048·매도 00047·취소 00192 로 문서에 나와 있지만, 접수의 증거는 번호다.
    번호 없이 성공으로 두면 접수되지 않은 주문을 기록이 가진 채로 남고, 반대로 코드만 보고 실패로 두면
    실제로는 들어간 주문을 기록에서 놓친다.
    """
    output = data.get("Output_0") or {}
    order_no = output.get("mkt_orr_no") if isinstance(output, dict) else None
    if not isinstance(order_no, (int, float)) or int(order_no) <= 0:
        if str(data.get("rsp_cd")) == CIRCUIT_BREAKER_CODE:
            raise BrokerApiError(_circuit_breaker_message(what, data))
        raise BrokerApiError(
            f"{what}이 접수되지 않았습니다: {_message_of(data) or '응답에 주문번호가 없습니다'} (코드 {data.get('rsp_cd')})"
        )
    return str(int(order_no))


def place_limit_order(account_no: str, side: str, ticker: str, quantity: int, price: int) -> dict[str, str]:
    """지정가 주문 한 건 — 접수되면 시장주문번호를 돌려준다. side 는 buy·sell."""
    base_url = _base_url(account_no)
    if side not in ORDER_PATH:
        raise BrokerApiError(f"알 수 없는 주문 구분입니다: {side}")
    if quantity <= 0 or price <= 0:
        raise BrokerApiError("수량과 가격은 0보다 커야 합니다.")
    data, _ = _post(
        base_url,
        ORDER_PATH[side],
        {
            "act_no": account_no,
            "iem_cd": ticker,
            "orr_qty": quantity,
            "orr_pr": price,
            "orr_amt": quantity * price,
            "nmn_pr_tp_cd": _PRICE_TYPE_LIMIT,
            "orr_cnd_dit_cd": _NO_CONDITION,
            "ssl_nmn_pr_dit_cd": _NO_CONDITION,
            "rmt_mkt_cd": ORDER_MARKET,
            "sor_mkt_sli_yn": "N",
        },
    )
    return {"order_no": _accepted_order_no(data, "주문"), "message": str(data.get("rsp_msg") or "")}


def cancel_order(account_no: str, order_no: str, ticker: str) -> dict[str, str]:
    """원주문을 전량 취소한다 — 돌려주는 번호는 취소 요청 자체의 주문번호다."""
    data, _ = _post(
        _base_url(account_no),
        CANCEL_PATH,
        {"act_no": account_no, "org_mkt_orr_no": int(order_no), "all_pat_dit_cd": "1", "iem_cd": ticker},
    )
    return {"cancel_order_no": _accepted_order_no(data, "취소"), "message": str(data.get("rsp_msg") or "")}


def order_status(row: dict[str, Any]) -> str:
    """체결 조회 한 줄 → 주문 상태(pending·partial·filled·canceled·rejected)."""
    quantity = float(row.get("orr_qty") or 0)
    filled = float(row.get("tot_cns_qty") or 0)
    canceled = float(row.get("can_qty") or 0)
    if "거부" in str(row.get("orr_rjt_rsn_cd_nm") or ""):
        return "rejected"
    if quantity > 0 and filled >= quantity:
        return "filled"
    if canceled > 0:
        return "canceled"
    if filled > 0:
        return "partial"
    return "pending"


def _day_order_row(item: dict[str, Any]) -> dict[str, Any]:
    """체결 조회 한 줄 → 주문 행. `origin_order_no` 는 정정·취소 요청일 때만 0 이 아니다."""
    return {
        "order_no": str(int(item.get("itg_orr_no") or 0)),
        "origin_order_no": str(int(item.get("org_itg_orr_no") or 0)),
        "ticker": str(item.get("iem_cd") or "").strip(),
        "name": str(item.get("iem_nm") or "").strip(),
        "side_label": str(item.get("sby_dit_cd_nm") or "").strip(),
        "quantity": int(float(item.get("orr_qty") or 0)),
        "price": float(item.get("orr_pr") or 0),
        "filled_quantity": int(float(item.get("tot_cns_qty") or 0)),
        "unfilled_quantity": int(float(item.get("ny_cns_qty") or 0)),
        "canceled_quantity": int(float(item.get("can_qty") or 0)),
        "average_fill_price": float(item.get("cns_avg_uit_pr") or 0),
        "status": order_status(item),
    }


def _root_order_no(row: dict[str, Any], by_no: dict[str, dict[str, Any]]) -> str | None:
    """정정이 이어진 주문의 맨 처음 원주문번호 — 원주문이 목록에 없으면 None."""
    current = row
    while current["origin_order_no"] != "0":
        current = by_no.get(current["origin_order_no"])
        if current is None:
            return None
    return current["order_no"]


def _merge_amendments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """정정 요청 행의 결과(가격·체결·상태)를 원주문 행에 합친다.

    증권사 앱에서 가격을 정정하면 새 주문번호의 행이 생겨 거기서 체결되고, 원주문 행은 취소수량도 체결도 없이
    미체결로 남는다. 원주문에 취소수량이 있으면 취소 요청이 이미 반영된 것이라 합치지 않는다.
    합친 행의 `order_no` 는 원주문번호 그대로이고, 정정·취소 요청을 보낼 번호는 `current_order_no`(가장 늦은 행)다.
    """
    by_no = {row["order_no"]: row for row in rows}
    chains: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        root = _root_order_no(row, by_no)
        if root is not None and root != row["order_no"]:
            chains.setdefault(root, []).append(row)
    merged: list[dict[str, Any]] = []
    for row in rows:
        if row["origin_order_no"] != "0":
            continue
        chain = chains.get(row["order_no"])
        if chain is None or row["canceled_quantity"] > 0:
            merged.append({**row, "current_order_no": row["order_no"]})
            continue
        tip = max(chain, key=lambda item: int(item["order_no"]))
        # 정정한 주문을 취소하면 취소수량은 정정 행에 남는다 — 그 결과를 우선한다.
        result = next((item for item in chain if item["canceled_quantity"] > 0), tip)
        merged.append(
            {**result, "order_no": row["order_no"], "origin_order_no": "0", "current_order_no": tip["order_no"]}
        )
    return merged


def list_day_orders(account_no: str, yyyymmdd: str) -> list[dict[str, Any]]:
    """그날의 주문 목록(체결 조회) — 원주문만이고, 정정은 최신 정정 행의 결과가 원주문번호로 합쳐져 나온다."""
    data = _post_paged(
        _base_url(account_no),
        DAY_ORDERS_PATH,
        {"orr_dt": yyyymmdd, "act_no": account_no, "orr_mkt_cd": "00", "ost_cns_dit": "0"},
        "Output_0",
        no_data_code=NO_DATA_CODE,
    )
    return _merge_amendments([_day_order_row(item) for item in data.get("Output_0") or []])
