"""나무증권 모의투자 주문 — 모의투자 서버로 **직접 요청**하는 주문·취소·조회.

SDK(`nhplug`)는 호출 대상 서버를 프로세스 전역 환경변수(`NHPLUG_BASE_URL`)로 읽는다. 요청 처리 중에 그 값을 바꾸면
같은 프로세스에서 도는 운영 조회(잔고 동기화 등)가 모의투자 서버로 새므로, 이 모듈은 SDK 를 거치지 않고
`requests` 로 모의투자 서버(`MOCK_BASE_URL`)에 직접 보낸다. 접근토큰은 SDK 호출과 같은 DB 공유 토큰이다
(`broker_api_service.namu_access_token`) — 키당 유효 토큰이 1개라 따로 발급하면 서로 죽인다.

안전 장치: 주문·취소·조회는 계좌구분 `03`(모의투자) 계좌만 받는다(`require_mock_account`). 운영 계좌에는
이 서버 주소로 보내도 실패하지만, 실수로 운영 주문이 나가는 경로를 코드에서 아예 만들지 않는다.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import requests

from services.broker_api_service import (
    BALANCE_PATH,
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
#: 나무증권 계좌구분 — 03 이 모의투자 전용이다.
MOCK_ACCOUNT_TYPE = "03"
#: 주문을 보내는 시장 — KRX 로 고정한다. SOR 는 KRX·NXT 로 쪼개져 주문번호가 둘이 되고, 사고팔기는 KRX 정규장 규칙이다.
ORDER_MARKET = "KRX"
_PRICE_TYPE_LIMIT = "01"  # 호가유형 — 보통가(지정가)
_NO_CONDITION = "00"  # 주문조건 없음 / 공매도 구분 정상

ORDER_PATH = {"buy": "/krstock/order/v1/cashBuy", "sell": "/krstock/order/v1/cashSell"}
CANCEL_PATH = "/krstock/order/v1/cancel"
DAY_ORDERS_PATH = "/krstock/inquiry/v1/dailyOrderExecution"

# 모든 호출을 최소 1초 간격으로 직렬화한다 — 운영 쪽(`_CALLS_PER_SECOND`)과 같은 보수적 간격이다.
_MIN_INTERVAL_SECONDS = 1.0
_MAX_PAGES = 20
_REQUEST_TIMEOUT_SECONDS = 10

_call_lock = threading.Lock()
_last_call_at = 0.0
_ACCOUNT_TYPE_CACHE = TtlCache(300, name="broker-account-types")


class OrderUncertainError(BrokerApiError):
    """요청이 서버에 닿았는지 알 수 없는 네트워크 오류 — 주문이 들어갔을 수도 있어 실패로 단정하면 안 된다."""


def require_mock_account(account_no: str) -> None:
    """모의투자(03) 계좌가 아니면 막는다 — 이 모듈의 모든 호출이 가장 먼저 통과해야 하는 관문이다."""
    key = _ACCOUNT_TYPE_CACHE.make_key("types")
    types = _ACCOUNT_TYPE_CACHE.get(key)
    if types is None:
        types = account_types()
        _ACCOUNT_TYPE_CACHE.set(key, types)
    if types.get(account_no) != MOCK_ACCOUNT_TYPE:
        raise BrokerApiError("모의투자(03) 계좌가 아니면 이 기능을 쓸 수 없습니다.")


def _message_of(data: dict[str, Any]) -> str:
    """오류 본문에서 사람이 읽을 메시지 — 업무 계층(rsp_msg)과 게이트웨이(error_description) 둘 다 훑는다."""
    for key in ("rsp_msg", "error_description", "message"):
        if data.get(key):
            return str(data[key])
    error = data.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return ""


def _post(path: str, payload: dict[str, Any], *, cts: str | None = None, cts_flag: str | None = None):
    """모의투자 서버로 한 건 보낸다 → (본문, 응답 헤더). HTTP 오류는 BrokerApiError 로 올린다."""
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
                    f"{MOCK_BASE_URL}{path}", headers=headers, data=body, timeout=_REQUEST_TIMEOUT_SECONDS
                )
            except requests.RequestException as exc:
                raise OrderUncertainError(f"모의투자 서버 응답을 받지 못했습니다: {exc}") from exc
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
        raise BrokerApiError(f"나무증권 모의투자 API 오류: {_message_of(data) or f'HTTP {response.status_code}'}")
    return data, response.headers


def _post_paged(path: str, payload: dict[str, Any], list_key: str) -> dict[str, Any]:
    """연속조회 — 목록(list_key)을 전 페이지 이어 붙인다. 마지막 페이지만 업무 성공을 판정한다."""
    merged: dict[str, Any] = {}
    items: list[Any] = []
    cts = cts_flag = None
    seen: set[str] = set()
    for page in range(1, _MAX_PAGES + 1):
        data, headers = _post(path, payload, cts=cts, cts_flag=cts_flag)
        next_cts = (headers.get("cts") or "").strip()
        has_next = (headers.get("cts_flag") or "").strip() == "Y" and bool(next_cts) and next_cts not in seen
        if not has_next and not is_business_success(
            str(data["rsp_cd"]) if data.get("rsp_cd") is not None else None, data.get("rsp_msg")
        ):
            raise BrokerApiError(
                f"나무증권 모의투자 API 오류: {_message_of(data) or '업무 오류'} (코드 {data.get('rsp_cd')})"
            )
        if not merged:
            merged = {key: value for key, value in data.items() if key != list_key}
        items.extend(data.get(list_key) or [])
        if not has_next:
            merged[list_key] = items
            return merged
        seen.add(next_cts)
        cts, cts_flag = next_cts, "Y"
    raise BrokerApiError(f"나무증권 모의투자 API 연속조회가 {_MAX_PAGES}페이지를 넘었습니다 — 응답을 확인하세요.")


def fetch_mock_balance(account_no: str) -> dict[str, Any]:
    """모의투자 계좌 잔고 — 운영 조회와 같은 표준형(예수금 D+2·총자산·보유)이다."""
    require_mock_account(account_no)
    return parse_balance(_post_paged(BALANCE_PATH, balance_request(account_no), "Output_1"))


def _accepted_order_no(data: dict[str, Any], what: str) -> str:
    """주문·취소 응답에서 시장주문번호를 꺼낸다 — **번호가 있어야 접수된 것이다**.

    응답 코드는 매수 00048·매도 00047·취소 00192 로 문서에 나와 있지만, 접수의 증거는 번호다.
    번호 없이 성공으로 두면 접수되지 않은 주문을 기록이 가진 채로 남고, 반대로 코드만 보고 실패로 두면
    실제로는 들어간 주문을 기록에서 놓친다.
    """
    output = data.get("Output_0") or {}
    order_no = output.get("mkt_orr_no") if isinstance(output, dict) else None
    if not isinstance(order_no, (int, float)) or int(order_no) <= 0:
        raise BrokerApiError(
            f"{what}이 접수되지 않았습니다: {_message_of(data) or '응답에 주문번호가 없습니다'} (코드 {data.get('rsp_cd')})"
        )
    return str(int(order_no))


def place_limit_order(account_no: str, side: str, ticker: str, quantity: int, price: int) -> dict[str, str]:
    """지정가 주문 한 건 — 접수되면 시장주문번호를 돌려준다. side 는 buy·sell."""
    require_mock_account(account_no)
    if side not in ORDER_PATH:
        raise BrokerApiError(f"알 수 없는 주문 구분입니다: {side}")
    if quantity <= 0 or price <= 0:
        raise BrokerApiError("수량과 가격은 0보다 커야 합니다.")
    data, _ = _post(
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
    require_mock_account(account_no)
    data, _ = _post(
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


def list_day_orders(account_no: str, yyyymmdd: str) -> list[dict[str, Any]]:
    """그날의 주문 목록(체결 조회) — 원주문만이다.

    정정·취소 요청도 자기 번호를 가진 행으로 나오지만(원주문번호 `org_itg_orr_no` 가 0 이 아님), 그 결과는 원주문의
    취소수량에 이미 반영되므로 목록에서 뺀다.
    """
    require_mock_account(account_no)
    data = _post_paged(
        DAY_ORDERS_PATH,
        {"orr_dt": yyyymmdd, "act_no": account_no, "orr_mkt_cd": "00", "ost_cns_dit": "0"},
        "Output_0",
    )
    rows: list[dict[str, Any]] = []
    for item in data.get("Output_0") or []:
        if int(item.get("org_itg_orr_no") or 0) != 0:
            continue
        rows.append(
            {
                "order_no": str(int(item.get("itg_orr_no") or 0)),
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
        )
    return rows
