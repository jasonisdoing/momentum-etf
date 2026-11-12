"""사고팔기 주문 경로 — 계좌 ID 로 증권사(나무·메리츠)를 골라 잔고·호가·지정가 주문·취소·주문내역을 부른다.

사고팔기·슬랙 주문·API 주문 모달은 이 모듈만 부른다. 증권사별 모듈은 같은 표준형을 돌려준다:
잔고 `{"holdings": [{ticker, quantity, ...}]}`, 호가 `{ask, bid, at}`,
주문 `{order_no, message}`, 주문 행(`broker_order_service._day_order_row` 모양).

모든 호출은 `account_env` 를 먼저 통과한다 — 계좌 설정에 증권사 연동이 저장된 계좌만 받는다.
"""

from __future__ import annotations

from typing import Any

from services.broker_api_service import MERITZ_PROVIDER, NAMU_PROVIDER, BrokerApiError


def _account_doc(account_id: str) -> dict[str, Any]:
    """증권사 연동이 저장된 계좌 설정 문서 — 계좌가 없거나 연동이 없으면 막는다."""
    from utils.account_settings_store import load_account_docs

    doc = next((doc for doc in load_account_docs() if doc["account_id"] == account_id), None)
    if doc is None:
        raise BrokerApiError(f"'{account_id}' 계좌가 계좌 설정에 없습니다.")
    provider = str((doc.get("broker_api") or {}).get("provider") or "").strip().upper()
    if provider not in (NAMU_PROVIDER, MERITZ_PROVIDER):
        raise BrokerApiError(f"'{account_id}' 계좌에 증권사 API 연동(나무·메리츠)이 저장돼 있지 않습니다.")
    return doc


def account_country(account_id: str) -> str:
    """계좌의 국가(kor·us) — 주문·체결 조회의 날짜와 통화가 이 값으로 갈린다."""
    return str(_account_doc(account_id)["country_code"]).strip().lower()


def require_trade_account(account_id: str) -> None:
    """증권사 연동이 저장된 계좌인지만 확인한다(증권사를 부르지 않는다) — 없으면 BrokerApiError."""
    _account_doc(account_id)


def _provider(doc: dict[str, Any]) -> str:
    return str(doc["broker_api"]["provider"]).strip().upper()


def _is_us(doc: dict[str, Any]) -> bool:
    """미국 계좌인지 — 계좌 국가가 증권사 주문 함수(국내·해외)를 가른다. 나무는 국내 전용이다."""
    return str(doc["country_code"]).strip().lower() == "us"


def _namu_account_no(doc: dict[str, Any]) -> str:
    return str(doc["broker_api"]["account_no"])


def account_env(account_id: str) -> str:
    """계좌의 구분(live·mock) — 나무는 증권사 계좌구분과 맞는지도 확인한다. 메리츠는 운영 계좌만 연동된다."""
    from services import broker_order_service

    doc = _account_doc(account_id)
    if _provider(doc) == NAMU_PROVIDER:
        return broker_order_service.account_env(_namu_account_no(doc))
    return str(doc["env"])


def fetch_order_balance(account_id: str) -> dict[str, Any]:
    """계좌 잔고(표준형) — 회차 원장은 보유 수량만 쓴다."""
    from services import broker_order_service, meritz_api_service

    doc = _account_doc(account_id)
    account_env(account_id)
    if _provider(doc) == NAMU_PROVIDER:
        return broker_order_service.fetch_order_balance(_namu_account_no(doc))
    return meritz_api_service.fetch_balance(account_id, str(doc["country_code"]))


def fetch_best_quote(account_id: str, ticker: str, venue: str) -> dict[str, Any]:
    """종목의 1호가 — ``venue`` 는 국내는 거래소(KRX·NXT), 미국은 시세 거래소 코드(OQ·NY·AX)다."""
    from services import broker_order_service, meritz_api_service

    doc = _account_doc(account_id)
    account_env(account_id)
    if _provider(doc) == NAMU_PROVIDER:
        return broker_order_service.fetch_best_quote(_namu_account_no(doc), ticker, venue)
    if _is_us(doc):
        return meritz_api_service.fetch_us_best_quote(account_id, ticker, venue)
    return meritz_api_service.fetch_best_quote(account_id, ticker, venue)


def place_limit_order(
    account_id: str, side: str, ticker: str, quantity: int, price: int | float, venue: str
) -> dict[str, str]:
    """지정가 주문 한 건 — 접수되면 주문번호를 돌려준다."""
    from services import broker_order_service, meritz_api_service

    doc = _account_doc(account_id)
    account_env(account_id)
    if _provider(doc) == NAMU_PROVIDER:
        return broker_order_service.place_limit_order(_namu_account_no(doc), side, ticker, quantity, price, venue)
    if _is_us(doc):
        return meritz_api_service.place_us_limit_order(account_id, side, ticker, quantity, price, venue)
    if int(price) != price:
        raise BrokerApiError(f"국내 주문 가격은 정수여야 합니다: {price}")
    return meritz_api_service.place_limit_order(account_id, side, ticker, quantity, int(price), venue)


def cancel_order(account_id: str, order_no: str, ticker: str, venue: str) -> dict[str, str]:
    """주문을 전량 취소한다."""
    from services import broker_order_service, meritz_api_service

    doc = _account_doc(account_id)
    account_env(account_id)
    if _provider(doc) == NAMU_PROVIDER:
        return broker_order_service.cancel_order(_namu_account_no(doc), order_no, ticker)
    # 나무 취소는 원주문번호로만 찾아 거래소를 받지 않는다. 메리츠 국내는 원주문의 거래소가 필요하고, 미국은 필요 없다.
    if _is_us(doc):
        return meritz_api_service.cancel_us_order(account_id, order_no, ticker)
    return meritz_api_service.cancel_order(account_id, order_no, ticker, venue)


def list_day_orders(account_id: str, yyyymmdd: str) -> list[dict[str, Any]]:
    """그날의 주문 목록 — 원주문만이고, 정정은 최신 정정 행의 결과가 원주문번호로 합쳐져 나온다."""
    from services import broker_order_service, meritz_api_service

    doc = _account_doc(account_id)
    account_env(account_id)
    if _provider(doc) == NAMU_PROVIDER:
        rows = broker_order_service.day_order_rows(_namu_account_no(doc), yyyymmdd)
    elif _is_us(doc):
        rows = meritz_api_service.us_day_order_rows(account_id, yyyymmdd)
    else:
        rows = meritz_api_service.day_order_rows(account_id, yyyymmdd)
    return merge_amendments(rows)


def _root_order_no(row: dict[str, Any], by_no: dict[str, dict[str, Any]]) -> str | None:
    """정정이 이어진 주문의 맨 처음 원주문번호 — 원주문이 목록에 없으면 None."""
    current = row
    while current["origin_order_no"] != "0":
        current = by_no.get(current["origin_order_no"])
        if current is None:
            return None
    return current["order_no"]


def merge_amendments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """정정 요청 행의 결과(가격·체결·상태)를 원주문 행에 합친다 — 두 증권사 모두 같은 규칙이다.

    증권사 앱에서 가격을 정정하면 새 주문번호의 행이 생겨 거기서 체결되고, 원주문 행은 취소수량도 체결도 없이
    미체결로 남는다. 원주문에 취소수량이 있으면 취소 요청이 이미 반영된 것이라 합치지 않는다.
    체결 수량·평균 체결가는 원주문과 정정 행 전체의 합계다 — 원주문이 정정 전에 일부 체결했을 수 있다.
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
        members = [row, *chain]
        filled = sum(item["filled_quantity"] for item in members)
        cost = sum(item["filled_quantity"] * item["average_fill_price"] for item in members)
        merged.append(
            {
                **result,
                "order_no": row["order_no"],
                "origin_order_no": "0",
                "current_order_no": tip["order_no"],
                "filled_quantity": filled,
                "average_fill_price": cost / filled if filled else 0.0,
            }
        )
    return merged


def resolve_us_exchange(account_id: str, ticker: str) -> str:
    """미국 종목의 시세·주문 거래소 코드(OQ·NY·AX) — 증권사 시세 조회로 판정한다. 메리츠 미국 계좌만 된다."""
    from services import meritz_api_service

    doc = _account_doc(account_id)
    if _provider(doc) != MERITZ_PROVIDER or not _is_us(doc):
        raise BrokerApiError(
            f"'{account_id}' 는 미국 종목 거래소를 조회할 수 있는 계좌가 아닙니다(메리츠 미국 계좌 필요)."
        )
    return meritz_api_service.resolve_us_exchange(account_id, ticker)
