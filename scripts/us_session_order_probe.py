"""[일회성] 메리츠 해외주식 세션별(프리·정규·애프터·데이장) 주문 동작 확인 — 확인이 끝나면 삭제한다.

운영 계좌에 **실제 주문**을 낸다. 현재가보다 30% 낮은 지정가 매수 1주를 넣고 곧바로 취소한다.
세션마다 한 번씩 돌려 접수·거부 사유·호가 유무를 비교한다.

    .venv/bin/python scripts/us_session_order_probe.py m_2 SCHD NY 정규장
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from services.broker_api_service import BrokerApiError, OrderUncertainError
from services.meritz_api_service import _accepted_order_no, _get, _request
from utils.env import load_env_if_present

# 시세 거래소(OQ·NY·AX) → 주문 거래소 벤더 번호.
VENDOR_BY_EXCHANGE = {"OQ": "0537", "NY": "0321", "AX": "0066"}
PRICE_DISCOUNT = 0.70
HISTORY_WAIT_SECONDS = 2


def _show(title: str, value: object) -> None:
    print(f"\n── {title}")
    print(json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else value)


def _try_get(title: str, account_id: str, path: str, params: dict[str, str], success: frozenset[str]) -> dict:
    """조회 한 건 — 실패해도 사유를 출력하고 계속한다(세션별로 없는 시세가 정상일 수 있다)."""
    try:
        data = _request(account_id, "GET", path, success=success, params=params).get("data") or {}
    except BrokerApiError as exc:
        _show(title, f"실패: {exc}")
        return {}
    _show(title, data)
    return data


def _history_rows(account_id: str) -> list[dict]:
    """당일 해외 주문·체결 내역 — 정정·취소 대상을 찾는 용도의 전용 조회다."""
    return (
        _get(account_id, "/trading/v1/overseas/orders/today", {"cntg_cls_code": "0", "seln_byov_clcd": "0"}).get("data")
        or []
    )


def _show_order_rows(account_id: str, order_no: str, title: str) -> dict | None:
    time.sleep(HISTORY_WAIT_SECONDS)
    rows = [
        row
        for row in _history_rows(account_id)
        if str(int(row.get("oder_no") or 0)) == order_no or str(int(row.get("orgl_oder_no") or 0)) == order_no
    ]
    _show(title, rows or "내역에 없음")
    return next((row for row in rows if str(int(row.get("oder_no") or 0)) == order_no), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("account_id", help="메리츠 미국 계좌 (예: m_2)")
    parser.add_argument("symbol", help="미국 종목 심볼 (예: SCHD)")
    parser.add_argument("exchange", choices=sorted(VENDOR_BY_EXCHANGE), help="OQ 나스닥 · NY 뉴욕 · AX 아멕스")
    parser.add_argument("label", help="지금 세션 이름(출력 표시용, 예: 프리 · 정규장 · 애프터 · 데이장)")
    parser.add_argument("--cancel-order", help="주문 없이 이 주문번호만 취소한다(이전 실행에서 취소가 안 된 경우)")
    args = parser.parse_args()
    load_env_if_present()
    account_id, symbol, exchange = args.account_id, args.symbol.upper(), args.exchange
    if args.cancel_order:
        return _cancel_order(account_id, args.cancel_order)

    now_kst = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    print(f"세션 {args.label} · 한국시각 {now_kst} · 계좌 {account_id} · {symbol}.{exchange}")

    _try_get(
        "장운영(미국 01)",
        account_id,
        "/reference/v1/overseas/market-hours",
        {"ovrs_stck_mrkt_tr_cls_code": "01"},
        frozenset({"5766"}),
    )
    quote_params = {"mrkt_div_code": "OV", "mrkt_cls_code": exchange, "iscd": symbol, "dely_rltm_cls_code": "0"}
    book = _try_get("호가(1·2호가만)", account_id, "/market/v1/overseas/orderbook", quote_params, frozenset({"0000"}))
    _try_get(
        "주간거래 현재가", account_id, "/market/v1/overseas/prices/daytime", quote_params, frozenset({"0000", "5766"})
    )
    _try_get(
        "애프터장 현재가", account_id, "/market/v1/overseas/prices/after", quote_params, frozenset({"0000", "5766"})
    )
    _try_get("정규장 현재가", account_id, "/market/v1/overseas/prices", quote_params, frozenset({"0000", "5766"}))
    if book:
        print(
            f"\n호가 요약: 매도1 {book.get('askp1')}({book.get('askp_rsqn1')}) · 매수1 {book.get('bidp1')}({book.get('bidp_rsqn1')})"
            f" · 현재가 {book.get('prpr')} · 호가시각 한국 {book.get('korea_hour')}"
        )
    last = float(book.get("prpr") or 0)
    if last <= 0:
        print("\n호가 응답에 현재가가 없어 주문 가격을 정할 수 없습니다 — 주문 없이 종료합니다.")
        return 1

    price = round(last * PRICE_DISCOUNT, 2)
    print(
        f"\n주문: {symbol} 지정가 매수 1주 @ {price} (현재가 {last} 의 {int(PRICE_DISCOUNT * 100)}%) — 운영 계좌 {account_id}"
    )
    if input("실제로 주문을 내고 바로 취소합니다. 진행하려면 y: ").strip().lower() != "y":
        print("취소했습니다.")
        return 0

    body = {
        "ovrs_stck_symb_iscd": symbol,
        "ovrs_exch_vndr_idno": VENDOR_BY_EXCHANGE[exchange],
        "oder_cls_code": "01",
        "oder_cond_cls_code": "0",
        "odqt": 1,
        "ovrs_stck_oder_prc": f"{price:.2f}",
        "warn_cnfr_yn": "N",
    }
    try:
        data = _request(account_id, "POST", "/trading/v1/overseas/orders/buy", success=frozenset({"0001"}), body=body)
        order_no = _accepted_order_no(data, "주문")
    except OrderUncertainError as exc:
        print(f"\n접수 여부 불명 — 메리츠 앱의 주문내역을 확인하세요: {exc}")
        return 2
    except BrokerApiError as exc:
        print(f"\n주문 거부/실패(세션 확인용 결과): {exc}")
        return 1
    print(f"\n주문 접수됨 — 주문번호 {order_no}")

    return _cancel_order(account_id, order_no)


def _cancel_order(account_id: str, order_no: str) -> int:
    """당일 주문내역에서 주문을 찾아(주문일자 필요) 전량 취소하고 결과를 출력한다."""
    row = _show_order_rows(account_id, order_no, "취소 전 주문내역")
    if row is None:
        print("\n주문내역에서 찾지 못해 취소하지 않았습니다 — 메리츠 앱에서 직접 확인·취소하세요.")
        return 2
    try:
        cancel = _request(
            account_id,
            "POST",
            "/trading/v1/overseas/orders/cancel",
            success=frozenset({"0001"}),
            body={
                "oder_date": str(row["oder_date"]),
                "orgl_oder_no": int(order_no),
                "oder_cls_code": "01",
                "odqt": 1,
                "warn_cnfr_yn": "N",
            },
        )
        print(f"\n취소 접수됨 — 취소 주문번호 {_accepted_order_no(cancel, '취소')}")
    except BrokerApiError as exc:
        print(f"\n취소 실패 — 메리츠 앱에서 주문 {order_no} 을 직접 취소하세요: {exc}")
        return 2
    _show_order_rows(account_id, order_no, "취소 후 주문내역")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
