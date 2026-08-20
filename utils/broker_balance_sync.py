"""증권사 잔고 동기화 배치 — `broker_api` 연동이 저장된 계좌를 순회해 잔고를 덮어쓴다.

수동 '증권사 데이터 덮어쓰기' 와 같은 코드(`apply_fetched_balance`)를 쓴다.
실패 슬랙은 폭주하지 않게 **실패 시작 1회 + 복구 1회**만 보낸다 — 10분 간격 배치라
장애가 이어지면 매번 알리는 게 소음이다. 상태는 DB(system_config)에 남긴다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()

_STATE_KEY = "broker_balance_sync_state"


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    return db


def _load_failing(account_id: str) -> bool:
    doc = _db().system_config.find_one({"_id": _STATE_KEY}) or {}
    return bool((doc.get("accounts") or {}).get(account_id, {}).get("failing"))


def _save_failing(account_id: str, failing: bool, message: str = "") -> None:
    _db().system_config.update_one(
        {"_id": _STATE_KEY},
        {
            "$set": {
                f"accounts.{account_id}": {
                    "failing": failing,
                    "message": message,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            }
        },
        upsert=True,
    )


def _notify(text: str) -> None:
    from utils.notification import send_slack_message_v2

    try:
        send_slack_message_v2(text)
    except Exception as exc:  # 알림 실패가 동기화 자체를 막으면 안 된다
        logger.warning("[BROKER-SYNC] 슬랙 발송 실패: %s", exc)


def _diff_lines(current: dict[str, Any] | None, fetched: dict[str, Any]) -> list[str]:
    """저장값 대비 증권사 값의 차이 — 슬랙 본문 줄 목록. 차이가 없으면 빈 목록.

    비교 기준은 화면의 대조 표와 같다 — 현금·수량은 원/주 단위, 평단은 원 단위 반올림.
    """
    lines: list[str] = []
    current = current or {"cash_balance": 0.0, "holdings": []}
    # 자산 화면과 같은 기준 — 다통화 맵의 KRW 값 우선(맵이 화면 표시의 단일 소스다).
    cash_map = current.get("cash") or {}
    cash_before = round(float(cash_map.get("KRW", current.get("cash_balance") or 0)))
    cash_after = round(float(fetched["cash"]))
    if cash_before != cash_after:
        lines.append(f"현금: {cash_before:,} → {cash_after:,}")

    before_by = {str(row.get("ticker") or ""): row for row in current.get("holdings") or []}
    after_by = {row["ticker"]: row for row in fetched["holdings"]}
    for ticker in sorted(set(before_by) | set(after_by)):
        before, after = before_by.get(ticker), after_by.get(ticker)
        if before is None and after is not None:
            lines.append(
                f"{after['name']}({ticker}): 신규 {after['quantity']:,.0f}주 @{after['average_buy_price']:,.0f}"
            )
            continue
        if after is None and before is not None:
            lines.append(
                f"{before.get('name') or ticker}({ticker}): {float(before.get('quantity') or 0):,.0f}주 → 삭제"
            )
            continue
        qty_before, qty_after = float(before.get("quantity") or 0), float(after["quantity"])
        avg_before = round(float(before.get("average_buy_price") or 0))
        avg_after = round(float(after["average_buy_price"]))
        parts = []
        if qty_before != qty_after:
            parts.append(f"{qty_before:,.0f}주 → {qty_after:,.0f}주")
        if avg_before != avg_after:
            parts.append(f"평단 {avg_before:,} → {avg_after:,}")
        if parts:
            lines.append(f"{after['name']}({ticker}): " + " · ".join(parts))
    return lines


def sync_all() -> dict[str, Any]:
    """연동된 전 계좌 동기화. 계좌별 결과 목록을 돌려준다 (스크립트가 로그로 남긴다).

    저장값과 차이가 없으면 저장도 슬랙도 건너뛴다 — 10분마다 updated_at 만 바뀌면
    '최종 변경' 표시가 의미를 잃고, 변화 없는 알림은 소음이다.
    """
    from services.broker_api_service import BrokerApiError, apply_fetched_balance, fetch_broker_balance
    from utils.account_settings_store import load_account_docs
    from utils.portfolio_io import load_portfolio_master

    results: list[dict[str, Any]] = []
    for doc in load_account_docs():
        linked = doc.get("broker_api") or {}
        provider = str(linked.get("provider") or "").strip().upper()
        account_no = str(linked.get("account_no") or "").strip()
        if not provider or not account_no:
            continue
        account_id = doc["account_id"]
        try:
            fetched = fetch_broker_balance(provider, account_no)
            diff = _diff_lines(load_portfolio_master(account_id), fetched)
            if not diff:
                results.append({"account_id": account_id, "ok": True, "changed": False})
            else:
                applied = apply_fetched_balance(account_id, provider, fetched)
                results.append({"account_id": account_id, "ok": True, "changed": True, **applied})
                _notify(f"🔄 증권사 잔고 반영 — {account_id} ({provider}) · 변경 {len(diff)}건\n" + "\n".join(diff))
            if _load_failing(account_id):
                _save_failing(account_id, False)
                _notify(f"✅ 증권사 잔고 동기화 복구 — {account_id} ({provider})")
        except (BrokerApiError, Exception) as exc:  # noqa: BLE001 — 계좌 하나의 실패가 다음 계좌를 막으면 안 된다
            message = str(exc)
            results.append({"account_id": account_id, "ok": False, "error": message})
            logger.warning("[BROKER-SYNC] %s 동기화 실패: %s", account_id, message)
            if not _load_failing(account_id):
                _save_failing(account_id, True, message)
                _notify(f"🚨 증권사 잔고 동기화 실패 — {account_id} ({provider})\n{message}")
    return {"accounts": results}
