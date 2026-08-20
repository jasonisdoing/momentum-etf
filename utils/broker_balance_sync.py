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


def sync_all() -> dict[str, Any]:
    """연동된 전 계좌 동기화. 계좌별 결과 목록을 돌려준다 (스크립트가 로그로 남긴다)."""
    from services.broker_api_service import BrokerApiError, apply_fetched_balance, fetch_broker_balance
    from utils.account_settings_store import load_account_docs

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
            applied = apply_fetched_balance(account_id, provider, fetched)
            results.append({"account_id": account_id, "ok": True, **applied})
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
