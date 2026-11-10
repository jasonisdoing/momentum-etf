"""증권사 잔고 동기화 배치 — `broker_api` 연동이 저장된 계좌를 순회해 잔고를 덮어쓴다.

수동 '증권사 데이터 덮어쓰기' 와 같은 저장·알림 함수(`apply_broker_balance`)를 쓴다 — 배치든 사람이든
현금·수량·평단이 하나라도 바뀌면 슬랙으로 알린다.
실패 슬랙은 폭주하지 않게 **실패 시작 1회 + 복구 1회**만 보낸다 — 주기 실행 배치라
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


def stored_cash_map(current: dict[str, Any], currencies: list[str]) -> dict[str, float]:
    """저장된 통화별 현금 — 자산 화면과 같은 기준(다통화 맵이 단일 소스).

    맵이 없던 옛 계좌는 원화만 레거시 `cash_balance` 를 본다. 외화의 레거시 값은 원화 환산액이라 쓰지 않는다.
    """
    cash_map = current.get("cash") or {}
    stored: dict[str, float] = {}
    for code in currencies:
        legacy = current.get("cash_balance") if code == "KRW" else 0
        stored[code] = float(cash_map.get(code, legacy or 0))
    return stored


# 평단 비교 자릿수 — 화면 대조 표(`AccountSettingsManager` 의 AVG_DIGITS)와 같아야 한다.
_AVG_DIGITS = 4


def _price(value: float) -> str:
    """평단 표기 — 소수 4자리까지, 끝의 0 은 뗀다(원화 85,085 · 달러 249.7371)."""
    return f"{value:,.{_AVG_DIGITS}f}".rstrip("0").rstrip(".")


def _account_valuation_label(account_id: str) -> str:
    """반영된 계좌의 원화 총자산과 전 거래일 대비 수익률을 표시한다."""
    from utils.daily_fund_service import calculate_period_return_pct
    from utils.dashboard_service import account_period_profit_and_base
    from utils.normalization import normalize_number
    from utils.portfolio_io import get_latest_daily_snapshot, load_portfolio_master, load_real_holdings_table

    try:
        account = load_portfolio_master(account_id)
        if account is None:
            raise RuntimeError("반영된 계좌를 찾지 못했습니다.")
        holdings = load_real_holdings_table(account_id)
        if holdings is None and account["holdings"]:
            raise RuntimeError("보유 종목 평가를 불러오지 못했습니다.")
        if holdings is not None and (holdings["수량"].gt(0) & ~holdings["현재가"].gt(0)).any():
            raise RuntimeError("현재가가 없는 보유 종목이 있습니다.")
        valuation = float(holdings["평가금액(KRW)"].sum()) if holdings is not None else 0.0
        total_assets = valuation + normalize_number(account["cash_balance"])
    except Exception as exc:
        logger.warning("[BROKER-SYNC] %s 원화 평가액 조회 실패: %s", account_id, exc)
        return "평가액 조회 불가"

    try:
        previous = get_latest_daily_snapshot(account_id, before_today=True)
        if previous and normalize_number(previous.get("total_assets")) > 0:
            profit, base = account_period_profit_and_base(
                total_assets, normalize_number(account["total_principal"]), previous
            )
            return f"{total_assets:,.0f}원({calculate_period_return_pct(profit, base):+.2f}%)"
    except Exception as exc:
        logger.warning("[BROKER-SYNC] %s 전 거래일 대비 수익률 조회 실패: %s", account_id, exc)
    return f"{total_assets:,.0f}원(비교 불가)"


def apply_broker_balance(account_id: str, fetched: dict[str, Any], *, trigger: str) -> dict[str, Any]:
    """불러온 잔고를 저장하고, 저장값과 달랐으면 슬랙으로 알린다 — 배치와 사람의 덮어쓰기가 공용.

    ``trigger`` 는 누가 바꿨는지(슬랙 문구) — "배치"·"수동 덮어쓰기".
    """
    from services.broker_api_service import apply_fetched_balance
    from utils.portfolio_io import load_portfolio_master

    diff = _diff_lines(load_portfolio_master(account_id), fetched)
    applied = apply_fetched_balance(account_id, fetched["provider"], fetched)
    if diff:
        valuation = _account_valuation_label(account_id)
        changes = "\n".join(f"{index}. {line}" for index, line in enumerate(diff, start=1))
        _notify(
            f"🔄 증권사 잔고 반영({trigger}) — {account_id} ({fetched['provider']}) {valuation}\n"
            f"변경 {len(diff)}건\n{changes}"
        )
    return {**applied, "changed": bool(diff)}


def _diff_lines(current: dict[str, Any] | None, fetched: dict[str, Any]) -> list[str]:
    """저장값 대비 증권사 값의 차이 — 슬랙 본문 줄 목록. 차이가 없으면 빈 목록.

    비교 기준은 화면의 대조 표와 같다 — 현금은 통화별(원화는 원 단위, 외화는 센트 단위), 수량은 주 단위,
    평단은 소수 4자리(미국 평단이 소수로 오기 때문 — 정수로 반올림하면 1달러 미만 변화가 묻힌다).
    """
    lines: list[str] = []
    current = current or {"cash_balance": 0.0, "holdings": []}
    before_cash = stored_cash_map(current, list(fetched["cash_map"]))
    for code, amount in fetched["cash_map"].items():
        digits = 0 if code == "KRW" else 2
        cash_before = round(before_cash[code], digits)
        cash_after = round(float(amount), digits)
        if cash_before != cash_after:
            lines.append(f"현금 {code}: {cash_before:,} → {cash_after:,}")

    before_by = {str(row.get("ticker") or ""): row for row in current.get("holdings") or []}
    after_by = {row["ticker"]: row for row in fetched["holdings"]}
    for ticker in sorted(set(before_by) | set(after_by)):
        before, after = before_by.get(ticker), after_by.get(ticker)
        if before is None and after is not None:
            lines.append(
                f"{after['name']}({ticker}): 신규 {after['quantity']:,.0f}주 @{_price(after['average_buy_price'])}"
            )
            continue
        if after is None and before is not None:
            lines.append(
                f"{before.get('name') or ticker}({ticker}): {float(before.get('quantity') or 0):,.0f}주 → 삭제"
            )
            continue
        qty_before, qty_after = float(before.get("quantity") or 0), float(after["quantity"])
        avg_before = round(float(before.get("average_buy_price") or 0), _AVG_DIGITS)
        avg_after = round(float(after["average_buy_price"]), _AVG_DIGITS)
        parts = []
        if qty_before != qty_after:
            parts.append(f"{qty_before:,.0f}주 → {qty_after:,.0f}주")
        if avg_before != avg_after:
            parts.append(f"평단 {_price(avg_before)} → {_price(avg_after)}")
        if parts:
            lines.append(f"{after['name']}({ticker}): " + " · ".join(parts))
    return lines


def sync_all() -> dict[str, Any]:
    """연동된 전 계좌 동기화. 계좌별 결과 목록을 돌려준다 (스크립트가 로그로 남긴다).

    저장값과 차이가 없으면 저장도 슬랙도 건너뛴다 — 주기마다 updated_at 만 바뀌면
    '최종 변경' 표시가 의미를 잃고, 변화 없는 알림은 소음이다.
    """
    from services.broker_api_service import BrokerApiError, fetch_broker_balance
    from utils.account_settings_store import load_live_account_docs
    from utils.portfolio_io import load_portfolio_master

    results: list[dict[str, Any]] = []
    for doc in load_live_account_docs():
        # 연동 여부는 커넥터로만 본다 — 계좌번호는 나무증권만 있다(메리츠는 키가 계좌에 묶여 없다).
        provider = str((doc.get("broker_api") or {}).get("provider") or "").strip().upper()
        if not provider:
            continue
        account_id = doc["account_id"]
        try:
            fetched = fetch_broker_balance(account_id)
            diff = _diff_lines(load_portfolio_master(account_id), fetched)
            if not diff:
                results.append({"account_id": account_id, "ok": True, "changed": False})
            else:
                applied = apply_broker_balance(account_id, fetched, trigger="배치")
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
