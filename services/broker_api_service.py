"""증권사 API 커넥터 — 계좌 설정의 `broker_api` 연동이 쓰는 단일 진입점.

커넥터는 레지스트리(`PROVIDERS`)에 등록한다. 환경변수는 `<PROVIDER>_API_KEY` /
`<PROVIDER>_API_SECRET` 규칙을 따른다 (예: NAMU_PLUG_API_KEY).

지금은 나무증권(NH PLUG, `nhplug` SDK) 하나다. 조회는 전부 읽기 전용이다 —
주문·이체 API 는 여기서 다루지 않는다.
"""

from __future__ import annotations

import os
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()

# 등록된 커넥터 — 화면 셀렉트가 이 목록을 그대로 쓴다.
PROVIDERS: tuple[dict[str, str], ...] = ({"id": "NAMU_PLUG", "name": "나무증권 (NH PLUG)"},)


class BrokerApiError(RuntimeError):
    """커넥터 검증/조회 오류 — 화면에 그대로 보여줄 한국어 메시지를 담는다."""


def _env_keys(provider: str) -> tuple[str, str]:
    return f"{provider}_API_KEY", f"{provider}_API_SECRET"


def list_providers() -> list[dict[str, Any]]:
    """커넥터 목록 + 환경변수 존재 여부 (값은 내려보내지 않는다)."""
    rows = []
    for provider in PROVIDERS:
        key_name, secret_name = _env_keys(provider["id"])
        rows.append({**provider, "env_ok": bool(os.environ.get(key_name)) and bool(os.environ.get(secret_name))})
    return rows


def _mask(account_no: str) -> str:
    return account_no[:3] + "***" + account_no[-2:] if len(account_no) > 5 else "***"


def _namu_call(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """nhplug 호출 — SDK 는 NHPLUG_* 환경변수를 보므로 우리 규칙(NAMU_PLUG_*)을 매핑한다."""
    key_name, secret_name = _env_keys("NAMU_PLUG")
    key, secret = os.environ.get(key_name), os.environ.get(secret_name)
    if not key or not secret:
        raise BrokerApiError(f".env 에 {key_name} / {secret_name} 가 필요합니다.")
    os.environ.setdefault("NHPLUG_APP_KEY", key)
    os.environ.setdefault("NHPLUG_APP_SECRET", secret)
    try:
        from nhplug import NhplugError, call
    except ImportError as exc:
        raise BrokerApiError("nhplug 패키지가 설치돼 있지 않습니다 (pip install nhplug).") from exc
    try:
        return call(path, payload)
    except NhplugError as exc:
        raise BrokerApiError(f"나무증권 API 오류: {exc.message} (코드 {exc.code})") from exc


def _namu_balance(account_no: str) -> dict[str, Any]:
    """국내주식 잔고 원본 응답 — Output_0 요약, Output_1 보유 목록."""
    return _namu_call(
        "/krstock/inquiry/v1/balance",
        {
            "act_no": account_no,
            "bnc_bse_cd": "5",  # 평가 기준(체결 기준)
            "ltg_aot_dit_cd": "9",
            "aet_bse": "2",
            "qut_dit_cd": "UNT",
        },
    )


def list_broker_accounts(provider: str) -> list[dict[str, Any]]:
    """커넥터 검증 + 계좌 나열. 잔고 조회가 되는 계좌에는 미리보기를 붙인다.

    화면의 '확인' 버튼이 부른다 — 토큰 발급까지 실제로 수행해 키가 유효한지 검증된다.
    """
    if provider != "NAMU_PLUG":
        raise BrokerApiError(f"등록되지 않은 커넥터입니다: {provider}")

    accounts = _namu_call("/n2/acctinfo", {}).get("Output_0", [])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in accounts:
        account_no = str(item.get("act_no") or item.get("acct_no") or "").strip()
        if not account_no or account_no in seen:
            continue
        seen.add(account_no)
        row: dict[str, Any] = {
            "account_no": account_no,
            "masked": _mask(account_no),
            "acct_type": str(item.get("acct_type") or ""),
            "ok": False,
        }
        # 계좌 유형에 따라 잔고 API 가 거부하는 계좌가 있다(종합/CMA 등) — 표시로 구분한다.
        try:
            data = _namu_balance(account_no)
            summary = data.get("Output_0", {}) or {}
            holdings = data.get("Output_1", []) or []
            row.update(
                {
                    "ok": True,
                    "cash": float(summary.get("dca") or 0),
                    "holdings_count": len(holdings),
                }
            )
        except BrokerApiError as exc:
            row["error"] = str(exc)
        rows.append(row)
    # 조회 가능한 계좌를 위로
    rows.sort(key=lambda r: (not r["ok"], r["account_no"]))
    return rows
