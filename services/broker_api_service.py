"""증권사 API 커넥터 — 계좌 설정의 `broker_api` 연동이 쓰는 단일 진입점.

커넥터는 레지스트리(`PROVIDERS`)에 등록한다. 환경변수는 `<PROVIDER>_API_KEY` /
`<PROVIDER>_API_SECRET` 규칙을 따른다 (예: NAMU_PLUG_API_KEY).

지금은 나무증권(NH PLUG, `nhplug` SDK) 하나다. 조회는 전부 읽기 전용이다 —
주문·이체 API 는 여기서 다루지 않는다.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from config import CACHE_TTL_COMPUTE
from utils.logger import get_app_logger
from utils.ttl_cache import TtlCache

logger = get_app_logger()

# 불러오기 결과를 잠시 보관 — '적용' 이 재호출 없이 이 값을 쓴다(일일 호출 제한 절약).
_FETCH_CACHE = TtlCache(CACHE_TTL_COMPUTE, name="broker-balance")

# NH API 호출 간격 — 유량 제한(IGW42902, 엔드포인트별 초당 제한)에 걸리지 않게
# **모든 호출을 최소 1초 간격**으로 직렬화한다. 연속조회 페이지·계좌 순회 포함.
_MIN_CALL_INTERVAL_SECONDS = 1.0
_throttle_lock = threading.Lock()
_last_call_at = 0.0


def _throttle() -> None:
    global _last_call_at
    with _throttle_lock:
        wait = _last_call_at + _MIN_CALL_INTERVAL_SECONDS - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


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


def _ensure_env() -> None:
    """우리 규칙(NAMU_PLUG_*)의 키를 SDK 가 보는 NHPLUG_* 로 매핑한다."""
    key_name, secret_name = _env_keys("NAMU_PLUG")
    key, secret = os.environ.get(key_name), os.environ.get(secret_name)
    if not key or not secret:
        raise BrokerApiError(f".env 에 {key_name} / {secret_name} 가 필요합니다.")
    os.environ.setdefault("NHPLUG_APP_KEY", key)
    os.environ.setdefault("NHPLUG_APP_SECRET", secret)


def _namu_call(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """nhplug 단건 호출 (연속조회 없는 API 용)."""
    _ensure_env()
    try:
        from nhplug import NhplugError, call
    except ImportError as exc:
        raise BrokerApiError("nhplug 패키지가 설치돼 있지 않습니다 (pip install nhplug).") from exc
    try:
        _throttle()
        return call(path, payload)
    except NhplugError as exc:
        raise BrokerApiError(f"나무증권 API 오류: {exc.message} (코드 {exc.code})") from exc


def _namu_call_paged(path: str, payload: dict[str, Any], list_key: str = "Output_1") -> dict[str, Any]:
    """연속조회(rsp_cd 00218) 지원 호출 — 목록(list_key)을 전 페이지 이어 붙여 돌려준다.

    NH 규약: 한 페이지를 넘는 목록은 rsp_cd=00218 로 오고, **응답 헤더 `cts`** 의
    연속키를 다음 요청 헤더에 실어 이어서 받는다. SDK 의 `call()` 은 응답 헤더를
    노출하지 않아 여기서만 requests 로 직접 호출한다 — 인증(토큰 캐시)·성공 판정은
    SDK 것을 그대로 재사용해 동작이 어긋나지 않게 한다.
    """
    import json

    import requests

    _ensure_env()
    from nhplug import clear_token, get_base_url, get_token
    from nhplug.client import is_success

    url = f"{get_base_url()}{path}"
    body = json.dumps({"Input_0": payload})
    merged: dict[str, Any] = {}
    items: list[Any] = []
    cts = ""
    refreshed = False
    for _page in range(20):  # 폭주 방지 — 잔고가 20페이지를 넘을 일은 없다
        headers = {
            "x-client-id": os.environ["NHPLUG_APP_KEY"],
            "x-client-secret": os.environ["NHPLUG_APP_SECRET"],
            "authorization": f"Bearer {get_token()}",
            "content-type": "application/json; charset=UTF-8",
        }
        if cts:
            # 연속 요청은 키(cts)와 플래그(cts_flag=Y) **둘 다** 필요하다 — 키만 보내면
            # 서버가 같은 첫 페이지를 반복해 돌려준다(실측).
            headers["cts"] = cts
            headers["cts_flag"] = "Y"
        try:
            _throttle()
            res = requests.post(url, headers=headers, data=body, timeout=10)
        except requests.RequestException as exc:
            raise BrokerApiError(f"나무증권 API 네트워크 오류: {exc}") from exc
        if res.status_code == 401 and not refreshed:
            clear_token()
            refreshed = True
            continue
        try:
            data = res.json()
        except Exception as exc:
            raise BrokerApiError(f"나무증권 API 응답 해석 실패 (HTTP {res.status_code})") from exc
        if not res.ok:
            raise BrokerApiError(
                f"나무증권 API 오류: {data.get('rsp_msg') or res.status_code} (코드 {data.get('rsp_cd')})"
            )
        code = str(data.get("rsp_cd") or "")
        has_more = code == "00218"
        if not has_more and not is_success(code, data.get("rsp_msg")):
            raise BrokerApiError(f"나무증권 API 오류: {data.get('rsp_msg') or '업무 오류'} (코드 {code})")
        if not merged:
            merged = {k: v for k, v in data.items() if k != list_key}
        items.extend(data.get(list_key) or [])
        if not has_more:
            break
        cts = res.headers.get("cts", "")
        if not cts:
            raise BrokerApiError("나무증권 API 연속조회 키(cts)가 응답 헤더에 없습니다.")
    else:
        raise BrokerApiError("나무증권 API 연속조회가 20페이지를 넘었습니다 — 응답을 확인하세요.")
    merged[list_key] = items
    return merged


def _namu_balance(account_no: str) -> dict[str, Any]:
    """국내주식 잔고 원본 응답 — Output_0 요약, Output_1 보유 목록(연속조회 병합)."""
    return _namu_call_paged(
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
            cash, holdings = _normalize_balance(data)
            summary = data.get("Output_0", {}) or {}
            row.update(
                {
                    "ok": True,
                    "cash": cash,
                    # 순자산금액(nas_amt) — 평가액+현금. 미리보기가 계좌 규모를 한눈에 보여준다.
                    "net_asset": float(summary.get("nas_amt") or 0),
                    "holdings_count": len(holdings),
                }
            )
        except BrokerApiError as exc:
            row["error"] = str(exc)
        rows.append(row)
    # 조회 가능한 계좌를 위로
    rows.sort(key=lambda r: (not r["ok"], r["account_no"]))
    return rows


def _normalize_balance(data: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    """잔고 원본 → (현금 D+2, 실보유 목록). 확인 미리보기와 동기화가 같은 기준을 쓴다.

    빈 티커 행(집계/공백)과 잔량 0 행(당일 전량 매도)은 보유가 아니다 — 원시 행을
    그대로 세면 종목 수가 부풀려 보인다.
    """
    summary = data.get("Output_0", {}) or {}
    holdings: list[dict[str, Any]] = []
    for item in data.get("Output_1", []) or []:
        ticker = str(item.get("iem_cd") or "").strip()
        quantity = float(item.get("rsdl_qty") or 0)
        if not ticker or quantity <= 0:
            continue
        holdings.append(
            {
                "ticker": ticker,
                "name": str(item.get("iem_nm") or "").strip(),
                "quantity": quantity,
                "average_buy_price": float(item.get("phs_pr") or 0),
                "current_price": float(item.get("now_pr") or 0) or None,
                "value": float(item.get("eal_amt") or 0) or None,
            }
        )
    return float(summary.get("nxt2_dd_dca") or 0), holdings


def fetch_broker_balance(provider: str, account_no: str) -> dict[str, Any]:
    """잔고를 표준형으로 정규화해 돌려주고, '적용' 용으로 잠시 캐시한다.

    필드 근거 (공식 openapi.json — /krstock/inquiry/v1/balance):
      - 현금 = `nxt2_dd_dca` (D+2 예수금) — 미결제 매수·매도가 반영된 실질 현금.
        `dca`(예수금)는 결제 전 금액이라 매수 직후에는 실제보다 크게 나온다.
      - 수량 = `rsdl_qty` (잔량수량) — 미결제 포함 현재 잔량.
      - 평단 = `phs_pr` (매입가격).
    """
    if provider != "NAMU_PLUG":
        raise BrokerApiError(f"등록되지 않은 커넥터입니다: {provider}")
    data = _namu_balance(account_no)
    summary = data.get("Output_0", {}) or {}
    cash, holdings = _normalize_balance(data)

    result = {
        "provider": provider,
        "account_no": account_no,
        "cash": cash,
        # 참고용 원본 요약 — 화면이 어떤 값을 썼는지 확인할 수 있게 함께 담는다.
        "cash_d0": float(summary.get("dca") or 0),
        "total_asset": float(summary.get("tot_aet_amt") or 0),
        "holdings": holdings,
    }
    _FETCH_CACHE.set(_FETCH_CACHE.make_key(provider, account_no), result)
    return result


def cached_broker_balance(provider: str, account_no: str) -> dict[str, Any] | None:
    """가장 최근 불러오기 결과 — '적용' 이 재호출 없이 쓴다. 없으면 None."""
    return _FETCH_CACHE.get(_FETCH_CACHE.make_key(provider, account_no))


def apply_fetched_balance(account_id: str, provider: str, fetched: dict[str, Any]) -> dict[str, Any]:
    """불러온 잔고를 portfolio_master 에 반영한다 — 수동 '덮어쓰기' 와 배치가 공용.

    기존 보유의 메모·매수일·정렬은 보존하고 수량·평단·현금만 증권사 값으로 바꾼다.
    변경 주체(updated_by)는 커넥터 id 로 남는다.
    """
    from utils.portfolio_io import load_portfolio_master, save_portfolio_master
    from utils.settings_loader import get_account_settings

    current = load_portfolio_master(account_id) or {"holdings": []}
    existing_by_ticker = {str(row.get("ticker") or ""): row for row in current.get("holdings") or []}
    currency = str((get_account_settings(account_id) or {}).get("currency") or "KRW").strip().upper()

    holdings = []
    for index, row in enumerate(fetched["holdings"]):
        base = existing_by_ticker.get(row["ticker"], {})
        holdings.append(
            {
                "ticker": row["ticker"],
                "name": row["name"] or base.get("name") or row["ticker"],
                "quantity": row["quantity"],
                "average_buy_price": row["average_buy_price"],
                "currency": base.get("currency") or currency,
                "first_buy_date": base.get("first_buy_date") or "",
                "last_buy_date": base.get("last_buy_date") or "",
                "memo": base.get("memo") or "",
                "sort_order": base.get("sort_order", index),
            }
        )

    ok = save_portfolio_master(account_id, holdings, cash_balance=fetched["cash"], updated_by=provider)
    if not ok:
        raise BrokerApiError("portfolio_master 저장에 실패했습니다.")
    return {"cash": fetched["cash"], "holdings_count": len(holdings)}
