"""증권사 API 커넥터 — 계좌 설정의 `broker_api` 연동이 쓰는 단일 진입점.

커넥터는 레지스트리(`PROVIDERS`)에 등록한다. 환경변수는 `<PROVIDER>_API_KEY` /
`<PROVIDER>_API_SECRET` 규칙을 따른다 (예: NAMU_PLUG_API_KEY).

지금은 나무증권(NH PLUG, `nhplug` SDK) 하나다. 조회는 전부 읽기 전용이다 —
주문·이체 API 는 여기서 다루지 않는다.
"""

from __future__ import annotations

import os
import re
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
# 스로틀 자체는 SDK 가 하고(`NHPLUG_RATE_LIMIT`, 슬라이딩 1초 창), 우리는 값만 정한다 —
# 여기서 따로 재면 SDK 호출분과 이중으로 걸려 간격이 두 배가 된다.
_CALLS_PER_SECOND = "1"

# 업무 성공 판정 — SDK 0.4.0 부터 `call()` 은 HTTP 상태만 보고 업무 판정을 하지 않는다
# (같은 rsp_cd 가 API 마다 뜻이 달라 SDK 가 판정하면 오판한다는 이유로 빠졌다).
# 나무증권 조회 API 의 성공 코드·메시지 규칙을 여기 한 곳에 둔다.
_SUCCESS_CODES = frozenset({"00000", "00166", "00221", "13578"})
#: 성공 메시지 안전망 — NH 성공 응답은 "…완료되었습니다" 형태다. 목록에 없는 정상 코드를
#: 실패로 오판하지 않기 위한 2차 방어.
_SUCCESS_MESSAGE_RE = re.compile(r"완료")

# 연속조회 폭주 방지 — 잔고가 이 페이지 수를 넘을 일은 없다.
_MAX_PAGES = 20

# 등록된 커넥터 — 화면 셀렉트가 이 목록을 그대로 쓴다.
PROVIDERS: tuple[dict[str, str], ...] = ({"id": "NAMU_PLUG", "name": "나무증권 (NH PLUG)"},)


def _is_business_success(rsp_cd: str | None, rsp_msg: str | None) -> bool:
    """업무 성공 여부 — 코드 allowlist 우선, 없으면 메시지의 '완료' 로 본다.

    `rsp_cd` 가 없는 응답(토큰 등)은 판정 대상이 아니라 성공으로 둔다.
    """
    if rsp_cd is None:
        return True
    if str(rsp_cd) in _SUCCESS_CODES:
        return True
    return bool(rsp_msg and _SUCCESS_MESSAGE_RE.search(rsp_msg))


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
    os.environ.setdefault("NHPLUG_RATE_LIMIT", _CALLS_PER_SECOND)


# ── 토큰 공유(DB) ─────────────────────────────────────────────────────────
# NH 는 앱키당 유효 토큰이 1개다(새 발급 = 이전 토큰 무효 + 고객 알림톡). 로컬·서버 워커가
# 각자 파일 캐시로 발급하면 서로 토큰을 죽이는 핑퐁이 되므로, **DB 문서 하나를 단일 소스**로
# 두고 호출 전 nhplug 파일 캐시에 심는다(SDK 단건 call() 과 연속조회가 같은 캐시를 읽는다).
# 발급이 일어나면(파일 캐시 값이 DB 와 달라짐) DB 에 올린다. 배치 큐가 잡을 한 워커만
# claim 하므로 동시 발급 경합은 없다.
_TOKEN_DOC_ID = "nhplug_token"


def _token_db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    return db["system_config"] if db is not None else None


def _read_local_token() -> tuple[str, float] | None:
    import json

    from nhplug.auth import cache_path

    try:
        data = json.loads(cache_path().read_text(encoding="utf-8"))
        if data.get("token"):
            return str(data["token"]), float(data.get("exp", 0))
    except Exception:
        return None
    return None


def _seed_token_from_db() -> None:
    """DB 의 공유 토큰이 유효하면 nhplug 파일 캐시에 심는다 (없으면 아무것도 안 함)."""
    import json

    from nhplug.auth import cache_path

    coll = _token_db()
    if coll is None:
        return
    doc = coll.find_one({"_id": _TOKEN_DOC_ID}) or {}
    token, exp = doc.get("token"), float(doc.get("exp") or 0)
    if not token or exp <= time.time() + 60:
        return
    local = _read_local_token()
    if local and local[0] == token:
        return
    try:
        path = cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({"token": token, "exp": exp}), encoding="utf-8")
        os.replace(tmp, path)
        # 메모리 캐시가 옛 토큰을 들고 있으면 파일을 안 읽는다 — 비워서 파일을 다시 읽게 한다.
        from nhplug.auth import _cache

        _cache["token"] = None
        _cache["exp"] = 0.0
    except Exception as exc:  # 공유 실패는 치명적이지 않다 — 각자 발급으로 동작은 한다
        logger.warning("[BROKER-SYNC] 공유 토큰 시딩 실패: %s", exc)


def _publish_token_to_db() -> None:
    """이번 호출에서 토큰이 새로 발급됐으면 DB 에 올린다."""
    local = _read_local_token()
    coll = _token_db()
    if not local or coll is None:
        return
    token, exp = local
    doc = coll.find_one({"_id": _TOKEN_DOC_ID}) or {}
    if doc.get("token") != token:
        coll.update_one({"_id": _TOKEN_DOC_ID}, {"$set": {"token": token, "exp": exp}}, upsert=True)
        logger.info(
            "[BROKER-SYNC] 새 나무증권 토큰을 DB 에 공유했습니다 (만료 %s)",
            time.strftime("%m-%d %H:%M", time.localtime(exp)),
        )


def _namu_call(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """nhplug 단건 호출 (연속조회 없는 API 용).

    SDK 는 HTTP 오류만 예외로 올린다 — 업무 오류(HTTP 200 + 실패 rsp_cd)는 여기서 가른다.
    """
    _ensure_env()
    try:
        from nhplug import NhplugError, call, status_of
    except ImportError as exc:
        raise BrokerApiError("nhplug 패키지가 설치돼 있지 않습니다 (pip install nhplug).") from exc
    _seed_token_from_db()
    try:
        result = call(path, payload)
    except NhplugError as exc:
        raise BrokerApiError(f"나무증권 API 오류: {exc.message} (코드 {exc.code})") from exc
    code, message = status_of(result)
    if not _is_business_success(code, message):
        raise BrokerApiError(f"나무증권 API 오류: {message or '업무 오류'} (코드 {code})")
    _publish_token_to_db()
    return result


def _namu_call_paged(path: str, payload: dict[str, Any], list_key: str = "Output_1") -> dict[str, Any]:
    """연속조회 지원 호출 — 목록(list_key)을 전 페이지 이어 붙여 돌려준다.

    연속키(cts·cts_flag) 주고받기·토큰 재발급·유량 스로틀은 전부 SDK `paginate()` 가 한다.
    여기서 하는 건 **업무 성공 판정과 페이지 병합**뿐이다 — 마지막 페이지만 판정한다
    (중간 페이지의 '계속' 코드는 오류가 아니다).
    """
    _ensure_env()
    try:
        from nhplug import NhplugError, paginate
    except ImportError as exc:
        raise BrokerApiError("nhplug 패키지가 설치돼 있지 않습니다 (pip install nhplug).") from exc

    _seed_token_from_db()
    merged: dict[str, Any] = {}
    items: list[Any] = []
    try:
        for page, (data, meta) in enumerate(paginate(path, payload, want_meta=True), start=1):
            if not meta.has_next and not _is_business_success(meta.rsp_cd, meta.rsp_msg):
                raise BrokerApiError(f"나무증권 API 오류: {meta.rsp_msg or '업무 오류'} (코드 {meta.rsp_cd})")
            if not merged:
                merged = {key: value for key, value in data.items() if key != list_key}
            items.extend(data.get(list_key) or [])
            # 상한에 닿았는데 더 남았으면 잘린 목록을 돌려주지 않고 막는다.
            if page >= _MAX_PAGES and meta.has_next:
                raise BrokerApiError(f"나무증권 API 연속조회가 {_MAX_PAGES}페이지를 넘었습니다 — 응답을 확인하세요.")
    except NhplugError as exc:
        raise BrokerApiError(f"나무증권 API 오류: {exc.message} (코드 {exc.code})") from exc
    merged[list_key] = items
    _publish_token_to_db()
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

    ok = save_portfolio_master(
        account_id,
        holdings,
        cash_balance=fetched["cash"],
        # 자산 화면이 우선하는 다통화 맵도 함께 — 계좌 통화만 갱신(다른 통화 잔액 보존).
        cash_map={currency: fetched["cash"]},
        updated_by=provider,
    )
    if not ok:
        raise BrokerApiError("portfolio_master 저장에 실패했습니다.")
    return {"cash": fetched["cash"], "holdings_count": len(holdings)}
