"""메리츠증권 Open API — 계좌 잔고 조회(국내·미국 계좌)와 국내주식 호가·지정가 주문·취소·주문내역(사고팔기).

키는 증권사 계좌 하나에 묶여 있다. 그래서 키 이름이 우리 계좌 ID 를 따르고(`{계좌ID}_API_KEY/SECRET`,
`broker_api_service.require_env_keys`), 요청에 계좌번호를 넣지 않는다. 모의투자 서버는 없다(운영만).

- 오류는 종류와 무관하게 HTTP 500 으로 온다 — 성공·실패는 본문 `rsp_cd` 로만 판정한다. 성공 코드는 API 계열마다
  다르다(잔고·주문내역 5766, 호가 0000, 주문·취소 0001).
- 접근토큰(유효 12시간)은 계좌별로 DB(`system_config`) 문서 하나에 공유한다 — 로컬·서버 워커가
  각자 발급하지 않게 한다(나무증권과 같은 이유).
- 모든 요청에 `mac_address`(호출 단말 MAC, 12자리) 헤더가 필수다. 환경변수 `MERITZ_MAC_ADDRESS` 로 정한다 —
  프로세스에서 읽는 MAC 은 macOS 사설 주소·컨테이너 재생성으로 실행마다 달라져 단말을 대표하지 못한다.
- 초당 호출 한도는 문서에 숫자가 없다 — 나무증권과 같이 호출 사이를 1초 띄운다.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Callable
from datetime import date, datetime
from math import isfinite
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import requests

from services.broker_api_service import MERITZ_PROVIDER, BrokerApiError, OrderUncertainError, require_env_keys
from utils.logger import get_app_logger
from utils.strategy_trade_signal_lock import SignalBusyError, broker_token_lock

logger = get_app_logger()

BASE_URL = "https://openapi.imeritz.com:9443"
TOKEN_PATH = "/oauth2/token"
VALUATION_PATH = "/accounts/v1/me/valuation"
HOLDINGS_PATH = "/accounts/v1/me/holdings"
OVERSEAS_HOLDINGS_PATH = "/accounts/v1/me/overseas/holdings"
CURRENCIES_PATH = "/accounts/v1/me/overseas/currencies"
ORDERBOOK_PATH = "/market/v1/orderbook"
ORDER_PATH = {"buy": "/trading/v1/orders/buy", "sell": "/trading/v1/orders/sell"}
MODIFY_PATH = "/trading/v1/orders/modify"
CANCEL_PATH = "/trading/v1/orders/cancel"
ORDER_HISTORY_PATH = "/trading/v1/orders/history"

# 조회 계열 성공 — 완료·연속조회 더 있음·자료 없음. 이 밖의 rsp_cd 는 전부 오류다.
_INQUIRY_CODES = frozenset({"5766", "5762", "5820"})
_QUOTE_CODES = frozenset({"0000"})
_ORDER_CODES = frozenset({"0001"})
# 토큰 만료·무효 — 받으면 한 번만 새로 발급해 다시 보낸다.
_TOKEN_EXPIRED_CODE = "EGW00123"
_TOKEN_ERROR_CODES = frozenset({"EGW00121", _TOKEN_EXPIRED_CODE})


class MeritzTokenExpiredError(BrokerApiError):
    """조회 토큰을 다시 발급한 뒤에도 메리츠가 만료로 거부했다."""


class MeritzTokenRefreshBusyError(MeritzTokenExpiredError):
    """유효한 인증을 아직 확보하지 못했고 다른 작업이 갱신 중이어서 다음 조회를 기다린다."""


# 잔고 행의 처리대상구분 — 국내주식("1")·해외주식("J")이 한 목록에 섞여 온다.
_DOMESTIC_ROW = "1"
# 해외 잔고구분 — "1" 일반, "2" 소수, "3" 대출.
_US_GENERAL_ROW = "1"

# 거래소(KRX·NXT)는 호출하는 쪽이 정한다(`utils.market_session.kor_order_venue`) — 주문 코드와 호가 시장 코드.
_EXCHANGE_CODES = {"KRX": "01", "NXT": "02"}
_ORDERBOOK_CODES = {"KRX": "J", "NXT": "NJ"}
_LIMIT_ORDER = "01"
_NO_CONDITION = "0"
# 주문 응답의 경고구분 — "0" 정상 접수, 아래 '보류'는 접수 여부를 단정할 수 없다. 그 밖은 경고(미접수, 사람이 확인).
_WARN_OK = "0"
_WARN_HOLD = frozenset({"2", "5", "8", "b", "e", "h"})
# 주문내역 행 — 정정취소구분(1 신규·2 정정·3 취소), 매매구분(1 매도·2 매수), 처리상태(3 거부).
_ROW_CANCEL = "3"
_SIDE_LABELS = {"1": "매도", "2": "매수"}
_STATE_REJECTED = "3"

_MIN_INTERVAL_SECONDS = 1.0
_REQUEST_TIMEOUT_SECONDS = 10
# 일시적 조회 통신 오류만 재시도한다. 기존 최소 호출 간격은 별도로 유지한다.
_INQUIRY_RETRY_DELAYS = (0.5, 1.0)
_MAX_PAGES = 20
# 만료 직전 토큰은 쓰지 않는다 — 요청 도중 만료되지 않게 여유를 둔다.
_TOKEN_MARGIN_SECONDS = 60
# 갱신 직후의 만료 응답은 GET만 잠깐 기다린다. 토큰 발급과 주문을 반복하지 않는다.
_TOKEN_RETRY_DELAYS = (1.0, 2.0)
_TOKEN_REISSUE_COOLDOWN_SECONDS = 5

_call_lock = threading.Lock()
_last_call_at = 0.0


MAC_ENV = "MERITZ_MAC_ADDRESS"
_MAC_RE = re.compile(r"[0-9A-F]{12}")


def _mac_address() -> str:
    """요청 헤더 `mac_address` — 환경변수의 MAC(대문자 16진수 12자리). 없거나 형식이 다르면 오류다."""
    value = os.environ.get(MAC_ENV, "").strip()
    if not _MAC_RE.fullmatch(value):
        raise BrokerApiError(f".env 에 {MAC_ENV}(대문자 16진수 12자리, 예: A1B2C3D4E5F6)가 필요합니다.")
    return value


# ── 토큰(DB 공유) ─────────────────────────────────────────────────────────


def _token_collection():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise BrokerApiError("메리츠증권 토큰 공유 DB 연결에 실패했습니다.")
    return db["system_config"]


def _token_doc_id(account_id: str) -> str:
    return f"meritz_token:{account_id}"


def _token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def _token_claims(token: str) -> dict[str, int | float]:
    """서명 인증은 서버에 맡기고 JWT의 발급·만료 시각만 진단에 쓴다. 불투명 토큰에는 클레임이 없다."""
    if token.count(".") != 2:
        return {}
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        exp = claims["exp"]
        if isinstance(exp, bool) or not isinstance(exp, (int, float)) or not isfinite(exp):
            raise ValueError("유효하지 않은 exp")
        result = {"exp": exp}
        if "iat" in claims:
            iat = claims["iat"]
            if isinstance(iat, bool) or not isinstance(iat, (int, float)) or not isfinite(iat):
                raise ValueError("유효하지 않은 iat")
            result["iat"] = iat
        return result
    except (ValueError, KeyError, TypeError):
        raise BrokerApiError("메리츠증권 JWT의 발급·만료 시각을 확인할 수 없습니다.") from None


def _usable_token(doc: dict[str, Any]) -> bool:
    if not doc.get("token"):
        return False
    if "exp" not in doc:
        raise BrokerApiError("공유 토큰의 만료 시각이 없습니다.")
    expiry = float(doc["exp"])
    claims = _token_claims(str(doc["token"]))
    if "exp" in claims:
        expiry = min(expiry, claims["exp"])
    return isfinite(expiry) and expiry > time.time() + _TOKEN_MARGIN_SECONDS


def _token_diagnostics(account_id: str, token: str) -> str:
    doc = _token_collection().find_one({"_id": _token_doc_id(account_id)}) or {}
    claims = _token_claims(token)
    shared = str(doc.get("token") or "")
    return (
        f"token_fp={_token_fingerprint(token)} token_iat={claims.get('iat')} token_exp={claims.get('exp')} "
        f"shared_fp={_token_fingerprint(shared) if shared else None} "
        f"shared_issued_at={doc.get('issued_at')} shared_exp={doc.get('exp')}"
    )


def _issue_token(account_id: str, rejected_token: str | None) -> str:
    """계좌별 선점 후 다시 읽어, 다른 작업이 발급한 토큰이나 갱신 직후 토큰은 재사용한다."""
    try:
        with broker_token_lock(account_id):
            collection = _token_collection()
            previous = collection.find_one({"_id": _token_doc_id(account_id)}) or {}
            recently_issued = previous.get("issued_at") is not None and (
                0 <= time.time() - float(previous["issued_at"]) < _TOKEN_REISSUE_COOLDOWN_SECONDS
            )
            if _usable_token(previous) and (
                rejected_token is None or previous["token"] != rejected_token or recently_issued
            ):
                token = str(previous["token"])
                logger.info(
                    "[MERITZ] account=%s stage=authentication outcome=reused %s",
                    account_id,
                    _token_diagnostics(account_id, token),
                )
                return token
            return _fetch_token(account_id, previous)
    except SignalBusyError:
        raise MeritzTokenRefreshBusyError(
            f"{account_id} 인증 토큰을 다른 작업이 갱신 중입니다. 다음 조회에서 확인합니다."
        ) from None


def _fetch_token(account_id: str, previous: dict[str, Any]) -> str:
    """인증 요청은 한 번만 보낸다. 새 응답이 같은 토큰이어도 기존 수명을 임의로 늘리지 않는다."""
    key, secret = require_env_keys(MERITZ_PROVIDER, account_id)
    try:
        response = requests.post(
            f"{BASE_URL}{TOKEN_PATH}",
            data={"grant_type": "client_credentials", "client_id": key, "client_secret": secret, "scope": "login"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        logger.warning(
            "[MERITZ] account=%s method=POST path=%s stage=authentication outcome=failed error=%s",
            account_id,
            TOKEN_PATH,
            _transport_detail(exc),
        )
        raise BrokerApiError(f"메리츠증권 인증 서버 통신에 실패했습니다: {_transport_detail(exc)}") from None
    data = _json_of(response)
    token, expires_in = data.get("access_token"), data.get("expires_in")
    if not response.ok or not isinstance(token, str) or not token or not expires_in:
        raise BrokerApiError(f"메리츠증권 토큰 발급 실패 ({account_id}): {_message_of(data, response)}")
    now = time.time()
    expiry = now + float(expires_in)
    claims = _token_claims(token)
    if "exp" in claims:
        expiry = min(expiry, claims["exp"])
    if previous.get("token") == token:
        expiry = min(expiry, float(previous["exp"]))
    if not isfinite(expiry) or expiry <= now + _TOKEN_MARGIN_SECONDS:
        logger.warning(
            "[MERITZ] account=%s stage=authentication outcome=expired token_fp=%s token_exp=%s",
            account_id,
            _token_fingerprint(token),
            claims.get("exp"),
        )
        raise MeritzTokenExpiredError("재발급된 메리츠증권 토큰의 사용 가능한 유효기간이 부족합니다.")
    _token_collection().update_one(
        {"_id": _token_doc_id(account_id)},
        {"$set": {"token": token, "exp": expiry, "issued_at": now}},
        upsert=True,
    )
    logger.info(
        "[MERITZ] account=%s stage=authentication outcome=issued previous_fp=%s same_token=%s %s",
        account_id,
        _token_fingerprint(str(previous["token"])) if previous.get("token") else None,
        previous.get("token") == token,
        _token_diagnostics(account_id, token),
    )
    return token


def token_age_seconds(account_id: str) -> int | None:
    """공유 토큰을 발급한 지 몇 초인지 — 발급 시각이 기록되기 전의 토큰이면 None. 만료 실험 기록용이다."""
    issued_at = _token_collection().find_one({"_id": _token_doc_id(account_id)}, {"issued_at": 1}) or {}
    return int(time.time() - float(issued_at["issued_at"])) if issued_at.get("issued_at") else None


def _access_token(account_id: str) -> str:
    """DB 에 공유된 유효 토큰 — 없거나 곧 만료면 새로 발급한다."""
    doc = _token_collection().find_one({"_id": _token_doc_id(account_id)}) or {}
    if _usable_token(doc):
        return str(doc["token"])
    return _issue_token(account_id, None)


# ── 호출 ─────────────────────────────────────────────────────────────────


def _json_of(response: requests.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _message_of(data: dict[str, Any], response: requests.Response) -> str:
    parts = [str(data[key]) for key in ("rsp_msg", "rsp_sub_msg", "error_description") if data.get(key)]
    return " · ".join(parts) or f"HTTP {response.status_code}"


def _transport_detail(exc: BaseException) -> str:
    """예외 메시지·요청 URL 대신 예외 종류와 시스템 오류 번호만 남긴다."""
    pending, seen, details = [exc], set(), []
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        errno = getattr(current, "errno", None)
        details.append(f"{type(current).__name__}(errno={errno})" if isinstance(errno, int) else type(current).__name__)
        pending.extend(arg for arg in current.args if isinstance(arg, BaseException))
        pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)
    return ">".join(dict.fromkeys(details))


def _retryable_inquiry_error(exc: requests.RequestException) -> bool:
    return isinstance(
        exc, (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError)
    ) and not isinstance(exc, requests.exceptions.SSLError)


def _send_request(
    method: str, path: str, token: str, mac: str, params: dict[str, str] | None, body: dict[str, Any] | None
) -> requests.Response:
    """호출 간격은 HTTP 요청마다 적용하고 재시도 대기 중에는 다른 요청을 막지 않는다."""
    global _last_call_at
    with _call_lock:
        wait = _MIN_INTERVAL_SECONDS - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        headers = {"authorization": f"Bearer {token}", "mac_address": mac}
        try:
            if method == "GET":
                return requests.get(
                    f"{BASE_URL}{path}", headers=headers, params=params, timeout=_REQUEST_TIMEOUT_SECONDS
                )
            return requests.post(
                f"{BASE_URL}{path}",
                headers={**headers, "content-type": "application/json; charset=utf-8"},
                json=body,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        finally:
            _last_call_at = time.monotonic()


def _request(
    account_id: str,
    method: str,
    path: str,
    *,
    success: frozenset[str],
    params: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """조회 통신 오류만 제한적으로 재시도한다. POST 응답 불명은 재전송하지 않는다.

    인증 거부 응답의 토큰 재발급은 통신 재시도와 별도로 한 번만 허용한다.
    """
    if method not in ("GET", "POST"):
        raise ValueError(f"지원하지 않는 메리츠 요청 방식입니다: {method}")
    request_id = uuid4().hex[:12]
    started = time.monotonic()
    context = f"request_id={request_id} account={account_id} method={method} path={path}"
    try:
        token = _access_token(account_id)
        mac = _mac_address()
    except Exception as exc:
        logger.error(
            "[MERITZ] %s stage=authentication outcome=failed error=%s elapsed_ms=%.1f",
            context,
            type(exc).__name__,
            (time.monotonic() - started) * 1000,
        )
        raise
    attempt, retries, refreshed, token_retries = 0, 0, False, 0
    rejected_tokens: set[str] = set()
    while True:
        attempt += 1
        attempt_started = time.monotonic()
        logger.info(
            "[MERITZ] %s stage=request attempt=%s outcome=started %s",
            context,
            attempt,
            _token_diagnostics(account_id, token),
        )
        try:
            response = _send_request(method, path, token, mac, params, body)
        except requests.RequestException as exc:
            retry = method == "GET" and _retryable_inquiry_error(exc) and retries < len(_INQUIRY_RETRY_DELAYS)
            logger.warning(
                "[MERITZ] %s stage=transport attempt=%s outcome=%s error=%s attempt_elapsed_ms=%.1f elapsed_ms=%.1f",
                context,
                attempt,
                "retrying" if retry else "failed",
                _transport_detail(exc),
                (time.monotonic() - attempt_started) * 1000,
                (time.monotonic() - started) * 1000,
            )
            if retry:
                delay = _INQUIRY_RETRY_DELAYS[retries]
                retries += 1
                time.sleep(delay)
                continue
            error = OrderUncertainError if method == "POST" else BrokerApiError
            # 원본 예외 문구에 인증 URL 등이 포함될 수 있어 안전한 진단 정보만 공개한다.
            raise error(
                f"메리츠증권 {'주문 접수 여부를 확인할 수 없습니다' if method == 'POST' else '조회 통신에 실패했습니다'}: "
                f"{method} {path} · {_transport_detail(exc)} · {attempt}회 시도 · 추적 ID {request_id}"
            ) from None
        data = _json_of(response)
        code = str(data.get("rsp_cd") or "")
        logger.info(
            "[MERITZ] %s stage=response attempt=%s http_status=%s rsp_cd=%s attempt_elapsed_ms=%.1f elapsed_ms=%.1f",
            context,
            attempt,
            response.status_code,
            code,
            (time.monotonic() - attempt_started) * 1000,
            (time.monotonic() - started) * 1000,
        )
        if (response.status_code == 401 or code in _TOKEN_ERROR_CODES) and not refreshed:
            refreshed = True
            rejected_tokens.add(token)
            logger.warning("[MERITZ] %s stage=authentication attempt=%s outcome=refreshing", context, attempt)
            try:
                token = _issue_token(account_id, token)
            except Exception as exc:
                logger.error(
                    "[MERITZ] %s stage=authentication outcome=failed error=%s elapsed_ms=%.1f",
                    context,
                    type(exc).__name__,
                    (time.monotonic() - started) * 1000,
                )
                raise
            continue
        if method == "GET" and code == _TOKEN_EXPIRED_CODE and token_retries < len(_TOKEN_RETRY_DELAYS):
            rejected_tokens.add(token)
            delay = _TOKEN_RETRY_DELAYS[token_retries]
            token_retries += 1
            logger.warning(
                "[MERITZ] %s stage=authentication outcome=deferred token_retry=%s delay_seconds=%s %s",
                context,
                token_retries,
                delay,
                _token_diagnostics(account_id, token),
            )
            time.sleep(delay)
            current = _token_collection().find_one({"_id": _token_doc_id(account_id)}) or {}
            if _usable_token(current) and current["token"] not in rejected_tokens:
                token = str(current["token"])
            continue
        if code in success:
            logger.info(
                "[MERITZ] %s outcome=%s attempts=%s retries=%s token_retries=%s elapsed_ms=%.1f",
                context,
                "recovered" if retries or token_retries else "success",
                attempt,
                retries,
                token_retries,
                (time.monotonic() - started) * 1000,
            )
            return data
        logger.error(
            "[MERITZ] %s stage=response outcome=rejected attempts=%s http_status=%s rsp_cd=%s",
            context,
            attempt,
            response.status_code,
            code,
        )
        error = MeritzTokenExpiredError if method == "GET" and code == _TOKEN_EXPIRED_CODE else BrokerApiError
        raise error(f"메리츠증권 API 오류: {_message_of(data, response)} (코드 {code or '-'}, 추적 ID {request_id})")


def _get(account_id: str, path: str, params: dict[str, str]) -> dict[str, Any]:
    """조회 계열 GET 한 건 → 본문(완료·연속·자료 없음)."""
    return _request(account_id, "GET", path, success=_INQUIRY_CODES, params=params)


def _get_paged(account_id: str, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
    """연속조회 — `data` 배열을 전 페이지 이어 붙인다. 자료 없음(5820)은 빈 목록이다."""
    rows: list[dict[str, Any]] = []
    page_params = dict(params)
    for _ in range(_MAX_PAGES):
        data = _get(account_id, path, page_params)
        rows.extend(data.get("data") or [])
        if str(data.get("tr_cont") or "0") != "1":
            return rows
        next_key = data.get("tr_cont_key")
        if not next_key:
            raise BrokerApiError("메리츠증권 연속조회 키가 없습니다 — 응답을 확인하세요.")
        page_params = {**params, "tr_cont": "1", "tr_cont_key": str(next_key)}
    raise BrokerApiError(f"메리츠증권 연속조회가 {_MAX_PAGES}페이지를 넘었습니다 — 응답을 확인하세요.")


# ── 잔고 ─────────────────────────────────────────────────────────────────


def _domestic_ticker(iscd: str) -> str:
    """국내 종목코드 "A005930" → "005930". 형식이 다르면 오류다."""
    if len(iscd) != 7 or not iscd.startswith("A"):
        raise BrokerApiError(f"메리츠증권 국내 종목코드 형식이 예상과 다릅니다: {iscd!r}")
    return iscd[1:]


def _require_unique(ticker: str, name: Any, seen: set[str]) -> None:
    """같은 종목이 여러 행(일반·신용·소수 등)이면 평단을 합쳐야 한다 — 임의로 합치지 않고 알린다."""
    if ticker in seen:
        raise BrokerApiError(f"같은 종목의 잔고가 여러 행입니다(일반·신용 등): {name}({ticker})")
    seen.add(ticker)


def _domestic_holdings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """국내 잔고 행(통합 잔고 목록) → 표준 보유 목록. 해외 행이 섞여 있으면 막는다(계좌 국가 하나)."""
    holdings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        iscd = str(row.get("iscd") or "").strip()
        if str(row.get("pros_trgt_cls_code") or "") != _DOMESTIC_ROW:
            raise BrokerApiError(
                f"국내 계좌에 해외 잔고가 있습니다: {row.get('isnm')}({iscd}) — 계좌 국가 설정을 확인하세요."
            )
        quantity = int(float(row.get("rmnd_qty") or 0))
        if quantity <= 0:
            continue
        ticker = _domestic_ticker(iscd)
        _require_unique(ticker, row.get("isnm"), seen)
        holdings.append(
            {
                "ticker": ticker,
                "name": str(row.get("isnm") or "").strip(),
                "quantity": quantity,
                "average_buy_price": float(row.get("byng_unpr") or 0),
                "current_price": float(row.get("psnt_prc") or 0) or None,
                "value": float(row.get("vltn_amt") or 0) or None,
            }
        )
    return holdings


def _us_holdings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """해외 잔고 행 → 표준 보유 목록(단가는 USD). 일반 잔고·USD·정수 수량만 받는다.

    티커는 거래소 접미사를 뗀 심볼(`ovrs_stck_symb_iscd`, "BRK.B")을 공통 시세 심볼("BRK-B")로 바꾼다.
    소수점 잔고는 자산 계산이 수량을 정수로 내려 평가가 빠지므로 막는다.
    """
    from utils.normalization import normalize_exchange_symbol

    holdings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        symbol = str(row.get("ovrs_stck_symb_iscd") or "").strip()
        name = str(row.get("isnm") or "").strip()
        if str(row.get("rmnd_clcd") or "") != _US_GENERAL_ROW:
            raise BrokerApiError(f"일반 잔고가 아닌 해외 잔고({row.get('prod_cls_name')})가 있습니다: {name}({symbol})")
        if str(row.get("crcd") or "") != "USD":
            raise BrokerApiError(f"USD 가 아닌 해외 잔고({row.get('crcd')})가 있습니다: {name}({symbol})")
        quantity = float(row.get("rmnd_dqty") or 0)
        if quantity != int(quantity):
            raise BrokerApiError(f"소수점 수량 잔고({quantity})는 동기화하지 않습니다: {name}({symbol})")
        if quantity <= 0:
            continue
        if not symbol:
            raise BrokerApiError(f"해외 잔고의 심볼이 비어 있습니다: {name}({row.get('iscd')})")
        ticker = normalize_exchange_symbol(symbol, suffix="")
        _require_unique(ticker, name, seen)
        holdings.append(
            {
                "ticker": ticker,
                "name": name,
                "quantity": int(quantity),
                "average_buy_price": float(row.get("avrg_unpr") or 0),
                "current_price": float(row.get("psnt_prc") or 0) or None,
                "value": float(row.get("frcr_vltn_amt") or 0) or None,
            }
        )
    return holdings


def _foreign_cash(account_id: str) -> dict[str, float]:
    """통화별 외화 예수금 — 추정 외화 예수금(`estm_frcr_amt`, 미결제 반영)이다. 원화 D+2 와 같은 기준."""
    rows = _get(account_id, CURRENCIES_PATH, {}).get("data") or []
    cash: dict[str, float] = {}
    for row in rows:
        code = str(row.get("crcd") or "").strip().upper()
        if not code or code in cash:
            raise BrokerApiError(f"메리츠증권 외화 잔고의 통화가 비었거나 겹칩니다: {code!r}")
        cash[code] = float(row.get("estm_frcr_amt") or 0)
    return cash


def fetch_balance(account_id: str, country_code: str) -> dict[str, Any]:
    """계좌 잔고 → 표준형(통화별 현금·총자산·보유). 계좌 국가(kor·us)의 잔고만 받는다.

    - 원화 현금은 D+2 예수금(`d2_aft_tfam`, 미결제 반영 — 나무 `nxt2_dd_dca` 와 같은 기준),
      외화는 추정 외화 예수금(`estm_frcr_amt`). 외화 예수금이 있을 때만 외화 잔고를 조회한다.
    - 잔고 목록은 국내·해외가 한 목록에 섞여 오므로, 계좌 국가와 다른 행이 있으면 막는다.
      미국 계좌의 종목 상세(평균단가 USD 등)는 해외주식 잔고 조회에서 받는다.
    """
    valuation = _get(account_id, VALUATION_PATH, {}).get("data") or {}
    if not valuation:
        raise BrokerApiError("메리츠증권 자산평가 응답이 비어 있습니다.")
    cash_map = {"KRW": float(valuation["d2_aft_tfam"])}
    if float(valuation.get("frcr_tfam") or 0) != 0:
        foreign = _foreign_cash(account_id)
        if "KRW" in foreign:
            raise BrokerApiError("메리츠증권 외화 잔고에 KRW 가 있습니다 — 응답을 확인하세요.")
        cash_map.update(foreign)

    combined = _get_paged(account_id, HOLDINGS_PATH, {"nthn_rmnd_incu_yn": "N"})
    if country_code == "kor":
        holdings = _domestic_holdings(combined)
    elif country_code == "us":
        domestic = [row for row in combined if str(row.get("pros_trgt_cls_code") or "") == _DOMESTIC_ROW]
        if domestic:
            names = ", ".join(f"{row.get('isnm')}({row.get('iscd')})" for row in domestic)
            raise BrokerApiError(f"미국 계좌에 국내 잔고가 있습니다: {names} — 계좌 국가 설정을 확인하세요.")
        holdings = _us_holdings(_get_paged(account_id, OVERSEAS_HOLDINGS_PATH, {}))
    else:
        raise BrokerApiError(f"메리츠증권 잔고 동기화는 국내(kor)·미국(us) 계좌만 지원합니다: {country_code}")
    return {
        "cash_map": cash_map,
        "total_asset": float(valuation["aset_vltn_amt"]),
        "holdings": holdings,
    }


# ── 국내주식 주문(사고팔기) ─────────────────────────────────────────────────


#: 호가 응답에 오는 단계 수 — 국내·미국 모두 매도·매수 10단계와 잔량을 준다.
_QUOTE_DEPTH_LEVELS = 10


def _quote_depth(
    data: dict[str, Any], price_key: Callable[[str, int], str], cast: Callable[[Any], Any], prev_close_key: str
) -> dict:
    """호가 응답의 단계별 가격·잔량과 전일 종가 — {"asks": [{price, quantity}, …1호가부터], "bids": […], "prev_close"}.

    가격이 0 인 단계는 빼고, 전일 종가가 0 이면(장 밖 등) None 이다 — 지어내지 않는다.
    """
    prev_close = cast(data[prev_close_key])
    depth: dict[str, Any] = {"prev_close": prev_close if prev_close > 0 else None}
    for side, prefix in (("asks", "askp"), ("bids", "bidp")):
        levels = []
        for level in range(1, _QUOTE_DEPTH_LEVELS + 1):
            price = cast(data[price_key(prefix, level)])
            if price > 0:
                levels.append({"price": price, "quantity": int(float(data[f"{prefix}_rsqn{level}"]))})
        depth[side] = levels
    return depth


def fetch_best_quote(account_id: str, ticker: str, venue: str) -> dict[str, Any]:
    """KRX 1호가 — 매도·매수 1호가와 호가 시각, 10단계 호가(`depth`). 호가가 비어 있으면 주문 가격을 못 정해 오류다(나무와 같은 표준형)."""
    data = (
        _request(
            account_id,
            "GET",
            ORDERBOOK_PATH,
            success=_QUOTE_CODES,
            params={"mrkt_div_code": _ORDERBOOK_CODES[venue], "iscd": ticker},
        ).get("data")
        or {}
    )
    # 없는 종목코드도 정상 코드로 온다 — 응답의 단축종목코드로 확인한다.
    if str(data.get("shrn_iscd") or "").strip() != ticker:
        raise BrokerApiError(f"메리츠증권 호가 응답의 종목이 다릅니다: {ticker} → {data.get('shrn_iscd')!r}")
    result = {
        "ask": int(data["askp"]),
        "bid": int(data["bidp"]),
        "at": str(data["prpr_hour"]),
        # 국내 응답은 1호가만 번호 없이(askp·bidp) 오고 2호가부터 번호가 붙는다.
        # 기준가(sdpr) — 국내 등락률의 기준인 전일 종가.
        "depth": _quote_depth(data, lambda prefix, level: prefix if level == 1 else f"{prefix}{level}", int, "sdpr"),
    }
    if result["ask"] <= 0 or result["bid"] <= 0:
        raise BrokerApiError(
            f"{ticker} 의 매도·매수 1호가가 없어 주문 가격을 정할 수 없습니다 (매도 {result['ask']}, 매수 {result['bid']})."
        )
    return result


def _accepted_order_no(data: dict[str, Any], what: str) -> str:
    """주문·취소 응답에서 주문번호를 꺼낸다 — 정상 접수(경고구분 "0")이고 번호가 있어야 접수된 것이다.

    경고는 증권사가 사람의 확인을 요구한 것이라 자동으로 재전송하지 않는다(미접수). 보류는 접수 여부를 모른다.
    """
    output = data.get("data") or {}
    warn = str(output.get("warn_cls_code") or "")
    message = str(output.get("warn_msg") or data.get("rsp_msg") or "").strip()
    if warn in _WARN_HOLD:
        raise OrderUncertainError(
            f"{what}이 보류됐습니다 — 증권사 주문내역에서 확인하세요: {message} (경고구분 {warn})"
        )
    if warn != _WARN_OK:
        raise BrokerApiError(f"{what}이 접수되지 않았습니다(증권사 경고): {message} (경고구분 {warn or '-'})")
    order_no = output.get("oder_no")
    if not isinstance(order_no, (int, float)) or int(order_no) <= 0:
        raise OrderUncertainError(f"{what} 응답에 주문번호가 없습니다 — 증권사 주문내역에서 확인하세요: {message}")
    return str(int(order_no))


def place_limit_order(account_id: str, side: str, ticker: str, quantity: int, price: int, venue: str) -> dict[str, str]:
    """KRX 지정가 주문 한 건 — 접수되면 주문번호를 돌려준다. side 는 buy·sell."""
    if side not in ORDER_PATH:
        raise BrokerApiError(f"알 수 없는 주문 구분입니다: {side}")
    if quantity <= 0 or price <= 0:
        raise BrokerApiError("수량과 가격은 0보다 커야 합니다.")
    data = _request(
        account_id,
        "POST",
        ORDER_PATH[side],
        success=_ORDER_CODES,
        body={
            "iscd": f"A{ticker}",
            "oder_unpr": price,
            "odqt": quantity,
            "oder_cls_code": _LIMIT_ORDER,
            "oder_cond_cls_code": _NO_CONDITION,
            "orgl_oder_no": 0,
            "whol_rctf_cncl_yn": "N",
            "warn_cnfr_yn": "N",
            "exch_kind_code": _EXCHANGE_CODES[venue],
        },
    )
    return {"order_no": _accepted_order_no(data, "주문"), "message": str(data.get("rsp_msg") or "")}


def modify_order(account_id: str, order_no: str, ticker: str, quantity: int, price: int, venue: str) -> dict[str, str]:
    """KRX 주문을 전량 지정가 정정한다 — 돌려주는 번호는 정정 요청의 새 주문번호다."""
    if quantity <= 0 or price <= 0:
        raise BrokerApiError("정정 수량과 가격은 0보다 커야 합니다.")
    data = _request(
        account_id,
        "POST",
        MODIFY_PATH,
        success=_ORDER_CODES,
        body={
            "iscd": f"A{ticker}",
            "oder_unpr": int(price),
            "odqt": int(quantity),
            "oder_cls_code": _LIMIT_ORDER,
            "oder_cond_cls_code": _NO_CONDITION,
            "orgl_oder_no": int(order_no),
            "whol_rctf_cncl_yn": "Y",
            "warn_cnfr_yn": "N",
            "exch_kind_code": _EXCHANGE_CODES[venue],
        },
    )
    return {"modify_order_no": _accepted_order_no(data, "정정"), "message": str(data.get("rsp_msg") or "")}


def cancel_order(account_id: str, order_no: str, ticker: str, venue: str) -> dict[str, str]:
    """주문을 전량 취소한다 — 돌려주는 번호는 취소 요청 자체의 주문번호다. ``venue`` 는 원주문의 거래소다(필수)."""
    data = _request(
        account_id,
        "POST",
        CANCEL_PATH,
        success=_ORDER_CODES,
        body={
            "iscd": f"A{ticker}",
            "orgl_oder_no": int(order_no),
            "whol_rctf_cncl_yn": "Y",
            "warn_cnfr_yn": "N",
            "exch_kind_code": _EXCHANGE_CODES[venue],
        },
    )
    return {"cancel_order_no": _accepted_order_no(data, "취소"), "message": str(data.get("rsp_msg") or "")}


def _order_status(row: dict[str, Any]) -> str:
    """주문내역 한 줄 → 주문 상태(pending·partial·filled·canceled·rejected) — 나무 `order_status` 와 같은 판정."""
    quantity = float(row.get("odqt") or 0)
    filled = float(row.get("total_cntg_qty") or 0)
    if str(row.get("oder_pros_stat_cls_code") or "") == _STATE_REJECTED:
        return "rejected"
    if quantity > 0 and filled >= quantity:
        return "filled"
    if float(row.get("cncl_qty") or 0) > 0:
        return "canceled"
    if filled > 0:
        return "partial"
    return "pending"


def _day_order_row(row: dict[str, Any]) -> dict[str, Any]:
    """주문내역 한 줄 → 표준 주문 행(나무 `_day_order_row` 와 같은 모양). `origin_order_no` 는 정정 행만 0 이 아니다."""
    side = str(row.get("deal_cls_code") or "")
    if side not in _SIDE_LABELS:
        raise BrokerApiError(f"메리츠증권 주문내역의 매매구분을 알 수 없습니다: {side!r}")
    return {
        "order_no": str(int(row.get("oder_no") or 0)),
        "origin_order_no": str(int(row.get("orgl_oder_no") or 0)),
        "ticker": _domestic_ticker(str(row.get("iscd") or "").strip()),
        "name": str(row.get("kor_issu_abwr_name") or "").strip(),
        "side_label": _SIDE_LABELS[side],
        "quantity": int(float(row.get("odqt") or 0)),
        "price": float(row.get("oder_unpr") or 0),
        "filled_quantity": int(float(row.get("total_cntg_qty") or 0)),
        "unfilled_quantity": int(float(row.get("ncnt_qty") or 0)),
        "canceled_quantity": int(float(row.get("cncl_qty") or 0)),
        "average_fill_price": float(row.get("cntg_unpr") or 0),
        "status": _order_status(row),
    }


def day_order_rows(account_id: str, yyyymmdd: str) -> list[dict[str, Any]]:
    """그날의 주문 행(주문내역 조회) — 표준 주문 행이고 정정 행은 합치지 않았다(`broker_trade_service.merge_amendments`).

    취소 요청 행은 뺀다 — 취소 결과(취소수량)는 취소된 주문 행 자신에 남는다.
    """
    rows = _get_paged(
        account_id,
        ORDER_HISTORY_PATH,
        {"oder_date": yyyymmdd, "deal_cls_code": "0", "cntg_cls_code": "0", "sort_stnd_code": "2"},
    )
    return [_day_order_row(row) for row in rows if str(row.get("rctf_cncl_cls_code") or "") != _ROW_CANCEL]


# ── 해외주식(미국) 주문(사고팔기) ───────────────────────────────────────────
#
# 국내 주문과 코드 체계가 달라(메리츠 명세: "코드를 공용으로 쓰지 마십시오") 함수를 따로 둔다. 표준형은 국내와 같다.
# 거래소는 시세용 코드(OQ 나스닥·NY 뉴욕·AX 아멕스)로 전략에 저장하고, 주문에는 벤더 번호로 바꿔 보낸다.
# 세션(프리·정규·애프터·데이장)은 주문 요청에 구분이 없다 — 같은 지정가 주문이 세션에 맞게 접수된다.

OVERSEAS_ORDER_PATH = {"buy": "/trading/v1/overseas/orders/buy", "sell": "/trading/v1/overseas/orders/sell"}
OVERSEAS_MODIFY_PATH = "/trading/v1/overseas/orders/modify"
OVERSEAS_CANCEL_PATH = "/trading/v1/overseas/orders/cancel"
OVERSEAS_ORDERBOOK_PATH = "/market/v1/overseas/orderbook"
OVERSEAS_PRICE_PATH = "/market/v1/overseas/prices"
OVERSEAS_ORDER_HISTORY_PATH = "/trading/v1/overseas/orders/history"
OVERSEAS_TODAY_ORDERS_PATH = "/trading/v1/overseas/orders/today"
#: 시세 거래소 코드 → 주문 거래소 벤더 번호.
US_EXCHANGE_VENDORS = {"OQ": "0537", "NY": "0321", "AX": "0066"}
_OVERSEAS_MARKET = "OV"
_REALTIME_QUOTE = "0"
_OVERSEAS_LIMIT_ORDER = "01"
# 주문내역 행 — 정정취소구분(0 원주문·1 정정·2 취소), 매매구분(1 매도·2 매수), 처리상태(1 거부·2 자동거부·4 미접수).
_OVERSEAS_ROW_MODIFY, _OVERSEAS_ROW_CANCEL = "1", "2"
_OVERSEAS_REJECTED_STATES = frozenset({"1", "2", "4"})
_US_MARKET_TIMEZONE = "America/New_York"
_US_PRICE_ORDER_DIGITS = 4


def _require_us_exchange(exchange: str) -> str:
    if exchange not in US_EXCHANGE_VENDORS:
        raise BrokerApiError(f"지원하지 않는 미국 거래소 코드입니다: {exchange!r}")
    return exchange


def resolve_us_exchange(account_id: str, ticker: str) -> str:
    """미국 종목이 상장된 시세 거래소 코드 — 현재가 조회에 종목코드가 채워지는 거래소가 맞는 곳이다.

    같은 심볼이 여러 거래소에 있으면(조회가 되는 곳이 둘 이상) 어느 쪽인지 알 수 없어 오류다.
    """
    found = []
    for exchange in US_EXCHANGE_VENDORS:
        data = (
            _request(
                account_id,
                "GET",
                OVERSEAS_PRICE_PATH,
                success=frozenset({"0000", "5766"}),
                params={
                    "mrkt_div_code": _OVERSEAS_MARKET,
                    "mrkt_cls_code": exchange,
                    "iscd": ticker,
                    "dely_rltm_cls_code": _REALTIME_QUOTE,
                },
            ).get("data")
            or {}
        )
        if str(data.get("iscd") or "").strip() == ticker:
            found.append(exchange)
    if not found:
        raise BrokerApiError(f"메리츠증권에서 {ticker} 를 찾을 수 없습니다(나스닥·뉴욕·아멕스).")
    if len(found) > 1:
        raise BrokerApiError(f"{ticker} 가 여러 거래소({', '.join(found)})에서 조회돼 거래소를 정할 수 없습니다.")
    return found[0]


def fetch_us_best_quote(account_id: str, ticker: str, exchange: str) -> dict[str, Any]:
    """미국 종목 1호가 — 매도·매수 1호가(달러)와 호가 시각(한국시간 HHMMSS), 10단계 호가(`depth`). 호가가 비어 있으면 주문 가격을 못 정해 오류다."""
    data = (
        _request(
            account_id,
            "GET",
            OVERSEAS_ORDERBOOK_PATH,
            success=_QUOTE_CODES,
            params={
                "mrkt_div_code": _OVERSEAS_MARKET,
                "mrkt_cls_code": _require_us_exchange(exchange),
                "iscd": ticker,
                "dely_rltm_cls_code": _REALTIME_QUOTE,
            },
        ).get("data")
        or {}
    )
    result = {
        "ask": float(data["askp1"]),
        "bid": float(data["bidp1"]),
        "at": str(data["korea_hour"]),
        "depth": _quote_depth(data, lambda prefix, level: f"{prefix}{level}", float, "prdy_clpr"),
    }
    if result["ask"] <= 0 or result["bid"] <= 0:
        raise BrokerApiError(
            f"{ticker} 의 매도·매수 1호가가 없어 주문 가격을 정할 수 없습니다 (매도 {result['ask']}, 매수 {result['bid']})."
        )
    return result


def place_us_limit_order(
    account_id: str, side: str, ticker: str, quantity: int, price: float, exchange: str
) -> dict[str, str]:
    """미국 지정가 주문 한 건 — 접수되면 주문번호를 돌려준다. side 는 buy·sell, 가격은 달러(소수 4자리까지)."""
    if side not in OVERSEAS_ORDER_PATH:
        raise BrokerApiError(f"알 수 없는 주문 구분입니다: {side}")
    if quantity <= 0 or price <= 0:
        raise BrokerApiError("수량과 가격은 0보다 커야 합니다.")
    data = _request(
        account_id,
        "POST",
        OVERSEAS_ORDER_PATH[side],
        success=_ORDER_CODES,
        body={
            "ovrs_stck_symb_iscd": ticker,
            "ovrs_exch_vndr_idno": US_EXCHANGE_VENDORS[_require_us_exchange(exchange)],
            "oder_cls_code": _OVERSEAS_LIMIT_ORDER,
            "oder_cond_cls_code": _NO_CONDITION,
            "odqt": quantity,
            "ovrs_stck_oder_prc": f"{price:.{_US_PRICE_ORDER_DIGITS}f}",
            "warn_cnfr_yn": "N",
        },
    )
    return {"order_no": _accepted_order_no(data, "주문"), "message": str(data.get("rsp_msg") or "")}


def _us_order_date(today: date) -> str:
    return f"{today:%Y%m%d}"


def _us_today() -> date:
    """메리츠 주문내역의 오늘 주문일자 — 거래 세션의 날짜다(한국 아침의 애프터 주문은 전날, 20:00 ET 이후 데이장 주문은 다음 날)."""
    from utils.strategy_trade_market import session_trade_date

    return session_trade_date("us", datetime.now(ZoneInfo(_US_MARKET_TIMEZONE)))


def _us_row_from_history(row: dict[str, Any]) -> dict[str, Any]:
    """기간 주문체결내역 행 → 공통 중간형. 이 조회에는 취소 확인 수량이 없어 취소 행의 수량으로 대신한다(아래 `_us_order_rows`)."""
    return {
        "date": str(row["oder_date"]),
        "order_no": int(row["oder_no"]),
        "origin_no": int(row["orgl_oder_no"]),
        "symbol": str(row["iscd"]).split(".")[0].strip(),
        "name": str(row.get("kor_isnm") or "").strip(),
        "side": str(row["seln_byov_clcd"]),
        "quantity": float(row["oder_dqty"]),
        "price": float(row["ovrs_stck_oder_prc"]),
        "filled": float(row["cntg_dqty"]),
        "fill_price": float(row["cntg_crnc_prc"]),
        "canceled": 0.0,
        "state": str(row["ovrs_stck_oder_stat_cls_code"]),
        "kind": str(row["oder_rctf_cncl_cls_code"]),
        "order_class": str(row["aspr_type_code"]),
    }


def _us_order_rows(account_id: str, day: date) -> list[dict[str, Any]]:
    """체결 이력의 공통 중간형. 주문의 현재 활성 여부는 당일 조회와 별도로 대조한다."""
    day_text = _us_order_date(day)
    raw = _get_paged(
        account_id,
        OVERSEAS_ORDER_HISTORY_PATH,
        {"oder_sta_date": day_text, "oder_fin_date": day_text, "cntg_cls_code": "0"},
    )
    return [row for row in map(_us_row_from_history, raw) if row["date"] == day_text]


def _us_status(row: dict[str, Any], canceled: float) -> str:
    """주문 상태(pending·partial·filled·canceled·rejected) — 국내 `_order_status` 와 같은 판정."""
    if row["state"] in _OVERSEAS_REJECTED_STATES:
        return "rejected"
    if row["quantity"] > 0 and row["filled"] >= row["quantity"]:
        return "filled"
    if canceled > 0:
        return "canceled"
    if row["state"] == "3":
        return "expired"
    if row["filled"] > 0:
        return "partial"
    return "pending"


def us_day_order_rows(account_id: str, yyyymmdd: str) -> list[dict[str, Any]]:
    """그날(미국 현지 날짜)의 주문 행 — 국내 `day_order_rows` 와 같은 표준 주문 행이고 정정 행은 합치지 않았다.

    취소 요청 행은 뺀다. 이 증권사는 취소 결과(취소수량)를 취소 행에 남기므로 원주문 행의 취소수량으로 옮긴다.
    """
    rows = _us_order_rows(account_id, date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:])))
    canceled: dict[int, float] = {}
    for row in rows:
        if row["kind"] == _OVERSEAS_ROW_CANCEL:
            canceled[row["origin_no"]] = canceled.get(row["origin_no"], 0.0) + (row["canceled"] or row["quantity"])
    result = []
    for row in rows:
        if row["kind"] == _OVERSEAS_ROW_CANCEL:
            continue
        if row["side"] not in _SIDE_LABELS:
            raise BrokerApiError(f"메리츠증권 해외 주문내역의 매매구분을 알 수 없습니다: {row['side']!r}")
        cancel_qty = canceled.get(row["order_no"], 0.0)
        result.append(
            {
                "order_no": str(row["order_no"]),
                "origin_order_no": str(row["origin_no"]),
                "order_date": row["date"],
                "ticker": row["symbol"],
                "name": row["name"],
                "side_label": _SIDE_LABELS[row["side"]],
                "quantity": int(row["quantity"]),
                "price": row["price"],
                "filled_quantity": int(row["filled"]),
                "unfilled_quantity": int(row["quantity"] - row["filled"] - cancel_qty),
                "canceled_quantity": int(cancel_qty),
                "average_fill_price": row["fill_price"],
                "status": _us_status(row, cancel_qty),
            }
        )
    return result


def _us_today_order_row(row: dict[str, Any]) -> dict[str, Any]:
    """당일 조회의 실제 미체결·정정취소 가능 수량으로만 활성 여부를 정한다."""
    ticker = str(row["iscd"]).split(".")[0].strip()
    order_no = str(int(row["oder_no"]))
    quantity, filled = float(row["odqt"]), float(row["cntg_qty"])
    remaining, amendable, canceled = float(row["ncnt_qty"]), float(row["oder_able_qty"]), float(row["cncl_cnfr_qty"])
    state, side = str(row["ovrs_stck_oder_stat_cls_code"]), str(row["seln_byov_clcd"])
    if side not in _SIDE_LABELS or state not in {"0", "1", "2", "3", "4"}:
        raise BrokerApiError(f"{ticker} 주문 {order_no}의 상태 확인이 필요합니다: 알 수 없는 증권사 상태 코드입니다.")
    values = (quantity, filled, remaining, amendable, canceled)
    if (
        quantity <= 0
        or any(not isfinite(value) or value < 0 for value in values)
        or filled + remaining + canceled > quantity
    ):
        raise BrokerApiError(f"{ticker} 주문 {order_no}의 상태 확인이 필요합니다: 증권사 잔량이 일치하지 않습니다.")
    if state in _OVERSEAS_REJECTED_STATES:
        status = "rejected"
    elif filled == quantity and quantity > 0:
        status = "filled"
    elif remaining > 0:
        if state == "3":
            raise BrokerApiError(f"{ticker} 주문 {order_no}의 자동취소 상태와 미체결 수량이 서로 다릅니다.")
        status = "partial" if filled > 0 else "pending"
    elif amendable > 0:
        raise BrokerApiError(f"{ticker} 주문 {order_no}의 미체결 수량과 정정·취소 가능 수량이 서로 다릅니다.")
    else:
        status = "canceled" if canceled > 0 else "expired"
    return {
        "order_no": order_no,
        "origin_order_no": str(int(row["orgl_oder_no"])),
        "order_date": str(row["oder_date"]),
        "ticker": ticker,
        "name": str(row["iscd_name"]),
        "side_label": _SIDE_LABELS[side],
        "quantity": quantity,
        "price": float(row["ovrs_stck_oder_prc"]),
        "filled_quantity": filled,
        "unfilled_quantity": remaining,
        "canceled_quantity": canceled,
        "average_fill_price": float(row["cntg_crnc_prc"]),
        "status": status,
    }


def us_current_order_rows(account_id: str) -> list[dict[str, Any]]:
    """기간 이력과 구분되는 실제 당일 주문 상태. 빈 조회와 조회 실패를 섞지 않는다."""
    data = _get(account_id, OVERSEAS_TODAY_ORDERS_PATH, {"cntg_cls_code": "0", "seln_byov_clcd": "0"})
    if str(data["rsp_cd"]) == "5820":
        return []
    if not isinstance(data.get("data"), list):
        raise BrokerApiError("해외 당일 주문 조회 응답에 주문 목록이 없습니다. 상태 확인이 필요합니다.")
    return [
        _us_today_order_row(row) for row in data["data"] if str(row["oder_rctf_cncl_cls_code"]) != _OVERSEAS_ROW_CANCEL
    ]


def modify_us_order(account_id: str, order_no: str, ticker: str, quantity: int, price: float) -> dict[str, str]:
    """미국 주문을 지정가 정정한다 — 주문일자는 당일 주문내역에서 찾는다."""
    if quantity <= 0 or price <= 0:
        raise BrokerApiError("정정 수량과 가격은 0보다 커야 합니다.")
    rows = _us_order_rows(account_id, _us_today())
    target = next((row for row in rows if str(row["order_no"]) == order_no and row["symbol"] == ticker), None)
    if target is None:
        raise BrokerApiError(f"오늘 주문내역에서 {ticker} 주문 {order_no} 을 찾을 수 없습니다 — 정정하지 않았습니다.")
    data = _request(
        account_id,
        "POST",
        OVERSEAS_MODIFY_PATH,
        success=_ORDER_CODES,
        body={
            "oder_date": target["date"],
            "orgl_oder_no": int(order_no),
            "oder_cls_code": _OVERSEAS_LIMIT_ORDER,
            "odqt": int(quantity),
            "ovrs_stck_oder_prc": f"{price:.4f}",
            "oder_cond_cls_code": _NO_CONDITION,
            "warn_cnfr_yn": "N",
        },
    )
    return {"modify_order_no": _accepted_order_no(data, "정정"), "message": str(data.get("rsp_msg") or "")}


def cancel_us_order(account_id: str, order_no: str, ticker: str) -> dict[str, str]:
    """미국 주문을 전량 취소한다 — 취소 요청에 필요한 주문일자·수량·주문 구분은 당일 주문내역에서 찾는다."""
    rows = _us_order_rows(account_id, _us_today())
    target = next((row for row in rows if str(row["order_no"]) == order_no and row["symbol"] == ticker), None)
    if target is None:
        raise BrokerApiError(f"오늘 주문내역에서 {ticker} 주문 {order_no} 을 찾을 수 없습니다 — 취소하지 않았습니다.")
    data = _request(
        account_id,
        "POST",
        OVERSEAS_CANCEL_PATH,
        success=_ORDER_CODES,
        body={
            "oder_date": target["date"],
            "orgl_oder_no": int(order_no),
            "oder_cls_code": _OVERSEAS_LIMIT_ORDER,
            "odqt": int(target["quantity"]),
            "warn_cnfr_yn": "N",
        },
    )
    return {"cancel_order_no": _accepted_order_no(data, "취소"), "message": str(data.get("rsp_msg") or "")}
