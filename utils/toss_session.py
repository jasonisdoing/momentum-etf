"""토스증권 웹 API 공용 세션 — 모든 토스 호출이 이 세션을 쓴다.

2026-10-22 부터 토스 API(시세·차트)가 게스트 세션 쿠키 없이 부르면 `400 invalid-request` 를 준다.
웹이 처음 여는 `wts-api …/api/v3/init` 이 SESSION·UTK·BTK 등 쿠키를 내려 주므로, 세션을 만들 때 한 번
그 초기화를 부르고 쿠키를 들고 다닌다. 세션이 만료돼 다시 400 이 오면 초기화를 한 번 더 하고 재시도한다.

2026-10-29 부터는 `deviceId` 쿠키도 있어야 한다(웹이 스스로 만들어 붙이는 값이라 init 이 내려주지 않는다).
없으면 init 을 거쳐도 400 이다. 값은 아무 uuid 면 통과한다.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

import requests

from config import TOSS_INVEST_HEADERS

_INIT_URL = "https://wts-api.tossinvest.com/api/v3/init"
_lock = threading.Lock()
_session: requests.Session | None = None


def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(TOSS_INVEST_HEADERS)
    session.cookies.set("deviceId", uuid.uuid4().hex, domain=".tossinvest.com", path="/")
    response = session.get(_INIT_URL, params={"tabId": f"tab-{uuid.uuid4().hex}"}, timeout=10)
    response.raise_for_status()
    if "SESSION" not in session.cookies:
        raise RuntimeError("토스 게스트 세션 초기화 응답에 SESSION 쿠키가 없습니다.")
    return session


def _current(*, renew: bool) -> requests.Session:
    global _session
    with _lock:
        if _session is None or renew:
            _session = _new_session()
        return _session


def _is_session_rejected(response: requests.Response) -> bool:
    if response.status_code != 400:
        return False
    try:
        return (response.json().get("error") or {}).get("code") == "invalid-request"
    except ValueError:
        return False


def toss_request(method: str, url: str, **kwargs: Any) -> requests.Response:
    """토스 API 요청. 세션 거절(400 invalid-request)이면 세션을 새로 만들어 한 번만 다시 보낸다."""
    response = _current(renew=False).request(method, url, **kwargs)
    if _is_session_rejected(response):
        response = _current(renew=True).request(method, url, **kwargs)
    return response
