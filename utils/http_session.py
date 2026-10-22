"""HTTP keep-alive 공유 Session.

매 외부 API 호출마다 새 TCP/TLS connection 을 만들면 handshake 비용(100~300ms)이
누적되어 직렬 배치 시간이 크게 늘어난다. 본 모듈은 프로세스 전체에서 재사용되는
`requests.Session` 한 개를 제공한다.

사용 예:
    from utils.http_session import shared_session

    resp = shared_session.get(url, headers=..., timeout=5)
    resp = shared_session.post(url, headers=..., json=...)

기존 코드의 `requests.get(...)` → `shared_session.get(...)` 으로 교체하면 된다.
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter

shared_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
shared_session.mount("http://", _adapter)
shared_session.mount("https://", _adapter)


# ── 브라우저 흉내 요청 ────────────────────────────────────────────────────
# Cloudflare 가 막는 운용사 사이트(WisdomTree·Schwab)는 브라우저 TLS 지문을 흉내 내는 curl_cffi 로 받는다.
# 어떤 지문이 통과하는지는 수시로 바뀐다(2026-10-22 WisdomTree: chrome·edge·firefox 403, chrome131·safari 200).
# 그래서 아래 순서로 403 이 아닌 응답이 나올 때까지 시도하고, 통과한 지문을 다음 요청부터 먼저 쓴다.
_BROWSER_PROFILES = (
    "chrome",
    "chrome136",
    "chrome131",
    "chrome124",
    "safari184",
    "safari180",
    "safari17_0",
    "edge101",
    "firefox135",
)
_browser_sessions: dict[str, object] = {}
_preferred_profile: dict[str, str] = {}


def browser_get(url: str, *, timeout: float):
    """브라우저 지문을 바꿔 가며 GET — 모든 지문이 403 이면 마지막 오류를 낸다. 403 외 오류는 바로 낸다."""
    from urllib.parse import urlparse

    from curl_cffi import requests as curl_requests

    host = urlparse(url).netloc
    preferred = _preferred_profile.get(host)
    profiles = ([preferred] if preferred else []) + [p for p in _BROWSER_PROFILES if p != preferred]
    response = None
    for profile in profiles:
        session = _browser_sessions.get(profile)
        if session is None:
            session = _browser_sessions[profile] = curl_requests.Session(impersonate=profile)
        response = session.get(url, timeout=timeout)
        if response.status_code != 403:
            response.raise_for_status()
            _preferred_profile[host] = profile
            return response
    response.raise_for_status()
    return response
