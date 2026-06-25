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
