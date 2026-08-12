"""yfinance 호출 직렬화 — 프로세스 전역 락 (단일 소스).

왜 필요한가
----------
yfinance 는 프로세스 전역 상태를 공유해서, 여러 스레드가 동시에 받으면 **서로 결과를
덮어쓴다**. FastAPI 는 동기 엔드포인트를 스레드풀에서 돌리므로 한 화면이 지수·시세를
여러 개 한꺼번에 요청하면 이 조건에 그대로 걸린다.

실제로 홈 화면의 '필라델피아 반도체' 카드에 나스닥 100 가격이 그려졌다(2026-08-04).
가격만 남의 것이고 티커별 설정은 제 것이 적용돼, 차트 모양과 배지가 서로 달라
한참 뒤에야 발견됐다. 4개 티커를 동시에 받게 하면 매번 재현된다.

쓰는 법
------
    from utils.yfinance_guard import yfinance_lock

    with yfinance_lock():
        df = yf.download(...)

`RLock` 이라 같은 스레드에서 중첩해 잡아도 안전하다(락 안에서 다른 락 함수를 불러도
교착되지 않는다).

주의
----
락은 **한 프로세스 안**에서만 유효하다. 배치는 각자 별도 프로세스라 서로 간섭하지
않으므로 문제가 없다. 반대로 한 프로세스 안에서 `ThreadPoolExecutor` 로 yfinance 를
병렬 호출하는 코드는 이 락을 쓰면 직렬화되어 느려진다 — 그런 곳은 속도와 정확성을
따져 개별로 판단한다.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_YF_LOCK = threading.RLock()


@contextmanager
def yfinance_lock() -> Iterator[None]:
    """yfinance 호출 구간을 감싼다 — 한 번에 한 스레드만 통과한다."""
    with _YF_LOCK:
        yield
