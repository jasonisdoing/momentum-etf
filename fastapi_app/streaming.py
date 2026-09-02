"""오래 걸리는 계산을 진행과 함께 흘려보내는 응답 — 튜닝이 쓴다.

계산이 다 끝나야 응답하면 두 가지가 깨진다.
  1. 화면이 10분 넘게 아무 소식을 못 받아 죽은 것처럼 보인다(진행 바가 추정으로만 움직인다).
  2. 응답 헤더가 그만큼 늦어, 프록시(undici)의 헤더 대기 한도(기본 5분)에 걸려 끊긴다.

형식은 **SSE**(`text/event-stream`)다. 처음에는 NDJSON 으로 만들었는데, FastAPI 는 줄을
제때 흘리는데도 Next 를 거치면서 한꺼번에 도착했다 — 중간 계층이 모르는 콘텐츠 타입은
모아 두기 때문이다. SSE 는 버퍼링하지 않는 것이 규약이라 그런 계층을 그대로 통과한다.

브라우저 `EventSource` 는 POST 를 못 해서 화면은 `fetch` 로 읽는다. 그래서 이벤트 이름은
쓰지 않고 `data:` 한 줄씩만 내보낸다.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from fastapi.responses import StreamingResponse

from utils.logger import get_app_logger


def sse_stream(source: Callable[[], Iterable[dict[str, Any]]]) -> StreamingResponse:
    """제너레이터가 내보내는 dict 를 SSE 이벤트로 하나씩 흘리는 응답.

    ``source`` 를 함수로 받는 이유: 제너레이터를 여기서 만들어야 첫 이벤트를 보내기 **전에**
    실패하는 경우까지 스트림 안에서 잡아 `{"type": "error"}` 로 알릴 수 있다. 헤더가 이미
    나간 뒤에는 HTTP 상태로 실패를 알릴 방법이 없다.
    """

    def events() -> Iterator[bytes]:
        # 연결 직후 주석 한 줄 — 첫 바이트가 바로 나가야 중간 계층이 스트림으로 다룬다.
        yield b": open\n\n"
        iterator: Iterator[dict[str, Any]] | None = None
        try:
            iterator = iter(source())
            for event in iterator:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()
        except Exception as error:
            get_app_logger().exception("[stream] 계산 중 실패")
            payload = json.dumps({"type": "error", "message": str(error)}, ensure_ascii=False)
            yield f"data: {payload}\n\n".encode()
        finally:
            # 브라우저가 중단하면 내부 튜닝 제너레이터도 닫아 워커 취소 신호를 즉시 보낸다.
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # 중간 프록시가 모아 두면 진행이 몰려서 온다 — 버퍼링·캐시를 끈다.
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
