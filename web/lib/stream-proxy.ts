/**
 * 오래 걸리는 계산의 **스트림 응답**을 그대로 통과시키는 프록시 — 튜닝이 쓴다.
 *
 * 일반 프록시(`fastapi-proxy`)는 응답을 통째로 읽어 JSON 으로 바꾼다. 그러면 진행이
 * 한꺼번에 도착해 스트리밍이 의미가 없어지므로, 여기서는 받은 청크를 곧바로 내보낸다.
 *
 * 받은 body 를 `new Response(upstream.body)` 로 그냥 넘기면 중간 계층이 모아 둘 수 있다
 * (진행 줄은 작아서 버퍼에 남고, 마지막 결과가 올 때 한꺼번에 밀려 나왔다). 그래서
 * 직접 읽어 청크마다 enqueue 한다.
 *
 * 첫 바이트가 곧바로 오므로 undici 의 헤더 대기 한도(기본 5분)에도 걸리지 않는다.
 */

function getBaseUrl(): string {
  const value = String(process.env.FASTAPI_INTERNAL_URL ?? "").trim();
  if (!value) throw new Error("FASTAPI_INTERNAL_URL 환경변수가 필요합니다.");
  return value.replace(/\/+$/, "");
}

function getToken(): string {
  const value = String(process.env.FASTAPI_INTERNAL_TOKEN ?? "").trim();
  if (!value) throw new Error("FASTAPI_INTERNAL_TOKEN 환경변수가 필요합니다.");
  return value;
}

/** 화면이 한 가지 형식만 읽으면 되게, 실패도 같은 SSE 이벤트로 돌려준다. */
function errorStream(message: string): Response {
  const body = `data: ${JSON.stringify({ type: "error", message })}\n\n`;
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache, no-transform" },
  });
}

export async function proxyStream(path: string, body: unknown): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(`${getBaseUrl()}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Internal-Token": getToken(),
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      cache: "no-store",
    });
  } catch (error) {
    return errorStream(error instanceof Error ? error.message : String(error));
  }

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text().catch(() => "");
    return errorStream(text.trim() || `요청에 실패했습니다. (${upstream.status})`);
  }

  const reader = upstream.body.getReader();
  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (done) {
          controller.close();
          return;
        }
        controller.enqueue(value);
      } catch (error) {
        controller.error(error);
      }
    },
    cancel(reason) {
      // 화면이 떠나면 위쪽 연결도 끊는다 — 서버가 헛돌지 않게.
      void reader.cancel(reason);
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
      Connection: "keep-alive",
    },
  });
}
