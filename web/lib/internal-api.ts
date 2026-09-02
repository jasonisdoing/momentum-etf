type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

function getFastApiBaseUrl(): string {
  const value = String(process.env.FASTAPI_INTERNAL_URL ?? "").trim();
  if (!value) {
    throw new Error("FASTAPI_INTERNAL_URL 환경변수가 필요합니다.");
  }
  return value.replace(/\/+$/, "");
}

function getFastApiToken(): string {
  const value = String(process.env.FASTAPI_INTERNAL_TOKEN ?? "").trim();
  if (!value) {
    throw new Error("FASTAPI_INTERNAL_TOKEN 환경변수가 필요합니다.");
  }
  return value;
}

const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * fetch 실패의 진짜 원인을 한 줄로 푼다.
 *
 * undici 는 연결 계열 실패를 전부 `TypeError: fetch failed` 로 감싸고, 실제 사유
 * (ECONNREFUSED·ECONNRESET·소켓 종료 등)는 `cause` 체인에 넣는다. 껍데기만 던지면
 * 화면에 "fetch failed" 한 줄이 떠서 서버가 죽은 건지, 끊긴 건지, 주소가 틀린 건지
 * 구분할 수가 없다. 체인을 훑어 메시지와 errno 코드를 함께 남긴다.
 */
function describeFetchFailure(error: unknown): string {
  const parts: string[] = [];
  let current: unknown = error;
  for (let depth = 0; current instanceof Error && depth < 5; depth += 1) {
    const code = (current as NodeJS.ErrnoException).code;
    parts.push(code ? `${current.message} [${code}]` : current.message);
    current = (current as { cause?: unknown }).cause;
  }
  return parts.join(" ← ") || String(error);
}

export async function fetchFastApiJson<T>(
  path: string,
  init?: RequestInit,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Content-Type", "application/json");
  headers.set("X-Internal-Token", getFastApiToken());

  const controller = new AbortController();
  if (init?.signal) {
    init.signal.addEventListener("abort", () => controller.abort(init.signal!.reason));
  }
  const timeoutId = setTimeout(() => controller.abort("timeout"), timeoutMs);

  let response: Response;
  try {
    response = await fetch(`${getFastApiBaseUrl()}${path}`, {
      ...init,
      headers,
      signal: controller.signal,
      cache: "no-store",
    });
  } catch (error) {
    clearTimeout(timeoutId);
    const isAbort =
      (error instanceof DOMException && error.name === "AbortError") ||
      (error instanceof Error && error.name === "AbortError") ||
      (error instanceof Error && error.message.includes("fetch failed") && String((error as any).cause).includes("AbortError"));

    if (isAbort) {
      try {
        await fetch(`${getFastApiBaseUrl()}/internal/health/report_error`, {
          method: "POST",
          headers: { "X-Internal-Token": getFastApiToken() },
        }).catch(() => { });
      } catch (e) { }
      throw new Error(`FastAPI 요청이 ${timeoutMs / 1_000}초 내에 응답하지 않았습니다. (${path})`);
    }
    throw new Error(`FastAPI 요청에 실패했습니다: ${describeFetchFailure(error)} (${path})`, { cause: error });
  } finally {
    clearTimeout(timeoutId);
  }

  const payload = (await response.json().catch(() => ({}))) as { detail?: string; error?: string } & JsonValue;
  if (!response.ok) {
    const message =
      (typeof payload === "object" && payload && "detail" in payload && typeof payload.detail === "string"
        ? payload.detail
        : null) ||
      (typeof payload === "object" && payload && "error" in payload && typeof payload.error === "string"
        ? payload.error
        : null) ||
      `FastAPI 요청에 실패했습니다. (${response.status})`;

    if (message.includes("NetworkTimeout") || message.includes("timed out") || message.includes("시간 초과") || message.includes("응답하지 않았습니다")) {
      try {
        await fetch(`${getFastApiBaseUrl()}/internal/health/report_error`, {
          method: "POST",
          headers: { "X-Internal-Token": getFastApiToken() },
        }).catch(() => { });
      } catch (e) { }
    }
    throw new Error(message);
  }

  return payload as T;
}
