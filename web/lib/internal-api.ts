import { Agent } from "undici";

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
 * 오래 걸리는 호출용 디스패처.
 *
 * undici 는 `AbortSignal` 과 **별개로** 자체 타임아웃을 건다 — 응답 헤더가 5분(기본
 * `headersTimeout`) 안에 안 오면 `UND_ERR_HEADERS_TIMEOUT` 으로 연결을 끊는다. 튜닝처럼
 * 계산이 다 끝나야 응답이 시작되는 호출은 우리가 1시간을 줘도 5분에 잘렸다.
 * 그래서 호출자가 정한 `timeoutMs` 를 undici 쪽에도 그대로 맞춘다.
 *
 * 기본 타임아웃(30초) 이하면 undici 기본값이 이미 넉넉하므로 만들지 않는다.
 * 같은 값의 Agent 는 재사용한다 — 요청마다 새로 만들면 연결 풀이 쌓인다.
 */
const AGENTS_BY_TIMEOUT = new Map<number, Agent>();

function dispatcherFor(timeoutMs: number): Agent | undefined {
  if (timeoutMs <= DEFAULT_TIMEOUT_MS) return undefined;
  const known = AGENTS_BY_TIMEOUT.get(timeoutMs);
  if (known) return known;
  // bodyTimeout 은 청크 사이 간격 기준이라 헤더와 같은 값이면 충분하다.
  const agent = new Agent({ headersTimeout: timeoutMs, bodyTimeout: timeoutMs });
  AGENTS_BY_TIMEOUT.set(timeoutMs, agent);
  return agent;
}

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
      dispatcher: dispatcherFor(timeoutMs),
    } as RequestInit & { dispatcher?: Agent });
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
