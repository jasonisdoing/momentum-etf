/**
 * FastAPI 프록시 라우트 공용 팩토리.
 *
 * `web/app/api/**` 의 순수 프록시 라우트는 전부 같은 모양(내부 경로 호출 → JSON 반환,
 * 실패 시 `{error}` 500)이라, 경로·에러 메시지 선언만 남기고 보일러플레이트를 여기로
 * 모은다. 쿼리스트링 가공·동적 세그먼트 등 로직이 있는 라우트는 이 팩토리를 쓰지 않고
 * 기존처럼 직접 구현한다.
 */

import type { NextRequest } from "next/server";

import { fetchFastApiJson } from "./internal-api";
import { jsonNoStore } from "./no-store-response";

type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

type MethodSpec = {
  /** FastAPI 내부 경로 (예: "/internal/memos"). */
  path: string;
  /** 실패했는데 에러 메시지를 알 수 없을 때의 기본 문구. */
  error: string;
  /** 요청 JSON body 를 그대로 FastAPI 로 전달할지 (POST/PUT 저장 계열). */
  forwardBody?: boolean;
  /** 기본(30초)보다 긴 타임아웃이 필요한 호출용 (ms). */
  timeoutMs?: number;
  /** 여기 적은 쿼리 파라미터만 그대로 FastAPI 로 넘긴다(예: 화면이 고른 종목풀). */
  forwardQuery?: readonly string[];
};

type ProxyHandler = (request: NextRequest) => Promise<Response>;

export function createFastApiProxy(
  spec: Partial<Record<HttpMethod, MethodSpec>>,
): Partial<Record<HttpMethod, ProxyHandler>> {
  const makeHandler = (method: HttpMethod, methodSpec: MethodSpec): ProxyHandler => {
    return async (request: NextRequest) => {
      try {
        const init: RequestInit = method === "GET" ? {} : { method };
        let path = methodSpec.path;
        if (methodSpec.forwardQuery?.length) {
          const forwarded = new URLSearchParams();
          for (const key of methodSpec.forwardQuery) {
            const value = request.nextUrl.searchParams.get(key);
            if (value) forwarded.set(key, value);
          }
          const query = forwarded.toString();
          if (query) path = `${path}?${query}`;
        }
        if (methodSpec.forwardBody) {
          const body = await request.json();
          init.headers = { "Content-Type": "application/json" };
          init.body = JSON.stringify(body);
        }
        const data = methodSpec.timeoutMs
          ? await fetchFastApiJson(path, init, methodSpec.timeoutMs)
          : await fetchFastApiJson(path, init);
        return jsonNoStore(data);
      } catch (error) {
        const message = error instanceof Error ? error.message : methodSpec.error;
        return jsonNoStore({ error: message }, { status: 500 });
      }
    };
  };

  const handlers: Partial<Record<HttpMethod, ProxyHandler>> = {};
  for (const [method, methodSpec] of Object.entries(spec) as [HttpMethod, MethodSpec][]) {
    handlers[method] = makeHandler(method, methodSpec);
  }
  return handlers;
}
