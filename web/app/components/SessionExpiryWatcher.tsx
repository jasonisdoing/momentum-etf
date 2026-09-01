"use client";

import { useEffect } from "react";

/**
 * 세션 만료 공통 처리 — `/api/*` 응답이 401 이면 로그인 화면으로 보낸다.
 *
 * 로그인이 풀리면 `proxy.ts` 가 API 요청을 FastAPI 로 넘기지 않고 그 자리에서
 * `401 {"error": "로그인이 필요합니다."}` 로 자른다. 그런데 화면마다 실패 문구가 달라
 * ("추가 실패: ALL, CRH", "데이터를 불러오지 못했습니다" …) 로그인 문제인 줄 알기 어려웠다.
 * 여기서 한 번만 처리해 모든 화면이 같게 동작하게 한다.
 *
 * 페이지 이동은 `proxy.ts` 가 이미 `/login` 으로 돌려보내므로, 여기서는 화면 안에서
 * 일어나는 API 호출만 본다.
 */

const LOGIN_PATH = "/login";
const EXPIRED_MESSAGE = "로그인 세션이 만료되었습니다. 다시 로그인해 주세요.";

/** 요청 URL 을 꺼낸다. fetch 의 첫 인자는 문자열·URL·Request 셋 다 될 수 있다. */
function requestUrl(input: RequestInfo | URL): string {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.href;
  return input.url;
}

/** 세션 만료로 볼 요청인지 — **같은 출처**의 `/api/*` 만. 로그인 자체 요청은 제외(무한 루프 방지).
 *
 * 출처를 안 보면 외부 API(예: 네이버 `finance.naver.com/api/...`)의 401 에도 로그인 화면으로
 * 튕긴다 — 경로만 보면 우리 API 와 구분되지 않는다.
 */
function isGuardedApiRequest(url: string): boolean {
  let parsed: URL;
  try {
    parsed = new URL(url, window.location.origin);
  } catch {
    return false;
  }
  if (parsed.origin !== window.location.origin) {
    return false;
  }
  return parsed.pathname.startsWith("/api/") && !parsed.pathname.startsWith("/api/auth/");
}

export function SessionExpiryWatcher() {
  useEffect(() => {
    const originalFetch = window.fetch;
    // 여러 번 이동하지 않게 한 번만 보낸다 — 화면이 동시에 여러 API 를 부르면 401 도 여러 번 온다.
    let redirecting = false;

    const patched: typeof window.fetch = async (input, init) => {
      const response = await originalFetch(input, init);
      if (
        response.status === 401 &&
        !redirecting &&
        window.location.pathname !== LOGIN_PATH &&
        isGuardedApiRequest(requestUrl(input))
      ) {
        redirecting = true;
        const next = `${window.location.pathname}${window.location.search}`;
        const params = new URLSearchParams({ next, error: EXPIRED_MESSAGE });
        window.location.assign(`${LOGIN_PATH}?${params.toString()}`);
      }
      return response;
    };

    window.fetch = patched;
    return () => {
      // 다른 코드가 그 사이에 또 감쌌다면 건드리지 않는다.
      if (window.fetch === patched) {
        window.fetch = originalFetch;
      }
    };
  }, []);

  return null;
}
