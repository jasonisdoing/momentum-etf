"use client";

import { useCallback, useEffect, useRef } from "react";

/**
 * 마지막 요청의 응답만 반영하게 해주는 훅 (늦게 온 옛 응답 폐기).
 *
 * 왜 필요한가: 필터를 바꾸면 새 요청이 나가는데, 먼저 보낸 요청이 더 느리면 나중에 도착해
 * 새 결과를 덮어쓴다. `/kor-market-stock` 에서 코스피(200종목)를 불러오는 중에 코스닥으로
 * 바꾸면, 코스닥 응답이 먼저 그려진 뒤 뒤늦게 온 코스피 응답이 화면을 다시 덮었다.
 * 요청 수가 적어 눈치채기 어렵고, 화면에는 "코스닥을 골랐는데 코스피가 보인다"로 나타난다.
 *
 * 사용법 — 상태를 넣기 직전에 `isLatest()` 로 확인한다.
 *
 * ```ts
 * const { begin, isLatest } = useLatestRequest();
 * const load = useCallback(async (market: string) => {
 *   const token = begin();
 *   const data = await fetchSomething(market);
 *   if (!isLatest(token)) return;   // 그 사이 다른 요청이 시작됐다 → 버린다
 *   setRows(data.rows);
 * }, [begin, isLatest]);
 * ```
 *
 * 언마운트 후에도 `isLatest` 가 거짓이 되어 사라진 컴포넌트의 setState 를 막는다.
 */
export function useLatestRequest() {
  const seqRef = useRef(0);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  /** 새 요청을 시작하고 토큰을 받는다. 이전 요청은 이 시점에 무효가 된다. */
  const begin = useCallback(() => {
    seqRef.current += 1;
    return seqRef.current;
  }, []);

  /** 이 토큰이 아직 마지막 요청인지 (그리고 컴포넌트가 살아 있는지). */
  const isLatest = useCallback((token: number) => mountedRef.current && token === seqRef.current, []);

  return { begin, isLatest };
}
