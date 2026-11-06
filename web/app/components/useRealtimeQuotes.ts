"use client";

import { useEffect, useState } from "react";

/** 표시용 실시간 시세 — 현재가·등락률만 주기적으로 받아온다.
 *
 * 전략 화면의 선정·판정은 무거워서 5분 캐시(`CACHE_TTL_COMPUTE`)를 쓴다. 그 안에서
 * 실제로 변하는 건 표시용 가격뿐이라, 계산 전체를 다시 돌리는 대신 이 훅으로 가격만
 * 덮어쓴다. 서버 시세도 60초 캐시(`CACHE_TTL_LIVE`)라 호출이 겹쳐도 부담이 없다.
 *
 * 세 전략 화면(모멘텀·신고가·합성)이 같은 규칙을 쓰도록 여기 한 곳에만 둔다.
 */
export type Quote = { price: number; change_pct: number | null };

/** 갱신 주기(ms) — 서버 시세 캐시(60초)와 같은 간격. 더 자주 불러도 같은 값이 온다. */
const REFRESH_MS = 60_000;

/** `missing` — 이번 갱신에서 시세를 못 받은 티커(요청 실패면 전부). 화면은 그 종목의 일간(%)을 비우고 경고한다. */
export type RealtimeQuotes = { quotes: Record<string, Quote>; missing: string[] };

export function useRealtimeQuotes(country: string, tickers: string[]): RealtimeQuotes {
  const [state, setState] = useState<RealtimeQuotes>({ quotes: {}, missing: [] });
  // 의존성은 티커 문자열로 잡는다 — 배열을 그대로 쓰면 매 렌더마다 이펙트가 다시 걸린다.
  const tickerKey = tickers.join(",");

  useEffect(() => {
    if (!country || !tickerKey) return;

    let alive = true;
    const refresh = async () => {
      try {
        const params = new URLSearchParams({ country, tickers: tickerKey });
        const response = await fetch(`/api/quotes?${params.toString()}`, { cache: "no-store" });
        const payload = (await response.json()) as { quotes?: Record<string, Quote>; error?: string };
        if (!alive) return;
        if (!response.ok || payload.error) throw new Error(payload.error ?? "시세 조회 실패");
        const received = payload.quotes ?? {};
        setState({ quotes: received, missing: tickerKey.split(",").filter((ticker) => !received[ticker]) });
      } catch {
        // 시세 갱신 실패 — 지난 시세를 그대로 두면 멈춘 값이 살아 있는 것처럼 보인다. 전부 「못 받음」으로 둔다.
        if (alive) setState({ quotes: {}, missing: tickerKey.split(",") });
      }
    };

    void refresh();
    const timer = setInterval(() => void refresh(), REFRESH_MS);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [country, tickerKey]);

  return state;
}
