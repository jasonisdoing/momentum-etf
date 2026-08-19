import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

/** 실시간 시세(현재가·등락률) — 화면이 표시용 가격만 자주 갱신할 때 쓴다. */
const proxy = createFastApiProxy({
  GET: {
    path: "/internal/quotes",
    error: "시세를 불러오지 못했습니다.",
    forwardQuery: ["country", "tickers"],
  },
});

export const GET = proxy.GET!;
