import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  GET: {
    path: "/internal/strategy-trade/signal-action",
    error: "슬랙 발송 주문 상태를 조회하지 못했습니다.",
    forwardQuery: ["account_id", "thread_ts", "side", "round_no", "ticker"],
  },
  POST: {
    path: "/internal/strategy-trade/signal-action",
    error: "슬랙 발송 주문 요청을 처리하지 못했습니다.",
    forwardBody: true,
  },
});

export const GET = proxy.GET!;
export const POST = proxy.POST!;
