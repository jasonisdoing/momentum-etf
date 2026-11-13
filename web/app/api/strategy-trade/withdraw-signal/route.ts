import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  POST: {
    path: "/internal/strategy-trade/withdraw-signal",
    error: "슬랙 신호를 회수하지 못했습니다.",
    forwardBody: true,
  },
});

export const POST = proxy.POST!;
