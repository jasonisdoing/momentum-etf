import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  POST: {
    path: "/internal/strategy-mix/order-signal",
    error: "슬랙 주문 신호를 보내지 못했습니다.",
    forwardBody: true,
  },
});

export const POST = proxy.POST!;
