import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

/** 모의 주문 취소 — 기록 id 로 전량 취소한다. */
const proxy = createFastApiProxy({
  POST: {
    path: "/internal/broker-api/mock/orders/cancel",
    error: "주문을 취소하지 못했습니다.",
    forwardBody: true,
  },
});

export const POST = proxy.POST!;
