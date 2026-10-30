import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

/** 모의 주문 메시지(버튼)를 슬랙으로 보낸다. */
const proxy = createFastApiProxy({
  POST: {
    path: "/internal/broker-api/mock/slack-order",
    error: "슬랙 주문 메시지를 보내지 못했습니다.",
    forwardBody: true,
  },
});

export const POST = proxy.POST!;
