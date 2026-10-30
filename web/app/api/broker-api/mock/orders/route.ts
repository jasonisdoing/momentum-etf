import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

/** 모의 주문 기록(GET: 증권사 상태 반영) · 사고팔기 회차 주문(POST). */
const proxy = createFastApiProxy({
  GET: {
    path: "/internal/broker-api/mock/orders",
    error: "주문 기록을 불러오지 못했습니다.",
    forwardQuery: ["account_no"],
  },
  POST: {
    path: "/internal/broker-api/mock/orders",
    error: "주문을 보내지 못했습니다.",
    forwardBody: true,
  },
});

export const GET = proxy.GET!;
export const POST = proxy.POST!;
