import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

/** 모의투자 계좌 잔고 — 모의 주문 테스트 모달이 부른다. */
const proxy = createFastApiProxy({
  GET: {
    path: "/internal/broker-api/mock/balance",
    error: "모의 계좌 잔고를 불러오지 못했습니다.",
    forwardQuery: ["account_no"],
  },
});

export const GET = proxy.GET!;
