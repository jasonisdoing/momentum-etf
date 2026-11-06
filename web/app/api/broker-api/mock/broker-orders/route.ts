import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

/** 증권사에 오늘 들어간 주문 전체 — 우리 기록과 대조하는 용도. */
const proxy = createFastApiProxy({
  GET: {
    path: "/internal/broker-api/mock/broker-orders",
    error: "증권사 주문내역을 불러오지 못했습니다.",
    forwardQuery: ["account_no"],
  },
});

export const GET = proxy.GET!;
