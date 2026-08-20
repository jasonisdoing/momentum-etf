import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

/** 연동 계좌 잔고 불러오기 — 현재 저장값과 나란히 돌려준다(차이 표시용). */
const proxy = createFastApiProxy({
  GET: {
    path: "/internal/broker-api/balance",
    error: "잔고를 불러오지 못했습니다.",
    forwardQuery: ["account_id"],
  },
});

export const GET = proxy.GET!;
