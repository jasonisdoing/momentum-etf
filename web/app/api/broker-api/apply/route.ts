import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

/** 불러온 잔고를 portfolio_master 에 반영 — 화면에서 차이를 확인한 뒤 누른다. */
const proxy = createFastApiProxy({
  POST: {
    path: "/internal/broker-api/apply",
    error: "잔고 반영에 실패했습니다.",
    forwardBody: true,
  },
});

export const POST = proxy.POST!;
