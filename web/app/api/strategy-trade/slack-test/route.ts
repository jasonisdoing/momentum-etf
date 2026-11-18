import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

/** 사고팔기 현황 즉시 발송(테스트) — 계좌 하나, 스위치와 무관하고 알림 상태는 건드리지 않는다. */
const proxy = createFastApiProxy({
  POST: {
    path: "/internal/strategy-trade/slack-test",
    error: "슬랙 테스트 발송에 실패했습니다.",
    forwardBody: true,
  },
});

export const POST = proxy.POST!;
