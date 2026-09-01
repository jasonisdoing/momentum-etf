import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

/** 합성 오늘의 액션 즉시 발송(테스트) — 변화 여부와 무관, 알람 상태는 건드리지 않는다. */
const proxy = createFastApiProxy({
  POST: {
    path: "/internal/strategy-mix/slack-test",
    error: "슬랙 테스트 발송에 실패했습니다.",
    forwardBody: true,
  },
});

export const POST = proxy.POST!;
