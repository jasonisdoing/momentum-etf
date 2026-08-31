import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

/** 신고가 알람 즉시 발송(테스트) — 슬랙 연결·풀 설정 경로 확인용. */
const proxy = createFastApiProxy({
  POST: {
    path: "/internal/strategy-new-high/slack-test",
    error: "슬랙 테스트 발송에 실패했습니다.",
    forwardBody: true,
  },
});

export const POST = proxy.POST!;
