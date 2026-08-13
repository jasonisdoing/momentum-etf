import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  POST: {
    path: "/internal/strategy-new-high/backtest",
    // 이벤트 기반이라 구간이 길면 체결 내역이 수백 건이 된다.
    error: "백테스트에 실패했습니다.",
    forwardBody: true,
    timeoutMs: 300_000,
  },
});

export const POST = proxy.POST!;
