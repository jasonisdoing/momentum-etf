import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  POST: {
    path: "/internal/strategy-new-high/tuning",
    // 조합 수십 개를 한 번에 백테스트한다.
    error: "튜닝에 실패했습니다.",
    forwardBody: true,
    timeoutMs: 3_600_000,
  },
});

export const POST = proxy.POST!;
