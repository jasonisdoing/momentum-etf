import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  POST: {
    path: "/internal/strategy-momentum/tuning",
    forwardQuery: ["pool"],
    // 조합 수백 개를 한 번에 백테스트한다 (후보 캐시 공유로 조합당 0.5초 수준).
    error: "튜닝에 실패했습니다.",
    forwardBody: true,
    timeoutMs: 3_600_000,
  },
});

export const POST = proxy.POST!;
