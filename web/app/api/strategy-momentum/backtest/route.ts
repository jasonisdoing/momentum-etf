import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  POST: { path: "/internal/strategy-momentum/backtest", forwardQuery: ["pool"], error: "백테스트에 실패했습니다.", forwardBody: true, timeoutMs: 120_000 },
});

export const POST = proxy.POST!;
