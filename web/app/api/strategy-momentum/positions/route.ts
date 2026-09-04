import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  POST: { path: "/internal/strategy-momentum/positions", forwardQuery: ["pool"], error: "운용 현황 조회에 실패했습니다.", timeoutMs: 120_000 },
});

export const POST = proxy.POST!;
