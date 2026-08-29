import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  POST: {
    path: "/internal/strategy-momentum/charts",
    error: "선정 종목 차트를 불러오지 못했습니다.",
    forwardBody: true,
    timeoutMs: 120_000,
  },
});

export const POST = proxy.POST!;
