import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  POST: { path: "/internal/strategy-sm/picks", error: "선정에 실패했습니다.", timeoutMs: 120_000 },
});

export const POST = proxy.POST!;
