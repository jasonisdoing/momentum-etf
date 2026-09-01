import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  GET: { path: "/internal/pool-backtest/options", error: "종목풀 백테스트 옵션을 불러오지 못했습니다." },
});

export const GET = proxy.GET!;
