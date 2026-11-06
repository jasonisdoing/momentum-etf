import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  GET: {
    path: "/internal/strategy-trade/backtest",
    error: "백테스트를 불러오지 못했습니다.",
    forwardQuery: ["pool", "months"],
  },
});

export const GET = proxy.GET!;
