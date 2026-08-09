import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  GET: { path: "/internal/market-trend/indices", error: "시장지수 목록을 불러오지 못했습니다." },
});

export const GET = proxy.GET!;
