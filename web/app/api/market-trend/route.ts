import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  GET: { path: "/internal/market-trend", error: "시장지수 추세 데이터를 불러오지 못했습니다." },
});

export const GET = proxy.GET!;
