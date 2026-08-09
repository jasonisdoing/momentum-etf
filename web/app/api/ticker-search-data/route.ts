import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  GET: { path: "/internal/ticker-detail/search-data", error: "전역 검색 데이터를 불러오지 못했습니다." },
});

export const GET = proxy.GET!;
