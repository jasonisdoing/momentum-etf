import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  GET: {
    path: "/internal/strategy-mix/meta",
    error: "종목풀 목록을 불러오지 못했습니다.",
  },
});

export const GET = proxy.GET!;
