import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  GET: { path: "/internal/strategy-sm", error: "설정을 불러오지 못했습니다." },
  PUT: { path: "/internal/strategy-sm/settings", error: "설정을 저장하지 못했습니다.", forwardBody: true },
});

export const GET = proxy.GET!;
export const PUT = proxy.PUT!;
