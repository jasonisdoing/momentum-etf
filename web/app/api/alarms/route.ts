import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  GET: { path: "/internal/alarms", error: "알람 설정을 불러오지 못했습니다." },
});

export const GET = proxy.GET!;
