import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  POST: { path: "/internal/alarms/send", error: "발송에 실패했습니다." },
});

export const POST = proxy.POST!;
