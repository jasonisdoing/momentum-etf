import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  POST: { path: "/internal/alarms/account", error: "저장에 실패했습니다.", forwardBody: true },
});

export const POST = proxy.POST!;
