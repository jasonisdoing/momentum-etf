import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  GET: { path: "/internal/memos", error: "메모를 불러오지 못했습니다." },
  POST: { path: "/internal/memos", error: "메모를 저장하지 못했습니다.", forwardBody: true },
});

export const GET = proxy.GET!;
export const POST = proxy.POST!;
