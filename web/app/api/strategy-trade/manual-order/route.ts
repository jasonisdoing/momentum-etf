import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";
const proxy = createFastApiProxy({
  POST: {
    path: "/internal/strategy-trade/manual-order",
    error: "수동 주문을 처리하지 못했습니다.",
    forwardBody: true,
  },
});
export const POST = proxy.POST!;
