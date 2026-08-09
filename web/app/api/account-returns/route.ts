import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  GET: { path: "/internal/snapshots/account-returns", error: "계좌 기간 수익률을 불러오지 못했습니다." },
});

export const GET = proxy.GET!;
