import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

/** 증권사 API 커넥터 목록 — 계좌 설정의 연동 셀렉트가 쓴다. */
const proxy = createFastApiProxy({
  GET: {
    path: "/internal/broker-api/providers",
    error: "커넥터 목록을 불러오지 못했습니다.",
  },
});

export const GET = proxy.GET!;
