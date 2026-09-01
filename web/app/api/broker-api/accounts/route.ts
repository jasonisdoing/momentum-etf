import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

/** 커넥터 검증 + 계좌 나열 — 계좌 설정의 '확인' 버튼이 부른다. */
const proxy = createFastApiProxy({
  GET: {
    path: "/internal/broker-api/accounts",
    error: "증권사 계좌를 불러오지 못했습니다.",
    forwardQuery: ["provider"],
  },
});

export const GET = proxy.GET!;
