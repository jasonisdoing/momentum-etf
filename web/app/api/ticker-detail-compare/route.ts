import { NextRequest } from "next/server";

import { proxyStream } from "@/lib/stream-proxy";

export const dynamic = "force-dynamic";

// 비교 계산은 SSE 스트림이다(튜닝과 같은 프록시) — 8종목 × 구성종목 시세 조회가 한
// 요청에 몰리면 JSON 프록시는 타임아웃에 걸렸고, 진행 바도 추정으로만 움직였다.
export async function POST(request: NextRequest) {
  const body = (await request.json()) as { items?: unknown; include_holdings?: unknown };
  return proxyStream("/internal/ticker-detail/compare", {
    items: Array.isArray(body?.items) ? body.items : [],
    // 성과분석·월간분석 탭은 구성종목 계산이 필요 없어 false 로 온다(기본값은 true).
    include_holdings: body?.include_holdings === undefined ? true : Boolean(body.include_holdings),
  });
}
