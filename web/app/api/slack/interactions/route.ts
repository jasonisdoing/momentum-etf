import { NextResponse, type NextRequest } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";

export const dynamic = "force-dynamic";

/**
 * 슬랙 인터랙션 웹훅(버튼 클릭·모달 제출) — 요청 원문과 서명 헤더를 그대로 FastAPI 로 넘긴다.
 * 서명은 원문 바이트 기준이라 여기서 파싱하지 않는다. 로그인 검사는 `proxy.ts` 에서 이 경로만 건너뛴다.
 */
export async function POST(request: NextRequest) {
  try {
    const data = await fetchFastApiJson<Record<string, unknown>>("/internal/slack/interactions", {
      method: "POST",
      body: JSON.stringify({
        raw_body: await request.text(),
        timestamp: request.headers.get("x-slack-request-timestamp") ?? "",
        signature: request.headers.get("x-slack-signature") ?? "",
      }),
    });
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "슬랙 요청을 처리하지 못했습니다." }, { status: 500 });
  }
}
