import { NextRequest, NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";

export const dynamic = "force-dynamic";

/** 포트폴리오 실험 실행 — FastAPI `/internal/portfolio-lab/run` 프록시. */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const data = await fetchFastApiJson("/internal/portfolio-lab/run", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "포트폴리오 실험 실행에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
