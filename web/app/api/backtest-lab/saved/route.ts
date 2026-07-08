import { NextRequest, NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

/** 저장된 포트폴리오 목록 — FastAPI `/internal/portfolio-lab/saved` 프록시. */
export async function GET() {
  try {
    const data = await fetchFastApiJson("/internal/portfolio-lab/saved", { method: "GET" });
    return jsonNoStore(data as Record<string, unknown>);
  } catch (error) {
    const message = error instanceof Error ? error.message : "저장된 포트폴리오를 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}

/** 포트폴리오 저장 — `{ name, tickers, months }`. */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const data = await fetchFastApiJson("/internal/portfolio-lab/saved", {
      method: "PUT",
      body: JSON.stringify(body),
    });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "포트폴리오 저장에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

/** 포트폴리오 삭제 — `{ name }`. */
export async function DELETE(request: NextRequest) {
  try {
    const { name } = (await request.json()) as { name?: string };
    const data = await fetchFastApiJson(`/internal/portfolio-lab/saved/${encodeURIComponent(name ?? "")}`, {
      method: "DELETE",
    });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "포트폴리오 삭제에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
