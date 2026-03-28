import { NextRequest, NextResponse } from "next/server";

import { loadSystemData, triggerSystemAction } from "../../../lib/system-store";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const data = await loadSystemData();
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "시스템정보 데이터를 불러오지 못했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const payload = (await request.json()) as { action?: "meta_all" | "cache_all" | "asset_summary" };
    const action = String(payload.action || "").trim() as "meta_all" | "cache_all" | "asset_summary";
    if (!action) {
      return NextResponse.json({ error: "action이 필요합니다." }, { status: 400 });
    }

    const message = await triggerSystemAction(action);
    return NextResponse.json({ message });
  } catch (error) {
    const message = error instanceof Error ? error.message : "시스템 작업 실행에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
