import { NextRequest, NextResponse } from "next/server";

import { SystemAction, loadSystemData, triggerSystemAction } from "../../../lib/system-store";

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
    const payload = (await request.json()) as { action?: string };
    const actionStr = String(payload.action || "").trim();
    if (!actionStr) {
      return NextResponse.json({ error: "실행할 배치를 지정해야 합니다." }, { status: 400 });
    }

    // 실행 가능한 배치 목록은 백엔드(`utils/system_service._SCRIPT_BY_ACTION`)가 단일 소스다.
    // 여기에 목록을 복사해 두면 배치를 추가할 때마다 어긋난다 (실제로 시장 폭 집계가 그랬다).
    const message = await triggerSystemAction(actionStr);
    return NextResponse.json({ message });
  } catch (error) {
    const message = error instanceof Error ? error.message : "시스템 작업 실행에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
