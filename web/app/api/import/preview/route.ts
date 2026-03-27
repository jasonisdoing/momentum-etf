import { NextRequest, NextResponse } from "next/server";

import { parseBulkImportText } from "@/lib/import-store";

export async function POST(request: NextRequest) {
  try {
    const payload = (await request.json()) as { text?: string };
    const preview = await parseBulkImportText(String(payload.text ?? ""));
    return NextResponse.json(preview);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "벌크 입력 파싱에 실패했습니다." },
      { status: 400 },
    );
  }
}
