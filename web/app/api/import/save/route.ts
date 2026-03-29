import { NextRequest, NextResponse } from "next/server";

import { saveBulkImportRows, type ParsedImportRow } from "@/lib/import-store";

export async function POST(request: NextRequest) {
  try {
    const payload = (await request.json()) as { rows?: ParsedImportRow[] };
    const rows = Array.isArray(payload.rows) ? payload.rows : [];
    const result = await saveBulkImportRows(rows);
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "벌크 입력 저장에 실패했습니다." },
      { status: 400 },
    );
  }
}
