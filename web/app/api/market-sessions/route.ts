import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  const internalToken = process.env.FASTAPI_INTERNAL_TOKEN || "";
  const baseUrl = process.env.FASTAPI_INTERNAL_URL || "http://127.0.0.1:8000";

  try {
    const response = await fetch(`${baseUrl}/internal/market/sessions`, {
      cache: "no-store",
      headers: { "X-Internal-Token": internalToken },
    });
    if (!response.ok) {
      return NextResponse.json({ error: "시장 세션 상태를 가져오지 못했습니다." }, { status: response.status });
    }
    return NextResponse.json(await response.json());
  } catch {
    return NextResponse.json({ error: "시장 세션 상태를 가져오지 못했습니다." }, { status: 500 });
  }
}
