import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

function getFastApiBaseUrl(): string {
  const value = String(process.env.FASTAPI_INTERNAL_URL ?? "").trim();
  if (!value) throw new Error("FASTAPI_INTERNAL_URL 환경변수가 필요합니다.");
  return value.replace(/\/+$/, "");
}

function getFastApiToken(): string {
  const value = String(process.env.FASTAPI_INTERNAL_TOKEN ?? "").trim();
  if (!value) throw new Error("FASTAPI_INTERNAL_TOKEN 환경변수가 필요합니다.");
  return value;
}

/**
 * 배치 실행 중단 요청 프록시.
 * `POST /api/system/cancel { key }` → FastAPI `/internal/system/cancel` 로 forward.
 * 응답 status 그대로 전달 (403 = 다른 인스턴스 작업 / 409 = 실행 중 항목 없음 / 200 = 요청 성공).
 */
export async function POST(request: NextRequest) {
  try {
    const payload = (await request.json().catch(() => ({}))) as { key?: string };
    const key = String(payload.key ?? "").trim();
    if (!key) {
      return NextResponse.json({ error: "key 필드가 필요합니다." }, { status: 400 });
    }

    const upstream = await fetch(`${getFastApiBaseUrl()}/internal/system/cancel`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Internal-Token": getFastApiToken(),
      },
      body: JSON.stringify({ key }),
      cache: "no-store",
    });

    const data = await upstream.json().catch(() => ({}));
    return NextResponse.json(data, { status: upstream.status });
  } catch (error) {
    const message = error instanceof Error ? error.message : "중단 요청에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
