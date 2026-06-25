import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

function getFastApiBaseUrl(): string {
  const value = String(process.env.FASTAPI_INTERNAL_URL ?? "").trim();
  if (!value) {
    throw new Error("FASTAPI_INTERNAL_URL 환경변수가 필요합니다.");
  }
  return value.replace(/\/+$/, "");
}

function getFastApiToken(): string {
  const value = String(process.env.FASTAPI_INTERNAL_TOKEN ?? "").trim();
  if (!value) {
    throw new Error("FASTAPI_INTERNAL_TOKEN 환경변수가 필요합니다.");
  }
  return value;
}

/**
 * 배치 작업 실행 로그 다운로드 프록시.
 * 클라이언트는 `/api/system/job-logs?key=...&started_at=...&ended_at=...` 를 호출하고,
 * 본 라우트가 FastAPI 의 `/internal/system/job-logs` 로 인증 헤더와 함께 전달한다.
 * 응답은 text/plain 으로 그대로 전달하고 Content-Disposition 도 보존한다.
 */
export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const key = url.searchParams.get("key");
    const startedAt = url.searchParams.get("started_at");
    const endedAt = url.searchParams.get("ended_at");

    if (!key || !startedAt) {
      return NextResponse.json({ error: "key 와 started_at 파라미터가 필요합니다." }, { status: 400 });
    }

    const params = new URLSearchParams({ key, started_at: startedAt });
    if (endedAt) params.set("ended_at", endedAt);

    const upstream = await fetch(`${getFastApiBaseUrl()}/internal/system/job-logs?${params.toString()}`, {
      method: "GET",
      headers: { "X-Internal-Token": getFastApiToken() },
      cache: "no-store",
    });

    if (!upstream.ok) {
      const text = await upstream.text().catch(() => "");
      return NextResponse.json(
        { error: text || `로그 다운로드 실패 (${upstream.status})` },
        { status: upstream.status },
      );
    }

    const body = await upstream.text();
    const headers = new Headers();
    headers.set("Content-Type", "text/plain; charset=utf-8");
    const disposition = upstream.headers.get("Content-Disposition");
    if (disposition) headers.set("Content-Disposition", disposition);
    return new NextResponse(body, { headers });
  } catch (error) {
    const message = error instanceof Error ? error.message : "로그 다운로드에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
