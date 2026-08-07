import { NextRequest, NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";

export const dynamic = "force-dynamic";

/** 종목풀 편집 가능 설정 조회 — FastAPI `/internal/pool-settings` 프록시. */
export async function GET() {
  try {
    const data = await fetchFastApiJson("/internal/pool-settings", { method: "GET" });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "설정을 불러오지 못했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

/** 종목풀 편집 가능 설정 저장 — `{ pool_id, values }`. */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const data = await fetchFastApiJson("/internal/pool-settings", {
      method: "PUT",
      body: JSON.stringify(body),
    });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "설정 저장에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

/** 신규 종목풀 생성 — `{ values }`. */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const data = await fetchFastApiJson("/internal/pool-settings/pools", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "종목풀 생성에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

/** 종목풀 정의 수정 — `{ pool_id, values }`. */
export async function PATCH(request: NextRequest) {
  try {
    const body = await request.json();
    const poolId = String(body.pool_id ?? "").trim();
    if (!poolId) {
      return NextResponse.json({ error: "pool_id가 필요합니다." }, { status: 400 });
    }
    const data = await fetchFastApiJson(`/internal/pool-settings/pools/${encodeURIComponent(poolId)}`, {
      method: "PATCH",
      body: JSON.stringify({ values: body.values ?? {}, save_method: body.save_method ?? "사용자" }),
    });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "종목풀 수정에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

/** 종목풀 하드 삭제 — `{ pool_id }`. */
export async function DELETE(request: NextRequest) {
  try {
    const body = await request.json();
    const poolId = String(body.pool_id ?? "").trim();
    if (!poolId) {
      return NextResponse.json({ error: "pool_id가 필요합니다." }, { status: 400 });
    }
    const data = await fetchFastApiJson(`/internal/pool-settings/pools/${encodeURIComponent(poolId)}`, {
      method: "DELETE",
    });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "종목풀 삭제에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
