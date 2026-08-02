import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../../lib/internal-api";
import { jsonNoStore } from "../../../../lib/no-store-response";

export const dynamic = "force-dynamic";

/** 메모 수정. body: {"title": "...", "content": "..."} */
export async function PUT(request: NextRequest, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    const body = await request.json();
    const data = await fetchFastApiJson(`/internal/memos/${encodeURIComponent(id)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "메모를 저장하지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}

/** 메모 삭제. */
export async function DELETE(_request: NextRequest, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    const data = await fetchFastApiJson(`/internal/memos/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "메모를 삭제하지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
