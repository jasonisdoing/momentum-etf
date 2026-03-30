import { NextRequest, NextResponse } from "next/server";
import { fetchFastApiJson } from "@/lib/internal-api";

export async function GET(request: NextRequest) {
  try {
    const accountId = request.nextUrl.searchParams.get("account") ?? undefined;
    const url = `/internal/account-stocks${accountId ? `?account_id=${encodeURIComponent(accountId)}` : ""}`;
    const payload = await fetchFastApiJson(url);
    return NextResponse.json(payload);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "포트폴리오 데이터를 불러오지 못했습니다." },
      { status: 400 },
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const payload = await request.json();
    const result = await fetchFastApiJson("/internal/account-stocks", {
      method: "POST",
      body: JSON.stringify({
        account_id: payload.account_id,
        ticker: payload.ticker,
        ratio: payload.ratio,
      }),
    });
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "종목 추가/수정에 실패했습니다." },
      { status: 400 },
    );
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const payload = await request.json();
    const result = await fetchFastApiJson("/internal/account-stocks", {
      method: "PATCH",
      body: JSON.stringify({
        account_id: payload.account_id,
        ticker: payload.ticker,
        ratio: payload.ratio,
      }),
    });
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "비중 수정에 실패했습니다." },
      { status: 400 },
    );
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const payload = await request.json();
    const result = await fetchFastApiJson("/internal/account-stocks", {
      method: "DELETE",
      body: JSON.stringify({
        account_id: payload.account_id,
        ticker: payload.ticker,
      }),
    });
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "종목 삭제에 실패했습니다." },
      { status: 400 },
    );
  }
}
