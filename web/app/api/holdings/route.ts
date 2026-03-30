import { NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";

type HoldingsRow = {
  account_name: string;
  currency: string;
  bucket: string;
  bucket_id: number;
  ticker: string;
  name: string;
  quantity: number;
  average_buy_price: string;
  current_price: string;
  pnl_krw: number;
  return_pct: number;
  buy_amount_krw: number;
  valuation_krw: number;
};

export async function GET() {
  try {
    const payload = await fetchFastApiJson<{ rows: HoldingsRow[] }>("/internal/holdings");
    return NextResponse.json(payload);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "보유 종목을 불러오지 못했습니다." },
      { status: 400 },
    );
  }
}

export const dynamic = "force-dynamic";
