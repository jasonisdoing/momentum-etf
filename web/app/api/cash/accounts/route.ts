import { NextResponse } from "next/server";

import { loadCashAccounts } from "@/lib/cash-store";
import { loadExchangeRates } from "@/lib/exchange-rates";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const [accounts, rates] = await Promise.all([loadCashAccounts(), loadExchangeRates()]);
    return NextResponse.json({ accounts, rates });
  } catch (error) {
    const message = error instanceof Error ? error.message : "자산관리 데이터를 불러오지 못했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
