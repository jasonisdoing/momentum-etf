import { NextRequest, NextResponse } from "next/server";

import { loadTickerDetailData } from "../../../lib/ticker-detail-store";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const ticker = searchParams.get("ticker");
    const tickerType = searchParams.get("ticker_type");
    const countryCode = searchParams.get("country_code") ?? undefined;
    const months = searchParams.get("months") ? Number(searchParams.get("months")) : undefined;

    if (!ticker || !tickerType) {
      return NextResponse.json({ error: "ticker, ticker_type 파라미터가 필요합니다." }, { status: 400 });
    }

    const data = await loadTickerDetailData({
      ticker,
      ticker_type: tickerType,
      country_code: countryCode,
      months,
    });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "종목 상세 데이터를 불러오지 못했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
