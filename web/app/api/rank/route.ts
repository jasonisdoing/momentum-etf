import { NextRequest, NextResponse } from "next/server";

import { loadRankData } from "../../../lib/rank-store";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const tickerType = searchParams.get("ticker_type") ?? undefined;
    const asOfDate = searchParams.get("as_of_date") ?? undefined;
    const heldBonusScore = searchParams.get("held_bonus_score");
    const maType = searchParams.get("ma_type");
    const maMonthsRaw = searchParams.get("ma_months");
    const maRuleOverride =
      maType || maMonthsRaw
        ? {
            ma_type: maType ?? "",
            ma_months: maMonthsRaw ? Number(maMonthsRaw) : 0,
            ma_days: 0,
            score_column: "추세",
          }
        : undefined;
    const data = await loadRankData({
      ticker_type: tickerType,
      ma_rule_override: maRuleOverride,
      as_of_date: asOfDate,
      held_bonus_score: heldBonusScore === null ? undefined : Number(heldBonusScore),
    });
    return jsonNoStore(data);
  } catch (error) {
    let message = error instanceof Error ? error.message : "순위 데이터를 불러오지 못했습니다.";
    if (message.includes("응답하지 않았습니다") || message.includes("fetch failed") || message.includes("timeout")) {
      message = "몽고디비 데이터베이스 응답 지연(타임아웃)으로 인해 순위 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.";
    }
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
