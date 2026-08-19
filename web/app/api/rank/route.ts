import { NextRequest, NextResponse } from "next/server";

import { loadRankData, type RankMaRuleOverride } from "../../../lib/rank-store";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const tickerType = searchParams.get("ticker_type") ?? undefined;
    const asOfDate = searchParams.get("as_of_date") ?? undefined;
    // 화면에서 임시로 바꾼 이평선 값. 넘어온 항목만 실어 보내고, 없는 항목은 저장 규칙을 그대로 쓴다.
    const maRuleOverride: RankMaRuleOverride = {};
    for (const key of ["short_ma_days", "long_ma_days"] as const) {
      const raw = searchParams.get(key);
      if (raw === null || raw === "") {
        continue;
      }
      const value = Number(raw);
      if (!Number.isInteger(value) || value <= 0) {
        throw new Error(`이평선 일수 '${key}' 값이 올바르지 않습니다: ${raw}`);
      }
      maRuleOverride[key] = value;
    }
    const hasMaRuleOverride = Object.keys(maRuleOverride).length > 0;
    const data = await loadRankData({
      ticker_type: tickerType,
      ma_rule_override: hasMaRuleOverride ? maRuleOverride : undefined,
      as_of_date: asOfDate,
    }, request.signal);
    return jsonNoStore(data);
  } catch (error) {
    let message = error instanceof Error ? error.message : "순위 데이터를 불러오지 못했습니다.";
    if (message.includes("응답하지 않았습니다") || message.includes("fetch failed") || message.includes("timeout")) {
      message = "몽고디비 데이터베이스 응답 지연(타임아웃)으로 인해 순위 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.";
    }
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
