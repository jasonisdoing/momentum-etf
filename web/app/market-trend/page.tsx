import { fetchFastApiJson } from "../../lib/internal-api";
import { MarketTrendClient } from "./MarketTrendClient";

export const dynamic = "force-dynamic";

type MarketTrendDefaults = {
  ma_days: number;
  score_anchor_percentile: number;
};

export default async function MarketTrendPage() {
  // config.py 의 MA/추세점수 설정을 받아온다 (표시 전용).
  const defaults = await fetchFastApiJson<MarketTrendDefaults>("/internal/market-trend/defaults");
  return (
    <MarketTrendClient
      maDays={defaults.ma_days}
      scoreAnchorPercentile={defaults.score_anchor_percentile}
    />
  );
}
