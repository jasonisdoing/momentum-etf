import { fetchFastApiJson } from "../../lib/internal-api";
import { MarketTrendClient } from "./MarketTrendClient";

export const dynamic = "force-dynamic";

type MarketTrendDefaults = {
  ma_type: string;
  ma_months: number;
  ma_types: string[];
  ma_months_max: number;
  score_anchor_percentile: number;
};

export default async function MarketTrendPage() {
  // config.py 의 단일 진실 소스에서 기본 MA 설정 + 추세점수 설정을 받아온다.
  const defaults = await fetchFastApiJson<MarketTrendDefaults>("/internal/market-trend/defaults");
  return (
    <MarketTrendClient
      defaultMaType={defaults.ma_type}
      defaultMaMonths={defaults.ma_months}
      maTypes={defaults.ma_types}
      maMonthsMax={defaults.ma_months_max}
      scoreAnchorPercentile={defaults.score_anchor_percentile}
    />
  );
}
