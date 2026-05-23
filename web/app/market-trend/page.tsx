import { fetchFastApiJson } from "../../lib/internal-api";
import { MarketTrendClient } from "./MarketTrendClient";

export const dynamic = "force-dynamic";

type MarketTrendDefaults = {
  ma_type: string;
  ma_months: number;
};

export default async function MarketTrendPage() {
  // config.py 의 단일 진실 소스에서 기본 MA 설정을 받아온다.
  const defaults = await fetchFastApiJson<MarketTrendDefaults>("/internal/market-trend/defaults");
  return <MarketTrendClient defaultMaType={defaults.ma_type} defaultMaMonths={defaults.ma_months} />;
}
