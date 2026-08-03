import { fetchFastApiJson } from "../lib/internal-api";
import { PageFrame } from "./components/PageFrame";
import { HomeTrendCharts } from "./components/HomeTrendCharts";

export const dynamic = "force-dynamic";

type MarketTrendDefaults = {
  ma_days: number;
  score_anchor_percentile: number;
  ma_type: string;
};

// 시장추세 기준값(MA 일수/종류) — 실패해도 홈은 떠야 하므로 실패 시 null.
async function loadTrendDefaults(): Promise<MarketTrendDefaults | null> {
  try {
    return await fetchFastApiJson<MarketTrendDefaults>("/internal/market-trend/defaults");
  } catch {
    return null;
  }
}

// 홈 = 메뉴 겸 대시보드(허브). 메인은 시장지수 추세 4차트(2×2), 우측 메뉴 레일은 AppShell 이 렌더.
export default async function HomePage() {
  const trendDefaults = await loadTrendDefaults();

  return (
    <PageFrame title="홈" fullWidth>
      {trendDefaults ? (
        <HomeTrendCharts maDays={trendDefaults.ma_days} maType={trendDefaults.ma_type} />
      ) : (
        <div className="text-secondary" style={{ fontSize: "var(--fs-base)" }}>
          시장추세 기준값을 불러오지 못했습니다. 시장지수 추세 화면에서 확인해주세요.
        </div>
      )}
    </PageFrame>
  );
}
