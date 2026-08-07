"use client";

import { PageFrame } from "../components/PageFrame";
import { AssetChartsManager } from "./AssetChartsManager";

type AssetChartsHeaderSummary = {
  latestWeekDate: string;
  rowCount: number;
  latestTotalAssets: number | null;
  totalAssetsDelta: number | null;
  totalAssetsDeltaPct: number | null;
};

export function AssetChartsPageClient() {
  return (
    <PageFrame title="자산 차트" fullHeight fullWidth>
      <AssetChartsManager />
    </PageFrame>
  );
}

export type { AssetChartsHeaderSummary };
