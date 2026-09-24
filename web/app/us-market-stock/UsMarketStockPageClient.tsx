"use client";

import { useCallback, useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { UsMarketStockManager } from "./UsMarketStockManager";

type StockSummary = {
  index: string;
  count: number;
  totalCount: number;
};

function formatIndexLabel(index: string): string {
  if (index === "COMBINED") return "S&P300 + 나스닥100";
  if (index === "SP500") return "S&P300";
  if (index === "NDX100") return "나스닥100";
  return index;
}

export function UsMarketStockPageClient() {
  const [summary, setSummary] = useState<StockSummary>({ index: "COMBINED", count: 0, totalCount: 0 });

  const handleSummaryChange = useCallback((s: StockSummary) => setSummary(s), []);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>표시:</span>
          <span className="appHeaderMetricValue">{summary.count}개</span>
        </div>
        <div className="appHeaderMetric">
          <span>인덱스:</span>
          <span className="appHeaderMetricValue">{formatIndexLabel(summary.index)}</span>
        </div>
      </div>
    ),
    [summary],
  );

  return (
    <PageFrame title="🇺🇸 미국 개별주" fullHeight fullWidth titleRight={titleRight}>
      <UsMarketStockManager onSummaryChange={handleSummaryChange} />
    </PageFrame>
  );
}
