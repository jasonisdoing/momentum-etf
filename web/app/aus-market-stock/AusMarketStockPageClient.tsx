"use client";

import { useCallback, useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { AusMarketStockManager } from "./AusMarketStockManager";

type StockSummary = {
  index: string;
  count: number;
  totalCount: number;
};

export function AusMarketStockPageClient() {
  const [summary, setSummary] = useState<StockSummary>({ index: "ASX200", count: 0, totalCount: 0 });

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
          <span className="appHeaderMetricValue">{summary.index}</span>
        </div>
      </div>
    ),
    [summary],
  );

  return (
    <PageFrame title="🇦🇺 호주 개별주" fullHeight fullWidth titleRight={titleRight}>
      <AusMarketStockManager onSummaryChange={handleSummaryChange} />
    </PageFrame>
  );
}
