"use client";

import { useCallback, useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { UsMarketStockManager } from "./UsMarketStockManager";

type StockSummary = {
  market: string;
  count: number;
  totalCount: number;
};

function formatMarketLabel(market: string): string {
  if (market === "NYS") return "뉴욕";
  if (market === "NSQ") return "나스닥";
  return market;
}

export function UsMarketStockPageClient() {
  const [summary, setSummary] = useState<StockSummary>({ market: "NSQ", count: 0, totalCount: 0 });

  const handleSummaryChange = useCallback((s: StockSummary) => setSummary(s), []);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>표시:</span>
          <span className="appHeaderMetricValue">{summary.count}개</span>
        </div>
        <div className="appHeaderMetric">
          <span>마켓:</span>
          <span className="appHeaderMetricValue">{formatMarketLabel(summary.market)}</span>
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
