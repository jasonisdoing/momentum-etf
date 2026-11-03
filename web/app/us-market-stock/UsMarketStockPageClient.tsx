"use client";

import { useCallback, useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { SlackAlertBasisMetric } from "../components/SlackAlertBasisMetric";
import { UsMarketStockManager } from "./UsMarketStockManager";

type StockSummary = {
  count: number;
  totalCount: number;
};

export function UsMarketStockPageClient() {
  const [summary, setSummary] = useState<StockSummary>({ count: 0, totalCount: 0 });
  const handleSummaryChange = useCallback((s: StockSummary) => setSummary(s), []);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>표시:</span>
          <span className="appHeaderMetricValue">{summary.count}개</span>
        </div>
        <SlackAlertBasisMetric apiPath="/api/us-market-stocks/alert-basis" />
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
