"use client";

import { useCallback, useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { KorMarketStockManager } from "./KorMarketStockManager";

type StockSummary = {
  market: string;
  count: number;
  totalCount: number;
};

export function KorMarketStockPageClient() {
  const [summary, setSummary] = useState<StockSummary>({ market: "KOSPI", count: 0, totalCount: 0 });

  const handleSummaryChange = useCallback((s: StockSummary) => setSummary(s), []);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>표시:</span>
          <span className="appHeaderMetricValue">{summary.count}개</span>
        </div>
        <div className="appHeaderMetric">
          <span>{summary.market} 전체:</span>
          <span className="appHeaderMetricValue">{new Intl.NumberFormat("ko-KR").format(summary.totalCount)}개</span>
        </div>
      </div>
    ),
    [summary],
  );

  return (
    <PageFrame title="🇰🇷 한국 개별주" fullHeight fullWidth titleRight={titleRight}>
      <KorMarketStockManager onSummaryChange={handleSummaryChange} />
    </PageFrame>
  );
}
