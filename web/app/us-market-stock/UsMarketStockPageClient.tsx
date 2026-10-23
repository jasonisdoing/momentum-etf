"use client";

import { useCallback, useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { UsMarketStockManager } from "./UsMarketStockManager";

type StockSummary = {
  index: string;
  count: number;
  totalCount: number;
  /** S&P500 시총 상위 절단 개수 — 서버 상수에서 온다(불러오기 전에는 null). */
  sp500Limit: number | null;
};

function formatIndexLabel(index: string, sp500Limit: number | null): string {
  const sp = `S&P${sp500Limit ?? ""}`;
  if (index === "COMBINED") return `${sp} + 나스닥100`;
  if (index === "SP500") return sp;
  if (index === "NDX100") return "나스닥100";
  return index;
}

export function UsMarketStockPageClient() {
  const [summary, setSummary] = useState<StockSummary>({ index: "COMBINED", count: 0, totalCount: 0, sp500Limit: null });

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
          <span className="appHeaderMetricValue">{formatIndexLabel(summary.index, summary.sp500Limit)}</span>
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
