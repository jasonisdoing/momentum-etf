"use client";

import { useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { MarketManager } from "./MarketManager";

type MarketHeaderSummary = {
  filteredCount: number;
  totalCount: number;
  updatedAt: string;
};

const DEFAULT_SUMMARY: MarketHeaderSummary = {
  filteredCount: 0,
  totalCount: 0,
  updatedAt: "-",
};

export function MarketPageClient() {
  const [summary, setSummary] = useState<MarketHeaderSummary>(DEFAULT_SUMMARY);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>총:</span>
          <span className="appHeaderMetricValue">{new Intl.NumberFormat("ko-KR").format(summary.filteredCount)}개</span>
        </div>
        <div className="appHeaderMetric">
          <span>전체:</span>
          <span className="appHeaderMetricValue">{new Intl.NumberFormat("ko-KR").format(summary.totalCount)}개</span>
        </div>
        <div className="appHeaderMetric">
          <span>KIS 마스터 갱신:</span>
          <span className="appHeaderMetricValue">{summary.updatedAt}</span>
        </div>
      </div>
    ),
    [summary],
  );

  return (
    <PageFrame title="ETF 마켓" fullHeight fullWidth titleRight={titleRight}>
      <MarketManager onHeaderSummaryChange={setSummary} />
    </PageFrame>
  );
}
