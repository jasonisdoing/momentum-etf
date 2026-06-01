"use client";

import { useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { HoldingsManager } from "./HoldingsManager";

type HoldingsHeaderSummary = {
  holdingCount: number;
};

const DEFAULT_SUMMARY: HoldingsHeaderSummary = {
  holdingCount: 0,
};

export function HoldingsPageClient() {
  const [summary, setSummary] = useState<HoldingsHeaderSummary>(DEFAULT_SUMMARY);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>종목:</span>
          <span className="appHeaderMetricValue">{summary.holdingCount}개</span>
        </div>
      </div>
    ),
    [summary],
  );

  return (
    <PageFrame title="보유종목" fullHeight fullWidth titleRight={titleRight}>
      <HoldingsManager onHeaderSummaryChange={setSummary} />
    </PageFrame>
  );
}
