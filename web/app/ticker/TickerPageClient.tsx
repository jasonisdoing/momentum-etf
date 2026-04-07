"use client";

import { useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { TickerDetailManager } from "./TickerDetailManager";

type TickerHeaderSummary = {
  displayTitle: string;
  priceText: string;
  changeText: string;
  changeClassName: string;
};

const DEFAULT_SUMMARY: TickerHeaderSummary = {
  displayTitle: "",
  priceText: "",
  changeText: "",
  changeClassName: "",
};

export function TickerPageClient() {
  const [summary, setSummary] = useState<TickerHeaderSummary>(DEFAULT_SUMMARY);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        {summary.displayTitle ? (
          <div className="appHeaderMetric">
            <span>종목:</span>
            <span className="appHeaderMetricValue">{summary.displayTitle}</span>
          </div>
        ) : null}
        {summary.priceText ? (
          <div className="appHeaderMetric">
            <span>현재가:</span>
            <span className="appHeaderMetricValue">{summary.priceText}</span>
            {summary.changeText ? <span className={summary.changeClassName}>{summary.changeText}</span> : null}
          </div>
        ) : null}
      </div>
    ),
    [summary],
  );

  return (
    <PageFrame title="개별종목" fullHeight fullWidth titleRight={titleRight}>
      <TickerDetailManager onHeaderSummaryChange={setSummary} />
    </PageFrame>
  );
}
