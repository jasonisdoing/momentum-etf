"use client";

import { useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { DailyManager } from "./DailyManager";

type DailyHeaderSummary = {
  latestDate: string;
  rowCount: number;
  dirtyCount: number;
};

const DEFAULT_SUMMARY: DailyHeaderSummary = {
  latestDate: "-",
  rowCount: 0,
  dirtyCount: 0,
};

export function DailyPageClient() {
  const [summary, setSummary] = useState<DailyHeaderSummary>(DEFAULT_SUMMARY);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>최신 일자:</span>
          <span className="appHeaderMetricValue">{summary.latestDate || "-"}</span>
        </div>
        <div className="appHeaderMetric">
          <span>행:</span>
          <span className="appHeaderMetricValue">{new Intl.NumberFormat("ko-KR").format(summary.rowCount)}</span>
        </div>
        {summary.dirtyCount > 0 ? (
          <div className="appHeaderMetric">
            <span>변경:</span>
            <span className="appHeaderMetricValue is-success">
              {new Intl.NumberFormat("ko-KR").format(summary.dirtyCount)}개
            </span>
          </div>
        ) : null}
      </div>
    ),
    [summary],
  );

  return (
    <PageFrame title="일별" fullHeight fullWidth titleRight={titleRight}>
      <DailyManager onHeaderSummaryChange={setSummary} />
    </PageFrame>
  );
}
