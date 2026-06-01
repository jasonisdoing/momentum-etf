"use client";

import { useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { YearlyManager } from "./YearlyManager";

type YearlyHeaderSummary = {
  activeYearDate: string;
  rowCount: number;
  dirtyCount: number;
};

const DEFAULT_SUMMARY: YearlyHeaderSummary = {
  activeYearDate: "-",
  rowCount: 0,
  dirtyCount: 0,
};

export function YearlyPageClient() {
  const [summary, setSummary] = useState<YearlyHeaderSummary>(DEFAULT_SUMMARY);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>활성 연도:</span>
          <span className="appHeaderMetricValue">{summary.activeYearDate || "-"}</span>
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
    <PageFrame title="년별" fullHeight fullWidth titleRight={titleRight}>
      <YearlyManager onHeaderSummaryChange={setSummary} />
    </PageFrame>
  );
}
