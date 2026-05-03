"use client";

import { useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { MonthlyManager } from "./MonthlyManager";

type MonthlyHeaderSummary = {
  activeMonthDate: string;
  rowCount: number;
  dirtyCount: number;
};

const DEFAULT_SUMMARY: MonthlyHeaderSummary = {
  activeMonthDate: "-",
  rowCount: 0,
  dirtyCount: 0,
};

export function MonthlyPageClient() {
  const [summary, setSummary] = useState<MonthlyHeaderSummary>(DEFAULT_SUMMARY);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>활성 월:</span>
          <span className="appHeaderMetricValue">{summary.activeMonthDate || "-"}</span>
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
    <PageFrame title="월별" fullHeight fullWidth titleRight={titleRight}>
      <MonthlyManager onHeaderSummaryChange={setSummary} />
    </PageFrame>
  );
}
