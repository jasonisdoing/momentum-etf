"use client";

import { useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { SystemManager } from "./SystemManager";

type SystemHeaderSummary = {
  scheduleCount: number;
};

const DEFAULT_SUMMARY: SystemHeaderSummary = {
  scheduleCount: 0,
};

export function SystemPageClient() {
  const [summary, setSummary] = useState<SystemHeaderSummary>(DEFAULT_SUMMARY);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>배치:</span>
          <span className="appHeaderMetricValue">{new Intl.NumberFormat("ko-KR").format(summary.scheduleCount)}개</span>
        </div>
      </div>
    ),
    [summary],
  );

  return (
    <PageFrame title="배치" titleRight={titleRight}>
      <SystemManager onHeaderSummaryChange={setSummary} />
    </PageFrame>
  );
}
