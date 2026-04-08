"use client";

import { useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { RankManager } from "./RankManager";

type RankHeaderSummary = {
  upCount: number;
  upPct: number;
  totalCount: number;
  ruleSummary: string;
};

const DEFAULT_SUMMARY: RankHeaderSummary = {
  upCount: 0,
  upPct: 0,
  totalCount: 0,
  ruleSummary: "-",
};

export function RankPageClient() {
  const [summary, setSummary] = useState<RankHeaderSummary>(DEFAULT_SUMMARY);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>점수 양수:</span>
          <span className="appHeaderMetricValue is-danger">
            {summary.upCount}개 ({summary.upPct}%)
          </span>
        </div>
        <div className="appHeaderMetric">
          <span>기준:</span>
          <span className="appHeaderMetricValue">{summary.ruleSummary}</span>
        </div>
        <div className="appHeaderMetric">
          <span>총 개수:</span>
          <span className="appHeaderMetricValue">{new Intl.NumberFormat("ko-KR").format(summary.totalCount)}개</span>
        </div>
      </div>
    ),
    [summary],
  );

  return (
    <PageFrame title="종목 관리" fullHeight fullWidth titleRight={titleRight}>
      <RankManager onHeaderSummaryChange={setSummary} />
    </PageFrame>
  );
}
