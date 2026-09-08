"use client";

import { useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { StocksManager } from "./StocksManager";

type StocksHeaderSummary = {
  upCount: number;
  upPct: number;
  totalCount: number;
  ruleSummary: string;
  /** 시장 ADR(모멘텀 ADR 게이트와 같은 소스) — 레짐 지수 없는 풀은 null. */
  adr: { market: string; value: number; floor: number | null } | null;
};

const DEFAULT_SUMMARY: StocksHeaderSummary = {
  upCount: 0,
  upPct: 0,
  totalCount: 0,
  ruleSummary: "-",
  adr: null,
};

export function StocksPageClient() {
  const [summary, setSummary] = useState<StocksHeaderSummary>(DEFAULT_SUMMARY);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>매수 후보:</span>
          <span className="appHeaderMetricValue is-danger">
            {summary.upCount}개 ({summary.upPct}%)
          </span>
        </div>
        <div className="appHeaderMetric">
          <span>기준:</span>
          <span className="appHeaderMetricValue">{summary.ruleSummary}</span>
        </div>
        {summary.adr ? (
          <div className="appHeaderMetric" title={`시장(${summary.adr.market})의 20일 등락 비율 — 모멘텀 ADR 게이트와 같은 값`}>
            <span>ADR:</span>
            <span className="appHeaderMetricValue is-danger">
              {summary.adr.value.toFixed(1)}
              {summary.adr.floor != null ? ` (하한 ${summary.adr.floor})` : ""}
            </span>
          </div>
        ) : null}
        <div className="appHeaderMetric">
          <span>총 개수:</span>
          <span className="appHeaderMetricValue">{new Intl.NumberFormat("ko-KR").format(summary.totalCount)}개</span>
        </div>
      </div>
    ),
    [summary],
  );

  return (
    <PageFrame title="순위" fullHeight fullWidth titleRight={titleRight}>
      <StocksManager onHeaderSummaryChange={setSummary} />
    </PageFrame>
  );
}
