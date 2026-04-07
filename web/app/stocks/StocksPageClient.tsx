"use client";

import { useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { StocksManager } from "./StocksManager";

type StocksHeaderSummary = {
  tickerTypeName: string;
  viewLabel: string;
  totalCount: number;
  selectedCount: number;
  dirtyCount: number;
};

const DEFAULT_SUMMARY: StocksHeaderSummary = {
  tickerTypeName: "-",
  viewLabel: "등록된 종목",
  totalCount: 0,
  selectedCount: 0,
  dirtyCount: 0,
};

export function StocksPageClient() {
  const [summary, setSummary] = useState<StocksHeaderSummary>(DEFAULT_SUMMARY);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>계정:</span>
          <span className="appHeaderMetricValue">{summary.tickerTypeName}</span>
        </div>
        <div className="appHeaderMetric">
          <span>보기:</span>
          <span className="appHeaderMetricValue">{summary.viewLabel}</span>
        </div>
        <div className="appHeaderMetric">
          <span>총 개수:</span>
          <span className="appHeaderMetricValue">{new Intl.NumberFormat("ko-KR").format(summary.totalCount)}개</span>
        </div>
        {summary.selectedCount > 0 ? (
          <div className="appHeaderMetric">
            <span>선택:</span>
            <span className="appHeaderMetricValue is-primary">
              {new Intl.NumberFormat("ko-KR").format(summary.selectedCount)}개
            </span>
          </div>
        ) : null}
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
    <PageFrame title="종목 관리" fullHeight fullWidth titleRight={titleRight}>
      <StocksManager onHeaderSummaryChange={setSummary} />
    </PageFrame>
  );
}
