"use client";

import { useCallback, useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { UsMarketStockManager } from "./UsMarketStockManager";

type StockSummary = {
  index: string;
  count: number;
  totalCount: number;
  /** 슬랙 미등록 알림의 대상 명단 설명 — 서버가 준다(불러오기 전에는 null). */
  alertBasis: string | null;
};

function formatIndexLabel(index: string): string {
  if (index === "COMBINED") return "시총상위1000 + S&P500 + 나스닥100";
  if (index === "USTOP1000") return "시총상위1000";
  if (index === "SP500") return "S&P500";
  if (index === "NDX100") return "나스닥100";
  return index;
}

export function UsMarketStockPageClient() {
  const [summary, setSummary] = useState<StockSummary>({ index: "COMBINED", count: 0, totalCount: 0, alertBasis: null });

  const handleSummaryChange = useCallback((s: StockSummary) => setSummary(s), []);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>표시:</span>
          <span className="appHeaderMetricValue">{summary.count}개</span>
        </div>
        <div className="appHeaderMetric">
          <span>인덱스:</span>
          <span className="appHeaderMetricValue">{formatIndexLabel(summary.index)}</span>
        </div>
        <div className="appHeaderMetric" title="이 명단 중 종목풀에 없는 종목을 매일 슬랙으로 알립니다.">
          <span>슬랙 기준:</span>
          <span className="appHeaderMetricValue">{summary.alertBasis ?? "-"}</span>
        </div>
      </div>
    ),
    [summary],
  );

  return (
    <PageFrame title="🇺🇸 미국 개별주" fullHeight fullWidth titleRight={titleRight}>
      <UsMarketStockManager onSummaryChange={handleSummaryChange} />
    </PageFrame>
  );
}
