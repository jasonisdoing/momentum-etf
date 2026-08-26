"use client";

import { useCallback, useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { KorMarketStockManager } from "./KorMarketStockManager";

type StockSummary = {
  market: string;
  count: number;
  totalCount: number;
};

/** 지수 구성종목 마켓의 총계 라벨 — 시총 상위 마켓은 "<마켓> 전체" 로 쓴다. */
const INDEX_MARKET_TOTAL_LABELS: Record<string, string> = {
  KOSPI200: "KODEX 200 구성",
  KOSDAQ150: "KODEX 코스닥150 구성",
};

export function KorMarketStockPageClient() {
  const [summary, setSummary] = useState<StockSummary>({ market: "KOSPI", count: 0, totalCount: 0 });

  const handleSummaryChange = useCallback((s: StockSummary) => setSummary(s), []);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>표시:</span>
          <span className="appHeaderMetricValue">{summary.count}개</span>
        </div>
        <div className="appHeaderMetric">
          <span>{INDEX_MARKET_TOTAL_LABELS[summary.market] ?? `${summary.market} 전체`}:</span>
          <span className="appHeaderMetricValue">{new Intl.NumberFormat("ko-KR").format(summary.totalCount)}개</span>
        </div>
      </div>
    ),
    [summary],
  );

  return (
    <PageFrame title="🇰🇷 한국 개별주" fullHeight fullWidth titleRight={titleRight}>
      <KorMarketStockManager onSummaryChange={handleSummaryChange} />
    </PageFrame>
  );
}
