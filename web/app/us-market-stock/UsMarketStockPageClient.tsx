"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { UsMarketStockManager } from "./UsMarketStockManager";

type StockSummary = {
  count: number;
  totalCount: number;
};

export function UsMarketStockPageClient() {
  const [summary, setSummary] = useState<StockSummary>({ count: 0, totalCount: 0 });
  // 슬랙 미등록 알림 기준 — 고정 문구라 종목 데이터 로딩(실패 포함)과 따로 한 번 받는다.
  const [alertBasis, setAlertBasis] = useState<string | null>(null);
  useEffect(() => {
    void fetch("/api/us-market-stocks/alert-basis", { cache: "no-store" })
      .then((response) => response.json() as Promise<{ alert_basis?: string }>)
      .then((payload) => setAlertBasis(payload.alert_basis ?? null))
      .catch(() => setAlertBasis(null));
  }, []);

  const handleSummaryChange = useCallback((s: StockSummary) => setSummary(s), []);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>표시:</span>
          <span className="appHeaderMetricValue">{summary.count}개</span>
        </div>
        <div className="appHeaderMetric" title="이 명단 중 종목풀에 없는 종목을 매일 슬랙으로 알립니다.">
          <span>슬랙 기준:</span>
          <span className="appHeaderMetricValue">{alertBasis ?? "-"}</span>
        </div>
      </div>
    ),
    [summary, alertBasis],
  );

  return (
    <PageFrame title="🇺🇸 미국 개별주" fullHeight fullWidth titleRight={titleRight}>
      <UsMarketStockManager onSummaryChange={handleSummaryChange} />
    </PageFrame>
  );
}
