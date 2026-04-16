"use client";

import { useMemo } from "react";

import { PageFrame } from "../components/PageFrame";
import { KorMarketStockManager } from "./KorMarketStockManager";

export function KorMarketStockPageClient() {
  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>총:</span>
          <span className="appHeaderMetricValue">0개</span>
        </div>
        <div className="appHeaderMetric">
          <span>전체:</span>
          <span className="appHeaderMetricValue">0개</span>
        </div>
      </div>
    ),
    [],
  );

  return (
    <PageFrame title="🇰🇷 한국 개별주" fullHeight fullWidth titleRight={titleRight}>
      <KorMarketStockManager />
    </PageFrame>
  );
}
