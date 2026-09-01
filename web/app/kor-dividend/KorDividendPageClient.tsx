"use client";

import { useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { KorDividendManager } from "./KorDividendManager";

export function KorDividendPageClient() {
  const [totalCount, setTotalCount] = useState(0);

  const titleRight = (
    <div className="appHeaderMetrics rankToolbarMeta">
      <div className="appHeaderMetric">
        <span>종목 수:</span>
        <span className="appHeaderMetricValue">{totalCount}개</span>
      </div>
    </div>
  );

  return (
    <PageFrame title="한국 배당주" fullHeight fullWidth titleRight={titleRight}>
      <KorDividendManager onSummaryChange={setTotalCount} />
    </PageFrame>
  );
}
