"use client";

import { useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { SettingsManager } from "./SettingsManager";

export function SettingsPageClient() {
  const [totalCount, setTotalCount] = useState(0);

  const titleRight = (
    <div className="appHeaderMetrics rankToolbarMeta">
      <div className="appHeaderMetric">
        <span>총 개수:</span>
        <span className="appHeaderMetricValue">{totalCount}개</span>
      </div>
    </div>
  );

  return (
    // 항목이 많아 가로를 최대한 쓴다 — 그리드가 14개 컬럼을 한 화면에 놓는다.
    <PageFrame title="설정" fullHeight fullWidth titleRight={titleRight}>
      <SettingsManager onSummaryChange={setTotalCount} />
    </PageFrame>
  );
}
