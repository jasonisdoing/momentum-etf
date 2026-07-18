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
    <PageFrame title="설정" titleRight={titleRight}>
      <SettingsManager onSummaryChange={setTotalCount} />
    </PageFrame>
  );
}
