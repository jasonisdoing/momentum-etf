"use client";

import { PageFrame } from "../components/PageFrame";
import { PoolBacktestManager } from "./PoolBacktestManager";

export function PoolBacktestPageClient() {
  return (
    <PageFrame title="백테스트">
      <PoolBacktestManager />
    </PageFrame>
  );
}
