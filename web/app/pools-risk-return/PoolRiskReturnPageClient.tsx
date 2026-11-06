"use client";

import { PageFrame } from "../components/PageFrame";
import { PoolRiskReturnManager } from "./PoolRiskReturnManager";

export function PoolRiskReturnPageClient() {
  return (
    <PageFrame title="위험·수익" fullHeight fullWidth>
      <PoolRiskReturnManager />
    </PageFrame>
  );
}
