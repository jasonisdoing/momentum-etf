"use client";

import type { ComponentProps } from "react";

import { AppModal } from "./AppModal";
import { StrategyHoldingCharts } from "./StrategyHoldingCharts";

type ChartProps = ComponentProps<typeof StrategyHoldingCharts>;

export function StrategyChartModal({
  open,
  onClose,
  title,
  charts,
  ...chartProps
}: ChartProps & { open: boolean; onClose: () => void; title: string }) {
  return (
    <AppModal open={open} onClose={onClose} title={title} subtitle="보유·진입 예정 종목 최대 10개" size="full">
      <div className="strategyChartModalBody">
        <StrategyHoldingCharts charts={charts?.slice(0, 10) ?? null} chartHeight={240} {...chartProps} />
      </div>
    </AppModal>
  );
}
