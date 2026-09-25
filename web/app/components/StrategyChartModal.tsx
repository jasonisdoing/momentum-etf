"use client";

import { useLayoutEffect, useRef, useState, type ComponentProps } from "react";

import { AppModal } from "./AppModal";
import { StrategyHoldingCharts } from "./StrategyHoldingCharts";

type ChartProps = ComponentProps<typeof StrategyHoldingCharts>;

function chartLayout(width: number, height: number, count: number) {
  if (count === 0 || width === 0 || height === 0) return { columns: 1, chartHeight: 240 };
  const desired = Math.ceil(Math.sqrt(count * width / height));
  const columns = Math.max(1, Math.min(count, 5, desired, Math.floor((width + 20) / 320)));
  const rows = Math.ceil(count / columns);
  const chartHeight = Math.max(110, Math.floor((height - 44 - 18 * (rows - 1)) / rows - 76));
  return { columns, chartHeight };
}

export function StrategyChartModal({
  open,
  onClose,
  title,
  charts,
  ...chartProps
}: ChartProps & { open: boolean; onClose: () => void; title: string }) {
  const bodyRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    if (!open || !bodyRef.current) return;
    const body = bodyRef.current;
    const resize = () => setSize({ width: body.clientWidth, height: body.clientHeight });
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(body);
    return () => observer.disconnect();
  }, [open]);

  const visibleCharts = charts?.slice(0, 10) ?? null;
  const layout = chartLayout(size.width, size.height, visibleCharts?.length ?? 0);
  return (
    <AppModal open={open} onClose={onClose} title={title} size="full">
      <div ref={bodyRef} className="strategyChartModalBody">
        <StrategyHoldingCharts charts={visibleCharts} chartHeight={layout.chartHeight} columns={layout.columns} {...chartProps} />
      </div>
    </AppModal>
  );
}
