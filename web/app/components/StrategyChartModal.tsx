"use client";

import { useLayoutEffect, useRef, useState, type ComponentProps } from "react";

import { AppModal } from "./AppModal";
import { StrategyHoldingCharts } from "./StrategyHoldingCharts";

type ChartProps = ComponentProps<typeof StrategyHoldingCharts>;

// 보유 종목 수 선택지의 최대가 10 이라 4열이면 10종목이 4+4+2 로 들어간다.
const MAX_COLUMNS = 4;

function chartLayout(width: number, height: number, count: number) {
  if (count === 0 || width === 0 || height === 0) return { columns: 1, chartHeight: 240 };
  const desired = Math.ceil(Math.sqrt(count * width / height));
  const widest = Math.max(1, Math.min(count, MAX_COLUMNS, desired, Math.floor((width + 20) / 420)));
  // 행 수가 정해지면 열을 그 행에 맞게 줄인다 — 5개를 4+1 로 두지 않고 3+2 로 고르게 놓는다.
  const rows = Math.ceil(count / widest);
  const columns = Math.ceil(count / rows);
  const chartHeight = Math.max(280, Math.min(650, Math.floor((height - 44 - 18 * (rows - 1)) / rows - 112)));
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

  const layout = chartLayout(size.width, size.height, charts?.length ?? 0);
  return (
    <AppModal open={open} onClose={onClose} title={title} size="full">
      <div ref={bodyRef} className="strategyChartModalBody">
        <StrategyHoldingCharts charts={charts} chartHeight={layout.chartHeight} columns={layout.columns} stackedHeader {...chartProps} />
      </div>
    </AppModal>
  );
}
