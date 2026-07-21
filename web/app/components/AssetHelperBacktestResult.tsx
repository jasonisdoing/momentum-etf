"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ColDef, GridOptions } from "ag-grid-community";
import { ColorType, LineSeries, createChart, createSeriesMarkers } from "lightweight-charts";
import type { Time } from "lightweight-charts";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { BUCKET_COLORS, BUCKET_THEME } from "@/lib/bucket-theme";
import { AppAgGrid } from "./AppAgGrid";
import { TickerDetailLink } from "./TickerDetailLink";
import { createAppGridTheme } from "./app-grid-theme";

// 자산 헬퍼 백테스트 결과 렌더링(차트+비중변화+요약+종목별 성과)을 공용화한 것.
// 자산 헬퍼 등 여러 화면이 동일한 백테스트 결과 UI 를 재사용한다(AGENTS.md 3-1).

type LabSummary = { total_return_pct: number; cagr_pct: number; mdd_pct: number; sortino: number };

type LabTicker = { ticker: string; name?: string; bucket?: number };

export type LabPosition = LabTicker & {
  buy_date: string;
  late_entry: boolean;
  shares: number;
  buy_price: number | null;
  last_price: number | null;
  return_pct: number | null;
  mdd_pct: number | null;
  mdd_start: string;
  mdd_end: string;
  sortino: number | null;
  profit: number;
  profit_contribution_pct?: number | null;
  value: number;
  min_weight?: number;
  max_weight?: number;
};

type AssetHelperWeightHistoryRow = { date: string; [key: string]: string | number };
type AssetHelperWeightItem = { key: string; label: string; bucket?: number };
type AssetHelperWeightHoverDetail = { date: string; items: Array<AssetHelperWeightItem & { weight: number; color: string }> };

export type LabResult = {
  months: number;
  rebalance?: string;
  buy_date: string;
  end_date: string;
  has_late_entry?: boolean;
  initial_capital: number;
  final_value: number;
  slippage?: { total_cost: number; total_cost_pct: number };
  summary: LabSummary;
  benchmark: LabTicker & { summary: LabSummary };
  positions: LabPosition[];
  cash_min_weight?: number;
  cash_max_weight?: number;
  chart: {
    dates: string[];
    portfolio_value?: number[];
    benchmark_value?: number[];
    portfolio_pct: number[];
    benchmark_pct: number[];
  };
  weight_history?: AssetHelperWeightHistoryRow[];
  weight_items?: AssetHelperWeightItem[];
  error?: string;
  detail?: string;
};

const previewGridTheme = createAppGridTheme();

const REBALANCE_LABELS: Record<string, string> = {
  none: "리밸런싱 없음 (보유)",
  weekly: "매주 (금요일)",
  monthly: "매월 (말일)",
  quarterly: "분기 (분기말)",
  yearly: "매년 (연말)",
};

function rebalanceLabel(value?: string): string {
  return REBALANCE_LABELS[value ?? "none"] ?? REBALANCE_LABELS.none;
}

function formatKrw(value: number): string {
  return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
}

function formatCompactKrw(value: number): string {
  const absValue = Math.abs(value);
  if (absValue >= 100_000_000) {
    return `${new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 1 }).format(value / 100_000_000)}억`;
  }
  if (absValue >= 10_000) {
    return `${new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 0 }).format(value / 10_000)}만`;
  }
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 0 }).format(value);
}

function formatReturnPct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "-";
  return `${value.toFixed(2)}%`;
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
}

function signedColor(value: number): string {
  if (value > 0) return "#d63939";
  if (value < 0) return "#206bc4";
  return "#475569";
}

function getBucketCellClass(bucketId: number | undefined): string {
  return bucketId ? `rankBucketCell rankBucketCell${bucketId}` : "rankBucketCell";
}

function getBucketName(bucketId: number | undefined): string {
  return bucketId ? BUCKET_THEME[String(bucketId)]?.name ?? "-" : "-";
}

function getAssetHelperWeightColor(items: AssetHelperWeightItem[] | undefined, key: string): string {
  if (key === "__CASH__") return BUCKET_COLORS[4];
  const bucket = items?.find((item) => item.key === key)?.bucket;
  return bucket && BUCKET_COLORS[bucket - 1] ? BUCKET_COLORS[bucket - 1] : "#64748b";
}

function formatMonthAxisLabel(value: string): string {
  const [year, month] = value.split("-").map((part) => Number(part));
  if (!year || !month) return value;
  return `${year}.${String(month).padStart(2, "0")}`;
}

function LabChart({ result }: { result: LabResult }) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const chart = createChart(container, {
      height: 320,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#5b6675" },
      grid: { vertLines: { color: "rgba(148,163,184,0.12)" }, horzLines: { color: "rgba(148,163,184,0.12)" } },
      rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.18, bottom: 0.15 } },
      timeScale: { borderVisible: false, rightOffset: 10 },
      localization: { priceFormatter: (price: number) => formatCompactKrw(price) },
      autoSize: true,
    });
    const toLine = (values: number[]) => result.chart.dates.map((date, index) => ({ time: date as Time, value: values[index] }));
    const portfolioValues = result.chart.portfolio_value ?? result.chart.portfolio_pct;
    const benchmarkValues = result.chart.benchmark_value ?? result.chart.benchmark_pct;
    const portfolioSeries = chart.addSeries(LineSeries, { color: "#2563eb", lineWidth: 2, lastValueVisible: false, priceLineVisible: false });
    const benchmarkSeries = chart.addSeries(LineSeries, { color: "#94a3b8", lineWidth: 1, lastValueVisible: false, priceLineVisible: false });
    portfolioSeries.setData(toLine(portfolioValues));
    benchmarkSeries.setData(toLine(benchmarkValues));

    const peakMarker = (values: number[], color: string, position: "aboveBar" | "belowBar") => {
      const peakIndex = values.reduce(
        (bestIndex, value, index) => (Number.isFinite(value) && value > values[bestIndex] ? index : bestIndex),
        0,
      );
      const peakValue = values[peakIndex];
      const finalValue = values[values.length - 1];
      const drawdown = peakValue > 0 ? (finalValue / peakValue - 1) * 100 : 0;
      const date = result.chart.dates[peakIndex];
      const displayDate = date ? date.replaceAll("-", ".") : "-";
      return {
        time: date as Time,
        position,
        color,
        shape: position === "aboveBar" ? ("arrowDown" as const) : ("arrowUp" as const),
        text: `${displayDate}(${drawdown.toFixed(2)}%)`,
        size: 1,
      };
    };

    if (portfolioValues.length > 0) createSeriesMarkers(portfolioSeries, [peakMarker(portfolioValues, "#2563eb", "aboveBar")]);
    if (benchmarkValues.length > 0) createSeriesMarkers(benchmarkSeries, [peakMarker(benchmarkValues, "#64748b", "belowBar")]);
    chart.timeScale().fitContent();

    return () => chart.remove();
  }, [result]);

  return <div ref={containerRef} style={{ width: "100%", height: 320 }} />;
}

function AssetHelperWeightHistoryChart({
  rows,
  items,
  onHoverChange,
}: {
  rows: AssetHelperWeightHistoryRow[];
  items: AssetHelperWeightItem[];
  onHoverChange: (detail: AssetHelperWeightHoverDetail | null) => void;
}) {
  if (!rows.length || !items.length) {
    return <div style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>비중 이력이 없습니다.</div>;
  }

  const sortedItems = [...items].sort((a, b) => {
    const aBucket = a.key === "__CASH__" ? 5 : a.bucket ?? 0;
    const bBucket = b.key === "__CASH__" ? 5 : b.bucket ?? 0;
    return bBucket - aBucket;
  });
  const showWeightsForDate = (date: string) => {
    const activeRow = rows.find((row) => String(row.date) === date);
    if (!activeRow) return;
    const total = items.reduce((sum, item) => sum + Number(activeRow[item.key] ?? 0), 0);
    const hoverItems = items
      .map((item) => ({
        ...item,
        weight: total > 0 ? (Number(activeRow[item.key] ?? 0) / total) * 100 : 0,
        color: getAssetHelperWeightColor(items, item.key),
      }))
      .filter((item) => item.weight > 0);
    onHoverChange({ date, items: hoverItems });
  };

  return (
    <div className="assetHelperWeightChartWrap">
      <div className="assetHelperWeightChartCanvas">
        <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={320}>
          <BarChart
            data={rows}
            margin={{ top: 10, right: 12, bottom: 6, left: 0 }}
            onMouseMove={(state) => {
              if (state?.activeLabel != null) showWeightsForDate(String(state.activeLabel));
            }}
            onMouseLeave={() => onHoverChange(null)}
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="date" tickFormatter={formatMonthAxisLabel} minTickGap={18} tick={{ fontSize: 12 }} />
            <YAxis tickFormatter={(value) => formatCompactKrw(Number(value))} width={52} tick={{ fontSize: 12 }} />
            <Tooltip content={() => null} />
            {sortedItems.map((item, index) => (
              <Bar
                key={item.key}
                dataKey={item.key}
                name={item.label}
                stackId="weight"
                fill={getAssetHelperWeightColor(items, item.key)}
                radius={index === sortedItems.length - 1 ? [3, 3, 0, 0] : [0, 0, 0, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function AssetHelperBacktestResult({ result }: { result: LabResult }) {
  const [weightHoverDetail, setWeightHoverDetail] = useState<AssetHelperWeightHoverDetail | null>(null);

  const columns = useMemo<ColDef<LabPosition>[]>(
    () => [
      {
        field: "bucket",
        headerName: "버킷",
        width: 108,
        minWidth: 108,
        valueGetter: (params) => getBucketName(params.data?.bucket),
        cellClass: (params) => getBucketCellClass(params.data?.bucket),
      },
      {
        field: "ticker",
        headerName: "티커",
        width: 95,
        minWidth: 95,
        cellRenderer: (params: { data?: LabPosition }) => {
          const row = params.data;
          if (!row) return "-";
          if (row.ticker === "__CASH__") return <span style={{ fontWeight: 800 }}>현금</span>;
          return <TickerDetailLink ticker={row.ticker} displayTicker={row.ticker} />;
        },
      },
      {
        field: "name",
        headerName: "종목명",
        minWidth: 180,
        flex: 1,
        cellClass: "assetHelperNameCell",
        valueGetter: (params) => {
          const row = params.data;
          if (!row) return "-";
          return row.ticker === "__CASH__" ? "" : row.name ?? row.ticker;
        },
        tooltipValueGetter: (params) => String(params.value ?? ""),
        cellRenderer: (params: { value: string | null | undefined }) => {
          const name = params.value || "-";
          return (
            <span className="assetHelperNameCellText" title={name}>
              {name}
            </span>
          );
        },
        cellStyle: (params) => ({
          color: getAssetHelperWeightColor(result.weight_items, params.data?.ticker ?? ""),
          fontWeight: 800,
        }),
      },
      {
        field: "buy_date",
        headerName: "매수일",
        width: 110,
        cellRenderer: (params: { data?: LabPosition; value?: string }) =>
          `${params.value ?? "-"}${params.data?.late_entry ? " ↩" : ""}`,
      },
      {
        field: "buy_price",
        headerName: "초기가",
        width: 105,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) =>
          params.value == null ? "-" : new Intl.NumberFormat("ko-KR").format(params.value),
      },
      {
        field: "last_price",
        headerName: "현재가",
        width: 105,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) =>
          params.value == null ? "-" : new Intl.NumberFormat("ko-KR").format(params.value),
      },
      {
        field: "return_pct",
        headerName: `수익률(${result.months ?? "-"}개월)`,
        width: 150,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) => (
          <span style={{ color: params.value == null ? "#475569" : signedColor(params.value) }}>{formatReturnPct(params.value)}</span>
        ),
      },
      {
        field: "mdd_pct",
        headerName: `MDD(${result.months ?? "-"}개월)`,
        width: 150,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) =>
          params.value == null ? "-" : <span style={{ color: signedColor(params.value) }}>{formatReturnPct(params.value)}</span>,
      },
      {
        field: "sortino",
        headerName: `Sortino(${result.months ?? "-"}개월)`,
        width: 150,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatNumber(params.value),
      },
      {
        field: "profit",
        headerName: "수익금",
        width: 130,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) =>
          params.value == null ? "-" : <span style={{ color: signedColor(params.value) }}>{formatKrw(params.value)}</span>,
      },
      {
        field: "profit_contribution_pct",
        headerName: "수익기여",
        width: 100,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) =>
          params.value == null ? "-" : <span style={{ color: signedColor(params.value) }}>{params.value.toFixed(1)}%</span>,
      },
      {
        field: "min_weight",
        headerName: "최저 비중",
        width: 100,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => (params.value == null ? "-" : `${formatNumber(params.value, 1)}%`),
      },
      {
        field: "max_weight",
        headerName: "최대 비중",
        width: 100,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => (params.value == null ? "-" : `${formatNumber(params.value, 1)}%`),
      },
    ],
    [result.months, result.weight_items],
  );

  const gridOptions = useMemo<GridOptions<LabPosition>>(() => ({ domLayout: "autoHeight", suppressMovableColumns: true }), []);

  const positionRows = useMemo<LabPosition[]>(() => {
    const lastWeightRow = result.weight_history?.[result.weight_history.length - 1];
    const cashValue = Number(lastWeightRow?.__CASH__ ?? 0);
    const totalProfit = result.final_value - result.initial_capital;
    const withContribution = (position: LabPosition): LabPosition => ({
      ...position,
      profit_contribution_pct: totalProfit === 0 ? null : (position.profit / totalProfit) * 100,
    });
    return [
      withContribution({
        ticker: "__CASH__",
        name: "현금",
        bucket: 5,
        buy_date: result.buy_date,
        late_entry: false,
        shares: 0,
        buy_price: null,
        last_price: null,
        return_pct: null,
        mdd_pct: null,
        mdd_start: "",
        mdd_end: "",
        sortino: null,
        profit: 0,
        value: Number.isFinite(cashValue) ? cashValue : 0,
        min_weight: result.cash_min_weight ?? 0,
        max_weight: result.cash_max_weight ?? 0,
      }),
      ...result.positions.map(withContribution),
    ];
  }, [result]);

  const summaryChip = (label: string, value: string, color?: string) => (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ fontSize: "0.76rem", color: "var(--text-muted)" }}>{label}</span>
      <strong style={{ fontSize: "0.98rem", color: color ?? "var(--text-normal)" }}>{value}</strong>
    </div>
  );

  return (
    <div className="assetHelperBacktestResultLayout">
      <div className="assetHelperBacktestTopLayout">
        <div className="assetHelperBacktestResultPanel">
          {weightHoverDetail ? (
            <div className="assetHelperWeightHoverOverlay">
              <div className="assetHelperWeightHoverDate">{weightHoverDetail.date}</div>
              <div className="assetHelperWeightHoverTitle">종목별 비중</div>
              <div className="assetHelperWeightHoverRows">
                {weightHoverDetail.items.map((item) => (
                  <div key={item.key} className="assetHelperWeightHoverRow">
                    <span style={{ color: item.color }}>{item.label}</span>
                    <strong>{item.weight.toFixed(1)}%</strong>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
          <h3 style={{ fontSize: "0.98rem", fontWeight: 800, marginBottom: 4 }}>백테스트 결과</h3>
          <p style={{ color: "var(--text-muted)", fontSize: "0.82rem", marginBottom: 4 }}>
            {result.buy_date} ~ {result.end_date} ({result.months}개월)
          </p>
          <p style={{ color: "var(--text-muted)", fontSize: "0.82rem", marginBottom: 12 }}>
            초기 {formatKrw(result.initial_capital)} → 최종 {formatKrw(result.final_value)} · 리밸런싱: {rebalanceLabel(result.rebalance)}
            {result.slippage ? (
              <>
                {" "}· 총 슬리피지 {formatKrw(result.slippage.total_cost)} (초기 대비 {result.slippage.total_cost_pct.toFixed(2)}%)
              </>
            ) : null}
          </p>
          <div className="assetHelperSummaryCompare">
            <div className="assetHelperSummaryGroup assetHelperSummaryPortfolio">
              <div className="assetHelperSummaryTitle">
                <span className="assetHelperSummaryLine" /> 포트폴리오
              </div>
              <div className="assetHelperSummaryMetrics">
                {summaryChip("총수익률", `${result.summary.total_return_pct.toFixed(2)}%`, signedColor(result.summary.total_return_pct))}
                {summaryChip("CAGR", `${result.summary.cagr_pct.toFixed(2)}%`, signedColor(result.summary.cagr_pct))}
                {summaryChip("MDD", `${result.summary.mdd_pct.toFixed(2)}%`, "#d63939")}
                {summaryChip("Sortino", result.summary.sortino.toFixed(2))}
              </div>
            </div>
            <div className="assetHelperSummaryGroup assetHelperSummaryBenchmark">
              <div className="assetHelperSummaryTitle">
                <span className="assetHelperSummaryLine" /> {result.benchmark.name}
              </div>
              <div className="assetHelperSummaryMetrics">
                {summaryChip("총수익률", `${result.benchmark.summary.total_return_pct.toFixed(2)}%`, signedColor(result.benchmark.summary.total_return_pct))}
                {summaryChip("CAGR", `${result.benchmark.summary.cagr_pct.toFixed(2)}%`, signedColor(result.benchmark.summary.cagr_pct))}
                {summaryChip("MDD", `${result.benchmark.summary.mdd_pct.toFixed(2)}%`, "#d63939")}
                {summaryChip("Sortino", result.benchmark.summary.sortino.toFixed(2))}
              </div>
            </div>
          </div>
          <LabChart result={result} />
        </div>
        <div className="assetHelperBacktestResultPanel">
          <h3 style={{ fontSize: "0.98rem", fontWeight: 800, marginBottom: 4 }}>비중 변화</h3>
          <p style={{ color: "var(--text-muted)", fontSize: "0.82rem", marginBottom: 10 }}>
            매주 금요일 기준 가격 변동이 반영된 종목별 평가금액
          </p>
          <AssetHelperWeightHistoryChart
            rows={result.weight_history ?? []}
            items={result.weight_items ?? []}
            onHoverChange={setWeightHoverDetail}
          />
        </div>
      </div>
      <div className="assetHelperBacktestPerformancePanel">
        <h3 style={{ fontSize: "0.98rem", fontWeight: 800, marginBottom: 8 }}>백테스트 종목별 성과</h3>
        {result.has_late_entry ? (
          <p style={{ color: "#b45309", background: "rgba(245,158,11,0.08)", fontSize: "0.8rem", padding: "6px 10px", borderRadius: 6, marginBottom: 10 }}>
            실험 시작 이후 상장된 종목은 배정 예산을 현금으로 대기시켰다가 상장일 종가에 편입합니다.
          </p>
        ) : null}
        <AppAgGrid<LabPosition>
          rowData={positionRows}
          columnDefs={columns}
          className="assetHelperPreviewGrid rankAgGrid"
          theme={previewGridTheme}
          getRowId={(params) => params.data.ticker}
          gridOptions={gridOptions}
        />
      </div>
      <style jsx global>{`
        .assetHelperBacktestResultLayout {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .assetHelperBacktestTopLayout {
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
          gap: 12px;
          align-items: stretch;
        }
        .assetHelperBacktestResultPanel {
          position: relative;
          display: flex;
          flex-direction: column;
          min-width: 0;
        }
        .assetHelperWeightHoverOverlay {
          position: absolute;
          inset: 0;
          z-index: 5;
          display: flex;
          flex-direction: column;
          padding: 18px 20px;
          border: 1px solid rgba(148, 163, 184, 0.42);
          border-radius: 10px;
          background: rgba(15, 23, 42, 0.96);
          box-shadow: 0 18px 40px rgba(15, 23, 42, 0.22);
          color: #f8fafc;
          pointer-events: none;
        }
        .assetHelperWeightHoverDate {
          font-size: 1.05rem;
          font-weight: 900;
        }
        .assetHelperWeightHoverTitle {
          margin-top: 3px;
          color: #94a3b8;
          font-size: 0.8rem;
          font-weight: 700;
        }
        .assetHelperWeightHoverRows {
          display: grid;
          grid-template-columns: minmax(0, 1fr);
          gap: 6px;
          margin-top: 12px;
        }
        .assetHelperWeightHoverRow {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          min-width: 0;
          font-size: 0.86rem;
        }
        .assetHelperWeightHoverRow span {
          overflow: hidden;
          font-weight: 800;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .assetHelperWeightHoverRow strong {
          flex: 0 0 auto;
          color: #f8fafc;
        }
        .assetHelperSummaryCompare {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px;
          margin-bottom: 10px;
        }
        .assetHelperSummaryGroup {
          min-width: 0;
          padding: 7px 10px;
          border: 1px solid rgba(148, 163, 184, 0.22);
          border-radius: 9px;
          background: rgba(248, 250, 252, 0.62);
        }
        .assetHelperSummaryPortfolio {
          border-top: 3px solid #2563eb;
        }
        .assetHelperSummaryBenchmark {
          border-top: 3px solid #94a3b8;
        }
        .assetHelperSummaryTitle {
          display: flex;
          align-items: center;
          gap: 7px;
          margin-bottom: 5px;
          color: #334155;
          font-size: 0.86rem;
          font-weight: 800;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .assetHelperSummaryLine {
          width: 18px;
          height: 3px;
          flex: 0 0 auto;
          border-radius: 2px;
          background: #2563eb;
        }
        .assetHelperSummaryBenchmark .assetHelperSummaryLine {
          background: #94a3b8;
        }
        .assetHelperSummaryMetrics {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 4px 12px;
        }
        .assetHelperBacktestPerformancePanel {
          min-width: 0;
          width: 100%;
        }
        .assetHelperWeightChartWrap {
          display: flex;
          flex-direction: column;
          flex: 1;
          min-width: 0;
          min-height: 380px;
          width: 100%;
        }
        .assetHelperWeightChartCanvas {
          flex: 1;
          min-height: 320px;
          min-width: 0;
        }
        .assetHelperPreviewGrid {
          height: auto !important;
        }
        .assetHelperPreviewGrid .appAgGridTheme {
          height: auto;
        }
        @media (max-width: 900px) {
          .assetHelperBacktestTopLayout {
            grid-template-columns: minmax(0, 1fr);
          }
          .assetHelperSummaryCompare {
            grid-template-columns: minmax(0, 1fr);
          }
        }
      `}</style>
    </div>
  );
}
