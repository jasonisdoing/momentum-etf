"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { BUCKET_COLORS, BUCKET_NAME_MAP } from "@/lib/bucket-theme";
import type { WeeklyRow, WeeklyTableData } from "@/lib/weekly-store";
import { AppLoadingState } from "../components/AppLoadingState";
import { ResponsiveFiltersSection } from "../components/ResponsiveFiltersSection";
import type { AssetChartsHeaderSummary } from "./AssetChartsPageClient";

type RangeKey = "1m" | "3m" | "6m" | "12m" | "all";

type ChartRow = {
  week_date: string;
  label: string;
  total_assets: number;
  // 누적 인출을 안 했다고 가정한 총자산 (= total_assets + 누적 인출)
  total_assets_if_no_withdraw: number;
  total_principal: number;
  bucket_1: number;
  bucket_2: number;
  bucket_3: number;
  bucket_4: number;
  bucket_5: number;
};

const RANGE_OPTIONS: Array<{ key: RangeKey; label: string; weeks: number | null }> = [
  { key: "1m", label: "1개월", weeks: 5 },
  { key: "3m", label: "3개월", weeks: 13 },
  { key: "6m", label: "6개월", weeks: 26 },
  { key: "12m", label: "12개월", weeks: 52 },
  { key: "all", label: "전체", weeks: null },
];

const BUCKET_KEYS = ["bucket_1", "bucket_2", "bucket_3", "bucket_4", "bucket_5"] as const;

function toNumber(value: unknown): number {
  const numeric = Number(value ?? 0);
  return Number.isFinite(numeric) ? numeric : 0;
}

function formatMoney(value: number): string {
  return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
}

function formatCompactMoney(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1_0000_0000) {
    return `${new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 0 }).format(value / 1_0000_0000)}억원`;
  }
  if (abs >= 1_0000) {
    return `${new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 0 }).format(value / 1_0000)}만원`;
  }
  return `${new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 0 }).format(value)}원`;
}

function formatDateLabel(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("ko-KR", { month: "numeric", day: "numeric" }).format(date);
}

function formatFullDateLabel(value: string): string {
  const [year, month, day] = value.split("-").map((part) => Number(part));
  if (!year || !month || !day) return value;
  return `${year}년 ${month}월 ${day}일`;
}

function formatMonthAxisLabel(value: string): string {
  const [year, month] = value.split("-").map((part) => Number(part));
  if (!year || !month) return value;
  return `${year}.${String(month).padStart(2, "0")}`;
}

function formatVisiblePeriod(rows: ChartRow[]): string {
  const first = rows[0];
  const last = rows[rows.length - 1];
  if (!first || !last) return "-";
  return `${formatFullDateLabel(first.week_date)} ~ ${formatFullDateLabel(last.week_date)}`;
}

function buildChartRows(rows: WeeklyRow[]): ChartRow[] {
  // 시간순 정렬 후 total_expense 누적합을 추적한다.
  const sorted = [...rows].sort((a, b) => a.week_date.localeCompare(b.week_date));
  let runningExpense = 0;
  return sorted.map((row) => {
    const totalAssets = toNumber(row.total_assets);
    const totalPrincipal = toNumber(row.total_principal);
    runningExpense += toNumber(row.total_expense);
    // 인출 미반영 가정 총자산 = 총자산 - 누적 지출
    // (지출이 음수로 기록되는 컨벤션이라 빼면 더해진다)
    const cashAmount = totalAssets * (toNumber(row.bucket_pct_cash) / 100);
    return {
      week_date: row.week_date,
      label: formatDateLabel(row.week_date),
      total_assets: totalAssets,
      total_assets_if_no_withdraw: totalAssets - runningExpense,
      total_principal: totalPrincipal,
      bucket_1: totalAssets * (toNumber(row.bucket_pct_momentum) / 100),
      bucket_2: totalAssets * (toNumber(row.bucket_pct_market) / 100),
      bucket_3: totalAssets * (toNumber(row.bucket_pct_dividend) / 100),
      bucket_4: totalAssets * (toNumber(row.bucket_pct_alternative) / 100),
      bucket_5: cashAmount,
    };
  });
}

function filterRowsByRange(rows: ChartRow[], rangeKey: RangeKey): ChartRow[] {
  const option = RANGE_OPTIONS.find((item) => item.key === rangeKey);
  if (!option?.weeks || rows.length <= option.weeks) return rows;
  return rows.slice(-option.weeks);
}

function getLatestSummary(rows: ChartRow[]): AssetChartsHeaderSummary {
  const latest = rows[rows.length - 1];
  const previous = rows[rows.length - 2];
  if (!latest) {
    return {
      latestWeekDate: "-",
      rowCount: 0,
      latestTotalAssets: null,
      totalAssetsDelta: null,
      totalAssetsDeltaPct: null,
    };
  }

  const delta = previous ? latest.total_assets - previous.total_assets : null;
  const deltaPct = previous && previous.total_assets !== 0 ? (delta! / previous.total_assets) * 100 : null;
  return {
    latestWeekDate: latest.label,
    rowCount: rows.length,
    latestTotalAssets: latest.total_assets,
    totalAssetsDelta: delta,
    totalAssetsDeltaPct: deltaPct,
  };
}

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; color?: string }>; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="assetChartsTooltip">
      <div className="assetChartsTooltipTitle">{label}</div>
      {payload.map((item) => (
        <div key={item.name} className="assetChartsTooltipRow">
          <span className="assetChartsTooltipLabel" style={{ color: item.color }}>{item.name}</span>
          <strong>{formatMoney(toNumber(item.value))}</strong>
        </div>
      ))}
    </div>
  );
}

export function AssetChartsManager({
  onHeaderSummaryChange,
}: {
  onHeaderSummaryChange?: (summary: AssetChartsHeaderSummary) => void;
}) {
  const [rows, setRows] = useState<WeeklyRow[]>([]);
  const [rangeKey, setRangeKey] = useState<RangeKey>("all");
  const [showAmounts, setShowAmounts] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch("/api/weekly", { cache: "no-store" });
        const payload = (await response.json()) as WeeklyTableData & { error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "주별 데이터를 불러오지 못했습니다.");
        }
        if (!cancelled) {
          setRows(payload.rows ?? []);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "주별 데이터를 불러오지 못했습니다.");
          setRows([]);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const chartRows = useMemo(() => buildChartRows(rows), [rows]);
  const visibleRows = useMemo(() => filterRowsByRange(chartRows, rangeKey), [chartRows, rangeKey]);
  const visiblePeriod = useMemo(() => formatVisiblePeriod(visibleRows), [visibleRows]);

  useEffect(() => {
    onHeaderSummaryChange?.(getLatestSummary(chartRows));
  }, [chartRows, onHeaderSummaryChange]);

  const latest = chartRows[chartRows.length - 1];

  return (
    <div className="appPageStack appPageStackFill assetChartsPage">
      {error ? <div className="bannerError">{error}</div> : null}

      <section className="appSection">
        <div className="card appCard">
          <div className="card-header">
            <ResponsiveFiltersSection>
              <div className="appMainHeader">
                <div className="appMainHeaderLeft">
                  <label className="appLabeledField">
                    <span className="appLabeledFieldLabel">기간</span>
                    <div className="appSegmentedToggle" role="group" aria-label="자산 차트 기간">
                      {RANGE_OPTIONS.map((option) => (
                        <button
                          key={option.key}
                          type="button"
                          className={rangeKey === option.key ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                          onClick={() => setRangeKey(option.key)}
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                  </label>
                </div>
                <div className="appMainHeaderRight">
                  <button
                    type="button"
                    className={`btn btn-sm shadow-sm ${showAmounts ? "btn-outline-secondary" : "btn-dark"}`}
                    onClick={() => setShowAmounts((previous) => !previous)}
                  >
                    {showAmounts ? "금액 가리기" : "금액 보기"}
                  </button>
                  <div className="appHeaderMetrics">
                    <div className="appHeaderMetric">
                      <span>기간:</span>
                      <span className="appHeaderMetricValue">{visiblePeriod}</span>
                    </div>
                    <div className="appHeaderMetric">
                      <span>최신 총자산:</span>
                      <span className="appHeaderMetricValue">{showAmounts && latest ? formatMoney(latest.total_assets) : "-"}</span>
                    </div>
                  </div>
                </div>
              </div>
            </ResponsiveFiltersSection>
          </div>
        </div>
      </section>

      <section className="appSection appSectionFill">
        <div className="assetChartsGrid">
          <div className="card appCard assetChartsCard">
            <div className="assetChartsCardHeader">
              <div>
                <h2>버킷별 자산</h2>
                <p>주별 총자산과 버킷 비중 기준</p>
              </div>
            </div>
            <div className="assetChartsBody">
              {loading ? (
                <AppLoadingState label="자산 차트를 불러오는 중입니다." />
              ) : (
                <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={400}>
                  <BarChart data={visibleRows} margin={{ top: 12, right: 18, bottom: 8, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="week_date" tickFormatter={formatMonthAxisLabel} minTickGap={18} tick={{ fontSize: 12 }} />
                    <YAxis hide={!showAmounts} tickFormatter={formatCompactMoney} width={showAmounts ? 88 : 0} tick={{ fontSize: 12 }} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend />
                    {BUCKET_KEYS.map((key, index) => (
                      <Bar
                        key={key}
                        dataKey={key}
                        name={BUCKET_NAME_MAP[index + 1]}
                        stackId="assets"
                        fill={BUCKET_COLORS[index]}
                        radius={index === BUCKET_KEYS.length - 1 ? [3, 3, 0, 0] : [0, 0, 0, 0]}
                      />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          <div className="card appCard assetChartsCard">
            <div className="assetChartsCardHeader">
              <div>
                <h2>총자산(인출 미반영) / 총자산 / 원금</h2>
                <p>주별 총자산·원금 흐름과 인출 안 했다면 총자산</p>
              </div>
            </div>
            <div className="assetChartsBody">
              {loading ? (
                <AppLoadingState label="자산 차트를 불러오는 중입니다." />
              ) : (
                <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={400}>
                  <AreaChart data={visibleRows} margin={{ top: 12, right: 18, bottom: 8, left: 8 }}>
                    <defs>
                      <linearGradient id="assetChartTotal" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#16a34a" stopOpacity={0.35} />
                        <stop offset="95%" stopColor="#16a34a" stopOpacity={0.04} />
                      </linearGradient>
                      <linearGradient id="assetChartPrincipal" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#7c3aed" stopOpacity={0.32} />
                        <stop offset="95%" stopColor="#7c3aed" stopOpacity={0.04} />
                      </linearGradient>
                      <linearGradient id="assetChartTotalNoWithdraw" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#2563eb" stopOpacity={0.32} />
                        <stop offset="95%" stopColor="#2563eb" stopOpacity={0.04} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="week_date" tickFormatter={formatMonthAxisLabel} minTickGap={18} tick={{ fontSize: 12 }} />
                    <YAxis hide={!showAmounts} tickFormatter={formatCompactMoney} width={showAmounts ? 88 : 0} tick={{ fontSize: 12 }} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="total_assets_if_no_withdraw"
                      name="총 자산(인출 미반영)"
                      stroke="#2563eb"
                      fill="url(#assetChartTotalNoWithdraw)"
                      strokeWidth={2}
                    />
                    <Area
                      type="monotone"
                      dataKey="total_principal"
                      name="총 원금"
                      stroke="#7c3aed"
                      fill="url(#assetChartPrincipal)"
                      strokeWidth={2}
                    />
                    <Area
                      type="monotone"
                      dataKey="total_assets"
                      name="총 자산"
                      stroke="#16a34a"
                      fill="url(#assetChartTotal)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
