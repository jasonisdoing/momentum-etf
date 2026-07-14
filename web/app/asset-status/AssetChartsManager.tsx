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

type DailyRow = {
  date: string;
  daily_return_pct: number;
  [key: string]: any;
};

type ChartRow = {
  week_date: string;
  label: string;
  total_assets: number;
  // 누적 인출을 안 했다고 가정한 총자산 (= total_assets + 누적 인출)
  total_assets_if_no_withdraw: number;
  total_principal: number;
  // 입출금 영향을 뺀 주간 전략 수익률(%) · 누적 손익(원) — 성과 지표 계산용
  weekly_return_pct: number;
  cumulative_profit: number;
  bucket_1: number;
  bucket_2: number;
  bucket_3: number;
  bucket_4: number;
  bucket_5: number;
};

type PerfMetrics = {
  weeks: number;
  totalReturn: number | null; // 누적 수익률(소수, 0.1=10%)
  cagr: number | null; // 연평균 수익률(소수)
  mdd: number | null; // 최대 낙폭(음수 소수)
  mddPeriod: string | null; // 최대 낙폭 발생 기간
  vol: number | null; // 연 변동성(소수)
  bestWeek: number | null;
  bestWeekDate: string | null; // 최고 수익 주간 날짜
  worstWeek: number | null;
  worstWeekDate: string | null; // 최저 수익 주간 날짜
  winRate: number | null;
  profitAmount: number | null; // 구간 누적 손익(원)
  // 새로 추가할 지표들
  sharpeRatio: number | null;
  sortinoRatio: number | null;
  calmarRatio: number | null;
  // 주기별 극단치 지표
  bestMonth: number | null;
  bestMonthDate: string | null;
  worstMonth: number | null;
  worstMonthDate: string | null;
  bestQuarter: number | null;
  bestQuarterDate: string | null;
  worstQuarter: number | null;
  worstQuarterDate: string | null;
  bestYear: number | null;
  bestYearDate: string | null;
  worstYear: number | null;
  worstYearDate: string | null;
};

function getQuarterKey(dateStr: string): string {
  const year = dateStr.substring(0, 4);
  const month = Number(dateStr.substring(5, 7));
  const q = Math.ceil(month / 3);
  return `${year}-Q${q}`;
}

function computeMetrics(rows: ChartRow[]): PerfMetrics {
  const empty: PerfMetrics = {
    weeks: rows.length,
    totalReturn: null, cagr: null, mdd: null, mddPeriod: null, vol: null,
    bestWeek: null, bestWeekDate: null, worstWeek: null, worstWeekDate: null, winRate: null, profitAmount: null,
    sharpeRatio: null, sortinoRatio: null, calmarRatio: null,
    bestMonth: null, bestMonthDate: null, worstMonth: null, worstMonthDate: null,
    bestQuarter: null, bestQuarterDate: null, worstQuarter: null, worstQuarterDate: null,
    bestYear: null, bestYearDate: null, worstYear: null, worstYearDate: null,
  };
  if (rows.length < 2) return empty;

  // 구간 진입점(첫 행) 이후의 주간 수익률을 체인 → 구간 내 NAV 곡선
  const steps = rows.slice(1).map((r) => ({
    val: toNumber(r.weekly_return_pct) / 100,
    date: r.week_date,
  }));
  const nav = [1];
  for (const r of steps) nav.push(nav[nav.length - 1] * (1 + r.val));
  const totalReturn = nav[nav.length - 1] - 1;

  const first = rows[0];
  const last = rows[rows.length - 1];
  const days = (new Date(last.week_date).getTime() - new Date(first.week_date).getTime()) / 86_400_000;
  const years = days / 365.25;
  const cagr = years > 0 && totalReturn > -1 ? Math.pow(1 + totalReturn, 1 / years) - 1 : null;

  // MDD 및 해당 고점/저점 날짜 추적
  let peakVal = nav[0];
  let peakIndex = 0;
  let mdd = 0;
  let mddPeakIndex = 0;
  let mddTroughIndex = 0;

  for (let i = 0; i < nav.length; i++) {
    const v = nav[i];
    if (v > peakVal) {
      peakVal = v;
      peakIndex = i;
    }
    const dd = v / peakVal - 1;
    if (dd < mdd) {
      mdd = dd;
      mddPeakIndex = peakIndex;
      mddTroughIndex = i;
    }
  }

  const mddPeriod = mdd < 0 
    ? `${rows[mddPeakIndex].week_date} ~ ${rows[mddTroughIndex].week_date}`
    : null;

  // 최고/최저 주간 수익률 및 발생일 추적
  let bestVal = -Infinity;
  let bestDate = null;
  let worstVal = Infinity;
  let worstDate = null;

  for (const step of steps) {
    if (step.val > bestVal) {
      bestVal = step.val;
      bestDate = step.date;
    }
    if (step.val < worstVal) {
      worstVal = step.val;
      worstDate = step.date;
    }
  }

  const mean = steps.reduce((sum, step) => sum + step.val, 0) / steps.length;
  const variance = steps.reduce((sum, step) => sum + (step.val - mean) ** 2, 0) / steps.length;
  const vol = Math.sqrt(variance) * Math.sqrt(52);

  // 1. 샤프 지수 (Sharpe Ratio) - 무위험 이자율 연 3%(0.03) 가정
  const sharpeRatio = (cagr !== null && vol > 0) ? (cagr - 0.03) / vol : null;

  // 2. 소르티노 지수 (Sortino Ratio) - 하방 변동성 기준
  const downsideVariance = steps.reduce((sum, step) => sum + Math.min(0, step.val) ** 2, 0) / steps.length;
  const downsideVol = Math.sqrt(downsideVariance) * Math.sqrt(52);
  const sortinoRatio = (cagr !== null && downsideVol > 0) ? (cagr - 0.03) / downsideVol : null;

  // 3. 캘마 지수 (Calmar Ratio)
  const calmarRatio = (cagr !== null && mdd !== null && mdd < 0) ? cagr / Math.abs(mdd) : null;

  // 4. 월간 / 분기별 / 년간 극단 수익률 계산 (NAV 체인 기준 복리 리샘플링)
  // 4-1. 월간 극단치
  const monthEndIndices = new Map<string, number>();
  for (let i = 0; i < rows.length; i++) {
    const key = rows[i].week_date.substring(0, 7); // "YYYY-MM"
    monthEndIndices.set(key, i);
  }
  const months = Array.from(monthEndIndices.keys()).sort();
  let bestMonthVal = -Infinity;
  let bestMonthDateStr = null;
  let worstMonthVal = Infinity;
  let worstMonthDateStr = null;

  for (let i = 0; i < months.length; i++) {
    const currIdx = monthEndIndices.get(months[i])!;
    const prevIdx = i > 0 ? monthEndIndices.get(months[i - 1])! : 0;
    const monthReturn = (nav[currIdx] / nav[prevIdx]) - 1;

    if (monthReturn > bestMonthVal) {
      bestMonthVal = monthReturn;
      bestMonthDateStr = months[i];
    }
    if (monthReturn < worstMonthVal) {
      worstMonthVal = monthReturn;
      worstMonthDateStr = months[i];
    }
  }

  // 4-2. 분기별 극단치
  const quarterEndIndices = new Map<string, number>();
  for (let i = 0; i < rows.length; i++) {
    const key = getQuarterKey(rows[i].week_date);
    quarterEndIndices.set(key, i);
  }
  const quarters = Array.from(quarterEndIndices.keys()).sort();
  let bestQuarterVal = -Infinity;
  let bestQuarterDateStr = null;
  let worstQuarterVal = Infinity;
  let worstQuarterDateStr = null;

  for (let i = 0; i < quarters.length; i++) {
    const currIdx = quarterEndIndices.get(quarters[i])!;
    const prevIdx = i > 0 ? quarterEndIndices.get(quarters[i - 1])! : 0;
    const quarterReturn = (nav[currIdx] / nav[prevIdx]) - 1;

    if (quarterReturn > bestQuarterVal) {
      bestQuarterVal = quarterReturn;
      bestQuarterDateStr = quarters[i];
    }
    if (quarterReturn < worstQuarterVal) {
      worstQuarterVal = quarterReturn;
      worstQuarterDateStr = quarters[i];
    }
  }

  // 4-3. 년간 극단치
  const yearEndIndices = new Map<string, number>();
  for (let i = 0; i < rows.length; i++) {
    const key = rows[i].week_date.substring(0, 4); // "YYYY"
    yearEndIndices.set(key, i);
  }
  const yearsList = Array.from(yearEndIndices.keys()).sort();
  let bestYearVal = -Infinity;
  let bestYearDateStr = null;
  let worstYearVal = Infinity;
  let worstYearDateStr = null;

  for (let i = 0; i < yearsList.length; i++) {
    const currIdx = yearEndIndices.get(yearsList[i])!;
    const prevIdx = i > 0 ? yearEndIndices.get(yearsList[i - 1])! : 0;
    const yearReturn = (nav[currIdx] / nav[prevIdx]) - 1;

    if (yearReturn > bestYearVal) {
      bestYearVal = yearReturn;
      bestYearDateStr = yearsList[i];
    }
    if (yearReturn < worstYearVal) {
      worstYearVal = yearReturn;
      worstYearDateStr = yearsList[i];
    }
  }

  return {
    weeks: rows.length,
    totalReturn,
    cagr,
    mdd,
    mddPeriod,
    vol,
    bestWeek: bestVal !== -Infinity ? bestVal : null,
    bestWeekDate: bestDate,
    worstWeek: worstVal !== Infinity ? worstVal : null,
    worstWeekDate: worstDate,
    winRate: steps.filter((step) => step.val > 0).length / steps.length,
    profitAmount: toNumber(last.cumulative_profit) - toNumber(first.cumulative_profit),
    sharpeRatio,
    sortinoRatio,
    calmarRatio,
    bestMonth: bestMonthVal !== -Infinity ? bestMonthVal : null,
    bestMonthDate: bestMonthDateStr,
    worstMonth: worstMonthVal !== Infinity ? worstMonthVal : null,
    worstMonthDate: worstMonthDateStr,
    bestQuarter: bestQuarterVal !== -Infinity ? bestQuarterVal : null,
    bestQuarterDate: bestQuarterDateStr,
    worstQuarter: worstQuarterVal !== Infinity ? worstQuarterVal : null,
    worstQuarterDate: worstQuarterDateStr,
    bestYear: bestYearVal !== -Infinity ? bestYearVal : null,
    bestYearDate: bestYearDateStr,
    worstYear: worstYearVal !== Infinity ? worstYearVal : null,
    worstYearDate: worstYearDateStr,
  };
}

function formatPct(value: number | null, withSign = false): string {
  if (value == null || !Number.isFinite(value)) return "-";
  const pct = value * 100;
  const sign = withSign && pct > 0 ? "+" : "";
  return `${sign}${pct.toFixed(2)}%`;
}

const RANGE_OPTIONS: Array<{ key: RangeKey; label: string; weeks: number | null }> = [
  { key: "1m", label: "1개월", weeks: 5 },
  { key: "3m", label: "3개월", weeks: 13 },
  { key: "6m", label: "6개월", weeks: 26 },
  { key: "12m", label: "12개월", weeks: 52 },
  { key: "all", label: "전체", weeks: null },
];

const BUCKET_STACK_KEYS = ["bucket_5", "bucket_4", "bucket_3", "bucket_2", "bucket_1"] as const;

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
      weekly_return_pct: toNumber(row.weekly_return_pct),
      cumulative_profit: toNumber(row.cumulative_profit),
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
  const [dailyRows, setDailyRows] = useState<DailyRow[]>([]);
  const [monthlyRows, setMonthlyRows] = useState<any[]>([]);
  const [yearlyRows, setYearlyRows] = useState<any[]>([]);
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
        const [weeklyRes, dailyRes, monthlyRes, yearlyRes] = await Promise.all([
          fetch("/api/weekly", { cache: "no-store" }),
          fetch("/api/daily", { cache: "no-store" }),
          fetch("/api/monthly", { cache: "no-store" }),
          fetch("/api/yearly", { cache: "no-store" }),
        ]);

        if (!weeklyRes.ok || !dailyRes.ok || !monthlyRes.ok || !yearlyRes.ok) {
          throw new Error("일부 데이터를 불러오는 데 실패했습니다.");
        }

        const [weeklyData, dailyData, monthlyData, yearlyData] = await Promise.all([
          weeklyRes.json(),
          dailyRes.json(),
          monthlyRes.json(),
          yearlyRes.json(),
        ]);

        if (!cancelled) {
          setRows(weeklyData.rows ?? []);
          setDailyRows(dailyData.rows ?? []);
          setMonthlyRows(monthlyData.rows ?? []);
          setYearlyRows(yearlyData.rows ?? []);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "데이터를 불러오지 못했습니다.");
          setRows([]);
          setDailyRows([]);
          setMonthlyRows([]);
          setYearlyRows([]);
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
  const metrics = useMemo(() => computeMetrics(visibleRows), [visibleRows]);

  // 일/주/월/년 극단값 실시간 계산 useMemo
  const extremeReturns = useMemo(() => {
    const startDate = visibleRows.length > 0 ? visibleRows[0].week_date : null;
    const endDate = visibleRows.length > 0 ? visibleRows[visibleRows.length - 1].week_date : null;

    const startMonth = startDate ? startDate.substring(0, 7) : "";
    const endMonth = endDate ? endDate.substring(0, 7) : "";

    const startYear = startDate ? startDate.substring(0, 4) : "";
    const endYear = endDate ? endDate.substring(0, 4) : "";

    // 1. 일간 극단치 (2026년 4월 27일 이전에는 일별 데이터가 부재하므로 해당 일자 이후 데이터만 산출에 포함)
    const visibleDaily = dailyRows.filter((r) => {
      const d = r.date || "";
      return d >= "2026-04-27" && (!startDate || d >= startDate) && (!endDate || d <= endDate);
    });
    let bestDailyVal = -Infinity;
    let bestDailyDate = null;
    let worstDailyVal = Infinity;
    let worstDailyDate = null;
    for (const r of visibleDaily) {
      const val = toNumber(r.daily_return_pct) / 100;
      if (val > bestDailyVal) {
        bestDailyVal = val;
        bestDailyDate = r.date;
      }
      if (val < worstDailyVal) {
        worstDailyVal = val;
        worstDailyDate = r.date;
      }
    }

    // 2. 주간 극단치
    let bestWeeklyVal = -Infinity;
    let bestWeeklyDate = null;
    let worstWeeklyVal = Infinity;
    let worstWeeklyDate = null;
    for (let i = 1; i < visibleRows.length; i++) {
      const val = toNumber(visibleRows[i].weekly_return_pct) / 100;
      const date = visibleRows[i].week_date;
      if (val > bestWeeklyVal) {
        bestWeeklyVal = val;
        bestWeeklyDate = date;
      }
      if (val < worstWeeklyVal) {
        worstWeeklyVal = val;
        worstWeeklyDate = date;
      }
    }

    // 3. 월간 극단치
    const visibleMonthly = monthlyRows.filter((r) => {
      const m = (r.month_date || "").substring(0, 7);
      return (!startMonth || m >= startMonth) && (!endMonth || m <= endMonth);
    });
    let bestMonthlyVal = -Infinity;
    let bestMonthlyDate = null;
    let worstMonthlyVal = Infinity;
    let worstMonthlyDate = null;
    for (const r of visibleMonthly) {
      const val = toNumber(r.monthly_return_pct) / 100;
      if (val > bestMonthlyVal) {
        bestMonthlyVal = val;
        bestMonthlyDate = r.month_date;
      }
      if (val < worstMonthlyVal) {
        worstMonthlyVal = val;
        worstMonthlyDate = r.month_date;
      }
    }

    // 4. 년간 극단치
    const visibleYearly = yearlyRows.filter((r) => {
      const y = (r.year_date || "").substring(0, 4);
      return y !== "2023" && (!startYear || y >= startYear) && (!endYear || y <= endYear);
    });
    let bestYearlyVal = -Infinity;
    let bestYearlyDate = null;
    let worstYearlyVal = Infinity;
    let worstYearlyDate = null;
    for (const r of visibleYearly) {
      const val = toNumber(r.yearly_return_pct) / 100;
      if (val > bestYearlyVal) {
        bestYearlyVal = val;
        bestYearlyDate = r.year_date;
      }
      if (val < worstYearlyVal) {
        worstYearlyVal = val;
        worstYearlyDate = r.year_date;
      }
    }

    return {
      bestDaily: bestDailyVal !== -Infinity ? bestDailyVal : null,
      bestDailyDate,
      worstDaily: worstDailyVal !== Infinity ? worstDailyVal : null,
      worstDailyDate,
      bestWeekly: bestWeeklyVal !== -Infinity ? bestWeeklyVal : null,
      bestWeeklyDate,
      worstWeekly: worstWeeklyVal !== Infinity ? worstWeeklyVal : null,
      worstWeeklyDate,
      bestMonthly: bestMonthlyVal !== -Infinity ? bestMonthlyVal : null,
      bestMonthlyDate,
      worstMonthly: worstMonthlyVal !== Infinity ? worstMonthlyVal : null,
      worstMonthlyDate,
      bestYearly: bestYearlyVal !== -Infinity ? bestYearlyVal : null,
      bestYearlyDate,
      worstYearly: worstYearlyVal !== Infinity ? worstYearlyVal : null,
      worstYearlyDate,
    };
  }, [visibleRows, dailyRows, monthlyRows, yearlyRows]);

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
                <ResponsiveContainer width="100%" height={420} minWidth={0}>
                  <BarChart data={visibleRows} margin={{ top: 12, right: 18, bottom: 8, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="week_date" tickFormatter={formatMonthAxisLabel} minTickGap={18} tick={{ fontSize: 12 }} />
                    <YAxis hide={!showAmounts} tickFormatter={formatCompactMoney} width={showAmounts ? 56 : 0} tick={{ fontSize: 12 }} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend itemSorter="value" />
                    {BUCKET_STACK_KEYS.map((key, index) => {
                      const bucketId = Number(key.slice(-1));
                      return (
                        <Bar
                          key={key}
                          dataKey={key}
                          name={BUCKET_NAME_MAP[bucketId]}
                          stackId="assets"
                          fill={BUCKET_COLORS[bucketId - 1]}
                          radius={index === BUCKET_STACK_KEYS.length - 1 ? [3, 3, 0, 0] : [0, 0, 0, 0]}
                        />
                      );
                    })}
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
                <ResponsiveContainer width="100%" height={420} minWidth={0}>
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
                    <YAxis hide={!showAmounts} tickFormatter={formatCompactMoney} width={showAmounts ? 56 : 0} tick={{ fontSize: 12 }} />
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

      <section className="appSection">
        <div className="assetChartsGrid" style={{ alignItems: "stretch" }}>
          <div className="card appCard assetChartsCard">
            <div className="assetChartsCardHeader">
              <div>
                <h2>포트폴리오 성과 지표</h2>
                <p>선택 기간 기준 · 입출금 영향 제외(주간 전략 수익률 기반)</p>
              </div>
            </div>
            <div className="assetChartsBody" style={{ display: "block" }}>
              {loading ? (
                <AppLoadingState label="성과 지표를 계산하는 중입니다." />
              ) : (
                <div style={{ padding: "0 1rem" }}>
                  {[
                    { label: "운용 기간", value: `${visiblePeriod} (${metrics.weeks}주)` },
                    { label: "누적 수익률", value: formatPct(metrics.totalReturn, true), color: signColor(metrics.totalReturn) },
                    { label: "연평균 수익률 (CAGR)", value: formatPct(metrics.cagr, true), color: signColor(metrics.cagr) },
                    { 
                      label: "최대 낙폭 (MDD)", 
                      value: formatPct(metrics.mdd), 
                      subValue: metrics.mddPeriod ? `${metrics.mddPeriod}` : undefined,
                      color: metrics.mdd != null ? "#dc2626" : undefined 
                    },
                    { label: "연 변동성", value: formatPct(metrics.vol), color: "#16a34a" },
                    { 
                      label: "샤프 지수 (Sharpe)", 
                      value: metrics.sharpeRatio !== null ? metrics.sharpeRatio.toFixed(2) : "-",
                      color: "#16a34a"
                    },
                    { label: "주간 승률 (Win Rate)", value: metrics.winRate != null ? `${(metrics.winRate * 100).toFixed(1)}%` : "-", color: "#16a34a" },
                    { label: "누적 손익", value: showAmounts ? (metrics.profitAmount != null ? formatMoney(metrics.profitAmount) : "-") : "•••", color: showAmounts ? signColor(metrics.profitAmount) : undefined },
                  ].map((item) => (
                    <div
                      key={item.label}
                      style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, padding: "4px 4px", borderBottom: "1px solid rgba(148,163,184,0.18)", height: "48px" }}
                    >
                      <span style={{ color: "#475569", fontWeight: 700, fontSize: "0.9rem" }}>{item.label}</span>
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
                        <span style={{ fontWeight: 800, color: item.color ?? "var(--text-strong)", fontSize: "0.98rem", textAlign: "right" }}>{item.value}</span>
                        {item.subValue && (
                          <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 2, fontWeight: 600 }}>
                            {item.subValue}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="card appCard assetChartsCard">
            <div className="assetChartsCardHeader">
              <div>
                <h2>최고 / 최저 수익률 요약</h2>
                <p>각 주기별 최고 및 최저 성과 지표(발생 시점)</p>
              </div>
            </div>
            <div className="assetChartsBody" style={{ display: "block" }}>
              {loading ? (
                <AppLoadingState label="극단 성과를 계산하는 중입니다." />
              ) : (
                <div style={{ padding: "0 1rem" }}>
                  {[
                    { 
                      label: "최고 일간 수익률", 
                      value: formatPct(extremeReturns.bestDaily, true), 
                      subValue: extremeReturns.bestDailyDate ? `${extremeReturns.bestDailyDate}` : undefined,
                      color: signColor(extremeReturns.bestDaily) 
                    },
                    { 
                      label: "최저 일간 수익률", 
                      value: formatPct(extremeReturns.worstDaily, true), 
                      subValue: extremeReturns.worstDailyDate ? `${extremeReturns.worstDailyDate}` : undefined,
                      color: signColor(extremeReturns.worstDaily) 
                    },
                    { 
                      label: "최고 주간 수익률", 
                      value: formatPct(extremeReturns.bestWeekly, true), 
                      subValue: extremeReturns.bestWeeklyDate ? `${extremeReturns.bestWeeklyDate} 주` : undefined,
                      color: signColor(extremeReturns.bestWeekly) 
                    },
                    { 
                      label: "최저 주간 수익률", 
                      value: formatPct(extremeReturns.worstWeekly, true), 
                      subValue: extremeReturns.worstWeeklyDate ? `${extremeReturns.worstWeeklyDate} 주` : undefined,
                      color: signColor(extremeReturns.worstWeekly) 
                    },
                    { 
                      label: "최고 월간 수익률", 
                      value: formatPct(extremeReturns.bestMonthly, true), 
                      subValue: extremeReturns.bestMonthlyDate ? `${extremeReturns.bestMonthlyDate}` : undefined,
                      color: signColor(extremeReturns.bestMonthly) 
                    },
                    { 
                      label: "최저 월간 수익률", 
                      value: formatPct(extremeReturns.worstMonthly, true), 
                      subValue: extremeReturns.worstMonthlyDate ? `${extremeReturns.worstMonthlyDate}` : undefined,
                      color: signColor(extremeReturns.worstMonthly) 
                    },
                    { 
                      label: "최고 년간 수익률", 
                      value: formatPct(extremeReturns.bestYearly, true), 
                      subValue: extremeReturns.bestYearlyDate ? `${extremeReturns.bestYearlyDate} 년` : undefined,
                      color: signColor(extremeReturns.bestYearly) 
                    },
                    { 
                      label: "최저 년간 수익률", 
                      value: formatPct(extremeReturns.worstYearly, true), 
                      subValue: extremeReturns.worstYearlyDate ? `${extremeReturns.worstYearlyDate} 년` : undefined,
                      color: signColor(extremeReturns.worstYearly) 
                    },
                  ].map((item) => (
                    <div
                      key={item.label}
                      style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, padding: "4px 4px", borderBottom: "1px solid rgba(148,163,184,0.18)", height: "48px" }}
                    >
                      <span style={{ color: "#475569", fontWeight: 700, fontSize: "0.9rem" }}>{item.label}</span>
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
                        <span style={{ fontWeight: 800, color: item.color ?? "var(--text-strong)", fontSize: "0.98rem", textAlign: "right" }}>{item.value}</span>
                        {item.subValue && (
                          <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 2, fontWeight: 600 }}>
                            {item.subValue}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

function signColor(value: number | null): string | undefined {
  if (value == null || !Number.isFinite(value) || value === 0) return undefined;
  return value > 0 ? "#dc2626" : "#2563eb";
}
