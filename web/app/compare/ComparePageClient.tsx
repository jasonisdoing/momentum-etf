"use client";

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createChart, ColorType, CrosshairMode, LineSeries } from "lightweight-charts";
import type { IChartApi, LineData, Time } from "lightweight-charts";

import { PageFrame } from "../components/PageFrame";
import { PortfolioChangeBreakdown } from "../components/PortfolioChangeBreakdown";
import { ResponsiveFiltersSection } from "../components/ResponsiveFiltersSection";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { calcPortfolioChange } from "@/lib/portfolio-change";
import type { PortfolioChangeResult } from "@/lib/portfolio-change";

type CompareTab = "performance" | "monthly" | "basic" | "holdings";
type PerformanceRange = "1m" | "3m" | "6m" | "ytd" | "1y" | "3y";
type CompareLoadingProgress = {
  percent: number;
  message: string;
};

type TickerItem = {
  ticker: string;
  name: string;
  ticker_type: string;
  country_code: string;
  is_etf?: boolean;
};

type TickerTypeItem = {
  ticker_type: string;
  order: number;
  name: string;
  icon: string;
};

type StocksTableData = {
  ticker_types: TickerTypeItem[];
  rows: unknown[];
  ticker_type: string;
};

type PriceRow = {
  date: string;
  close: number | null;
  volume?: number | null;
  change_pct: number | null;
};

type TickerFxRate = {
  currency: string;
  rate?: number | null;
  change_pct?: number | null;
};

type TickerEtfInfo = {
  nav?: number | null;
  nav_change?: number | null;
  nav_change_pct?: number | null;
  deviation?: number | null;
  expense_ratio?: number | null;
  dividend_yield_ttm?: number | null;
  market_cap_krw?: number | null;
  volume?: number | null;
  fx_rates?: TickerFxRate[];
  portfolio_change_base_date?: string | null;
  portfolio_change_base_is_open?: boolean | null;
};

type TickerHoldingRow = {
  ticker: string;
  name: string;
  weight: number | null;
  raw_code?: string | null;
  yahoo_symbol?: string | null;
  change_pct?: number | null;
  cumulative_change_pct?: number | null;
  price_currency?: string | null;
};

type TickerDetailResponse = {
  ticker: string;
  rows: PriceRow[];
  etf_info?: TickerEtfInfo | null;
  holdings: TickerHoldingRow[];
  error?: string;
};

type SelectedProduct = {
  item: TickerItem;
  detail: TickerDetailResponse;
};

type CompareHoldingExposureRow = {
  code: string;
  name: string;
  totalWeight: number;
  changePct: number | null;
  holdingsByProductKey: Map<string, TickerHoldingRow>;
};

type ChartDateRange = {
  startDate: string;
  endDate: string;
  shortened: boolean;
};

type CompareGroupMap = Record<string, string[]>;

type PerformanceMetricRange =
  | { key: string; label: string; kind: "period"; days: number }
  | { key: string; label: string; kind: "ytd" };

const MAX_PRODUCTS = 8;
const COMPARE_GROUPS_KEY = "momentum-etf:compare:groups";
const COMPARE_ACTIVE_GROUP_KEY = "momentum-etf:compare:active-group";
const COMPARE_TEMP_SELECTION_KEY = "momentum-etf:compare:temp-selection";
const CHART_COLORS = ["#ef4444", "#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#db2777"];
const CHART_TINTS = [
  "rgba(239, 68, 68, 0.08)",
  "rgba(37, 99, 235, 0.08)",
  "rgba(22, 163, 74, 0.08)",
  "rgba(245, 158, 11, 0.08)",
  "rgba(124, 58, 237, 0.08)",
  "rgba(219, 39, 119, 0.08)",
];
// 옅은 파스텔 (셀 배경 매칭용). 진한 텍스트 버전과 1:1 대응되도록 유지한다.
const HOLDING_MATCH_COLORS = [
  "#dbeafe",
  "#dcfce7",
  "#fef3c7",
  "#ede9fe",
  "#cffafe",
  "#fee2e2",
  "#e0e7ff",
  "#fce7f3",
  "#ecfccb",
  "#f3e8ff",
];
// HOLDING_MATCH_COLORS 와 1:1 대응되는 진한 톤 (큰 숫자 텍스트용).
const HOLDING_MATCH_TEXT_COLORS = [
  "#2563eb",
  "#16a34a",
  "#d97706",
  "#7c3aed",
  "#0891b2",
  "#dc2626",
  "#4f46e5",
  "#db2777",
  "#65a30d",
  "#9333ea",
];
const PERFORMANCE_RANGES: { key: PerformanceRange; label: string; days: number; ytd?: boolean }[] = [
  { key: "1m", label: "1개월", days: 31 },
  { key: "3m", label: "3개월", days: 92 },
  { key: "6m", label: "6개월", days: 183 },
  { key: "ytd", label: "연초이후", days: 0, ytd: true },
  { key: "1y", label: "1년", days: 365 },
  { key: "3y", label: "3년", days: 365 * 3 },
];
const PERFORMANCE_METRIC_RANGES: PerformanceMetricRange[] = [
  { key: "1m", label: "1개월", kind: "period", days: 31 },
  { key: "3m", label: "3개월", kind: "period", days: 92 },
  { key: "6m", label: "6개월", kind: "period", days: 183 },
  { key: "ytd", label: "연초이후", kind: "ytd" },
  { key: "1y", label: "1년", kind: "period", days: 365 },
  { key: "3y", label: "3년", kind: "period", days: 365 * 3 },
];
const BASIC_INFO_METRICS = [
  { label: "현재가", multiline: true },
  { label: "iNAV", multiline: true },
  { label: "괴리율", multiline: false },
  { label: "거래량", multiline: false },
  { label: "운용보수", multiline: false },
  { label: "시가총액", multiline: false },
  { label: "배당 수익률", multiline: false },
  { label: "포트폴리오 변동", multiline: true },
];

function tickerKey(item: TickerItem): string {
  return `${item.ticker_type}::${item.country_code}::${item.ticker}`;
}

function readCompareGroups(): CompareGroupMap {
  if (typeof window === "undefined") return {};
  const raw = window.localStorage.getItem(COMPARE_GROUPS_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      const result: CompareGroupMap = {};
      for (const [name, value] of Object.entries(parsed as Record<string, unknown>)) {
        if (Array.isArray(value)) {
          result[name] = value
            .filter((key): key is string => typeof key === "string")
            .slice(0, MAX_PRODUCTS);
        }
      }
      return result;
    }
  } catch {
    return {};
  }
  return {};
}

function writeCompareGroups(groups: CompareGroupMap): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(COMPARE_GROUPS_KEY, JSON.stringify(groups));
}

function readActiveGroupName(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(COMPARE_ACTIVE_GROUP_KEY);
}

function writeActiveGroupName(name: string | null): void {
  if (typeof window === "undefined") return;
  if (name === null) {
    window.localStorage.removeItem(COMPARE_ACTIVE_GROUP_KEY);
  } else {
    window.localStorage.setItem(COMPARE_ACTIVE_GROUP_KEY, name);
  }
}

function readTempSelection(): string[] {
  if (typeof window === "undefined") return [];
  const raw = window.localStorage.getItem(COMPARE_TEMP_SELECTION_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (Array.isArray(parsed)) {
      return parsed.filter((key): key is string => typeof key === "string").slice(0, MAX_PRODUCTS);
    }
  } catch {
    return [];
  }
  return [];
}

function writeTempSelection(keys: string[]): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(COMPARE_TEMP_SELECTION_KEY, JSON.stringify(keys.slice(0, MAX_PRODUCTS)));
}

function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function formatPrice(value: number | null | undefined, countryCode: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (countryCode === "us") return `$${formatNumber(value, 2)}`;
  if (countryCode === "au") return `A$${formatNumber(value, 2)}`;
  return `${formatNumber(value, 0)}원`;
}

function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}`;
}

function getSignedClass(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) return "";
  return value > 0 ? "metricPositive" : "metricNegative";
}

function formatRatioPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value.toFixed(2)}%`;
}

function formatSignedPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function formatEokFromKrw(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const eokValue = value / 100_000_000;
  const jo = Math.floor(eokValue / 10_000);
  const eok = Math.round(eokValue % 10_000);
  if (jo <= 0) return `${formatNumber(eok, 0)}억`;
  if (eok <= 0) return `${formatNumber(jo, 0)}조`;
  return `${formatNumber(jo, 0)}조 ${formatNumber(eok, 0)}억`;
}

function formatUsdMarketCap(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  if (value >= 1_000_000_000_000) {
    return `${formatNumber(value / 1_000_000_000_000, 2)}조 달러`;
  }
  if (value >= 100_000_000) {
    return `${formatNumber(value / 100_000_000, 1)}억 달러`;
  }
  return `${formatNumber(value, 0)}달러`;
}

function formatAudMarketCap(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  if (value >= 100_000_000) {
    return `${formatNumber(value / 100_000_000, 1)}억 AUD`;
  }
  return `${formatNumber(value, 0)} AUD`;
}

function formatVolume(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (value >= 10_000) return `${formatNumber(Math.floor(value / 10_000), 0)}만`;
  return formatNumber(value, 0);
}

function formatSignedPriceDelta(value: number | null | undefined, countryCode: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const absValue = Math.abs(value);
  if (absValue === 0) return "0";
  return `${value > 0 ? "▲ " : "▼ "}${formatPrice(absValue, countryCode)}`;
}

function formatKoreanDateLabel(value: string | null): string {
  if (!value) return "";
  const date = new Date(`${value}T00:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  return `${date.getFullYear()}년 ${date.getMonth() + 1}월 ${date.getDate()}일`;
}

function getLatestClose(detail: TickerDetailResponse): number | null {
  for (let index = detail.rows.length - 1; index >= 0; index -= 1) {
    const close = detail.rows[index]?.close;
    if (close !== null && close !== undefined && !Number.isNaN(close)) return close;
  }
  return null;
}

function getPreviousClose(detail: TickerDetailResponse): number | null {
  let foundLatest = false;
  for (let index = detail.rows.length - 1; index >= 0; index -= 1) {
    const close = detail.rows[index]?.close;
    if (close === null || close === undefined || Number.isNaN(close)) continue;
    if (foundLatest) return close;
    foundLatest = true;
  }
  return null;
}

function getLatestRow(detail: TickerDetailResponse): PriceRow | null {
  for (let index = detail.rows.length - 1; index >= 0; index -= 1) {
    const row = detail.rows[index];
    if (row?.close !== null && row?.close !== undefined && !Number.isNaN(row.close)) return row;
  }
  return null;
}

function getLatestChangeAmount(detail: TickerDetailResponse): number | null {
  const latestClose = getLatestClose(detail);
  const latestChangePct = getLatestRow(detail)?.change_pct ?? null;
  const previousClose = getPreviousClose(detail);
  if (latestClose === null || latestChangePct === null) return null;
  if (previousClose !== null) return latestClose - previousClose;
  const calculatedPreviousClose = latestClose / (1 + latestChangePct / 100);
  if (!Number.isFinite(calculatedPreviousClose)) return null;
  return latestClose - calculatedPreviousClose;
}

function getPortfolioChange(detail: TickerDetailResponse): PortfolioChangeResult {
  return calcPortfolioChange(
    detail.holdings ?? [],
    detail.etf_info?.fx_rates ?? [],
  );
}

function getPricedRows(rows: PriceRow[]): PriceRow[] {
  return rows.filter((row) => row.close !== null && row.close !== undefined && row.close > 0);
}

function toDateKey(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function formatDateKey(dateKey: string): string {
  return dateKey.replaceAll("-", ".");
}

function getReturnPct(rows: PriceRow[], calendarDays: number): number | null {
  const pricedRows = getPricedRows(rows);
  if (pricedRows.length < 2) return null;
  const endRow = pricedRows.at(-1);
  const end = endRow?.close;
  if (!endRow?.date || !end) return null;
  const cutoff = new Date(endRow.date);
  cutoff.setDate(cutoff.getDate() - calendarDays);
  if (new Date(pricedRows[0].date) > cutoff) return null;
  const start = pricedRows.find((row) => new Date(row.date) >= cutoff)?.close;
  if (!start) return null;
  return ((end / start) - 1) * 100;
}

function getYearToDateReturnPct(rows: PriceRow[]): number | null {
  const pricedRows = getPricedRows(rows);
  if (pricedRows.length < 2) return null;
  const endRow = pricedRows.at(-1);
  const end = endRow?.close;
  if (!endRow?.date || !end) return null;
  const yearStart = new Date(`${endRow.date.slice(0, 4)}-01-01`);
  if (new Date(pricedRows[0].date) > yearStart) return null;
  const start = pricedRows.find((row) => new Date(row.date) >= yearStart)?.close;
  if (!start) return null;
  return ((end / start) - 1) * 100;
}

function getMetricReturnPct(rows: PriceRow[], period: PerformanceMetricRange): number | null {
  if (period.kind === "ytd") return getYearToDateReturnPct(rows);
  return getReturnPct(rows, period.days);
}

/**
 * 선택된 종목들의 공통 가용 기간을 계산한다.
 * - start: 각 종목의 첫 영업일 중 가장 늦은 날(가장 짧은 종목 기준)
 * - end:   각 종목의 마지막 영업일 중 가장 이른 날
 */
function getCommonAvailableRange(products: SelectedProduct[]): {
  startDate: Date;
  endDate: Date;
  days: number;
} | null {
  const pricedRowsList = products.map((product) => getPricedRows(product.detail.rows));
  if (pricedRowsList.length === 0 || pricedRowsList.some((rows) => rows.length < 2)) return null;
  const startTimes = pricedRowsList.map((rows) => new Date(rows[0].date).getTime());
  const endTimes = pricedRowsList.map((rows) => new Date(rows.at(-1)?.date ?? "").getTime());
  if (startTimes.some(Number.isNaN) || endTimes.some(Number.isNaN)) return null;
  const startDate = new Date(Math.max(...startTimes));
  const endDate = new Date(Math.min(...endTimes));
  if (startDate >= endDate) return null;
  const days = Math.floor((endDate.getTime() - startDate.getTime()) / (24 * 3600 * 1000));
  return { startDate, endDate, days };
}

function formatMaxRangeLabel(days: number): string {
  if (days < 30) return `${days}일`;
  if (days < 365) {
    const months = Math.round(days / 30);
    return `${months}개월`;
  }
  const years = days / 365;
  const rounded = Math.round(years * 10) / 10;
  return rounded % 1 === 0 ? `${rounded.toFixed(0)}년` : `${rounded.toFixed(1)}년`;
}

function getChartDateRange(
  products: SelectedProduct[],
  options: { calendarDays?: number; ytd?: boolean },
): ChartDateRange | null {
  const pricedRowsList = products.map((product) => getPricedRows(product.detail.rows));
  if (pricedRowsList.length === 0 || pricedRowsList.some((rows) => rows.length < 2)) return null;

  const startTimes = pricedRowsList.map((rows) => new Date(rows[0].date).getTime());
  const endTimes = pricedRowsList.map((rows) => new Date(rows.at(-1)?.date ?? "").getTime());
  if (startTimes.some(Number.isNaN) || endTimes.some(Number.isNaN)) return null;

  const comparableEnd = new Date(Math.min(...endTimes));
  let requestedStart: Date;
  if (options.ytd) {
    // 올해 초(1월 1일)부터
    requestedStart = new Date(`${comparableEnd.getFullYear()}-01-01T00:00:00`);
  } else {
    requestedStart = new Date(comparableEnd);
    requestedStart.setDate(requestedStart.getDate() - (options.calendarDays ?? 0));
  }

  const firstCommonStart = new Date(Math.max(...startTimes));
  const comparableStart = firstCommonStart > requestedStart ? firstCommonStart : requestedStart;
  if (comparableStart >= comparableEnd) return null;

  return {
    startDate: toDateKey(comparableStart),
    endDate: toDateKey(comparableEnd),
    shortened: firstCommonStart > requestedStart,
  };
}

/**
 * 주어진 기간의 일간 종가에서 MDD(%) 와 그 구간(고점일~저점일)을 계산.
 * MDD = min((price - running_max) / running_max) × 100
 */
type MaxDrawdownResult = { pct: number; peakDate: string | null; troughDate: string | null };

function getMaxDrawdown(rows: PriceRow[], dateRange: ChartDateRange | null): MaxDrawdownResult | null {
  if (!dateRange) return null;
  const seriesRows = getPricedRows(rows).filter(
    (row) => row.date >= dateRange.startDate && row.date <= dateRange.endDate,
  );
  if (seriesRows.length < 2) return null;
  let runningMax = seriesRows[0].close ?? 0;
  let runningMaxDate = seriesRows[0].date;
  let maxDrawdown = 0; // 0 이하의 값(음수)으로 갱신됨
  let peakDate: string | null = null;
  let troughDate: string | null = null;
  for (const row of seriesRows) {
    const price = row.close ?? 0;
    if (price <= 0) continue;
    if (price > runningMax) {
      runningMax = price;
      runningMaxDate = row.date;
      continue;
    }
    if (runningMax <= 0) continue;
    const drawdown = (price - runningMax) / runningMax;
    if (drawdown < maxDrawdown) {
      maxDrawdown = drawdown;
      peakDate = runningMaxDate; // 낙폭이 시작된 고점일
      troughDate = row.date; // 최대 낙폭이 찍힌 저점일
    }
  }
  return { pct: maxDrawdown * 100, peakDate, troughDate };
}

/**
 * 월별 변동률(%) — 각 월의 마지막 종가를 직전 월 마지막 종가와 비교한다.
 * 첫 월은 비교 기준(직전 월 종가)이 없어 제외한다. 반환: { "2026-07": 1.23, ... }
 */
function getMonthlyReturnMap(rows: PriceRow[]): Record<string, number> {
  const lastCloseByMonth = new Map<string, number>();
  for (const row of getPricedRows(rows)) {
    const month = String(row.date).slice(0, 7); // YYYY-MM
    if (month.length !== 7) continue;
    lastCloseByMonth.set(month, row.close as number); // rows 는 날짜 오름차순 → 마지막 값이 월말 종가
  }
  const months = [...lastCloseByMonth.keys()].sort();
  const result: Record<string, number> = {};
  for (let index = 1; index < months.length; index += 1) {
    const prevClose = lastCloseByMonth.get(months[index - 1]);
    const close = lastCloseByMonth.get(months[index]);
    if (prevClose && close && prevClose > 0) {
      result[months[index]] = (close / prevClose - 1) * 100;
    }
  }
  return result;
}

/** "2026-07" → "2026년 07월" */
function formatMonthLabel(month: string): string {
  const [year, mm] = month.split("-");
  return year && mm ? `${year}년 ${mm}월` : month;
}

/**
 * 샤프지수 = (일간 수익률 평균 / 표준편차) × √252.
 * 무위험 수익률 = 0 으로 단순화 (편차로만 위험 측정).
 */
function getSharpeRatio(rows: PriceRow[], dateRange: ChartDateRange | null): number | null {
  if (!dateRange) return null;
  const seriesRows = getPricedRows(rows).filter(
    (row) => row.date >= dateRange.startDate && row.date <= dateRange.endDate,
  );
  if (seriesRows.length < 3) return null;
  const dailyReturns: number[] = [];
  for (let i = 1; i < seriesRows.length; i++) {
    const prev = seriesRows[i - 1].close ?? 0;
    const curr = seriesRows[i].close ?? 0;
    if (prev <= 0 || curr <= 0) continue;
    dailyReturns.push(curr / prev - 1);
  }
  if (dailyReturns.length < 2) return null;
  const mean = dailyReturns.reduce((acc, v) => acc + v, 0) / dailyReturns.length;
  const variance =
    dailyReturns.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (dailyReturns.length - 1);
  const std = Math.sqrt(variance);
  if (std === 0) return null;
  return (mean / std) * Math.sqrt(252);
}

/**
 * 소르티노지수 = (일간 수익률 평균 / 하방 표준편차) × √252.
 * 무위험 수익률 = 0 으로 단순화.
 */
function getSortinoRatio(rows: PriceRow[], dateRange: ChartDateRange | null): number | null {
  if (!dateRange) return null;
  const seriesRows = getPricedRows(rows).filter(
    (row) => row.date >= dateRange.startDate && row.date <= dateRange.endDate,
  );
  if (seriesRows.length < 3) return null;
  const dailyReturns: number[] = [];
  for (let i = 1; i < seriesRows.length; i++) {
    const prev = seriesRows[i - 1].close ?? 0;
    const curr = seriesRows[i].close ?? 0;
    if (prev <= 0 || curr <= 0) continue;
    dailyReturns.push(curr / prev - 1);
  }
  if (dailyReturns.length < 2) return null;
  const mean = dailyReturns.reduce((acc, v) => acc + v, 0) / dailyReturns.length;

  // 하방 편차(Downside Deviation) 계산: 음수 수익률의 제곱합을 구함
  const downsideSquareSum = dailyReturns.reduce((acc, v) => {
    const negativeRet = Math.min(0, v);
    return acc + negativeRet ** 2;
  }, 0);

  const downsideStd = Math.sqrt(downsideSquareSum / (dailyReturns.length - 1));
  if (downsideStd === 0) return null;
  return (mean / downsideStd) * Math.sqrt(252);
}

function buildReturnSeries(rows: PriceRow[], dateRange: ChartDateRange | null): LineData[] {
  if (!dateRange) return [];
  const seriesRows = getPricedRows(rows).filter((row) => row.date >= dateRange.startDate && row.date <= dateRange.endDate);
  if (seriesRows.length < 2) return [];
  const first = seriesRows[0]?.close;
  if (!first) return [];
  return seriesRows.map((row) => ({
    time: row.date as Time,
    value: Number((((row.close ?? first) / first - 1) * 100).toFixed(4)),
  }));
}

function getHoldingCode(row: TickerHoldingRow): string {
  return String(row.yahoo_symbol || row.raw_code || row.ticker || "").trim().toUpperCase();
}

function buildHoldingExposureRows(products: SelectedProduct[]): CompareHoldingExposureRow[] {
  if (products.length === 0) return [];
  const rowsByCode = new Map<string, CompareHoldingExposureRow>();
  const productCount = products.length;

  products.forEach((product) => {
    const productKey = tickerKey(product.item);
    product.detail.holdings.forEach((holding) => {
      const code = getHoldingCode(holding);
      if (!code) return;
      const weight = Number(holding.weight ?? 0);
      const currentRow = rowsByCode.get(code) ?? {
        code,
        name: holding.name || holding.ticker || code,
        totalWeight: 0,
        changePct: holding.change_pct ?? null,
        holdingsByProductKey: new Map<string, TickerHoldingRow>(),
      };
      currentRow.totalWeight += weight / productCount;
      if (currentRow.changePct === null && holding.change_pct !== null && holding.change_pct !== undefined) {
        currentRow.changePct = holding.change_pct;
      }
      if (!currentRow.holdingsByProductKey.has(productKey)) {
        currentRow.holdingsByProductKey.set(productKey, holding);
      }
      rowsByCode.set(code, currentRow);
    });
  });

  return Array.from(rowsByCode.values()).sort((a, b) => {
    if (b.totalWeight !== a.totalWeight) return b.totalWeight - a.totalWeight;
    return a.code.localeCompare(b.code);
  });
}

// 여러 ETF를 한 번에 — 서버에서 구성종목 합집합을 1회 조회해 공유한다.
// 같은 종목(예: SK스퀘어)이 여러 ETF에 나와도 동일 시세/변동률로 나오고, 중복 조회가 사라진다.
async function loadTickerDetailsBatch(
  items: TickerItem[],
  includeHoldings: boolean = true,
): Promise<TickerDetailResponse[]> {
  const response = await fetch(`/api/ticker-detail-compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    body: JSON.stringify({
      items: items.map((item) => ({
        ticker: item.ticker,
        ticker_type: item.ticker_type,
        country_code: item.country_code,
      })),
      // 성과분석 탭은 가격 시계열만 필요 — 구성종목 평가를 건너뛰어 초기 로딩을 줄인다.
      include_holdings: includeHoldings,
    }),
  });
  const payload = (await response.json()) as { results?: TickerDetailResponse[]; error?: string };
  if (!response.ok || payload.error) {
    throw new Error(payload.error ?? "비교 데이터를 불러오지 못했습니다.");
  }
  return payload.results ?? [];
}

function ProductSearchField({
  value,
  items,
  disabledKeys,
  inputRef,
  onChange,
}: {
  value: string;
  items: TickerItem[];
  disabledKeys: string[];
  inputRef?: { current: HTMLInputElement | null };
  onChange: (value: string) => void;
}) {
  const [focused, setFocused] = useState(false);
  const normalized = value.trim().toLowerCase();
  const suggestions = useMemo(() => {
    if (!normalized) return [];
    const filtered = items.filter((item) => `${item.ticker} ${item.name}`.toLowerCase().includes(normalized));
    return filtered.filter((item) => !disabledKeys.includes(tickerKey(item))).slice(0, 12);
  }, [disabledKeys, items, normalized]);

  return (
    <label className="appLabeledField compareTickerSearchField">
      <span className="appLabeledFieldLabel">상품 검색</span>
      <input
        ref={inputRef}
        className="form-control"
        value={value}
        placeholder="티커 또는 종목명"
        onFocus={() => setFocused(true)}
        onBlur={() => window.setTimeout(() => setFocused(false), 120)}
        onChange={(event) => {
          onChange(event.target.value);
          setFocused(true);
        }}
      />
      {focused && normalized ? (
        <div className="compareTickerSearchMenu">
          {suggestions.length > 0 ? (
            suggestions.map((item) => (
              <button
                key={tickerKey(item)}
                type="button"
                className="compareTickerSearchOption"
                onMouseDown={(event) => {
                  event.preventDefault();
                  onChange(`${item.ticker} · ${item.name}`);
                  window.dispatchEvent(new CustomEvent("compare:add-product", { detail: tickerKey(item) }));
                  setFocused(false);
                }}
              >
                <span>{item.ticker}</span>
                <small>{item.name}</small>
              </button>
            ))
          ) : (
            <div className="compareTickerSearchEmpty">검색 결과 없음</div>
          )}
        </div>
      ) : null}
    </label>
  );
}

function BasicInfoValue({ product, metric }: { product: SelectedProduct; metric: string }) {
  const latestRow = getLatestRow(product.detail);
  const latestClose = getLatestClose(product.detail);
  const latestChangePct = latestRow?.change_pct ?? null;
  const latestChangeAmount = getLatestChangeAmount(product.detail);
  const etfInfo = product.detail.etf_info ?? null;
  const portfolioChange = getPortfolioChange(product.detail);

  if (metric === "현재가") {
    return (
      <div className="compareBasicValue">
        <strong>{formatPrice(latestClose, product.item.country_code)}</strong>
        <span className={getSignedClass(latestChangeAmount ?? latestChangePct)}>
          {formatSignedPriceDelta(latestChangeAmount, product.item.country_code)}
        </span>
        <span className={getSignedClass(latestChangePct)}>{formatSignedPercent(latestChangePct)}</span>
      </div>
    );
  }

  if (metric === "iNAV") {
    // NAV 도 현재가와 같은 종목 통화로 표기한다(미국 $, 호주 A$, 국내 원).
    return (
      <div className="compareBasicValue">
        <strong>{formatPrice(etfInfo?.nav ?? null, product.item.country_code)}</strong>
        <span className={getSignedClass(etfInfo?.nav_change ?? null)}>
          {formatSignedPriceDelta(etfInfo?.nav_change ?? null, product.item.country_code)}
        </span>
        <span className={getSignedClass(etfInfo?.nav_change_pct ?? null)}>
          {formatSignedPercent(etfInfo?.nav_change_pct ?? null)}
        </span>
      </div>
    );
  }

  if (metric === "괴리율") {
    // 괴리율도 다른 증감 값과 같은 규칙(+빨강 / -파랑)으로 색을 준다.
    const deviation = etfInfo?.deviation ?? null;
    return <strong className={getSignedClass(deviation)}>{formatSignedPercent(deviation)}</strong>;
  }

  if (metric === "거래량") {
    return <strong>{formatVolume(etfInfo?.volume ?? latestRow?.volume ?? null)}</strong>;
  }

  if (metric === "운용보수") {
    return <strong>{formatRatioPercent(etfInfo?.expense_ratio ?? null)}</strong>;
  }

  if (metric === "시가총액") {
    const country = String(product.item.country_code || "").toLowerCase();
    const ticker = String(product.item.ticker || "").toUpperCase();
    const isUs = country === "us";
    const isAu = country === "au" || ticker.startsWith("ASX:") || ticker.endsWith(".AX");
    const val = etfInfo?.market_cap_krw ?? null;
    return (
      <strong>
        {isUs
          ? formatUsdMarketCap(val)
          : isAu
            ? formatAudMarketCap(val)
            : formatEokFromKrw(val)}
      </strong>
    );
  }

  if (metric === "배당 수익률") {
    return <strong>{formatRatioPercent(etfInfo?.dividend_yield_ttm ?? null)}</strong>;
  }

  if (metric === "포트폴리오 변동") {
    return (
      <div className="compareBasicValue">
        <PortfolioChangeBreakdown
          items={portfolioChange.breakdown}
          fxRates={etfInfo?.fx_rates ?? []}
          variant="compact"
          emptyText="-"
        />
      </div>
    );
  }

  return <strong>-</strong>;
}

function CompareChart({ products, dateRange }: { products: SelectedProduct[]; dateRange: ChartDateRange | null }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    chartRef.current?.remove();
    chartRef.current = null;
    if (products.length === 0 || !dateRange) return;

    const container = containerRef.current;
    const chart = createChart(container, {
      width: container.clientWidth,
      height: 360,
      layout: { background: { type: ColorType.Solid, color: "#ffffff" }, textColor: "#475569", fontSize: 12 },
      grid: { vertLines: { color: "#eef2f7" }, horzLines: { color: "#d8e0ec", style: 2 } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#e2e8f0" },
      timeScale: { borderColor: "#e2e8f0", timeVisible: false },
    });
    chartRef.current = chart;

    products.forEach((product, index) => {
      chart.addSeries(LineSeries, {
        color: CHART_COLORS[index % CHART_COLORS.length],
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: true,
      }).setData(buildReturnSeries(product.detail.rows, dateRange));
    });
    chart.timeScale().fitContent();

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) chart.applyOptions({ width: entry.contentRect.width });
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [dateRange, products]);

  return <div ref={containerRef} className="compareChart" />;
}

export function ComparePageClient() {
  const [tickerItems, setTickerItems] = useState<TickerItem[]>([]);
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  const [products, setProducts] = useState<SelectedProduct[]>([]);
  const [activeTab, setActiveTab] = useState<CompareTab>("performance");
  // 구성 종목 탭에서 종목 정렬 기준: weight(비중, 기본) / change(상승률 내림차순)
  const [holdingsSortBy, setHoldingsSortBy] = useState<"weight" | "change">("weight");
  const [performanceRange, setPerformanceRange] = useState<string>("1y"); // 성과분석 기본 기간
  const [searchText, setSearchText] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState<CompareLoadingProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [groups, setGroups] = useState<CompareGroupMap>({});
  const [activeGroupName, setActiveGroupName] = useState<string | null>(null);
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const hydratedRef = useRef(false);

  const itemByKey = useMemo(() => {
    const map = new Map<string, TickerItem>();
    tickerItems.forEach((item) => map.set(tickerKey(item), item));
    return map;
  }, [tickerItems]);

  /** is_etf === true 인 ETF 만 검색 대상. (국내상장 국내/해외 ETF) */
  const searchableItems = useMemo(
    () => tickerItems.filter((item) => item.is_etf === true),
    [tickerItems],
  );

  // 마지막으로 로드한 조합 + 범위("종목키::full|price"). 탭 전환 시 중복 요청을 막는다.
  const loadedSignatureRef = useRef<string | null>(null);

  const loadSelectedProducts = useCallback(async (keys: string[], includeHoldings: boolean = true) => {
    const items = keys.map((key) => itemByKey.get(key)).filter((item): item is TickerItem => Boolean(item));
    if (items.length === 0) {
      setProducts([]);
      return;
    }
    setLoading(true);
    setLoadingProgress({ percent: 10, message: `${items.length}개 ETF 비교 요청 준비 중` });
    setError(null);
    const progressTimers: number[] = [];
    let progressInterval: number | null = null;
    try {
      progressTimers.push(
        window.setTimeout(() => {
          setLoadingProgress({ percent: 35, message: "구성종목 합집합과 가격 스냅샷 조회 중" });
        }, 400),
      );
      progressTimers.push(
        window.setTimeout(() => {
          setLoadingProgress((previous) => ({
            percent: Math.max(previous?.percent ?? 0, 55),
            message: "ETF별 비교 데이터 계산 중",
          }));
          progressInterval = window.setInterval(() => {
            setLoadingProgress((previous) => {
              if (!previous) return previous;
              return {
                ...previous,
                percent: Math.min(90, previous.percent + 5),
              };
            });
          }, 1200);
        }, 1400),
      );
      // 한 번의 일괄 호출 — 서버가 구성종목 합집합을 1회 조회해 공유하므로
      // 같은 종목은 ETF 간 동일 값이 되고, 중복 조회/전역 lock 직렬화 문제도 사라진다.
      const details = await loadTickerDetailsBatch(items, includeHoldings);
      setLoadingProgress({ percent: 100, message: "비교 데이터 반영 중" });
      setProducts(items.map((item, index) => ({ item, detail: details[index] })));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "비교 데이터를 불러오지 못했습니다.");
      loadedSignatureRef.current = null; // 실패한 요청은 '로드됨'으로 두지 않는다(재시도 가능하게).
    } finally {
      progressTimers.forEach((timer) => window.clearTimeout(timer));
      if (progressInterval !== null) window.clearInterval(progressInterval);
      setLoading(false);
      setLoadingProgress(null);
    }
  }, [itemByKey]);

  useEffect(() => {
    let alive = true;
    async function loadInitialData() {
      try {
        const tickersResponse = await fetch("/api/ticker-tickers", { cache: "no-store" });
        const tickersPayload = (await tickersResponse.json()) as TickerItem[] | { error?: string };
        if (!tickersResponse.ok || !Array.isArray(tickersPayload)) {
          throw new Error(Array.isArray(tickersPayload) ? "종목 목록을 불러오지 못했습니다." : tickersPayload.error);
        }
        if (!alive) return;
        const items = tickersPayload.filter((item) => item.ticker && item.ticker_type && item.country_code);
        setTickerItems(items);

        // 그룹/활성 그룹/임시 선택 복원
        const savedGroups = readCompareGroups();
        const savedActive = readActiveGroupName();
        setGroups(savedGroups);
        const isValidActive = savedActive && savedGroups[savedActive] !== undefined;
        if (isValidActive) {
          setActiveGroupName(savedActive);
          const savedKeys = savedGroups[savedActive] ?? [];
          const validKeys = savedKeys.filter((key) => items.some((it) => tickerKey(it) === key));
          setSelectedKeys(validKeys.slice(0, MAX_PRODUCTS));
        } else {
          setActiveGroupName(null);
          const tempKeys = readTempSelection();
          const validKeys = tempKeys.filter((key) => items.some((it) => tickerKey(it) === key));
          setSelectedKeys(validKeys.slice(0, MAX_PRODUCTS));
        }
        hydratedRef.current = true;
      } catch (loadError) {
        if (alive) {
          hydratedRef.current = true;
          setError(loadError instanceof Error ? loadError.message : "종목 목록을 불러오지 못했습니다.");
        }
      }
    }
    void loadInitialData();
    return () => {
      alive = false;
    };
  }, []);

  // 성과분석 탭은 가격 시계열만 있으면 되므로 구성종목 계산을 생략해 초기 로딩을 줄인다.
  // 기본정보/구성종목 탭을 열면(또는 종목이 바뀌면) 구성종목까지 포함해 다시 요청한다.
  // 이미 전체 데이터를 받아둔 조합이면 재요청하지 않는다(성과분석으로 되돌아와도 유지).
  useEffect(() => {
    // 성과분석·월간분석은 가격 시계열만 쓴다 → 구성종목 계산 없이 가볍게 로드.
    const needsHoldings = activeTab !== "performance" && activeTab !== "monthly";
    const key = selectedKeys.join("|");
    const signature = `${key}::${needsHoldings ? "full" : "price"}`;
    // 전체(full)를 이미 받았으면 가격만 필요한 요청은 건너뛴다(같은 종목 조합인 동안).
    if (loadedSignatureRef.current === signature) return;
    if (!needsHoldings && loadedSignatureRef.current === `${key}::full`) return;
    loadedSignatureRef.current = signature;
    void loadSelectedProducts(selectedKeys, needsHoldings);
  }, [loadSelectedProducts, selectedKeys, activeTab]);

  // selectedKeys 가 변경될 때 활성 그룹이면 해당 그룹에 자동 저장, 아니면 임시 저장
  useEffect(() => {
    if (!hydratedRef.current) return;
    if (activeGroupName) {
      setGroups((current) => {
        const next = { ...current, [activeGroupName]: selectedKeys.slice(0, MAX_PRODUCTS) };
        writeCompareGroups(next);
        return next;
      });
    } else {
      writeTempSelection(selectedKeys);
    }
  }, [selectedKeys, activeGroupName]);

  useEffect(() => {
    const handler = (event: Event) => {
      const key = (event as CustomEvent<string>).detail;
      if (!key || selectedKeys.includes(key) || selectedKeys.length >= MAX_PRODUCTS) return;
      setSelectedKeys((prev) => [...prev, key]);
      setSearchText("");
    };
    window.addEventListener("compare:add-product", handler);
    return () => window.removeEventListener("compare:add-product", handler);
  }, [selectedKeys]);

  /** 그룹 선택 변경 (드롭다운). */
  const handleSelectGroup = useCallback((name: string | null) => {
    setActiveGroupName(name);
    writeActiveGroupName(name);
    if (name) {
      const keys = groups[name] ?? [];
      setSelectedKeys(keys.slice(0, MAX_PRODUCTS));
    } else {
      const tempKeys = readTempSelection();
      setSelectedKeys(tempKeys.slice(0, MAX_PRODUCTS));
    }
    setSearchText("");
  }, [groups]);

  /** 빈 새 그룹 생성. 기존에 선택된 종목은 제거됨. */
  const handleCreateGroup = useCallback(() => {
    const rawName = window.prompt("그룹 이름을 입력하세요 (예: 배당 ETF 모음)");
    if (rawName === null) return;
    const name = rawName.trim();
    if (!name) return;
    if (groups[name] !== undefined) {
      window.alert(`"${name}" 그룹이 이미 존재합니다.`);
      return;
    }
    const nextGroups = { ...groups, [name]: [] };
    setGroups(nextGroups);
    writeCompareGroups(nextGroups);
    setActiveGroupName(name);
    writeActiveGroupName(name);
    setSelectedKeys([]);
    setSearchText("");
  }, [groups]);

  /** 현재 그룹 이름 변경. */
  const handleRenameGroup = useCallback(() => {
    if (!activeGroupName) return;
    const rawName = window.prompt("새 그룹 이름을 입력하세요", activeGroupName);
    if (rawName === null) return;
    const name = rawName.trim();
    if (!name || name === activeGroupName) return;
    if (groups[name] !== undefined) {
      window.alert(`"${name}" 그룹이 이미 존재합니다.`);
      return;
    }
    const { [activeGroupName]: oldKeys, ...rest } = groups;
    const nextGroups = { ...rest, [name]: oldKeys };
    setGroups(nextGroups);
    writeCompareGroups(nextGroups);
    setActiveGroupName(name);
    writeActiveGroupName(name);
  }, [activeGroupName, groups]);

  /** 현재 그룹 삭제. */
  const handleDeleteGroup = useCallback(() => {
    if (!activeGroupName) return;
    if (!window.confirm(`"${activeGroupName}" 그룹을 삭제하시겠습니까?`)) return;
    const { [activeGroupName]: _, ...rest } = groups;
    setGroups(rest);
    writeCompareGroups(rest);
    setActiveGroupName(null);
    writeActiveGroupName(null);
    const tempKeys = readTempSelection();
    setSelectedKeys(tempKeys.slice(0, MAX_PRODUCTS));
  }, [activeGroupName, groups]);

  // commonAvailableRange 는 sortedProducts 와 무관(순서만 다름)하므로 products 로 미리 계산
  const commonAvailableRange = useMemo(() => getCommonAvailableRange(products), [products]);

  const selectedPerformanceRange = useMemo<{
    key: string;
    label: string;
    days: number;
    ytd?: boolean;
  }>(() => {
    const fixed = PERFORMANCE_RANGES.find((range) => range.key === performanceRange);
    if (fixed) return fixed;
    if (performanceRange === "__max__" && commonAvailableRange) {
      return {
        key: "__max__",
        label: `최대 ${formatMaxRangeLabel(commonAvailableRange.days)}`,
        days: commonAvailableRange.days,
      };
    }
    return PERFORMANCE_RANGES[1];
  }, [commonAvailableRange, performanceRange]);
  const performanceMetricRows = useMemo<PerformanceMetricRange[]>(() => {
    if (selectedPerformanceRange.key !== "__max__") return PERFORMANCE_METRIC_RANGES;
    const maxRow: PerformanceMetricRange = {
      key: "__max__",
      label: selectedPerformanceRange.label,
      kind: "period",
      days: selectedPerformanceRange.days,
    };
    const insertIndex = PERFORMANCE_METRIC_RANGES.findIndex(
      (row) => row.kind === "period" && row.days > selectedPerformanceRange.days,
    );
    if (insertIndex < 0) return [...PERFORMANCE_METRIC_RANGES, maxRow];
    return [
      ...PERFORMANCE_METRIC_RANGES.slice(0, insertIndex),
      maxRow,
      ...PERFORMANCE_METRIC_RANGES.slice(insertIndex),
    ];
  }, [selectedPerformanceRange.days, selectedPerformanceRange.key, selectedPerformanceRange.label]);
  const sortedProducts = useMemo(() => {
    return products
      .map((product, index) => ({
        product,
        index,
        returnPct: selectedPerformanceRange.ytd
          ? getYearToDateReturnPct(product.detail.rows)
          : getReturnPct(product.detail.rows, selectedPerformanceRange.days),
      }))
      .sort((a, b) => {
        const aValue = a.returnPct;
        const bValue = b.returnPct;
        if (aValue === null && bValue === null) return a.index - b.index;
        if (aValue === null) return 1;
        if (bValue === null) return -1;
        if (bValue === aValue) return a.index - b.index;
        return bValue - aValue;
      })
      .map(({ product }) => product);
  }, [products, selectedPerformanceRange.days, selectedPerformanceRange.ytd]);

  // 월간 분석 탭 — 종목별 월간 변동률 맵 + 표시할 월 목록(최신 월이 위)
  const monthlyReturnMaps = useMemo(() => {
    const map = new Map<string, Record<string, number>>();
    for (const product of sortedProducts) {
      map.set(tickerKey(product.item), getMonthlyReturnMap(product.detail.rows));
    }
    return map;
  }, [sortedProducts]);
  const monthlyRows = useMemo(() => {
    const months = new Set<string>();
    for (const returns of monthlyReturnMaps.values()) {
      for (const month of Object.keys(returns)) months.add(month);
    }
    return [...months].sort().reverse(); // 최신 월 → 과거 월
  }, [monthlyReturnMaps]);
  const chartDateRange = useMemo(
    () => getChartDateRange(sortedProducts, {
      calendarDays: selectedPerformanceRange.days,
      ytd: selectedPerformanceRange.ytd,
    }),
    [selectedPerformanceRange.days, selectedPerformanceRange.ytd, sortedProducts],
  );

  /** 토글 버튼 목록 — 가용 기간보다 긴 옵션은 disabled, 적당한 max 옵션 동적 삽입. */
  const performanceRangeButtons = useMemo<
    Array<{ key: string; label: string; enabled: boolean; days: number; ytd?: boolean }>
  >(() => {
    const maxDays = commonAvailableRange?.days ?? Infinity;
    const ytdStart = commonAvailableRange
      ? new Date(`${commonAvailableRange.endDate.getFullYear()}-01-01T00:00:00`)
      : null;
    const ytdFullyCovered =
      commonAvailableRange && ytdStart ? commonAvailableRange.startDate <= ytdStart : true;

    // 표준 버튼 + enabled 플래그
    const base: Array<{ key: string; label: string; enabled: boolean; days: number; ytd?: boolean }> =
      PERFORMANCE_RANGES.map((range) => {
        let enabled: boolean;
        if (range.ytd) {
          enabled = ytdFullyCovered;
        } else {
          // 5일 정도 tolerance (영업일/달력일 차이)
          enabled = maxDays + 5 >= range.days;
        }
        return { key: range.key as string, label: range.label, enabled, days: range.days, ytd: range.ytd };
      });

    // 마지막으로 enabled 된 표준 범위(가장 긴 것)를 찾고, max 와 차이 크면 "max" 버튼 삽입
    if (commonAvailableRange) {
      // YTD 제외 가장 긴 enabled 고정 범위 days
      const lastEnabledFixed = base
        .filter((b) => b.enabled && !b.ytd)
        .reduce((acc, b) => Math.max(acc, b.days), 0);
      const TOLERANCE_DAYS = 14; // 2주 이상 차이 나야 별도 버튼 의미 있음
      const hasMeaningfulGap = maxDays - lastEnabledFixed > TOLERANCE_DAYS;
      if (hasMeaningfulGap) {
        // 첫 번째 disabled 위치 직전에 삽입
        const firstDisabledIdx = base.findIndex((b) => !b.enabled && !b.ytd);
        const maxButton = {
          key: "__max__",
          label: `최대 ${formatMaxRangeLabel(commonAvailableRange.days)}`,
          enabled: true,
          days: commonAvailableRange.days,
          ytd: undefined as boolean | undefined,
        };
        if (firstDisabledIdx >= 0) {
          base.splice(firstDisabledIdx, 0, maxButton);
        } else {
          base.push(maxButton);
        }
      }
    }
    return base;
  }, [commonAvailableRange]);

  // 현재 선택된 범위가 disabled 가 됐다면 자동으로 가장 큰 enabled 로 폴백
  useEffect(() => {
    const current = performanceRangeButtons.find((b) => b.key === performanceRange);
    if (current && current.enabled) return;
    const lastEnabled = [...performanceRangeButtons].reverse().find((b) => b.enabled);
    if (lastEnabled && lastEnabled.key !== performanceRange) {
      setPerformanceRange(lastEnabled.key);
    }
  }, [performanceRange, performanceRangeButtons]);

  const portfolioChangeBaseDate = useMemo(() => {
    for (const p of sortedProducts) {
      if (p.detail.etf_info?.portfolio_change_base_date) {
        return p.detail.etf_info.portfolio_change_base_date;
      }
    }
    return null;
  }, [sortedProducts]);

  // base_date 를 뽑은 것과 동일한 상품 기준 — 당일 시초가 baseline 이면 라벨을 "시초가 이후"로.
  const portfolioChangeBaseIsOpen = useMemo(() => {
    for (const p of sortedProducts) {
      if (p.detail.etf_info?.portfolio_change_base_date) {
        return !!p.detail.etf_info.portfolio_change_base_is_open;
      }
    }
    return false;
  }, [sortedProducts]);

  // 매칭 색상 계산용: 전체 종목 (이전엔 상위 10개만이었지만 컬럼 스크롤로 전부 노출되므로 전부 대상).
  const holdingExposureRows = useMemo(() => buildHoldingExposureRows(sortedProducts), [sortedProducts]);
  const holdingColorByCode = useMemo(() => {
    const counts = new Map<string, number>();
    holdingExposureRows.forEach((row) => {
      counts.set(row.code, row.holdingsByProductKey.size);
    });

    const colors = new Map<string, string>();
    holdingExposureRows.forEach((row) => {
      const code = row.code;
      if ((counts.get(code) ?? 0) < 2) return;
      colors.set(code, HOLDING_MATCH_COLORS[colors.size % HOLDING_MATCH_COLORS.length]);
    });
    return colors;
  }, [holdingExposureRows]);
  // 매칭 종목의 텍스트용 진한 색 (큰 순위 숫자 색으로 사용).
  const holdingTextColorByCode = useMemo(() => {
    const counts = new Map<string, number>();
    holdingExposureRows.forEach((row) => {
      counts.set(row.code, row.holdingsByProductKey.size);
    });
    const colors = new Map<string, string>();
    holdingExposureRows.forEach((row) => {
      const code = row.code;
      if ((counts.get(code) ?? 0) < 2) return;
      colors.set(code, HOLDING_MATCH_TEXT_COLORS[colors.size % HOLDING_MATCH_TEXT_COLORS.length]);
    });
    return colors;
  }, [holdingExposureRows]);

  const titleRight = (
    <div className="compareTitleMeta">
      <span>비교 상품:</span>
      <span>{products.length}/{MAX_PRODUCTS}</span>
    </div>
  );

  return (
    <PageFrame title="ETF 비교" fullHeight fullWidth titleRight={titleRight}>
      <div className="appPageStack appPageStackFill comparePage">
        {error ? <div className="alert alert-danger mb-0">{error}</div> : null}

        <section className="appSection">
          <div className="card appCard">
            <div className="card-header">
              <ResponsiveFiltersSection>
                <div className="appMainHeader">
                  <div className="appMainHeaderLeft compareMainHeaderLeft">
                    <label className="appLabeledField">
                      <span className="appLabeledFieldLabel">그룹</span>
                      <div className="compareGroupControl">
                        <select
                          className="form-select"
                          value={activeGroupName ?? ""}
                          onChange={(event) => {
                            const next = event.target.value;
                            handleSelectGroup(next === "" ? null : next);
                          }}
                        >
                          <option value="">(저장 안 됨)</option>
                          {Object.keys(groups).sort().map((name) => (
                            <option key={name} value={name}>{name}</option>
                          ))}
                        </select>
                        <button
                          type="button"
                          className="btn btn-sm btn-outline-primary"
                          onClick={handleCreateGroup}
                          title="현재 선택을 새 그룹으로 저장"
                        >
                          + 새 그룹
                        </button>
                        {activeGroupName ? (
                          <>
                            <button
                              type="button"
                              className="btn btn-sm btn-outline-secondary"
                              onClick={handleRenameGroup}
                              title="그룹 이름 변경"
                            >
                              이름 변경
                            </button>
                            <button
                              type="button"
                              className="btn btn-sm btn-outline-danger"
                              onClick={handleDeleteGroup}
                              title="현재 그룹 삭제"
                            >
                              그룹 삭제
                            </button>
                          </>
                        ) : null}
                      </div>
                    </label>
                    <ProductSearchField
                      value={searchText}
                      items={searchableItems}
                      disabledKeys={selectedKeys}
                      inputRef={searchInputRef}
                      onChange={setSearchText}
                    />
                    <label className="appLabeledField" style={{ minWidth: 0, width: "auto" }}>
                      <span className="appLabeledFieldLabel">탭</span>
                      <div className="compareHeaderTabs appSegmentedToggle" role="group" aria-label="비교 보기 선택">
                        <button
                          type="button"
                          className={activeTab === "performance" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                          onClick={() => setActiveTab("performance")}
                        >
                          성과 분석
                        </button>
                        <button
                          type="button"
                          className={activeTab === "monthly" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                          onClick={() => setActiveTab("monthly")}
                        >
                          월간 분석
                        </button>
                        <button
                          type="button"
                          className={activeTab === "basic" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                          onClick={() => setActiveTab("basic")}
                        >
                          기본 정보
                        </button>
                        <button
                          type="button"
                          className={activeTab === "holdings" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                          onClick={() => setActiveTab("holdings")}
                        >
                          구성 종목
                        </button>
                      </div>
                    </label>
                    {activeTab === "holdings" ? (
                      <label className="appLabeledField" style={{ marginLeft: "0.75rem", minWidth: 0, width: "auto" }}>
                        <span className="appLabeledFieldLabel">구성종목 정렬</span>
                        <div className="appSegmentedToggle" role="group" aria-label="구성종목 정렬 기준">
                          <button
                            type="button"
                            className={
                              holdingsSortBy === "weight"
                                ? "btn appSegmentedToggleButton is-active"
                                : "btn appSegmentedToggleButton"
                            }
                            onClick={() => setHoldingsSortBy("weight")}
                          >
                            비중
                          </button>
                          <button
                            type="button"
                            className={
                              holdingsSortBy === "change"
                                ? "btn appSegmentedToggleButton is-active"
                                : "btn appSegmentedToggleButton"
                            }
                            onClick={() => setHoldingsSortBy("change")}
                          >
                            상승률
                          </button>
                        </div>
                      </label>
                    ) : null}
                  </div>
                </div>
              </ResponsiveFiltersSection>
            </div>
          </div>
        </section>

        {loading ? (
          <div className="compareLoading">
            <div className="compareLoadingText">
              <span>비교 데이터를 불러오는 중...</span>
              <strong>{loadingProgress?.percent ?? 0}%</strong>
            </div>
            <div className="compareLoadingBar" aria-hidden="true">
              <div style={{ width: `${loadingProgress?.percent ?? 0}%` }} />
            </div>
            <small>{loadingProgress?.message ?? "비교 데이터 요청 중"}</small>
          </div>
        ) : null}

        {/* 월간 분석은 행이 많아 종목 헤더를 고정(sticky)하고 아래 행만 스크롤한다. */}
        <section className={`compareMatrix${activeTab === "monthly" ? " compareMatrixStickyHeader" : ""}`}>
          <div className="compareMatrixLabel compareMatrixLabelWide compareProductHeaderLabel">종목</div>
          {sortedProducts.map((product, index) => (
            <div
              key={tickerKey(product.item)}
              className="compareProductCard"
              style={{
                borderTopColor: CHART_COLORS[index % CHART_COLORS.length],
                background: CHART_TINTS[index % CHART_TINTS.length],
              }}
            >
              <button
                type="button"
                className="compareProductRemove"
                aria-label={`${product.item.ticker} 제거`}
                onClick={() => setSelectedKeys((prev) => prev.filter((key) => key !== tickerKey(product.item)))}
              >
                ×
              </button>
              <div className="compareProductCode" style={{ color: CHART_COLORS[index % CHART_COLORS.length] }}>
                <TickerDetailLink ticker={product.item.ticker} />
              </div>
              <div className="compareProductName">{product.item.name}</div>
              <div className="compareProductPrice">
                {formatPrice(getLatestClose(product.detail), product.item.country_code)}
                <span className={getSignedClass(getLatestRow(product.detail)?.change_pct ?? null)}>
                  {formatSignedPercent(getLatestRow(product.detail)?.change_pct ?? null)}
                </span>
              </div>
            </div>
          ))}
          {Array.from({ length: Math.max(0, MAX_PRODUCTS - sortedProducts.length) }).map((_, index) => (
            <div key={`empty-${index}`} className="compareProductEmpty compareProductHeaderEmpty">비교 상품을 추가해 주세요.</div>
          ))}
        </section>

        {activeTab === "performance" ? (
          <section className="compareMatrix compareMatrixBody">
            <div className="compareMatrixLabel compareMatrixLabelWide">수익률 추이</div>
            <div className="compareMatrixWide">
              <div className="comparePerformanceToolbar">
                <div className="appSegmentedToggle" role="group" aria-label="수익률 추이 기간">
                  {performanceRangeButtons.map((range) => {
                    const isActive = performanceRange === range.key;
                    const className = [
                      "btn",
                      "appSegmentedToggleButton",
                      isActive ? "is-active" : "",
                      !range.enabled ? "is-disabled" : "",
                    ]
                      .filter(Boolean)
                      .join(" ");
                    return (
                      <button
                        key={range.key}
                        type="button"
                        className={className}
                        disabled={!range.enabled}
                        title={!range.enabled ? "선택된 종목의 데이터가 부족합니다" : undefined}
                        onClick={() => range.enabled && setPerformanceRange(range.key)}
                      >
                        {range.label}
                      </button>
                    );
                  })}
                </div>
              </div>
              <CompareChart products={sortedProducts} dateRange={chartDateRange} />
            </div>
            <div className="compareMatrixLabel compareMetricsGroupLabel" style={{ gridRow: `span ${performanceMetricRows.length}` }}>
              수익률(%)
            </div>
            {performanceMetricRows.map((period) => {
              const isActiveMetricRow = period.key === selectedPerformanceRange.key;
              return (
                <Fragment key={period.key}>
                  <div className={`compareMetricPeriodLabel ${isActiveMetricRow ? "is-active-range" : ""}`}>{period.label}</div>
                  {sortedProducts.map((product) => {
                    const value = getMetricReturnPct(product.detail.rows, period);
                    return (
                      <div
                        key={tickerKey(product.item)}
                        className={`compareMetricCell ${getSignedClass(value)} ${isActiveMetricRow ? "is-active-range" : ""}`}
                      >
                        {formatPercent(value)}
                      </div>
                    );
                  })}
                  {Array.from({ length: Math.max(0, MAX_PRODUCTS - sortedProducts.length) }).map((_, index) => (
                    <div
                      key={`empty-metric-${period.label}-${index}`}
                      className={`compareMetricCell ${isActiveMetricRow ? "is-active-range" : ""}`}
                    >
                      -
                    </div>
                  ))}
                </Fragment>
              );
            })}

            <div className="compareMatrixLabel compareMetricsGroupLabel compareMetricsGroupLabelSingle">
              MDD(%)
              <div className="compareMatrixLabelHint">
                {chartDateRange?.shortened
                  ? `${formatDateKey(chartDateRange.startDate)} ~ ${formatDateKey(chartDateRange.endDate)}`
                  : `${selectedPerformanceRange.label} 기준`}
              </div>
            </div>
            {sortedProducts.map((product) => {
              const mdd = getMaxDrawdown(product.detail.rows, chartDateRange);
              const value = mdd?.pct ?? null;
              // MDD 값 아래에 낙폭 구간(고점일~저점일)을 작게 표기한다.
              const period =
                mdd?.peakDate && mdd?.troughDate
                  ? `${formatDateKey(mdd.peakDate)}~${formatDateKey(mdd.troughDate)}`
                  : null;
              return (
                <div key={`mdd-${tickerKey(product.item)}`} className={`compareMetricCell ${getSignedClass(value)}`}>
                  {formatPercent(value)}
                  {period ? <div className="compareMetricCellHint">({period})</div> : null}
                </div>
              );
            })}
            {Array.from({ length: Math.max(0, MAX_PRODUCTS - sortedProducts.length) }).map((_, index) => (
              <div key={`empty-mdd-${index}`} className="compareMetricCell">-</div>
            ))}

            <div className="compareMatrixLabel compareMetricsGroupLabel compareMetricsGroupLabelSingle">
              소르티노 지수
              <div className="compareMatrixLabelHint">
                {chartDateRange?.shortened
                  ? `${formatDateKey(chartDateRange.startDate)} ~ ${formatDateKey(chartDateRange.endDate)}`
                  : `${selectedPerformanceRange.label} 기준`}
              </div>
            </div>
            {sortedProducts.map((product) => {
              const value = getSortinoRatio(product.detail.rows, chartDateRange);
              return (
                <div key={`sortino-${tickerKey(product.item)}`} className={`compareMetricCell ${getSignedClass(value)}`}>
                  {value === null || Number.isNaN(value) ? "-" : value.toFixed(2)}
                </div>
              );
            })}
            {Array.from({ length: Math.max(0, MAX_PRODUCTS - sortedProducts.length) }).map((_, index) => (
              <div key={`empty-sortino-${index}`} className="compareMetricCell">-</div>
            ))}
            <div className="compareSharpeLegend">
              소르티노 지수 해석: <strong>0 미만</strong> 손실 · <strong>0~1</strong> 평범 · <strong>1~2</strong> 양호 · <strong>2~3</strong> 우수 · <strong>3 이상</strong> 매우 우수 (하방 변동성(Downside Deviation)만을 기준으로 손실 위험 대비 초과수익을 측정, 연율화)
            </div>
          </section>
        ) : activeTab === "monthly" ? (
          <section className="compareMatrix compareMatrixBody compareMatrixMonthlyScroll">
            {monthlyRows.length === 0 ? (
              <div className="compareMatrixWide" style={{ color: "var(--text-muted)" }}>
                월간 변동률을 계산할 데이터가 없습니다.
              </div>
            ) : (
              monthlyRows.map((month, monthIndex) => {
                // 해가 바뀌는 지점(위에서 아래로 내려가며 연도가 달라지는 첫 행)에 굵은 구분선을 넣는다.
                const prevMonth = monthIndex > 0 ? monthlyRows[monthIndex - 1] : null;
                const isYearBoundary = Boolean(prevMonth && prevMonth.slice(0, 4) !== month.slice(0, 4));
                const boundaryClass = isYearBoundary ? " compareMonthlyYearBoundary" : "";
                return (
                  <Fragment key={month}>
                    <div className={`compareMatrixLabel compareMatrixLabelWide compareBasicCompactLabel${boundaryClass}`}>
                      <div className="compareMatrixLabelText">{formatMonthLabel(month)}</div>
                    </div>
                    {sortedProducts.map((product) => {
                      const value = monthlyReturnMaps.get(tickerKey(product.item))?.[month] ?? null;
                      return (
                        <div
                          key={`monthly-${month}-${tickerKey(product.item)}`}
                          className={`compareMetricCell ${getSignedClass(value)}${boundaryClass}`}
                        >
                          {formatPercent(value)}
                        </div>
                      );
                    })}
                    {Array.from({ length: Math.max(0, MAX_PRODUCTS - sortedProducts.length) }).map((_, index) => (
                      <div key={`empty-monthly-${month}-${index}`} className={`compareMetricCell${boundaryClass}`}>
                        -
                      </div>
                    ))}
                  </Fragment>
                );
              })
            )}
          </section>
        ) : activeTab === "basic" ? (
          <section className="compareMatrix compareMatrixBody">
            {BASIC_INFO_METRICS.map((metric) => (
              <Fragment key={metric.label}>
                <div
                  className={metric.multiline ? "compareMatrixLabel compareMatrixLabelWide" : "compareMatrixLabel compareMatrixLabelWide compareBasicCompactLabel"}
                  style={metric.label === "포트폴리오 변동" ? { flexDirection: "column" } : undefined}
                >
                  <div className="compareMatrixLabelText">{metric.label}</div>
                  {metric.label === "포트폴리오 변동" && portfolioChangeBaseDate && (
                    <div className="compareMatrixLabelHint" style={{ fontSize: "11px", color: "var(--text-muted)", fontWeight: "normal", marginTop: "4px" }}>
                      ({formatKoreanDateLabel(portfolioChangeBaseDate)}
                      {portfolioChangeBaseIsOpen ? " 시초가" : ""} 이후)
                    </div>
                  )}
                </div>
                {sortedProducts.map((product) => (
                  <div
                    key={tickerKey(product.item)}
                    className={
                      metric.label === "포트폴리오 변동"
                        ? "compareBasicCell comparePortfolioCell"
                        : metric.multiline
                          ? "compareBasicCell"
                          : "compareBasicCell compareBasicCompactCell"
                    }
                  >
                    <BasicInfoValue product={product} metric={metric.label} />
                  </div>
                ))}
                {Array.from({ length: Math.max(0, MAX_PRODUCTS - sortedProducts.length) }).map((_, index) => (
                  <div key={`empty-basic-${metric.label}-${index}`} className={metric.multiline ? "compareProductEmpty" : "compareProductEmpty compareBasicCompactCell"}>-</div>
                ))}
              </Fragment>
            ))}
          </section>
        ) : (
          <section className="compareMatrix compareMatrixBody">
            <div className="compareMatrixLabel compareMatrixLabelWide" style={{ flexDirection: "column" }}>
              <div className="compareMatrixLabelText">포트폴리오 변동</div>
              {portfolioChangeBaseDate && (
                <div className="compareMatrixLabelHint" style={{ fontSize: "11px", color: "var(--text-muted)", fontWeight: "normal", marginTop: "4px" }}>
                  ({formatKoreanDateLabel(portfolioChangeBaseDate)} 이후)
                </div>
              )}
            </div>
            {sortedProducts.map((product) => (
              <div key={tickerKey(product.item)} className="compareBasicCell comparePortfolioCell">
                <BasicInfoValue product={product} metric="포트폴리오 변동" />
              </div>
            ))}
            {Array.from({ length: Math.max(0, MAX_PRODUCTS - sortedProducts.length) }).map((_, index) => (
              <div key={`empty-holding-portfolio-change-${index}`} className="compareProductEmpty">-</div>
            ))}
            <div className="compareMatrixLabel compareMatrixLabelWide compareHoldingsGroupLabel">구성종목</div>
            {sortedProducts.map((product) => {
              // 정렬 기준에 따라 표시용 holdings 를 만든다. 원본은 변형하지 않는다.
              const sortedHoldings = (() => {
                if (holdingsSortBy === "change") {
                  return [...product.detail.holdings].sort((a, b) => {
                    const av = a.change_pct ?? Number.NEGATIVE_INFINITY;
                    const bv = b.change_pct ?? Number.NEGATIVE_INFINITY;
                    return bv - av; // 상승률 내림차순
                  });
                }
                // 기본: 백엔드가 이미 비중순으로 내려준다고 가정. 안정성 위해 명시적 정렬.
                return [...product.detail.holdings].sort(
                  (a, b) => Number(b.weight ?? 0) - Number(a.weight ?? 0),
                );
              })();
              return (
                <div
                  key={tickerKey(product.item)}
                  className="compareHoldingScrollColumn"
                  style={{ maxHeight: "60vh", overflowY: "auto" }}
                >
                  {sortedHoldings.length === 0 ? (
                    <div className="compareHoldingCell">-</div>
                  ) : (
                    sortedHoldings.map((holding, idx) => {
                      const holdingCode = getHoldingCode(holding);
                      // 공통 종목 매칭 색상은 카드 좌측 strip 으로 표시 (배경/텍스트 색에 안 섞임).
                      const matchStripColor = holdingCode
                        ? holdingTextColorByCode.get(holdingCode)
                        : undefined;
                      const weight = Number(holding.weight ?? 0);
                      // 가운데 큰 비중% 색: 항상 슬레이트 그레이 (매칭 색은 좌측 strip 이 담당).
                      const weightTextColor = "#475569";
                      // 변동률 기반 배경: 양수=빨강, 음수=파랑. 5% 절댓값에서 alpha 최대치(0.18) 도달.
                      const changePct = holding.change_pct;
                      let changeBg: string | undefined;
                      if (changePct != null && !Number.isNaN(changePct) && changePct !== 0) {
                        const changeAlpha = Math.min(0.18, (Math.abs(changePct) / 5) * 0.18);
                        changeBg =
                          changePct > 0
                            ? `rgba(239, 68, 68, ${changeAlpha})`
                            : `rgba(37, 99, 235, ${changeAlpha})`;
                      }
                      const cellStyle = {
                        position: "relative",
                        overflow: "hidden",
                        backgroundColor: changeBg,
                        boxShadow: matchStripColor ? `inset 4px 0 0 0 ${matchStripColor}` : undefined,
                      } as const;
                      return (
                        <div
                          key={`${holdingCode || holding.ticker}-${idx}`}
                          className="compareHoldingCell"
                          style={cellStyle}
                        >
                          {/* 윗줄: 종목명 (전체 너비) */}
                          <div className="compareHoldingName" style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                            {holding.name || holding.ticker}
                          </div>
                          {/* 아랫줄: 티커 + 비중 + 변동률 */}
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "space-between",
                              gap: "0.5rem",
                              marginTop: "0.15rem",
                            }}
                          >
                            <div className="compareHoldingCode" style={{ flex: "0 0 auto" }}>
                              {holdingCode}
                            </div>
                            <span style={{ color: weightTextColor, fontWeight: 900, fontSize: "0.95rem" }}>
                              {weight.toFixed(2)}%
                            </span>
                            <span
                              className={getSignedClass(holding.change_pct ?? null)}
                              style={{ fontSize: "0.95rem", fontWeight: 800 }}
                            >
                              {formatSignedPercent(holding.change_pct ?? null)}
                            </span>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              );
            })}
            {Array.from({ length: Math.max(0, MAX_PRODUCTS - sortedProducts.length) }).map((_, index) => (
              <div key={`empty-holding-scroll-${index}`} className="compareHoldingCell">-</div>
            ))}
          </section>
        )}
      </div>
    </PageFrame>
  );
}
