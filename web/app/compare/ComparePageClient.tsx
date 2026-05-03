"use client";

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createChart, ColorType, CrosshairMode, LineSeries } from "lightweight-charts";
import type { IChartApi, LineData, Time } from "lightweight-charts";

import { PageFrame } from "../components/PageFrame";
import { ResponsiveFiltersSection } from "../components/ResponsiveFiltersSection";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { calcPortfolioChange, getCurrencyRegionLabel } from "@/lib/portfolio-change";

type CompareTab = "performance" | "basic" | "holdings";
type PerformanceRange = "1m" | "3m" | "6m" | "1y" | "3y";

type TickerItem = {
  ticker: string;
  name: string;
  ticker_type: string;
  country_code: string;
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

type SavedCompareState = {
  tickerType: string;
  selectedKeys: string[];
};

type PerformanceMetricRange =
  | { label: string; kind: "period"; days: number }
  | { label: string; kind: "ytd" };

const MAX_PRODUCTS = 5;
const COMPARE_STORAGE_KEY = "momentum-etf:compare:selected";
const COMPARE_SELECTED_TICKER_TYPE_KEY = "momentum-etf:compare:selected-ticker-type";
const COMPARE_STORAGE_KEY_PREFIX = "momentum-etf:compare:selected:";
const CHART_COLORS = ["#ef4444", "#2563eb", "#16a34a", "#f59e0b", "#7c3aed"];
const CHART_TINTS = [
  "rgba(239, 68, 68, 0.08)",
  "rgba(37, 99, 235, 0.08)",
  "rgba(22, 163, 74, 0.08)",
  "rgba(245, 158, 11, 0.08)",
  "rgba(124, 58, 237, 0.08)",
];
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
const PERFORMANCE_RANGES: { key: PerformanceRange; label: string; days: number }[] = [
  { key: "1m", label: "1개월", days: 31 },
  { key: "3m", label: "3개월", days: 92 },
  { key: "6m", label: "6개월", days: 183 },
  { key: "1y", label: "1년", days: 365 },
  { key: "3y", label: "3년", days: 365 * 3 },
];
const PERFORMANCE_METRIC_RANGES: PerformanceMetricRange[] = [
  ...PERFORMANCE_RANGES.map(({ label, days }) => ({ label, kind: "period" as const, days })),
  { label: "연초이후", kind: "ytd" },
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

function readSavedCompareState(): SavedCompareState | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(COMPARE_STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<SavedCompareState>;
    if (!parsed.tickerType || !Array.isArray(parsed.selectedKeys)) return null;
    return {
      tickerType: parsed.tickerType,
      selectedKeys: parsed.selectedKeys.filter((key): key is string => typeof key === "string").slice(0, MAX_PRODUCTS),
    };
  } catch {
    return null;
  }
}

function getComparePoolStorageKey(tickerType: string): string {
  return `${COMPARE_STORAGE_KEY_PREFIX}${tickerType}`;
}

function readSavedTickerType(): string | null {
  if (typeof window === "undefined") return null;
  const savedTickerType = window.localStorage.getItem(COMPARE_SELECTED_TICKER_TYPE_KEY);
  if (savedTickerType) return savedTickerType;
  return readSavedCompareState()?.tickerType ?? null;
}

function readSavedCompareKeys(tickerType: string): string[] {
  if (typeof window === "undefined" || !tickerType) return [];
  const raw = window.localStorage.getItem(getComparePoolStorageKey(tickerType));
  if (raw) {
    try {
      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed)) {
        return parsed.filter((key): key is string => typeof key === "string").slice(0, MAX_PRODUCTS);
      }
    } catch {
      return [];
    }
  }
  const legacyState = readSavedCompareState();
  if (legacyState?.tickerType === tickerType) {
    return legacyState.selectedKeys;
  }
  return [];
}

function writeSavedCompareState(tickerType: string, selectedKeys: string[]): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(COMPARE_SELECTED_TICKER_TYPE_KEY, tickerType);
  window.localStorage.setItem(getComparePoolStorageKey(tickerType), JSON.stringify(selectedKeys.slice(0, MAX_PRODUCTS)));
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

// getCurrencyRegionLabel は @/lib/portfolio-change から import

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

function getPortfolioChange(detail: TickerDetailResponse): {
  totalPct: number | null;
  breakdown: { currency: string; label: string; changePct: number; weight: number }[];
} {
  const result = calcPortfolioChange(
    detail.holdings ?? [],
    detail.etf_info?.fx_rates ?? [],
  );
  return {
    totalPct: result.totalPct,
    breakdown: result.breakdown.map((item) => ({
      currency: item.currency,
      label: item.label,
      changePct: item.change_pct,
      weight: item.weight,
    })),
  };
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

function getChartDateRange(products: SelectedProduct[], calendarDays: number): ChartDateRange | null {
  const pricedRowsList = products.map((product) => getPricedRows(product.detail.rows));
  if (pricedRowsList.length === 0 || pricedRowsList.some((rows) => rows.length < 2)) return null;

  const startTimes = pricedRowsList.map((rows) => new Date(rows[0].date).getTime());
  const endTimes = pricedRowsList.map((rows) => new Date(rows.at(-1)?.date ?? "").getTime());
  if (startTimes.some(Number.isNaN) || endTimes.some(Number.isNaN)) return null;

  const comparableEnd = new Date(Math.min(...endTimes));
  const requestedStart = new Date(comparableEnd);
  requestedStart.setDate(requestedStart.getDate() - calendarDays);

  const firstCommonStart = new Date(Math.max(...startTimes));
  const comparableStart = firstCommonStart > requestedStart ? firstCommonStart : requestedStart;
  if (comparableStart >= comparableEnd) return null;

  return {
    startDate: toDateKey(comparableStart),
    endDate: toDateKey(comparableEnd),
    shortened: firstCommonStart > requestedStart,
  };
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

async function loadTickerDetail(item: TickerItem): Promise<TickerDetailResponse> {
  const params = new URLSearchParams({
    ticker: item.ticker,
    ticker_type: item.ticker_type,
    country_code: item.country_code,
  });
  const response = await fetch(`/api/ticker-detail?${params.toString()}`, { cache: "no-store" });
  const payload = (await response.json()) as TickerDetailResponse;
  if (!response.ok || payload.error) {
    throw new Error(payload.error ?? "비교 데이터를 불러오지 못했습니다.");
  }
  return payload;
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
    return (
      <div className="compareBasicValue">
        <strong>{formatPrice(etfInfo?.nav ?? null, "kor")}</strong>
        <span className={getSignedClass(etfInfo?.nav_change ?? null)}>
          {formatSignedPriceDelta(etfInfo?.nav_change ?? null, "kor")}
        </span>
        <span className={getSignedClass(etfInfo?.nav_change_pct ?? null)}>
          {formatSignedPercent(etfInfo?.nav_change_pct ?? null)}
        </span>
      </div>
    );
  }

  if (metric === "괴리율") {
    return <strong>{formatSignedPercent(etfInfo?.deviation ?? null)}</strong>;
  }

  if (metric === "거래량") {
    return <strong>{formatVolume(etfInfo?.volume ?? latestRow?.volume ?? null)}</strong>;
  }

  if (metric === "운용보수") {
    return <strong>{formatRatioPercent(etfInfo?.expense_ratio ?? null)}</strong>;
  }

  if (metric === "시가총액") {
    return <strong>{formatEokFromKrw(etfInfo?.market_cap_krw ?? null)}</strong>;
  }

  if (metric === "배당 수익률") {
    return <strong>{formatRatioPercent(etfInfo?.dividend_yield_ttm ?? null)}</strong>;
  }

  if (metric === "포트폴리오 변동") {
    const fxRateByCurrency = new Map<string, TickerFxRate>();
    (etfInfo?.fx_rates ?? []).forEach((fx) => {
      const currency = String(fx.currency || "").trim().toUpperCase();
      if (currency) fxRateByCurrency.set(currency, fx);
    });
    return (
      <div className="compareBasicValue">
        <strong className={getSignedClass(portfolioChange.totalPct)}>{formatSignedPercent(portfolioChange.totalPct)}</strong>
        {portfolioChange.breakdown.length > 0 ? (
          <span className="comparePortfolioBreakdownList">
            {portfolioChange.breakdown.map((item) => (
              <span key={item.currency} className="comparePortfolioBreakdownItem">
                <span>{item.label}({formatNumber(item.weight, 0)}%)</span>
                <span className={getSignedClass(item.changePct)}>{formatSignedPercent(item.changePct)}</span>
                {fxRateByCurrency.has(item.currency) ? (
                  <span className="comparePortfolioBreakdownFx">
                    · 환율{" "}
                    <span className={getSignedClass(fxRateByCurrency.get(item.currency)?.change_pct ?? null)}>
                      {formatSignedPercent(fxRateByCurrency.get(item.currency)?.change_pct ?? null)}
                    </span>
                  </span>
                ) : null}
              </span>
            ))}
          </span>
        ) : null}
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
  const [tickerTypes, setTickerTypes] = useState<TickerTypeItem[]>([]);
  const [selectedTickerType, setSelectedTickerType] = useState("");
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  const [products, setProducts] = useState<SelectedProduct[]>([]);
  const [activeTab, setActiveTab] = useState<CompareTab>("performance");
  const [performanceRange, setPerformanceRange] = useState<PerformanceRange>("3m");
  const [searchText, setSearchText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const hydratedRef = useRef(false);

  const itemByKey = useMemo(() => {
    const map = new Map<string, TickerItem>();
    tickerItems.forEach((item) => map.set(tickerKey(item), item));
    return map;
  }, [tickerItems]);

  const poolTickerItems = useMemo(
    () => tickerItems.filter((item) => item.ticker_type === selectedTickerType),
    [selectedTickerType, tickerItems],
  );

  const loadSelectedProducts = useCallback(async (keys: string[]) => {
    const items = keys.map((key) => itemByKey.get(key)).filter((item): item is TickerItem => Boolean(item));
    if (items.length === 0) {
      setProducts([]);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const details = await Promise.all(items.map((item) => loadTickerDetail(item)));
      setProducts(items.map((item, index) => ({ item, detail: details[index] })));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "비교 데이터를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [itemByKey]);

  useEffect(() => {
    let alive = true;
    async function loadInitialData() {
      try {
        const [tickersResponse, stocksResponse] = await Promise.all([
          fetch("/api/ticker-tickers", { cache: "no-store" }),
          fetch("/api/stocks", { cache: "no-store" }),
        ]);
        const tickersPayload = (await tickersResponse.json()) as TickerItem[] | { error?: string };
        const stocksPayload = (await stocksResponse.json()) as StocksTableData | { error?: string };
        if (!tickersResponse.ok || !Array.isArray(tickersPayload)) {
          throw new Error(Array.isArray(tickersPayload) ? "종목 목록을 불러오지 못했습니다." : tickersPayload.error);
        }
        if (!stocksResponse.ok || !("ticker_types" in stocksPayload)) {
          throw new Error("종목풀 목록을 불러오지 못했습니다.");
        }
        if (!alive) return;
        const items = tickersPayload.filter((item) => item.ticker && item.ticker_type && item.country_code);
        const types = stocksPayload.ticker_types ?? [];
        setTickerItems(items);
        setTickerTypes(types);
        const savedTickerType = readSavedTickerType();
        const savedType = savedTickerType
          ? types.find((type) => type.ticker_type === savedTickerType && items.some((item) => item.ticker_type === savedTickerType))
          : null;
        const initialType = savedType ?? types.find((type) => items.some((item) => item.ticker_type === type.ticker_type));
        if (initialType) {
          setSelectedTickerType(initialType.ticker_type);
          const savedKeys = readSavedCompareKeys(initialType.ticker_type);
          const validKeys = savedKeys.filter((key) => {
            const item = items.find((candidate) => tickerKey(candidate) === key);
            return item?.ticker_type === initialType.ticker_type;
          });
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

  useEffect(() => {
    void loadSelectedProducts(selectedKeys);
  }, [loadSelectedProducts, selectedKeys]);

  useEffect(() => {
    if (!hydratedRef.current || !selectedTickerType) return;
    writeSavedCompareState(selectedTickerType, selectedKeys);
  }, [selectedKeys, selectedTickerType]);

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

  const selectedPerformanceRange = PERFORMANCE_RANGES.find((range) => range.key === performanceRange) ?? PERFORMANCE_RANGES[1];
  const sortedProducts = useMemo(() => {
    return products
      .map((product, index) => ({
        product,
        index,
        returnPct: getReturnPct(product.detail.rows, selectedPerformanceRange.days),
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
  }, [products, selectedPerformanceRange.days]);
  const chartDateRange = useMemo(
    () => getChartDateRange(sortedProducts, selectedPerformanceRange.days),
    [selectedPerformanceRange.days, sortedProducts],
  );
  
  const portfolioChangeBaseDate = useMemo(() => {
    for (const p of sortedProducts) {
      if (p.detail.etf_info?.portfolio_change_base_date) {
        return p.detail.etf_info.portfolio_change_base_date;
      }
    }
    return null;
  }, [sortedProducts]);

  const holdingExposureRows = useMemo(() => buildHoldingExposureRows(sortedProducts).slice(0, 10), [sortedProducts]);
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

  const titleRight = (
    <div className="compareTitleMeta">
      <span>비교 상품:</span>
      <span>{products.length}/{MAX_PRODUCTS}</span>
    </div>
  );

  return (
    <PageFrame title="종목 비교" fullHeight fullWidth titleRight={titleRight}>
      <div className="appPageStack appPageStackFill comparePage">
        {error ? <div className="alert alert-danger mb-0">{error}</div> : null}

        <section className="appSection">
          <div className="card appCard">
            <div className="card-header">
              <ResponsiveFiltersSection>
                <div className="appMainHeader">
                  <div className="appMainHeaderLeft compareMainHeaderLeft">
                    <label className="appLabeledField">
                      <span className="appLabeledFieldLabel">종목풀</span>
                      <select
                        className="form-select"
                        value={selectedTickerType}
                        onChange={(event) => {
                          const nextTickerType = event.target.value;
                          const savedKeys = readSavedCompareKeys(nextTickerType);
                          const validKeys = savedKeys.filter((key) => itemByKey.get(key)?.ticker_type === nextTickerType);
                          setSelectedTickerType(nextTickerType);
                          setProducts([]);
                          setSelectedKeys(validKeys.slice(0, MAX_PRODUCTS));
                          setSearchText("");
                        }}
                        disabled={tickerTypes.length === 0}
                      >
                        {tickerTypes.length === 0 ? (
                          <option value="">종목풀 불러오는 중...</option>
                        ) : (
                          tickerTypes.map((type) => (
                            <option key={type.ticker_type} value={type.ticker_type}>
                              {type.icon ? `${type.icon} ` : ""}{type.name}
                            </option>
                          ))
                        )}
                      </select>
                    </label>
                    <ProductSearchField
                      value={searchText}
                      items={poolTickerItems}
                      disabledKeys={selectedKeys}
                      inputRef={searchInputRef}
                      onChange={setSearchText}
                    />
                    <div className="compareHeaderTabs appSegmentedToggle" role="group" aria-label="비교 보기 선택">
                      <button
                        type="button"
                        className={activeTab === "performance" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                        onClick={() => setActiveTab("performance")}
                      >
                        성과분석
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
                  </div>
                </div>
              </ResponsiveFiltersSection>
            </div>
          </div>
        </section>

        {loading ? <div className="compareLoading">비교 데이터를 불러오는 중...</div> : null}

        <section className={activeTab === "holdings" ? "compareMatrix compareMatrixWithTotal" : "compareMatrix"}>
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
          {activeTab === "holdings" ? (
            <div className="compareProductEmpty compareProductHeaderEmpty compareHoldingTotalHeader">합계</div>
          ) : null}
        </section>

        {activeTab === "performance" ? (
          <section className="compareMatrix compareMatrixBody">
            <div className="compareMatrixLabel compareMatrixLabelWide">수익률 추이</div>
            <div className="compareMatrixWide">
              <div className="comparePerformanceToolbar">
                <div className="appSegmentedToggle" role="group" aria-label="수익률 추이 기간">
                  {PERFORMANCE_RANGES.map((range) => (
                    <button
                      key={range.key}
                      type="button"
                      className={performanceRange === range.key ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                      onClick={() => setPerformanceRange(range.key)}
                    >
                      {range.label}
                    </button>
                  ))}
                </div>
                {chartDateRange?.shortened ? (
                  <span className="comparePerformanceHint is-warning">
                    선택 기간보다 데이터가 짧아 {formatDateKey(chartDateRange.startDate)} ~ {formatDateKey(chartDateRange.endDate)} 기준 비교
                  </span>
                ) : null}
              </div>
              <CompareChart products={sortedProducts} dateRange={chartDateRange} />
            </div>
            <div className="compareMatrixLabel compareMetricsGroupLabel" style={{ gridRow: `span ${PERFORMANCE_METRIC_RANGES.length}` }}>
              수익률(%)
            </div>
            {PERFORMANCE_METRIC_RANGES.map((period) => (
              <Fragment key={period.label}>
                <div className="compareMetricPeriodLabel">{period.label}</div>
                {sortedProducts.map((product) => {
                  const value = getMetricReturnPct(product.detail.rows, period);
                  return (
                    <div key={tickerKey(product.item)} className={`compareMetricCell ${getSignedClass(value)}`}>
                      {formatPercent(value)}
                    </div>
                  );
                })}
                {Array.from({ length: Math.max(0, MAX_PRODUCTS - sortedProducts.length) }).map((_, index) => (
                  <div key={`empty-metric-${period.label}-${index}`} className="compareMetricCell">-</div>
                ))}
              </Fragment>
            ))}
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
                    <div className="compareMatrixLabelHint" style={{ fontSize: "11px", color: "#64748b", fontWeight: "normal", marginTop: "4px" }}>
                      ({formatKoreanDateLabel(portfolioChangeBaseDate)} 이후)
                    </div>
                  )}
                </div>
                {sortedProducts.map((product) => (
                  <div key={tickerKey(product.item)} className={metric.multiline ? "compareBasicCell" : "compareBasicCell compareBasicCompactCell"}>
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
          <section className="compareMatrix compareMatrixBody compareMatrixWithTotal">
            <div className="compareMatrixLabel compareMatrixLabelWide compareHoldingCountLabel">구성종목 수</div>
            {sortedProducts.map((product) => (
              <div key={tickerKey(product.item)} className="compareHoldingCount">{product.detail.holdings.length}</div>
            ))}
            {Array.from({ length: Math.max(0, MAX_PRODUCTS - sortedProducts.length) }).map((_, index) => (
              <div key={`empty-count-${index}`} className="compareProductEmpty compareHoldingCountEmpty" />
            ))}
            <div className="compareHoldingCount compareHoldingTotalCount">합계비중</div>
            <div className="compareMatrixLabel compareHoldingsGroupLabel" style={{ gridRow: "span 10" }}>종목비중 TOP10</div>
            {Array.from({ length: 10 }).map((_, rowIndex) => (
              <Fragment key={rowIndex}>
                <div className="compareHoldingRankLabel">{rowIndex + 1}</div>
                {sortedProducts.map((product) => {
                  const holding = product.detail.holdings[rowIndex];
                  const holdingCode = holding ? getHoldingCode(holding) : "";
                  const matchColor = holdingCode ? holdingColorByCode.get(holdingCode) : undefined;
                  return (
                    <div
                      key={tickerKey(product.item)}
                      className={matchColor ? "compareHoldingCell is-matched" : "compareHoldingCell"}
                      style={matchColor ? { backgroundColor: matchColor } : undefined}
                    >
                      {holding ? (
                        <>
                          <div className="compareHoldingName">{holding.name || holding.ticker}</div>
                          <div className="compareHoldingCode">{holdingCode}</div>
                          <div className="compareHoldingFooter">
                            <span className={getSignedClass(holding.change_pct ?? null)}>
                              {formatSignedPercent(holding.change_pct ?? null)}
                            </span>
                            <strong>{Number(holding.weight ?? 0).toFixed(2)}%</strong>
                          </div>
                        </>
                      ) : "-"}
                    </div>
                  );
                })}
                {Array.from({ length: Math.max(0, MAX_PRODUCTS - sortedProducts.length) }).map((_, index) => (
                  <div key={`empty-holding-${rowIndex}-${index}`} className="compareHoldingCell">-</div>
                ))}
                {holdingExposureRows[rowIndex] ? (
                  <div
                    className={
                      holdingColorByCode.get(holdingExposureRows[rowIndex].code)
                        ? "compareHoldingCell compareHoldingTotalCell is-matched"
                        : "compareHoldingCell compareHoldingTotalCell"
                    }
                    style={
                      holdingColorByCode.get(holdingExposureRows[rowIndex].code)
                        ? { backgroundColor: holdingColorByCode.get(holdingExposureRows[rowIndex].code) }
                        : undefined
                    }
                  >
                    <div className="compareHoldingName">{holdingExposureRows[rowIndex].name}</div>
                    <div className="compareHoldingCode">{holdingExposureRows[rowIndex].code}</div>
                    <div className="compareHoldingFooter">
                      <span className={getSignedClass(holdingExposureRows[rowIndex].changePct)}>
                        {formatSignedPercent(holdingExposureRows[rowIndex].changePct)}
                      </span>
                      <strong>{holdingExposureRows[rowIndex].totalWeight.toFixed(2)}%</strong>
                    </div>
                  </div>
                ) : (
                  <div className="compareHoldingCell compareHoldingTotalCell">-</div>
                )}
              </Fragment>
            ))}
          </section>
        )}
      </div>
    </PageFrame>
  );
}
