"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { IconCheck, IconPlus } from "@tabler/icons-react";
import { useSearchParams } from "next/navigation";
import {
  createChart,
  ColorType,
  CrosshairMode,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
} from "lightweight-charts";
import type {
  IChartApi,
  LineData,
  Logical,
  Time,
  MouseEventParams,
} from "lightweight-charts";
import type { ColDef, GridApi, GridReadyEvent } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";
import { useToast } from "../components/ToastProvider";
import { persistRecentTickerSearch } from "@/lib/recent-ticker-searches";
import { addStockCandidate } from "@/lib/stocks-store";
import { readRememberedTickerType } from "../components/account-selection";
import { createAppGridTheme } from "../components/app-grid-theme";

// --- 타입 ---

type TickerItem = {
  ticker: string;
  name: string;
  ticker_type: string;
  country_code: string;
  is_etf?: boolean;
  has_holdings?: boolean;
};

type PriceRow = {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
  change_pct: number | null;
};

type MonthlyPriceRow = {
  month: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
  change_pct: number | null;
};

type ChartInterval = "day" | "week" | "month";
type HistoryTab = "daily" | "monthly";

type TickerDetailResponse = {
  ticker: string;
  rows: PriceRow[];
  etf_info?: TickerEtfInfo | null;
  holdings: TickerHoldingRow[];
  holdings_as_of_date?: string | null;
  holdings_price_as_of_date?: string | null;
  holdings_error?: string | null;
  error?: string;
};

type TickerEtfInfo = {
  nav?: number | null;
  nav_change?: number | null;
  nav_change_pct?: number | null;
  deviation?: number | null;
  expense_ratio?: number | null;
  dividend_yield_ttm?: number | null;
  total_net_assets_eok?: number | null;
  market_cap_krw?: number | null;
  volume?: number | null;
  fx_rate?: number | null;
  fx_change_pct?: number | null;
  fx_rates?: TickerFxRate[];
};

type TickerFxRate = {
  currency: string;
  rate?: number | null;
  change_pct?: number | null;
};

type PortfolioChangeBreakdownItem = {
  currency: string;
  label: string;
  change_pct: number;
  weight: number;
};

type TickerHoldingRow = {
  ticker: string;
  name: string;
  contracts: number | null;
  amount: number | null;
  raw_code?: string | null;
  raw_name?: string | null;
  reuters_code?: string | null;
  yahoo_symbol?: string | null;
  current_price?: number | null;
  previous_close?: number | null;
  change_pct?: number | null;
  price_currency?: string | null;
  weight: number | null;
  is_us_pool_candidate?: boolean;
  in_us_pool?: boolean;
  is_kor_pool_candidate?: boolean;
  in_kor_pool?: boolean;
};

type TickerResolveItem = {
  ticker: string;
  name: string;
  ticker_type: string;
  country_code: string;
  is_etf?: boolean;
  has_holdings?: boolean;
};

type CrosshairInfo = {
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  change_pct: number | null;
};

type ChartRangeBadge = {
  text: string;
  left: number;
  top: number;
  anchorLeft: number;
  tone: "high" | "low";
};

// --- 상수 ---

const MA_PERIODS = [
  { period: 5, color: "#2196F3", label: "5" },
  { period: 20, color: "#FF9800", label: "20" },
  { period: 60, color: "#E91E63", label: "60" },
  { period: 120, color: "#9C27B0", label: "120" },
];

const gridTheme = createAppGridTheme({
  rowHeight: 34,
  headerHeight: 36,
});

// --- 유틸 ---

function formatNumber(value: number | null, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
}

function formatPercent(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value.toFixed(2)}%`;
}

function formatUnsignedPercent(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value.toFixed(2)}%`;
}

function formatRatioPercent(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value.toFixed(2)}%`;
}

function formatEok(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  const jo = Math.floor(value / 10_000);
  const eok = Math.round(value % 10_000);
  if (jo <= 0) {
    return `${formatNumber(eok, 0)}억`;
  }
  if (eok <= 0) {
    return `${formatNumber(jo, 0)}조`;
  }
  return `${formatNumber(jo, 0)}조 ${formatNumber(eok, 0)}억`;
}

function formatEokFromKrw(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return formatEok(value / 100_000_000);
}

function formatSignedPriceDelta(value: number | null, countryCode: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const absValue = Math.abs(value);
  if (absValue === 0) {
    return "0";
  }
  return `${value > 0 ? "▲ " : "▼ "}${formatTickerPrice(absValue, countryCode)}`;
}

function formatDateWithWeekday(value: string): string {
  if (!value) return "-";
  const date = new Date(`${value}T00:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  const weekdays = ["일", "월", "화", "수", "목", "금", "토"];
  return `${value}(${weekdays[date.getDay()]})`;
}

function getSignedClass(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) return "";
  return value > 0 ? "metricPositive" : "metricNegative";
}

function calculateMA(data: PriceRow[], period: number): LineData[] {
  const result: LineData[] = [];
  for (let i = period - 1; i < data.length; i++) {
    let sum = 0;
    let count = 0;
    for (let j = i - period + 1; j <= i; j++) {
      if (data[j].close !== null) { sum += data[j].close!; count++; }
    }
    if (count === period) {
      result.push({ time: data[i].date as Time, value: sum / count });
    }
  }
  return result;
}

function getWeekBucketStart(value: string): string {
  const date = new Date(`${value}T00:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  const day = date.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  date.setDate(date.getDate() + diff);
  return date.toISOString().slice(0, 10);
}

function aggregatePriceRows(data: PriceRow[], bucket: "week" | "month"): PriceRow[] {
  const aggregatedRows: PriceRow[] = [];
  let currentKey = "";
  let currentRow: PriceRow | null = null;
  let previousClose: number | null = null;

  for (const row of data) {
    const nextKey = bucket === "week" ? getWeekBucketStart(row.date) : row.date.slice(0, 7);
    if (!nextKey) {
      continue;
    }

    if (nextKey !== currentKey) {
      if (currentRow) {
        if (currentRow.close !== null && previousClose !== null && previousClose !== 0) {
          currentRow.change_pct = Number((((currentRow.close - previousClose) / previousClose) * 100).toFixed(2));
        }
        if (currentRow.close !== null) {
          previousClose = currentRow.close;
        }
        aggregatedRows.push(currentRow);
      }

      currentKey = nextKey;
      currentRow = { ...row };
      continue;
    }

    if (!currentRow) {
      continue;
    }

    if (currentRow.open === null && row.open !== null) {
      currentRow.open = row.open;
    }
    if (row.high !== null) {
      currentRow.high = currentRow.high === null ? row.high : Math.max(currentRow.high, row.high);
    }
    if (row.low !== null) {
      currentRow.low = currentRow.low === null ? row.low : Math.min(currentRow.low, row.low);
    }
    if (row.close !== null) {
      currentRow.close = row.close;
    }
    if (row.volume !== null) {
      currentRow.volume = (currentRow.volume ?? 0) + row.volume;
    }
    currentRow.date = row.date;
  }

  if (currentRow) {
    if (currentRow.close !== null && previousClose !== null && previousClose !== 0) {
      currentRow.change_pct = Number((((currentRow.close - previousClose) / previousClose) * 100).toFixed(2));
    }
    aggregatedRows.push(currentRow);
  }

  return aggregatedRows;
}

function aggregateMonthlyRows(data: PriceRow[]): MonthlyPriceRow[] {
  return aggregatePriceRows(data, "month").map((row) => ({
    month: row.date.slice(0, 7),
    open: row.open,
    high: row.high,
    low: row.low,
    close: row.close,
    volume: row.volume,
    change_pct: row.change_pct,
  }));
}

function formatTickerPrice(value: number | null, countryCode: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }

  const normalized = String(countryCode || "").trim().toLowerCase();
  if (normalized === "au" || normalized === "aud") {
    return `A$${new Intl.NumberFormat("en-AU", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)}`;
  }

  if (normalized === "us" || normalized === "usd") {
    return `$${new Intl.NumberFormat("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)}`;
  }

  if (normalized === "eur") {
    return `€${new Intl.NumberFormat("de-DE", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)}`;
  }

  if (normalized === "twd") {
    return `${new Intl.NumberFormat("zh-TW", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)} TWD`;
  }

  if (normalized === "hkd") {
    return `HK$${new Intl.NumberFormat("en-HK", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)}`;
  }

  if (normalized === "jpy") {
    return `¥${new Intl.NumberFormat("ja-JP", {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value)}`;
  }

  if (normalized === "gbp") {
    return `£${new Intl.NumberFormat("en-GB", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)}`;
  }

  if (normalized === "cny") {
    return `${new Intl.NumberFormat("zh-CN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)} CNY`;
  }

  return `${new Intl.NumberFormat("ko-KR", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value)}원`;
}

function buildRangeBadgeText(
  anchorPrice: number,
  currentPrice: number,
  date: string,
  countryCode: string,
): string {
  const pct = anchorPrice === 0 ? null : ((currentPrice / anchorPrice) - 1) * 100;
  return `${formatTickerPrice(anchorPrice, countryCode)} (${formatPercent(pct)}, ${date})`;
}

function getCurrencyRegionLabel(currency: string): string {
  const normalized = String(currency || "").trim().toUpperCase();
  if (normalized === "KRW") return "국내";
  if (normalized === "TWD") return "대만";
  if (normalized === "JPY") return "일본";
  if (normalized === "USD") return "미국";
  if (normalized === "HKD") return "홍콩";
  if (normalized === "CNY") return "중국";
  if (normalized === "AUD") return "호주";
  if (normalized === "GBP") return "영국";
  if (normalized === "EUR") return "유럽";
  return normalized;
}

function getInitialVisibleLogicalRange(
  data: PriceRow[],
  interval: ChartInterval,
): { from: number; to: number } | null {
  if (data.length === 0) {
    return null;
  }

  const lastIndex = data.length - 1;
  if (interval === "month") {
    return {
      from: -3,
      to: lastIndex + 3.5,
    };
  }

  const lastRow = data[lastIndex];

  const lastDate = new Date(`${lastRow.date}T00:00:00`);
  if (Number.isNaN(lastDate.getTime())) {
    return null;
  }

  const startDate = new Date(lastDate);
  if (interval === "day") {
    startDate.setFullYear(startDate.getFullYear() - 1);
  } else {
    startDate.setFullYear(startDate.getFullYear() - 2);
  }

  const startDateText = startDate.toISOString().slice(0, 10);
  const visibleStartIndex = data.findIndex((row) => row.date >= startDateText);
  return {
    from: Math.max((visibleStartIndex >= 0 ? visibleStartIndex : 0) - 1, -1),
    to: lastIndex + 2.5,
  };
}

// --- 컴포넌트 ---

export function TickerDetailManager({
}: {
  }) {
  const searchParams = useSearchParams();
  const toast = useToast();

  // URL query params
  const qTicker = searchParams.get("ticker") ?? "";
  const qTickerType = searchParams.get("ticker_type") ?? "";
  const qCountryCode = searchParams.get("country_code") ?? "";
  const qName = searchParams.get("name") ?? "";

  // 전체 종목 목록
  const [allTickers, setAllTickers] = useState<TickerItem[]>([]);

  // 현재 선택된 종목
  const [selectedTicker, setSelectedTicker] = useState<TickerItem | null>(null);

  // 데이터
  const [rows, setRows] = useState<PriceRow[]>([]);
  const [holdings, setHoldings] = useState<TickerHoldingRow[]>([]);
  const [etfInfo, setEtfInfo] = useState<TickerEtfInfo | null>(null);
  const [holdingsAsOfDate, setHoldingsAsOfDate] = useState<string | null>(null);
  const [holdingsPriceAsOfDate, setHoldingsPriceAsOfDate] = useState<string | null>(null);
  const [holdingsError, setHoldingsError] = useState<string | null>(null);
  const [addingPoolKeys, setAddingPoolKeys] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const holdingsGridApiRef = useRef<GridApi<TickerHoldingRow> | null>(null);
  const addingPoolKeysRef = useRef<string[]>([]);
  const [chartInterval, setChartInterval] = useState<ChartInterval>("day");
  const [historyTab, setHistoryTab] = useState<HistoryTab>("daily");

  // 차트
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [crosshairInfo, setCrosshairInfo] = useState<CrosshairInfo | null>(null);
  const [chartBadges, setChartBadges] = useState<ChartRangeBadge[]>([]);

  // --- 종목 목록 로드 ---

  useEffect(() => {
    let alive = true;
    async function fetchAll() {
      try {
        const res = await fetch("/api/ticker-tickers", { cache: "no-store" });
        if (!res.ok) return;
        const data = (await res.json()) as TickerItem[];
        if (alive && Array.isArray(data)) setAllTickers(data);
      } catch { /* 무시 */ }
    }
    fetchAll();
    return () => { alive = false; };
  }, []);

  // URL에서 ticker가 있으면 자동 조회
  useEffect(() => {
    if (!qTicker) {
      setSelectedTicker(null);
      setRows([]);
      setHoldings([]);
      setHoldingsAsOfDate(null);
      setHoldingsPriceAsOfDate(null);
      setHoldingsError(null);
      setError(null);
      return;
    }

    if (qTickerType) {
      const item: TickerItem = { ticker: qTicker, name: qName, ticker_type: qTickerType, country_code: qCountryCode };
      setSelectedTicker(item);
      void loadTickerData(item);
      return;
    }

    if (allTickers.length === 0) {
      return;
    }

    const matches = allTickers.filter((item) => item.ticker.toLowerCase() === qTicker.toLowerCase());
    if (matches.length === 1) {
      setSelectedTicker(matches[0]);
      void loadTickerData(matches[0]);
      return;
    }

    setRows([]);
    setHoldings([]);
    setEtfInfo(null);
    setHoldingsAsOfDate(null);
    setHoldingsPriceAsOfDate(null);
    setHoldingsError(null);
    if (matches.length > 1) {
      const rememberedType = readRememberedTickerType();
      const bestMatch = matches.find((m) => m.ticker_type === rememberedType);

      if (bestMatch) {
        setSelectedTicker(bestMatch);
        void loadTickerData(bestMatch);
        return;
      }

      setSelectedTicker(null);
      setError(`동일한 티커 ${qTicker}가 여러 종목풀(${matches.map(m => m.ticker_type).join(", ")})에 등록되어 있습니다.`);
      return;
    }

    void (async () => {
      try {
        const response = await fetch(`/api/ticker-resolve?ticker=${encodeURIComponent(qTicker)}`, {
          cache: "no-store",
        });
        const resolved = (await response.json()) as TickerResolveItem & { error?: string };
        if (!response.ok) {
          throw new Error(resolved.error || `${qTicker} 티커를 찾지 못했습니다.`);
        }
        const resolvedItem: TickerItem = {
          ticker: resolved.ticker,
          name: resolved.name,
          ticker_type: resolved.ticker_type,
          country_code: resolved.country_code,
          is_etf: resolved.is_etf,
          has_holdings: resolved.has_holdings,
        };
        setSelectedTicker(resolvedItem);
        await loadTickerData(resolvedItem);
      } catch (error) {
        setSelectedTicker(null);
        setError(error instanceof Error ? error.message : `${qTicker} 티커를 찾지 못했습니다.`);
      }
    })();
  }, [allTickers, qTicker, qTickerType, qCountryCode, qName]);

  // --- 데이터 로드 ---

  async function loadTickerData(item: TickerItem) {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);
    setRows([]);
    setHoldings([]);
    setHoldingsAsOfDate(null);
    setHoldingsPriceAsOfDate(null);
    setHoldingsError(null);
    setAddingPoolKeys([]);
    setCrosshairInfo(null);
    setChartBadges([]);

    try {
      const search = new URLSearchParams({
        ticker: item.ticker,
      });
      if (item.ticker_type) {
        search.set("ticker_type", item.ticker_type);
      }
      if (item.country_code) {
        search.set("country_code", item.country_code);
      }
      const response = await fetch(`/api/ticker-detail?${search.toString()}`, {
        cache: "no-store",
        signal: controller.signal,
      });
      const payload = (await response.json()) as TickerDetailResponse;
      if (!response.ok) throw new Error(payload.error ?? "데이터를 불러오지 못했습니다.");
      if (payload.error) throw new Error(payload.error);
      const latestRow = payload.rows[payload.rows.length - 1] ?? null;
      const matchedItem =
        allTickers.find(
          (candidate) => candidate.ticker === item.ticker && candidate.ticker_type === item.ticker_type,
        ) ?? item;
      setRows(payload.rows);
      setEtfInfo(payload.etf_info ?? null);
      setHoldings(payload.holdings ?? []);
      setHoldingsAsOfDate(payload.holdings_as_of_date ?? null);
      setHoldingsPriceAsOfDate(payload.holdings_price_as_of_date ?? null);
      setHoldingsError(payload.holdings_error ?? null);
      persistRecentTickerSearch({
        ticker: matchedItem.ticker,
        name: matchedItem.name,
        ticker_type: matchedItem.ticker_type,
        country_code: matchedItem.country_code,
        current_price: latestRow?.close ?? null,
        change_pct: latestRow?.change_pct ?? null,
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof Error ? err.message : "데이터를 불러오지 못했습니다.");
      setRows([]);
      setEtfInfo(null);
      setHoldings([]);
      setHoldingsAsOfDate(null);
      setHoldingsPriceAsOfDate(null);
      setHoldingsError(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    addingPoolKeysRef.current = addingPoolKeys;
    holdingsGridApiRef.current?.refreshCells({
      force: true,
      columns: ["ticker"],
    });
  }, [addingPoolKeys]);

  // --- 핸들러 ---

  async function handleAddPoolTicker(row: TickerHoldingRow, targetPool: "us" | "kor") {
    const ticker = String(row.ticker || "").trim().toUpperCase();
    if (!ticker) {
      return;
    }
    const poolKey = `${targetPool}:${ticker}`;
    setAddingPoolKeys((current) => (
      current.includes(poolKey) ? current : [...current, poolKey]
    ));
    const poolName = targetPool === "us" ? "미국 종목풀" : "한국 종목풀";
    try {
      await addStockCandidate(targetPool, ticker, 1);
      setHoldings((current) =>
        current.map((row) =>
          row.ticker === ticker
            ? targetPool === "us"
              ? { ...row, in_us_pool: true }
              : { ...row, in_kor_pool: true }
            : row,
        ),
      );
      toast.success(`${ticker}를 ${poolName}의 1. 모멘텀에 추가하였습니다.`);
    } catch (addError) {
      const message = addError instanceof Error ? addError.message : `${poolName} 추가에 실패했습니다.`;
      if (message.includes("이미 등록된 종목입니다.")) {
        setHoldings((current) =>
          current.map((row) =>
            row.ticker === ticker
              ? targetPool === "us"
                ? { ...row, in_us_pool: true }
                : { ...row, in_kor_pool: true }
              : row,
          ),
        );
        toast.success(`${ticker}를 ${poolName}의 1. 모멘텀에 추가하였습니다.`);
        return;
      }
      toast.error(`${ticker} ${poolName} 추가에 실패했습니다. ${message}`);
    } finally {
      setAddingPoolKeys((current) => current.filter((item) => item !== poolKey));
    }
  }

  // --- 차트 관련 ---

  const selectedCountryCode = selectedTicker?.country_code ?? "kor";
  const priceDigits =
    selectedTicker?.country_code === "au" || selectedTicker?.country_code === "us"
      ? 2
      : 0;
  const priceMinMove = priceDigits > 0 ? 0.01 : 1;

  const chartRows = useMemo(() => {
    if (chartInterval === "week") {
      return aggregatePriceRows(rows, "week");
    }
    if (chartInterval === "month") {
      return aggregatePriceRows(rows, "month");
    }
    return rows;
  }, [chartInterval, rows]);

  const dateRowMap = useMemo(() => {
    const map = new Map<string, PriceRow>();
    for (const row of chartRows) map.set(row.date, row);
    return map;
  }, [chartRows]);

  const lastInfo = useMemo<CrosshairInfo | null>(() => {
    if (chartRows.length === 0) return null;
    const last = chartRows[chartRows.length - 1];
    return { open: last.open, high: last.high, low: last.low, close: last.close, change_pct: last.change_pct };
  }, [chartRows]);

  useEffect(() => {
    if (!chartContainerRef.current || chartRows.length === 0) {
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
      setCrosshairInfo(null);
      setChartBadges([]);
      return;
    }
    if (chartRef.current) { chartRef.current.remove(); chartRef.current = null; }
    setCrosshairInfo(null);
    setChartBadges([]);

    const container = chartContainerRef.current;
    const chart = createChart(container, {
      width: container.clientWidth,
      height: 420,
      layout: { background: { type: ColorType.Solid, color: "#ffffff" }, textColor: "#5b6778", fontSize: 12 },
      grid: { vertLines: { color: "#f0f2f5" }, horzLines: { color: "#f0f2f5" } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#e6e8ec", scaleMargins: { top: 0.12, bottom: 0.25 } },
      timeScale: { borderColor: "#e6e8ec", timeVisible: false },
    });
    chartRef.current = chart;

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#e03131", downColor: "#206bc4",
      borderUpColor: "#e03131", borderDownColor: "#206bc4",
      wickUpColor: "#e03131", wickDownColor: "#206bc4",
      priceFormat: {
        type: "custom",
        minMove: priceMinMove,
        formatter: (price: number) => formatTickerPrice(price, selectedCountryCode),
      },
    });
    candleSeries.setData(
      chartRows
        .filter((r) => r.open !== null && r.high !== null && r.low !== null && r.close !== null)
        .map((r) => ({ time: r.date as Time, open: r.open!, high: r.high!, low: r.low!, close: r.close! })),
    );

    function updateRangeBadges(logicalRange?: { from: number; to: number } | null) {
      const safeFrom = logicalRange ? Math.max(Math.floor(logicalRange.from), 0) : 0;
      const safeTo = logicalRange ? Math.min(Math.ceil(logicalRange.to), chartRows.length - 1) : chartRows.length - 1;
      const visibleRows = chartRows.slice(safeFrom, safeTo + 1).filter(
        (row) => row.high !== null && row.low !== null && row.close !== null,
      );

      if (visibleRows.length === 0) {
        setChartBadges([]);
        return;
      }

      const currentRow = visibleRows[visibleRows.length - 1];
      if (currentRow.close === null) {
        setChartBadges([]);
        return;
      }

      let highRow = visibleRows[0];
      let lowRow = visibleRows[0];
      for (const row of visibleRows) {
        if ((row.high ?? Number.NEGATIVE_INFINITY) > (highRow.high ?? Number.NEGATIVE_INFINITY)) {
          highRow = row;
        }
        if ((row.low ?? Number.POSITIVE_INFINITY) < (lowRow.low ?? Number.POSITIVE_INFINITY)) {
          lowRow = row;
        }
      }

      const highIndex = chartRows.findIndex((row) => row.date === highRow.date);
      const lowIndex = chartRows.findIndex((row) => row.date === lowRow.date);
      const highX = chart.timeScale().logicalToCoordinate(highIndex as Logical);
      const lowX = chart.timeScale().logicalToCoordinate(lowIndex as Logical);
      const highY = candleSeries.priceToCoordinate(highRow.high!);
      const lowY = candleSeries.priceToCoordinate(lowRow.low!);

      if (highX === null || lowX === null || highY === null || lowY === null) {
        setChartBadges([]);
        return;
      }

      const chartWidth = container.clientWidth;
      const badgeWidth = 240;
      const clampLeft = (value: number, width = badgeWidth) => Math.max(12, Math.min(value, chartWidth - width - 12));
      const getBadgeLeft = (anchorX: number) => clampLeft(anchorX - badgeWidth / 2);

      setChartBadges([
        {
          tone: "high",
          text: buildRangeBadgeText(highRow.high!, currentRow.close, highRow.date, selectedTicker?.country_code ?? "kor"),
          left: getBadgeLeft(highX),
          top: Math.max(highY - 78, 10),
          anchorLeft: Math.max(10, Math.min(highX - getBadgeLeft(highX), badgeWidth - 10)),
        },
        {
          tone: "low",
          text: buildRangeBadgeText(lowRow.low!, currentRow.close, lowRow.date, selectedTicker?.country_code ?? "kor"),
          left: getBadgeLeft(lowX),
          top: Math.min(lowY + 40, 382),
          anchorLeft: Math.max(10, Math.min(lowX - getBadgeLeft(lowX), badgeWidth - 10)),
        },
      ]);
    }

    const volumeSeries = chart.addSeries(HistogramSeries, { priceFormat: { type: "volume" }, priceScaleId: "volume" });
    chart.priceScale("volume").applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
    volumeSeries.setData(
      chartRows
        .filter((r) => r.volume !== null && r.close !== null)
        .map((r, i) => {
          const prevClose = i > 0 ? chartRows[i - 1].close : null;
          const isUp = prevClose !== null && r.close !== null ? r.close >= prevClose : true;
          return { time: r.date as Time, value: r.volume!, color: isUp ? "rgba(224, 49, 49, 0.32)" : "rgba(32, 107, 196, 0.32)" };
        }),
    );

    for (const ma of MA_PERIODS) {
      if (chartRows.length < ma.period) continue;
      const maData = calculateMA(chartRows, ma.period);
      if (maData.length === 0) continue;
      chart.addSeries(LineSeries, {
        color: ma.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
        priceFormat: {
          type: "custom",
          minMove: priceMinMove,
          formatter: (price: number) => formatTickerPrice(price, selectedCountryCode),
        },
      }).setData(maData);
    }

    chart.subscribeCrosshairMove((param: MouseEventParams) => {
      if (!param.time) { setCrosshairInfo(null); return; }
      const row = dateRowMap.get(String(param.time));
      if (row) setCrosshairInfo({ open: row.open, high: row.high, low: row.low, close: row.close, change_pct: row.change_pct });
    });

    chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      updateRangeBadges(range);
    });

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        chart.applyOptions({ width: entry.contentRect.width });
        updateRangeBadges(chart.timeScale().getVisibleLogicalRange());
      }
    });
    observer.observe(container);
    const initialVisibleRange = getInitialVisibleLogicalRange(chartRows, chartInterval);
    if (initialVisibleRange) {
      chart.timeScale().setVisibleLogicalRange(initialVisibleRange);
    } else {
      chart.timeScale().fitContent();
    }
    updateRangeBadges(chart.timeScale().getVisibleLogicalRange());

    return () => {
      observer.disconnect();
      setChartBadges([]);
      chart.remove();
      chartRef.current = null;
    };
  }, [chartInterval, chartRows, dateRowMap, priceMinMove, selectedCountryCode]);

  const reversedRows = useMemo(
    () => [...rows].reverse().map((r, i) => ({ ...r, id: `${r.date}-${i}` })),
    [rows],
  );

  const monthlyRows = useMemo(
    () => aggregateMonthlyRows(rows).reverse().map((row, i) => ({ ...row, id: `${row.month}-${i}` })),
    [rows],
  );

  const holdingsRows = useMemo(
    () => holdings.map((row, index) => ({ ...row, id: `${row.ticker}-${index}` })),
    [holdings],
  );
  const showHoldingsWeightColumn = useMemo(
    () => holdingsRows.some((row) => (row.weight ?? 0) > 0),
    [holdingsRows],
  );

  const holdingsPanelTitle = showHoldingsWeightColumn ? "구성종목비중" : "구성종목";
  const holdingsPanelMeta = useMemo(() => {
    if (holdingsRows.length === 0) return "데이터 없음";
    return `상위 ${new Intl.NumberFormat("ko-KR").format(holdingsRows.length)}개`;
  }, [holdingsRows.length]);
  const holdingsDirectionCounts = useMemo(() => {
    let rising = 0;
    let neutral = 0;
    let falling = 0;
    for (const row of holdingsRows) {
      const changePct = row.change_pct;
      if (changePct == null || Number.isNaN(changePct) || changePct === 0) {
        neutral += 1;
        continue;
      }
      if (changePct > 0) {
        rising += 1;
        continue;
      }
      falling += 1;
    }
    return { rising, neutral, falling };
  }, [holdingsRows]);

  const lastPriceRow = useMemo(() => rows[rows.length - 1] ?? null, [rows]);
  const previousPriceRow = useMemo(() => rows[rows.length - 2] ?? null, [rows]);
  const latestClose = lastPriceRow?.close ?? null;
  const latestChangePct = lastPriceRow?.change_pct ?? null;
  const latestChangeAmount = useMemo(() => {
    if (latestClose === null || latestChangePct === null) {
      return null;
    }
    if (previousPriceRow?.close != null) {
      return latestClose - previousPriceRow.close;
    }
    const previousClose = latestClose / (1 + latestChangePct / 100);
    if (!Number.isFinite(previousClose)) {
      return null;
    }
    return latestClose - previousClose;
  }, [latestClose, latestChangePct, previousPriceRow]);
  const latestVolumeText = useMemo(() => {
    const volume = etfInfo?.volume ?? lastPriceRow?.volume ?? null;
    if (volume === null || Number.isNaN(volume)) {
      return "-";
    }
    if (volume >= 10_000) {
      return `${formatNumber(Math.floor(volume / 10_000), 0)}만`;
    }
    return formatNumber(volume, 0);
  }, [etfInfo?.volume, lastPriceRow]);

  const showKoreanEtfInfoSection = Boolean(
    selectedTicker?.country_code === "kor" && selectedTicker?.is_etf && selectedTicker?.has_holdings,
  );
  const navDelta = etfInfo?.nav_change ?? null;
  const navChangePct = etfInfo?.nav_change_pct ?? null;

  const displayFxRates = useMemo<TickerFxRate[]>(() => {
    return etfInfo?.fx_rates ?? [];
  }, [etfInfo?.fx_rates]);
  const fxChangePctByCurrency = useMemo(() => {
    const map = new Map<string, number>();
    displayFxRates.forEach((fx) => {
      const currency = String(fx.currency || "").trim().toUpperCase();
      const changePct = fx.change_pct;
      if (!currency || changePct == null || Number.isNaN(changePct)) {
        return;
      }
      map.set(currency, changePct);
    });
    return map;
  }, [displayFxRates]);

  const portfolioChange = useMemo<{
    total_pct: number | null;
    breakdown: PortfolioChangeBreakdownItem[];
  }>(() => {
    if (!holdings || holdings.length === 0) {
      return { total_pct: null, breakdown: [] };
    }

    const groups = new Map<string, { weight: number; weightedSum: number }>();

    holdings.forEach((h) => {
      const weight = h.weight ?? 0;
      if (weight <= 0) return;
      if (h.change_pct == null || Number.isNaN(h.change_pct)) return;

      const currency = String(h.price_currency || "").trim().toUpperCase();
      const isForeign = currency !== "" && currency !== "KRW";
      let changePctKrw = h.change_pct;
      if (isForeign) {
        const fxChangePct = fxChangePctByCurrency.get(currency);
        if (fxChangePct == null || Number.isNaN(fxChangePct)) {
          return;
        }
        // (1 + 현지통화 변동률) × (1 + 환율 변동률) - 1
        const fxFactor = 1 + fxChangePct / 100;
        changePctKrw = ((1 + h.change_pct / 100) * fxFactor - 1) * 100;
      }

      const group = groups.get(currency || "KRW") ?? { weight: 0, weightedSum: 0 };
      group.weight += weight;
      group.weightedSum += weight * changePctKrw;
      groups.set(currency || "KRW", group);
    });

    let totalWeight = 0;
    let totalWeightedSum = 0;
    const breakdown: PortfolioChangeBreakdownItem[] = [];
    for (const [currency, group] of groups.entries()) {
      if (group.weight <= 0) {
        continue;
      }
      const changePct = group.weightedSum / group.weight;
      breakdown.push({
        currency,
        label: getCurrencyRegionLabel(currency),
        change_pct: changePct,
        weight: group.weight,
      });
      totalWeight += group.weight;
      totalWeightedSum += group.weight * changePct;
    }

    breakdown.sort((a, b) => b.weight - a.weight);
    if (totalWeight <= 0) {
      return { total_pct: null, breakdown };
    }

    const divisor = Math.max(totalWeight, 100);
    return { total_pct: totalWeightedSum / divisor, breakdown };
  }, [holdings, fxChangePctByCurrency]);
  const portfolioChangePct = portfolioChange.total_pct;
  const portfolioChangeBreakdown = portfolioChange.breakdown;
  const dailyColumns = useMemo<ColDef[]>(
    () => [
      {
        field: "date",
        headerName: "날짜",
        minWidth: 138,
        flex: 1.45,
        cellStyle: { fontWeight: 600 },
        cellRenderer: (params: { value: string }) => formatDateWithWeekday(params.value),
      },
      {
        field: "close", headerName: "종가", minWidth: 84, flex: 0.95, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatTickerPrice(params.value, selectedCountryCode)
      },
      {
        field: "change_pct", headerName: "등락률", minWidth: 92, flex: 0.95, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => (
          <span className={getSignedClass(params.value)}>{formatPercent(params.value)}</span>
        )
      },
      {
        field: "volume", headerName: "거래량", minWidth: 118, flex: 1.2, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatNumber(params.value, 0)
      },
    ],
    [selectedCountryCode],
  );

  const monthlyColumns = useMemo<ColDef[]>(
    () => [
      {
        field: "month",
        headerName: "년월",
        minWidth: 92,
        flex: 1.1,
        cellStyle: { fontWeight: 600 },
      },
      {
        field: "close", headerName: "월말 종가", minWidth: 88, flex: 0.95, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatTickerPrice(params.value, selectedCountryCode)
      },
      {
        field: "change_pct", headerName: "월간 등락률", minWidth: 96, flex: 0.95, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => (
          <span className={getSignedClass(params.value)}>{formatPercent(params.value)}</span>
        )
      },
      {
        field: "volume", headerName: "월간 거래량", minWidth: 120, flex: 1.25, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatNumber(params.value, 0)
      },
    ],
    [selectedCountryCode],
  );

  const holdingColumns = useMemo<ColDef[]>(
    () => {
      const columns: ColDef[] = [
        {
          field: "ticker",
          headerName: "종목코드",
          minWidth: 120,
          width: 120,
          cellClass: "tickerDetailCodeCell",
          cellStyle: { fontWeight: 700 },
          cellRenderer: (params: { value: string; data?: TickerHoldingRow }) => {
            const ticker = String(params.value ?? "").trim();
            const row = params.data;
            const isUsCandidate = Boolean(row?.is_us_pool_candidate);
            const inUsPool = Boolean(row?.in_us_pool);
            const isKorCandidate = Boolean(row?.is_kor_pool_candidate);
            const inKorPool = Boolean(row?.in_kor_pool);
            const currentPoolKey = isUsCandidate ? `us:${ticker}` : isKorCandidate ? `kor:${ticker}` : "";
            const isLoading = currentPoolKey ? addingPoolKeysRef.current.includes(currentPoolKey) : false;
            const isDone = isUsCandidate ? inUsPool : isKorCandidate ? inKorPool : false;
            const poolName = isUsCandidate ? "미국 종목풀" : isKorCandidate ? "한국 종목풀" : "";
            const targetPool = isUsCandidate ? "us" : isKorCandidate ? "kor" : null;

            return (
              <div className="tickerDetailCodeContent">
                <span className="tickerDetailCodeText">{ticker || "-"}</span>
                {targetPool ? (
                  isDone ? (
                    <span className="tickerDetailPoolState is-done" title={`${poolName}에 이미 등록됨`} aria-label={`${poolName} 등록 완료`}>
                      <IconCheck size={14} stroke={2.2} />
                    </span>
                  ) : isLoading ? (
                    <span className="tickerDetailPoolState is-loading" title={`${poolName} 추가 중`} aria-label={`${poolName} 추가 중`}>
                      <span className="spinner-border spinner-border-sm" />
                    </span>
                  ) : (
                    <button
                      type="button"
                      className="tickerDetailPoolState is-add"
                      title={`${poolName}에 추가`}
                      aria-label={`${ticker} ${poolName} 추가`}
                      onClick={(event) => {
                        event.stopPropagation();
                        if (row && targetPool) {
                          void handleAddPoolTicker(row, targetPool);
                        }
                      }}
                    >
                      <IconPlus size={14} stroke={2.2} />
                    </button>
                  )
                ) : null}
              </div>
            );
          },
        },
        {
          field: "name",
          headerName: "종목명",
          minWidth: 148,
          flex: 1.2,
          cellClass: "tickerDetailNameCell",
        },
      ];

      if (showHoldingsWeightColumn) {
        columns.push({
          field: "weight",
          headerName: "비중",
          minWidth: 76,
          width: 76,
          type: "rightAligned",
          cellRenderer: (params: { value: number | null }) => formatUnsignedPercent(params.value),
        });
      }

      columns.push(
        {
          field: "current_price",
          headerName: "현재가",
          minWidth: 108,
          width: 108,
          type: "rightAligned",
          cellRenderer: (params: { value: number | null; data?: TickerHoldingRow }) =>
            formatTickerPrice(params.value, String(params.data?.price_currency || "kor")),
        },
        {
          field: "change_pct",
          headerName: "일간(%)",
          minWidth: 88,
          width: 88,
          type: "rightAligned",
          cellRenderer: (params: { value: number | null }) => (
            <span className={getSignedClass(params.value)}>{formatPercent(params.value)}</span>
          ),
        },
      );

      return columns;
    },
    [showHoldingsWeightColumn],
  );

  const displayTitle = selectedTicker
    ? (selectedTicker.name ? `${selectedTicker.name}(${selectedTicker.ticker})` : selectedTicker.ticker)
    : null;

  return (
    <div className="appPageStack appPageStackFill">
      {error ? (
        <div className="appBannerStack">
          <div className="bannerError alert alert-danger mb-0">{error}</div>
        </div>
      ) : null}

      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-body appCardBodyTight appTableCardBodyFill">
            {!selectedTicker && !loading ? (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", flex: 1, color: "#5b6778" }}>
                <span style={{ fontSize: 15 }}>티커 또는 종목명을 검색하세요.</span>
              </div>
            ) : (
              <>
                {displayTitle ? (
                  <div className="tickerDetailHero">
                    <div className="tickerDetailHeroLeft">
                      <div className="tickerDetailHeroTitle">{displayTitle}</div>
                      {lastInfo?.close != null ? (
                        <span className="tickerDetailHeroPrice">{formatTickerPrice(lastInfo.close, selectedCountryCode)}</span>
                      ) : null}
                      {lastInfo?.change_pct != null ? (
                        <span className={`tickerDetailHeroChange ${getSignedClass(lastInfo.change_pct)}`}>
                          {formatPercent(lastInfo.change_pct)}
                        </span>
                      ) : null}
                    </div>
                  </div>
                ) : null}

                {/* 캔들스틱 차트 / 구성종목 / 가격 테이블 */}
                {loading && rows.length === 0 ? (
                  <div style={{ height: 420, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                    <span className="spinner-border" />
                  </div>
                ) : rows.length > 0 ? (
                  <div className={showKoreanEtfInfoSection ? "tickerDetailLayoutGrid" : "tickerDetailClassicLayout"}>
                    {showKoreanEtfInfoSection ? (
                      <>
                        <div className="tickerDetailInfoPanel">
                          <div className="tickerDetailTableHeader">
                            <span className="tickerDetailTableTitle">ETF정보</span>
                          </div>
                          <div className="tickerDetailInfoCard">
                            <div className="tickerDetailInfoSummary">
                              <div className="tickerDetailInfoSummaryRow">
                                <span className="tickerDetailInfoLabel">현재가</span>
                                <div className="tickerDetailInfoMain">
                                  <strong>{formatTickerPrice(latestClose, selectedCountryCode)}</strong>
                                  <span className={getSignedClass(latestChangeAmount ?? latestChangePct)}>
                                    {formatSignedPriceDelta(latestChangeAmount, selectedCountryCode)}
                                  </span>
                                  <span className={getSignedClass(latestChangePct)}>{formatPercent(latestChangePct)}</span>
                                </div>
                              </div>
                              <div className="tickerDetailInfoSummaryRow">
                                <span className="tickerDetailInfoLabel">iNAV</span>
                                <div className="tickerDetailInfoMain">
                                  <strong>{formatTickerPrice(etfInfo?.nav ?? null, "kor")}</strong>
                                  <span className={getSignedClass(navDelta)}>{formatSignedPriceDelta(navDelta, "kor")}</span>
                                  <span className={getSignedClass(navChangePct)}>{formatPercent(navChangePct)}</span>
                                </div>
                              </div>
                              <div className="tickerDetailInfoSummaryGrid">
                                <div className="tickerDetailInfoMetric">
                                  <span className="tickerDetailInfoLabel">괴리율</span>
                                  <strong className={getSignedClass(etfInfo?.deviation ?? null)}>{formatPercent(etfInfo?.deviation ?? null)}</strong>
                                </div>
                                <div className="tickerDetailInfoMetric">
                                  <span className="tickerDetailInfoLabel">거래량</span>
                                  <strong>{latestVolumeText}</strong>
                                </div>
                                <div className="tickerDetailInfoMetric">
                                  <span className="tickerDetailInfoLabel">운용보수</span>
                                  <strong>{formatRatioPercent(etfInfo?.expense_ratio ?? null)}</strong>
                                </div>
                                <div className="tickerDetailInfoMetric">
                                  <span className="tickerDetailInfoLabel">시가총액</span>
                                  <strong>{formatEokFromKrw(etfInfo?.market_cap_krw ?? null)}</strong>
                                </div>
                                <div className="tickerDetailInfoMetric">
                                  <span className="tickerDetailInfoLabel">배당수익률</span>
                                  <strong>{formatRatioPercent(etfInfo?.dividend_yield_ttm ?? null)}</strong>
                                </div>
                                <div className="tickerDetailInfoMetric">
                                  <span className="tickerDetailInfoLabel">배당주기</span>
                                  <strong>-</strong>
                                </div>
                              </div>
                            </div>
                            <div className="tickerDetailInfoTracker">
                              <div className="tickerDetailInfoTrackerRow">
                                <div>
                                  <div className="tickerDetailInfoTrackerLabel">포트폴리오 변동</div>
                                  <div className="tickerDetailInfoTrackerHint">
                                    {portfolioChangeBreakdown.length > 0 ? (
                                      <span className="tickerDetailInfoBreakdownList">
                                        {portfolioChangeBreakdown.map((item, index) => (
                                          <span key={item.currency} className="tickerDetailInfoBreakdownItem">
                                            {index > 0 ? <span className="tickerDetailInfoBreakdownSeparator">/</span> : null}
                                            <span>{item.label}</span>
                                            <span className={getSignedClass(item.change_pct)}>{formatPercent(item.change_pct)}</span>
                                          </span>
                                        ))}
                                      </span>
                                    ) : (
                                      "구성종목 가중 평균"
                                    )}
                                  </div>
                                </div>
                                <strong className={getSignedClass(portfolioChangePct)}>
                                  {portfolioChangePct !== null ? formatPercent(portfolioChangePct) : "-"}
                                </strong>
                              </div>
                              <div className="tickerDetailInfoTrackerRow">
                                <div>
                                  <div className="tickerDetailInfoTrackerLabel">환율</div>
                                  <div className="tickerDetailInfoTrackerHint">구성종목 통화</div>
                                </div>
                                {displayFxRates.length > 0 ? (
                                  <div className="tickerDetailInfoFxList">
                                    {displayFxRates.map((fx, index) => (
                                      <div key={fx.currency} className="tickerDetailInfoFxItem">
                                        {index > 0 ? <span className="tickerDetailInfoFxSeparator">/</span> : null}
                                        <span>{fx.currency}</span>
                                        <strong>
                                          {fx.rate != null ? `${formatNumber(fx.rate, 2)}원` : "-"}
                                          <span className={getSignedClass(fx.change_pct ?? null)}>
                                            ({formatPercent(fx.change_pct ?? null)})
                                          </span>
                                        </strong>
                                      </div>
                                    ))}
                                  </div>
                                ) : (
                                  <strong>-</strong>
                                )}
                              </div>
                              <div className="tickerDetailInfoTrackerRow tickerDetailInfoTrackerRowLast">
                                <div>
                                  <div className="tickerDetailInfoTrackerLabel">구성종목 방향</div>
                                  <div className="tickerDetailInfoTrackerHint">상승/보합/하락</div>
                                </div>
                                <div className="tickerDetailInfoTrackerCounts">
                                  <span className="metricPositive">▲ {holdingsDirectionCounts.rising}종목</span>
                                  <span>■ {holdingsDirectionCounts.neutral}종목</span>
                                  <span className="metricNegative">▼ {holdingsDirectionCounts.falling}종목</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="tickerDetailHoldingsPanel">
                          <div className="tickerDetailTableHeader">
                            <span className="tickerDetailTableTitle">{holdingsPanelTitle}</span>
                            <span className="tickerDetailTableMeta">{holdingsPanelMeta}</span>
                          </div>
                          {holdingsRows.length > 0 ? (
                            <div className="appGridFillWrap">
                              <AppAgGrid
                                className="tickerDetailHoldingsGrid"
                                rowData={holdingsRows}
                                columnDefs={holdingColumns}
                                loading={loading}
                                theme={gridTheme}
                                gridOptions={{
                                  suppressMovableColumns: true,
                                  rowHeight: 40,
                                  getRowId: (params) => String(params.data.id),
                                  onGridReady: (event: GridReadyEvent<TickerHoldingRow>) => {
                                    holdingsGridApiRef.current = event.api;
                                  },
                                }}
                              />
                            </div>
                          ) : (
                            <div className="tickerDetailHoldingsEmpty">
                              {holdingsError ?? "구성종목 데이터를 확인할 수 없습니다."}
                            </div>
                          )}
                        </div>
                      </>
                    ) : null}
                    <div className="tickerDetailChartWrap">
                      <div className="tickerDetailChartToolbar">
                        <div className="appSegmentedToggle appSegmentedToggleCompact" role="group" aria-label="차트 봉 기준">
                          {[
                            { value: "day", label: "일" },
                            { value: "week", label: "주" },
                            { value: "month", label: "월" },
                          ].map((option) => (
                            <button
                              key={option.value}
                              type="button"
                              className={chartInterval === option.value ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                              onClick={() => setChartInterval(option.value as ChartInterval)}
                            >
                              {option.label}
                            </button>
                          ))}
                        </div>
                        <div className="tickerDetailChartMaLegend" />
                      </div>
                      <div ref={chartContainerRef} style={{ width: "100%", position: "relative" }} />
                      {chartBadges.map((badge, index) => (
                        <div
                          key={`${badge.tone}-${index}`}
                          className={badge.tone === "high" ? "tickerDetailChartBadge is-high" : "tickerDetailChartBadge is-low"}
                          style={{ left: badge.left, top: badge.top }}
                        >
                          {badge.tone === "low" ? <span className="tickerDetailChartBadgeArrow" style={{ left: badge.anchorLeft }}>↑</span> : null}
                          <span className="tickerDetailChartBadgeText">{badge.text}</span>
                          {badge.tone === "high" ? <span className="tickerDetailChartBadgeArrow" style={{ left: badge.anchorLeft }}>↓</span> : null}
                        </div>
                      ))}
                    </div>
                    {showKoreanEtfInfoSection ? (
                      <div className="tickerDetailTablePanel">
                        <div className="tickerDetailTableHeader tickerDetailTableHeaderBetween">
                          <div className="appSegmentedToggle appSegmentedToggleCompact" role="tablist" aria-label="가격 이력 탭">
                            <button
                              type="button"
                              role="tab"
                              aria-selected={historyTab === "daily"}
                              className={historyTab === "daily" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                              onClick={() => setHistoryTab("daily")}
                            >
                              일별
                            </button>
                            <button
                              type="button"
                              role="tab"
                              aria-selected={historyTab === "monthly"}
                              className={historyTab === "monthly" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                              onClick={() => setHistoryTab("monthly")}
                            >
                              월별
                            </button>
                          </div>
                          <span className="text-muted tickerDetailTableMeta">
                            {historyTab === "daily"
                              ? `총 ${new Intl.NumberFormat("ko-KR").format(rows.length)}일`
                              : `총 ${new Intl.NumberFormat("ko-KR").format(monthlyRows.length)}개월`}
                          </span>
                        </div>
                        <div className="appGridFillWrap">
                          <AppAgGrid
                            rowData={historyTab === "daily" ? reversedRows : monthlyRows}
                            columnDefs={historyTab === "daily" ? dailyColumns : monthlyColumns}
                            loading={loading}
                            theme={gridTheme}
                            gridOptions={{ suppressMovableColumns: true }}
                          />
                        </div>
                      </div>
                    ) : (
                      <div className="tickerDetailTables">
                        <div className="tickerDetailTablePanel">
                          <div className="tickerDetailTableHeader tickerDetailTableHeaderBetween">
                            <span className="tickerDetailTableTitle">일별</span>
                            <span className="text-muted tickerDetailTableMeta">
                              총 {new Intl.NumberFormat("ko-KR").format(rows.length)}일
                            </span>
                          </div>
                          <div className="appGridFillWrap">
                            <AppAgGrid
                              rowData={reversedRows}
                              columnDefs={dailyColumns}
                              loading={loading}
                              theme={gridTheme}
                              gridOptions={{ suppressMovableColumns: true }}
                            />
                          </div>
                        </div>
                        <div className="tickerDetailTablePanel">
                          <div className="tickerDetailTableHeader tickerDetailTableHeaderBetween">
                            <span className="tickerDetailTableTitle">월별</span>
                            <span className="text-muted tickerDetailTableMeta">
                              총 {new Intl.NumberFormat("ko-KR").format(monthlyRows.length)}개월
                            </span>
                          </div>
                          <div className="appGridFillWrap">
                            <AppAgGrid
                              rowData={monthlyRows}
                              columnDefs={monthlyColumns}
                              loading={loading}
                              theme={gridTheme}
                              gridOptions={{ suppressMovableColumns: true }}
                            />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : null}
              </>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
