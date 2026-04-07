"use client";

import { useEffect, useMemo, useRef, useState } from "react";
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
  Time,
  MouseEventParams,
} from "lightweight-charts";
import { iconSetQuartzBold, themeQuartz } from "ag-grid-community";
import type { ColDef } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";

// --- 타입 ---

type TickerItem = {
  ticker: string;
  name: string;
  ticker_type: string;
  country_code: string;
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

type TickerDetailResponse = {
  ticker: string;
  rows: PriceRow[];
  error?: string;
};

type CrosshairInfo = {
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  change_pct: number | null;
};

// --- 상수 ---

const MA_PERIODS = [
  { period: 5, color: "#2196F3", label: "5" },
  { period: 20, color: "#FF9800", label: "20" },
  { period: 60, color: "#E91E63", label: "60" },
  { period: 120, color: "#9C27B0", label: "120" },
];

const gridTheme = themeQuartz
  .withPart(iconSetQuartzBold)
  .withParams({
    accentColor: "#206bc4",
    backgroundColor: "#ffffff",
    foregroundColor: "#182433",
    headerBackgroundColor: "#f8fafc",
    headerTextColor: "#5b6778",
    spacing: 8,
    fontSize: 14,
    wrapperBorderRadius: 10,
    rowHeight: 34,
    headerHeight: 36,
    cellHorizontalPadding: 12,
    headerColumnBorder: true,
    headerColumnBorderHeight: "70%",
    columnBorder: true,
    oddRowBackgroundColor: "#fbfdff",
  });

// --- 유틸 ---

function formatNumber(value: number | null, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
}

function formatPercent(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
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
      to: lastIndex + 0.5,
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
    to: lastIndex + 0.5,
  };
}

// --- 컴포넌트 ---

export function TickerDetailManager() {
  const searchParams = useSearchParams();

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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [chartInterval, setChartInterval] = useState<ChartInterval>("day");

  // 차트
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [crosshairInfo, setCrosshairInfo] = useState<CrosshairInfo | null>(null);

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
    if (matches.length > 1) {
      setSelectedTicker(null);
      setError(`동일한 티커 ${qTicker}가 여러 종목 타입에 등록되어 있습니다.`);
      return;
    }

    setSelectedTicker(null);
    setError(`${qTicker} 티커를 찾지 못했습니다.`);
  }, [allTickers, qTicker, qTickerType, qCountryCode, qName]);

  // --- 데이터 로드 ---

  async function loadTickerData(item: TickerItem) {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);

    try {
      const search = new URLSearchParams({
        ticker: item.ticker,
        ticker_type: item.ticker_type,
        country_code: item.country_code || "kor",
      });
      const response = await fetch(`/api/ticker-detail?${search.toString()}`, {
        cache: "no-store",
        signal: controller.signal,
      });
      const payload = (await response.json()) as TickerDetailResponse;
      if (!response.ok) throw new Error(payload.error ?? "데이터를 불러오지 못했습니다.");
      if (payload.error) throw new Error(payload.error);
      setRows(payload.rows);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof Error ? err.message : "데이터를 불러오지 못했습니다.");
      setRows([]);
    } finally {
      setLoading(false);
    }
  }

  // --- 핸들러 ---

  // --- 차트 관련 ---

  const priceDigits = selectedTicker?.country_code === "au" ? 2 : 0;

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
    if (!chartContainerRef.current || chartRows.length === 0) return;
    if (chartRef.current) { chartRef.current.remove(); chartRef.current = null; }
    setCrosshairInfo(null);

    const container = chartContainerRef.current;
    const chart = createChart(container, {
      width: container.clientWidth,
      height: 420,
      layout: { background: { type: ColorType.Solid, color: "#ffffff" }, textColor: "#5b6778", fontSize: 12 },
      grid: { vertLines: { color: "#f0f2f5" }, horzLines: { color: "#f0f2f5" } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#e6e8ec", scaleMargins: { top: 0.05, bottom: 0.25 } },
      timeScale: { borderColor: "#e6e8ec", timeVisible: false },
    });
    chartRef.current = chart;

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#26a69a", downColor: "#ef5350",
      borderUpColor: "#26a69a", borderDownColor: "#ef5350",
      wickUpColor: "#26a69a", wickDownColor: "#ef5350",
      priceFormat: { type: "price", precision: 0, minMove: 1 },
    });
    candleSeries.setData(
      chartRows
        .filter((r) => r.open !== null && r.high !== null && r.low !== null && r.close !== null)
        .map((r) => ({ time: r.date as Time, open: r.open!, high: r.high!, low: r.low!, close: r.close! })),
    );

    const volumeSeries = chart.addSeries(HistogramSeries, { priceFormat: { type: "volume" }, priceScaleId: "volume" });
    chart.priceScale("volume").applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
    volumeSeries.setData(
      chartRows
        .filter((r) => r.volume !== null && r.close !== null)
        .map((r, i) => {
          const prevClose = i > 0 ? chartRows[i - 1].close : null;
          const isUp = prevClose !== null && r.close !== null ? r.close >= prevClose : true;
          return { time: r.date as Time, value: r.volume!, color: isUp ? "rgba(38, 166, 154, 0.3)" : "rgba(239, 83, 80, 0.3)" };
        }),
    );

    for (const ma of MA_PERIODS) {
      if (chartRows.length < ma.period) continue;
      const maData = calculateMA(chartRows, ma.period);
      if (maData.length === 0) continue;
      chart.addSeries(LineSeries, {
        color: ma.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
        priceFormat: { type: "price", precision: 0, minMove: 1 },
      }).setData(maData);
    }

    chart.subscribeCrosshairMove((param: MouseEventParams) => {
      if (!param.time) { setCrosshairInfo(null); return; }
      const row = dateRowMap.get(String(param.time));
      if (row) setCrosshairInfo({ open: row.open, high: row.high, low: row.low, close: row.close, change_pct: row.change_pct });
    });

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) chart.applyOptions({ width: entry.contentRect.width });
    });
    observer.observe(container);
    const initialVisibleRange = getInitialVisibleLogicalRange(chartRows, chartInterval);
    if (initialVisibleRange) {
      chart.timeScale().setVisibleLogicalRange(initialVisibleRange);
    } else {
      chart.timeScale().fitContent();
    }

    return () => { observer.disconnect(); chart.remove(); chartRef.current = null; };
  }, [chartInterval, chartRows, dateRowMap]);

  const displayInfo = crosshairInfo ?? lastInfo;

  const reversedRows = useMemo(
    () => [...rows].reverse().map((r, i) => ({ ...r, id: `${r.date}-${i}` })),
    [rows],
  );

  const monthlyRows = useMemo(
    () => aggregateMonthlyRows(rows).reverse().map((row, i) => ({ ...row, id: `${row.month}-${i}` })),
    [rows],
  );

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
      { field: "close", headerName: "종가", minWidth: 84, flex: 0.95, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatNumber(params.value, priceDigits) },
      { field: "change_pct", headerName: "등락률", minWidth: 92, flex: 0.95, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => (
          <span className={getSignedClass(params.value)}>{formatPercent(params.value)}</span>
        ) },
      { field: "volume", headerName: "거래량", minWidth: 118, flex: 1.2, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatNumber(params.value, 0) },
    ],
    [priceDigits],
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
      { field: "close", headerName: "월말 종가", minWidth: 88, flex: 0.95, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatNumber(params.value, priceDigits) },
      { field: "change_pct", headerName: "월간 등락률", minWidth: 96, flex: 0.95, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => (
          <span className={getSignedClass(params.value)}>{formatPercent(params.value)}</span>
        ) },
      { field: "volume", headerName: "월간 거래량", minWidth: 120, flex: 1.25, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatNumber(params.value, 0) },
    ],
    [priceDigits],
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
          {/* 상단 헤더 */}
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                {/* 선택된 종목 정보 */}
                {displayTitle ? (
                  <span style={{ fontWeight: 700, fontSize: "1em", whiteSpace: "nowrap" }}>
                    {displayTitle}
                  </span>
                ) : null}

                {displayInfo?.close != null ? (
                  <span style={{ fontSize: "1.1em", fontWeight: 700 }}>
                    {formatNumber(displayInfo.close, priceDigits)}
                    {displayInfo.change_pct != null ? (
                      <span className={getSignedClass(displayInfo.change_pct)} style={{ marginLeft: 8, fontSize: "0.85em" }}>
                        {formatPercent(displayInfo.change_pct)}
                      </span>
                    ) : null}
                  </span>
                ) : null}

              </div>
            </div>
          </div>

          {/* 차트 + 일별 테이블 */}
          <div className="card-body appCardBodyTight appTableCardBodyFill">
            {!selectedTicker && !loading ? (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", flex: 1, color: "#5b6778" }}>
                <span style={{ fontSize: 15 }}>티커 또는 종목명을 검색하세요.</span>
              </div>
            ) : (
              <>
                {/* 캔들스틱 차트 (고정 높이) */}
                {loading && rows.length === 0 ? (
                  <div style={{ height: 420, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                    <span className="spinner-border" />
                  </div>
                ) : rows.length > 0 ? (
                  <div style={{ flexShrink: 0 }}>
                    <div style={{ display: "flex", gap: 12, padding: "8px 16px 0", flexWrap: "wrap", alignItems: "center" }}>
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
                      <span style={{ fontSize: 12, color: "#5b6778", fontWeight: 600 }}>이동평균선</span>
                      {MA_PERIODS.map((ma) => (
                        chartRows.length >= ma.period ? (
                          <span key={ma.period} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12 }}>
                            <span style={{ width: 14, height: 2, backgroundColor: ma.color, display: "inline-block" }} />
                            <span style={{ color: ma.color, fontWeight: 600 }}>{ma.label}</span>
                          </span>
                        ) : null
                      ))}
                    </div>
                    <div ref={chartContainerRef} style={{ width: "100%", position: "relative" }} />
                  </div>
                ) : null}

                {/* 일별/월별 테이블 (나머지 공간 채움) */}
                {selectedTicker ? (
                  <div className="tickerDetailTables">
                    <div className="tickerDetailTablePanel">
                      <div className="tickerDetailTableHeader">
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
                      <div className="tickerDetailTableHeader">
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
                ) : null}
              </>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
