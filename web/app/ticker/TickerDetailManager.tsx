"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
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
  CandlestickData,
  HistogramData,
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

const MONTHS_OPTIONS = [3, 6, 12, 24, 36];

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

// --- 컴포넌트 ---

export function TickerDetailManager() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // URL query params
  const qTicker = searchParams.get("ticker") ?? "";
  const qTickerType = searchParams.get("ticker_type") ?? "";
  const qCountryCode = searchParams.get("country_code") ?? "";
  const qName = searchParams.get("name") ?? "";

  // 전체 종목 목록
  const [allTickers, setAllTickers] = useState<TickerItem[]>([]);

  // 검색 입력
  const [searchInput, setSearchInput] = useState("");
  const [showDropdown, setShowDropdown] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(-1);
  const searchRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  // 현재 선택된 종목
  const [selectedTicker, setSelectedTicker] = useState<TickerItem | null>(
    qTicker ? { ticker: qTicker, name: qName, ticker_type: qTickerType, country_code: qCountryCode } : null,
  );

  // 기간
  const [months, setMonths] = useState(12);

  // 데이터
  const [rows, setRows] = useState<PriceRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

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
    if (qTicker && qTickerType) {
      const item: TickerItem = { ticker: qTicker, name: qName, ticker_type: qTickerType, country_code: qCountryCode };
      setSelectedTicker(item);
      loadTickerData(item, months);
    }
  }, [qTicker, qTickerType]);

  // 클릭 외부 감지 → 드롭다운 닫기
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // --- 검색 필터 ---

  const filteredTickers = useMemo(() => {
    const q = searchInput.trim().toLowerCase();
    if (!q) return [];
    return allTickers
      .filter((t) => t.ticker.toLowerCase().includes(q) || t.name.toLowerCase().includes(q))
      .slice(0, 20);
  }, [searchInput, allTickers]);

  // 정확한 티커 매칭 시 자동 선택
  useEffect(() => {
    const q = searchInput.trim();
    if (!q) return;
    const exactMatch = allTickers.find((t) => t.ticker.toLowerCase() === q.toLowerCase());
    if (exactMatch && filteredTickers.length <= 1) {
      selectTicker(exactMatch);
    }
  }, [filteredTickers]);

  // --- 데이터 로드 ---

  async function loadTickerData(item: TickerItem, m: number) {
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
        months: String(m),
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

  function selectTicker(item: TickerItem) {
    setSelectedTicker(item);
    setSearchInput("");
    setShowDropdown(false);
    setHighlightIndex(-1);
    loadTickerData(item, months);
    // URL 업데이트
    const params = new URLSearchParams({
      ticker: item.ticker,
      ticker_type: item.ticker_type,
      country_code: item.country_code,
      name: item.name,
    });
    router.replace(`/ticker?${params.toString()}`);
  }

  function handleSearchInputChange(value: string) {
    setSearchInput(value);
    setShowDropdown(true);
    setHighlightIndex(-1);
  }

  function handleSearchKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightIndex((prev) => Math.min(prev + 1, filteredTickers.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (highlightIndex >= 0 && highlightIndex < filteredTickers.length) {
        selectTicker(filteredTickers[highlightIndex]);
      } else if (filteredTickers.length === 1) {
        selectTicker(filteredTickers[0]);
      }
    } else if (e.key === "Escape") {
      setShowDropdown(false);
    }
  }

  function handleMonthsChange(next: number) {
    setMonths(next);
    if (selectedTicker) loadTickerData(selectedTicker, next);
  }

  // --- 차트 관련 ---

  const priceDigits = selectedTicker?.country_code === "au" ? 2 : 0;

  const dateRowMap = useMemo(() => {
    const map = new Map<string, PriceRow>();
    for (const row of rows) map.set(row.date, row);
    return map;
  }, [rows]);

  const lastInfo = useMemo<CrosshairInfo | null>(() => {
    if (rows.length === 0) return null;
    const last = rows[rows.length - 1];
    return { open: last.open, high: last.high, low: last.low, close: last.close, change_pct: last.change_pct };
  }, [rows]);

  useEffect(() => {
    if (!chartContainerRef.current || rows.length === 0) return;
    if (chartRef.current) { chartRef.current.remove(); chartRef.current = null; }

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
    });
    candleSeries.setData(
      rows
        .filter((r) => r.open !== null && r.high !== null && r.low !== null && r.close !== null)
        .map((r) => ({ time: r.date as Time, open: r.open!, high: r.high!, low: r.low!, close: r.close! })),
    );

    const volumeSeries = chart.addSeries(HistogramSeries, { priceFormat: { type: "volume" }, priceScaleId: "volume" });
    chart.priceScale("volume").applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
    volumeSeries.setData(
      rows
        .filter((r) => r.volume !== null && r.close !== null)
        .map((r, i) => {
          const prevClose = i > 0 ? rows[i - 1].close : null;
          const isUp = prevClose !== null && r.close !== null ? r.close >= prevClose : true;
          return { time: r.date as Time, value: r.volume!, color: isUp ? "rgba(38, 166, 154, 0.3)" : "rgba(239, 83, 80, 0.3)" };
        }),
    );

    for (const ma of MA_PERIODS) {
      if (rows.length < ma.period) continue;
      const maData = calculateMA(rows, ma.period);
      if (maData.length === 0) continue;
      chart.addSeries(LineSeries, {
        color: ma.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
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
    chart.timeScale().fitContent();

    return () => { observer.disconnect(); chart.remove(); chartRef.current = null; };
  }, [rows, dateRowMap]);

  const displayInfo = crosshairInfo ?? lastInfo;

  const reversedRows = useMemo(
    () => [...rows].reverse().map((r, i) => ({ ...r, id: `${r.date}-${i}` })),
    [rows],
  );

  const columns = useMemo<ColDef[]>(
    () => [
      { field: "date", headerName: "날짜", minWidth: 120, width: 120, cellStyle: { fontWeight: 600 } },
      { field: "close", headerName: "종가", minWidth: 110, width: 110, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatNumber(params.value, priceDigits) },
      { field: "change_pct", headerName: "등락률", minWidth: 100, width: 100, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => (
          <span className={getSignedClass(params.value)}>{formatPercent(params.value)}</span>
        ) },
      { field: "volume", headerName: "거래량", minWidth: 120, width: 120, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatNumber(params.value, 0) },
      { field: "open", headerName: "시가", minWidth: 110, width: 110, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatNumber(params.value, priceDigits) },
      { field: "high", headerName: "고가", minWidth: 110, width: 110, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatNumber(params.value, priceDigits) },
      { field: "low", headerName: "저가", minWidth: 110, width: 110, type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatNumber(params.value, priceDigits) },
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
                {/* 종목 검색 자동완성 */}
                <div ref={searchRef} style={{ position: "relative" }}>
                  <input
                    ref={inputRef}
                    className="form-control"
                    type="text"
                    style={{ width: "280px", fontWeight: 600 }}
                    value={searchInput}
                    placeholder="티커 또는 종목명 검색"
                    onChange={(e) => handleSearchInputChange(e.target.value)}
                    onFocus={() => { if (searchInput.trim()) setShowDropdown(true); }}
                    onKeyDown={handleSearchKeyDown}
                  />
                  {showDropdown && filteredTickers.length > 0 ? (
                    <div style={{
                      position: "absolute",
                      top: "100%",
                      left: 0,
                      right: 0,
                      zIndex: 1000,
                      backgroundColor: "#fff",
                      border: "1px solid #e6e8ec",
                      borderRadius: 8,
                      boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
                      maxHeight: 320,
                      overflowY: "auto",
                      marginTop: 4,
                    }}>
                      {filteredTickers.map((item, idx) => (
                        <button
                          key={`${item.ticker_type}-${item.ticker}`}
                          type="button"
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 8,
                            width: "100%",
                            padding: "8px 12px",
                            border: "none",
                            backgroundColor: idx === highlightIndex ? "#eef4fb" : "transparent",
                            cursor: "pointer",
                            textAlign: "left",
                            fontSize: 14,
                          }}
                          onMouseEnter={() => setHighlightIndex(idx)}
                          onMouseDown={(e) => { e.preventDefault(); selectTicker(item); }}
                        >
                          <span className="appCodeText" style={{ fontWeight: 700, minWidth: 72 }}>{item.ticker}</span>
                          <span style={{ color: "#182433", flex: 1 }}>{item.name}</span>
                        </button>
                      ))}
                    </div>
                  ) : null}
                </div>

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

                {selectedTicker ? (
                  <div className="appSegmentedToggle appSegmentedToggleCompact" role="group" aria-label="조회 기간">
                    {MONTHS_OPTIONS.map((m) => (
                      <button
                        key={m}
                        type="button"
                        className={months === m ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                        onClick={() => handleMonthsChange(m)}
                      >
                        {m >= 12 ? `${m / 12}년` : `${m}개월`}
                      </button>
                    ))}
                  </div>
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
                      <span style={{ fontSize: 12, color: "#5b6778", fontWeight: 600 }}>이동평균선</span>
                      {MA_PERIODS.map((ma) => (
                        rows.length >= ma.period ? (
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

                {/* 일별 테이블 (나머지 공간 채움) */}
                {selectedTicker ? (
                  <>
                    <div style={{ padding: "12px 16px 4px", display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                      <span style={{ fontWeight: 700, fontSize: 15 }}>일별</span>
                      <span className="text-muted" style={{ fontSize: 13 }}>
                        총 {new Intl.NumberFormat("ko-KR").format(rows.length)}일
                      </span>
                    </div>
                    <div className="appGridFillWrap">
                      <AppAgGrid
                        rowData={reversedRows}
                        columnDefs={columns}
                        loading={loading}
                        theme={gridTheme}
                        gridOptions={{ suppressMovableColumns: true }}
                      />
                    </div>
                  </>
                ) : null}
              </>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
