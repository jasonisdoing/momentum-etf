"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  ColorType,
  CrosshairMode,
  LineSeries,
  LineStyle,
  CandlestickSeries,
  HistogramSeries,
  createChart,
  createSeriesMarkers,
} from "lightweight-charts";
import type { IChartApi, LineData, CandlestickData, HistogramData, Time } from "lightweight-charts";

import {
  ADR_LINE_COLOR,
  ADR_NEUTRAL_COLOR,
  ADR_OVERHEATED_COLOR,
  ADR_OVERSOLD_COLOR,
  describeAdrLevel,
} from "./adr-types";
import type { AdrPoint, AdrResponse } from "./adr-types";


type RegimeKey = "accel_up" | "accel_down";

// 그 일자 기준 SuperTrend 전환 가격선(상승/하락). 즉시 전환 기준이다.
type ForecastThresholds = {
  up_pct: number | null;
  up_price: number | null;
  dn_pct: number | null;
  dn_price: number | null;
  raw_regime: RegimeKey | null;
};

type HistoryPoint = {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
  ma: number | null;
  trend_pct: number | null;
  trend_score: number | null;
  regime: RegimeKey | null;
  forecast: ForecastThresholds | null;
  supertrend: number | null;
  supertrend_dir: number | null;
};

type HistoryResponse = {
  ticker: string;
  name: string;
  ma_days: number;
  ma_short_days: number;
  history: HistoryPoint[];
  trend_min_12m: number | null;
  trend_max_12m: number | null;
  /** 한국 지수만 값이 있다. 구성종목 데이터가 없는 지수는 null. */
  adr: AdrResponse | null;
  error?: string;
};

type MarketTrendChartProps = {
  ticker: string;
  name: string;
  maType: string;
  maDays: number;
  /** compact: 좌패널·게이지·추세전환·레짐·기간버튼을 숨기고 타이틀 + 차트만 표시(홈 대시보드용). */
  compact?: boolean;
};

type RegimeRange = {
  regime: RegimeKey;
  startDate: string;
  endDate: string;
  isCurrent: boolean;
  days: number;
};

type ChartRangeKey = "1m" | "3m" | "6m" | "ytd" | "1y" | "3y" | "5y";

const CHART_RANGES: Array<{ key: ChartRangeKey; label: string; days?: number; ytd?: boolean }> = [
  { key: "1m", label: "1개월", days: 31 },
  { key: "3m", label: "3개월", days: 92 },
  { key: "6m", label: "6개월", days: 183 },
  { key: "ytd", label: "연초이후", ytd: true },
  { key: "1y", label: "1년", days: 365 },
  { key: "3y", label: "3년", days: 365 * 3 },
  { key: "5y", label: "5년", days: 365 * 5 },
];

// 2단계: 상승(빨강) / 하락(파랑).
const REGIME_COLOR: Record<RegimeKey, string> = {
  accel_up: "#d62828",   // 빨강 — 상승
  accel_down: "#1971c2", // 파랑 — 하락
};

const REGIME_LABEL: Record<RegimeKey, string> = {
  accel_up: "⬆️ 상승",
  accel_down: "⬇️ 하락",
};

function parseDateKey(date: string): Date {
  return new Date(`${date}T00:00:00`);
}

function formatKoreanAxisMonth(time: Time, tickMarkType: number): string {
  // tickMarkType이 0(Year) 또는 1(Month) 일 때만 년월을 출력하고,
  // 2(Day) 이하의 상세 일별 틱마크는 빈 문자열로 리턴하여 중복 출력을 차단합니다.
  if (tickMarkType !== 0 && tickMarkType !== 1) {
    return "";
  }

  if (typeof time === "string") {
    const [year, month] = time.split("-");
    if (year && month) return `${year}년 ${Number(month)}월`;
    return time;
  }

  if (typeof time === "number") {
    const date = new Date(time * 1000);
    return `${date.getFullYear()}년 ${date.getMonth() + 1}월`;
  }

  return `${time.year}년 ${time.month}월`;
}

function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 2 }).format(value);
}

function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const rounded = Math.round(value);
  return `${rounded > 0 ? "+" : ""}${rounded}`;
}

/** 추세% 값을 부호 포함 1자리 + % 로 포맷. 범례용. */
function formatPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
}

function formatSignedPct(value: number): string {
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
}

/**
 * 현재 레짐에서 ``target`` 레짐으로 넘어가는 SuperTrend 전환 가격선을 표기한다.
 */
function regimeEntryText(fc: ForecastThresholds, current: RegimeKey | null, target: RegimeKey): string {
  let pct: number | null = null;
  let price: number | null = null;
  if (target === "accel_up") {
    pct = fc.up_pct;
    price = fc.up_price;
  } else if (target === "accel_down") {
    pct = fc.dn_pct;
    price = fc.dn_price;
  } else if (current === "accel_up") {
    pct = fc.up_pct;
    price = fc.up_price;
  } else if (current === "accel_down") {
    pct = fc.dn_pct;
    price = fc.dn_price;
  }
  if (pct === null || price === null) return "-";
  const rank: Record<RegimeKey, number> = { accel_up: 1, accel_down: 0 };
  const suffix = current !== null && rank[target] < rank[current] ? " 미만" : " 이상";
  return `${formatSignedPct(pct)} (${formatNumber(price)})${suffix}`;
}

function filterHistoryByRange(history: HistoryPoint[], rangeKey: ChartRangeKey): HistoryPoint[] {
  if (history.length === 0) return [];
  const range = CHART_RANGES.find((item) => item.key === rangeKey);
  if (!range) return history;

  const lastDate = parseDateKey(history[history.length - 1].date);
  let startDate: Date;
  if (range.ytd) {
    startDate = new Date(lastDate.getFullYear(), 0, 1);
  } else {
    startDate = new Date(lastDate);
    startDate.setDate(startDate.getDate() - (range.days ?? 365));
  }

  return history.filter((point) => parseDateKey(point.date) >= startDate);
}

function buildRawRegimeRanges(history: HistoryPoint[]): RegimeRange[] {
  type Raw = { regime: RegimeKey; startIdx: number; endIdx: number };
  const raw: Raw[] = [];
  let current: Raw | null = null;

  history.forEach((point, index) => {
    if (!point.regime) {
      if (current) {
        raw.push(current);
        current = null;
      }
      return;
    }
    if (!current || current.regime !== point.regime) {
      if (current) raw.push(current);
      current = { regime: point.regime, startIdx: index, endIdx: index };
    } else {
      current.endIdx = index;
    }
  });

  if (current) raw.push(current);

  return raw.map((range) => ({
    regime: range.regime,
    startDate: history[range.startIdx].date,
    endDate: history[range.endIdx].date,
    isCurrent: range.endIdx === history.length - 1,
    days: range.endIdx - range.startIdx + 1,
  }));
}

function buildRecentRegimeRanges(history: HistoryPoint[], limit: number): RegimeRange[] {
  return buildRawRegimeRanges(history).slice(-limit).reverse();
}

function buildBandRegimeRanges(history: HistoryPoint[]): RegimeRange[] {
  const ranges = buildRawRegimeRanges(history);
  return ranges.map((range, index) => {
    const next = ranges[index + 1];
    return {
      ...range,
      endDate: next?.startDate ?? range.endDate,
    };
  });
}

function buildLineData(history: HistoryPoint[], key: "close" | "ma"): LineData<Time>[] {
  return history
    .filter((point) => point[key] !== null)
    .map((point) => ({
      time: point.date as Time,
      value: point[key] as number,
    }));
}

function buildSuperTrendSegments(
  history: HistoryPoint[],
  targetDir: number
): LineData<Time>[][] {
  const segments: LineData<Time>[][] = [];
  let currentSegment: LineData<Time>[] = [];

  history.forEach((point) => {
    if (point.supertrend_dir === targetDir && point.supertrend !== null) {
      currentSegment.push({
        time: point.date as Time,
        value: point.supertrend,
      });
    } else if (currentSegment.length > 0) {
      segments.push(currentSegment);
      currentSegment = [];
    }
  });

  if (currentSegment.length > 0) {
    segments.push(currentSegment);
  }

  return segments;
}

function buildCandleData(history: HistoryPoint[]): CandlestickData<Time>[] {
  return history
    .filter(
      (point) =>
        point.open !== null &&
        point.high !== null &&
        point.low !== null &&
        point.close !== null
    )
    .map((point) => ({
      time: point.date as Time,
      open: point.open as number,
      high: point.high as number,
      low: point.low as number,
      close: point.close as number,
    }));
}

function buildVolumeData(history: HistoryPoint[]): HistogramData<Time>[] {
  return history
    .filter(
      (point) =>
        point.volume !== null &&
        point.open !== null &&
        point.close !== null
    )
    .map((point) => {
      const open = point.open as number;
      const close = point.close as number;
      const color = close >= open ? "rgba(220, 38, 38, 0.4)" : "rgba(37, 99, 235, 0.4)";
      return {
        time: point.date as Time,
        value: point.volume as number,
        color,
      };
    });
}

function renderRegimeBands(
  chart: IChartApi,
  overlay: HTMLDivElement,
  history: HistoryPoint[],
): void {
  overlay.innerHTML = "";
  const segments = buildBandRegimeRanges(history);
  const width = overlay.clientWidth;
  if (width <= 0) return;

  for (const segment of segments) {
    const start = chart.timeScale().timeToCoordinate(segment.startDate as Time);
    const end = chart.timeScale().timeToCoordinate(segment.endDate as Time);
    if (start === null && end === null) continue;
    const left = Math.max(0, start ?? 0);
    const right = Math.min(width, end ?? width);
    if (right <= left) continue;

    const band = document.createElement("div");
    band.style.position = "absolute";
    band.style.top = "0";
    band.style.bottom = "0";
    band.style.left = `${left}px`;
    band.style.width = `${right - left}px`;
    band.style.background = REGIME_COLOR[segment.regime];
    band.style.opacity = "0.16";
    overlay.appendChild(band);
  }
}



function formatShortMonthDay(date: string): string {
  const parts = date.split("-");
  if (parts.length !== 3) return date;
  const [, m, d] = parts;
  return `${Number(m)}/${Number(d)}`;
}

export function MarketTrendChart({
  ticker,
  name,
  maType,
  maDays,
  compact = false,
}: MarketTrendChartProps) {
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rangeKey, setRangeKey] = useState<ChartRangeKey>("1y");
  const showSuperTrend = true;
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const bandOverlayRef = useRef<HTMLDivElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);

  // 차트 판에는 라벨을 넣을 자리가 없어, 기간 버튼 줄 오른쪽에 ADR 현재값을 적는다.
  const adr = data?.adr ?? null;
  const adrSummary = useMemo(() => {
    if (!adr) return null;
    const level = describeAdrLevel(adr.latest_adr, adr);
    return (
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: 8 }}>
        <span style={{ fontWeight: 700, color: ADR_LINE_COLOR }}>ADR</span>
        <span style={{ fontSize: "var(--fs-lg)", fontWeight: 700, color: level.color }}>
          {adr.latest_adr != null ? adr.latest_adr.toFixed(1) : "-"}
        </span>
        <span style={{ fontSize: "var(--fs-sm)", fontWeight: 700, color: level.color }}>{level.label}</span>
        <span style={{ fontSize: "var(--fs-sm)", color: "var(--text-muted)" }}>
          {adr.window_days}일 누적 · 대상 {adr.universe_size}종목 · 과열 {adr.overheated} / 침체 {adr.oversold}
        </span>
      </div>
    );
  }, [adr]);

  useEffect(() => {
    let alive = true;
    // 요청 시점의 ticker 를 고정해 둔다 — 응답이 늦게 도착해도 다른 지수 데이터를 그리지 않는다.
    const requestedTicker = ticker;
    async function load() {
      try {
        // ticker 가 바뀌면 이전 지수 데이터를 먼저 비운다(다른 지수 화면이 잠시 남는 것 방지).
        setData(null);
        setLoading(true);
        setError(null);
        const response = await fetch(
          `/api/market-trend/history?ticker=${encodeURIComponent(requestedTicker)}`,
          { cache: "no-store" },
        );
        const payload = (await response.json()) as HistoryResponse;
        if (!response.ok) {
          throw new Error(payload.error ?? "히스토리를 불러오지 못했습니다.");
        }
        // 응답의 ticker 가 요청과 다르면 폐기한다(프록시·캐시 혼선 방어).
        const responseTicker = String(payload.ticker ?? requestedTicker);
        if (responseTicker !== requestedTicker) {
          throw new Error(`응답 지수가 요청과 다릅니다: ${responseTicker} ≠ ${requestedTicker}`);
        }
        if (alive) setData(payload);
      } catch (loadError) {
        if (alive)
          setError(loadError instanceof Error ? loadError.message : "히스토리를 불러오지 못했습니다.");
      } finally {
        if (alive) setLoading(false);
      }
    }
    load();
    return () => {
      alive = false;
    };
  }, [ticker]);

  const visibleHistory = useMemo(
    () => (data?.history ? filterHistoryByRange(data.history, rangeKey) : []),
    [data, rangeKey],
  );

  // 최신 SuperTrend 기준으로 다음 레짐 전환에 필요한 가격선을 보여준다.
  const forecastTransitions = useMemo(() => {
    const latest = data?.history.at(-1) ?? null;
    const fc = latest?.forecast;
    const current = latest?.regime;
    if (!fc || !current) return [];
    const rank: Record<RegimeKey, number> = { accel_up: 1, accel_down: 0 };
    const out: {
      next_regime: RegimeKey;
      target_price: number | null;
      change_pct: number | null;
      mode: "drop_below" | "rise_above";
    }[] = [];
    (["accel_up", "accel_down"] as RegimeKey[]).forEach((rg) => {
      if (rg === current) return;
      let pct: number | null = null;
      let price: number | null = null;
      if (rg === "accel_up") {
        pct = fc.up_pct;
        price = fc.up_price;
      } else if (rg === "accel_down") {
        pct = fc.dn_pct;
        price = fc.dn_price;
      }
      const mode: "drop_below" | "rise_above" = rank[rg] < rank[current] ? "drop_below" : "rise_above";
      out.push({ next_regime: rg, target_price: price, change_pct: pct, mode });
    });
    // 상승 → 하락 고정 순서로 배치.
    const regimeOrder: Record<RegimeKey, number> = { accel_up: 0, accel_down: 1 };
    out.sort((a, c) => regimeOrder[a.next_regime] - regimeOrder[c.next_regime]);
    return out.filter((t) => t.target_price !== null && t.change_pct !== null);
  }, [data]);

  const recentRegimeRanges = useMemo(() => {
    return visibleHistory.length > 0 ? buildRecentRegimeRanges(visibleHistory, 4) : [];
  }, [visibleHistory]);

  const latestPoint = data?.history.at(-1) ?? null;

  // 추세 전환 조건 아래에 붙는 기간 수익률 — 최신 종가 대비 N거래일 전 종가.
  const periodReturns = useMemo(() => {
    const hist = data?.history ?? [];
    const lastClose = hist.at(-1)?.close;
    if (lastClose == null || !(lastClose > 0)) return [];
    return [
      { label: "1개월", days: 20 },
      { label: "3개월", days: 60 },
      { label: "1년", days: 240 },
    ].map((p) => {
      const past = hist[hist.length - 1 - p.days]?.close;
      const pct = past != null && past > 0 ? (lastClose / past - 1) * 100 : null;
      return { label: p.label, pct };
    });
  }, [data]);

  useEffect(() => {
    const container = chartContainerRef.current;
    const overlay = bandOverlayRef.current;
    const tooltip = tooltipRef.current;
    if (!container || !overlay || !tooltip) return;

    chartRef.current?.remove();
    chartRef.current = null;
    tooltip.style.display = "none";
    overlay.innerHTML = "";

    if (visibleHistory.length < 2) return;

    const chart = createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#475569",
        fontSize: 12,
      },
      grid: {
        vertLines: { color: "rgba(226, 232, 240, 0.55)" },
        horzLines: { color: "rgba(203, 213, 225, 0.75)", style: 2 },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#e2e8f0" },
      timeScale: {
        borderColor: "#e2e8f0",
        timeVisible: false,
        secondsVisible: false,
        tickMarkFormatter: (time: Time, tickMarkType: number) => formatKoreanAxisMonth(time, tickMarkType),
        rightOffset: 0,
        barSpacing: Math.max(6, Math.min(12, container.clientWidth / Math.max(visibleHistory.length, 1))),
      },
      handleScroll: true,
      handleScale: true,
    });
    chartRef.current = chart;

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#dc2626",
      downColor: "#2563eb",
      borderUpColor: "#dc2626",
      borderDownColor: "#2563eb",
      wickUpColor: "#dc2626",
      wickDownColor: "#2563eb",
      priceLineVisible: false,
      // 마지막 종가 라벨은 아래 "현재가" 가로선 라벨과 중복이라 끈다.
      lastValueVisible: false,
    });
    candleSeries.setData(buildCandleData(visibleHistory));

    // 가로 점선: 현재가(검정) / 상승 전환 조건(빨강) / 하락 전환 조건(파랑).
    const latestForPriceLines = visibleHistory.at(-1) ?? null;
    const currentClose = latestForPriceLines?.close ?? null;
    const upTurnPrice = latestForPriceLines?.forecast?.up_price ?? null;
    const dnTurnPrice = latestForPriceLines?.forecast?.dn_price ?? null;
    // 제목은 라벨을 넓혀 최근 캔들을 가리므로 생략하고, 우측 축에는 값 라벨만 색으로 구분해 표시한다.
    if (currentClose != null) {
      candleSeries.createPriceLine({
        price: currentClose,
        color: "#111827",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
      });
    }
    if (upTurnPrice != null) {
      candleSeries.createPriceLine({
        price: upTurnPrice,
        color: "#dc2626",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
      });
    }
    if (dnTurnPrice != null) {
      candleSeries.createPriceLine({
        price: dnTurnPrice,
        color: "#2563eb",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
      });
    }

    const markerLineSeries = chart.addSeries(LineSeries, {
      color: "rgba(0, 0, 0, 0)",
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      autoscaleInfoProvider: () => null,
    });

    const markerLineData: any[] = [];
    const markers: any[] = [];
    if (showSuperTrend) {
      let prevDir: number | null = null;
      visibleHistory.forEach((point) => {
        if (point.supertrend_dir === null || point.supertrend_dir === undefined) return;
        if (prevDir !== null && point.supertrend_dir !== prevDir) {
          const isUp = point.supertrend_dir === 1;
          const lowVal = ((point.low !== null && point.low !== undefined) ? point.low : point.close) as number;
          const highVal = ((point.high !== null && point.high !== undefined) ? point.high : point.close) as number;
          const offsetPrice = isUp ? lowVal * 0.935 : highVal * 1.065;

          markerLineData.push({
            time: point.date as Time,
            value: offsetPrice,
          });

          markers.push({
            time: point.date as Time,
            position: "inBar",
            color: isUp ? "#fa5252" : "#228be6",
            shape: isUp ? "arrowUp" : "arrowDown",
            size: 1.5,
          });
        }
        prevDir = point.supertrend_dir;
      });
    }
    markerLineSeries.setData(markerLineData);
    createSeriesMarkers(markerLineSeries, markers);

    // 거래량 데이터가 없는 지수(예: 필라델피아 반도체)는 히스토그램의 빈 바닥선만 남으므로 아예 그리지 않는다.
    const hasVolume = visibleHistory.some((point) => point.volume != null && point.volume > 0);
    if (hasVolume) {
      const volumeSeries = chart.addSeries(HistogramSeries, {
        priceFormat: {
          type: "volume",
        },
        priceScaleId: "volume",
        priceLineVisible: false,
        lastValueVisible: false,
      });
      volumeSeries.setData(buildVolumeData(visibleHistory));

      chart.priceScale("volume").applyOptions({
        scaleMargins: {
          top: 0.75,
          bottom: 0,
        },
      });
    }

    // ADR 은 지수와 단위가 달라(60~200 대 수천 pt) 같은 축에 얹으면 축 설정에 따라
    // 교차점이 멋대로 움직인다. 별도 판(pane)에 두면 축은 나뉘고 시간축만 공유해
    // 세로로 같은 위치가 항상 같은 날짜가 된다 — 지수와 어긋나는 구간을 그대로 읽을 수 있다.
    if (adr && adr.points.length >= 2) {
      // 지수와 같은 구간만 남긴다. ADR 이 지수보다 앞선 날짜를 갖고 있으면 시간축이 그만큼
      // 늘어나, 기간 버튼으로 고른 구간(예: 6개월)의 캔들이 오른쪽으로 밀려 좁아진다.
      const fromDate = visibleHistory[0].date;
      const toDate = visibleHistory[visibleHistory.length - 1].date;
      const adrLine = adr.points
        .filter((point): point is AdrPoint & { adr: number } => point.adr != null)
        .filter((point) => point.date >= fromDate && point.date <= toDate)
        .map((point) => ({ time: point.date as Time, value: point.adr }));

      if (adrLine.length >= 2) {
        const ADR_PANE_INDEX = 1;
        const adrSeries = chart.addSeries(
          LineSeries,
          {
            color: ADR_LINE_COLOR,
            lineWidth: 2,
            priceLineVisible: false,
            lastValueVisible: true,
          },
          ADR_PANE_INDEX,
        );
        adrSeries.setData(adrLine);

        // 과열·강세약세·침체 경계. 이 선을 넘나드는 지점이 판단 기준이라 축 라벨까지 남긴다.
        // 가운데 100 은 상승 종목수와 하락 종목수가 같아지는 지점 — 강세/약세를 가른다.
        for (const [price, color] of [
          [adr.overheated, ADR_OVERHEATED_COLOR],
          [adr.neutral, ADR_NEUTRAL_COLOR],
          [adr.oversold, ADR_OVERSOLD_COLOR],
        ] as const) {
          adrSeries.createPriceLine({
            price,
            color,
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            axisLabelVisible: true,
          });
        }

        // 지수 판 4 : ADR 판 1 — 거래량이 차지하던 정도의 높이.
        chart.panes()[0]?.setStretchFactor(4);
        chart.panes()[ADR_PANE_INDEX]?.setStretchFactor(1);
      }
    }

    chart.addSeries(LineSeries, {
      color: "#16a34a",
      lineWidth: 2,
      lineStyle: 0,
      priceLineVisible: false,
      lastValueVisible: false,
    }).setData(buildLineData(visibleHistory, "ma"));

    if (showSuperTrend) {
      buildSuperTrendSegments(visibleHistory, 1).forEach((segment) => {
        chart.addSeries(LineSeries, {
          color: "#fa5252",
          lineWidth: 2,
          lineStyle: 0,
          priceLineVisible: false,
          lastValueVisible: false,
        }).setData(segment);
      });

      buildSuperTrendSegments(visibleHistory, -1).forEach((segment) => {
        chart.addSeries(LineSeries, {
          color: "#228be6",
          lineWidth: 2,
          lineStyle: 0,
          priceLineVisible: false,
          lastValueVisible: false,
        }).setData(segment);
      });
    }

    chart.timeScale().fitContent();

    const pointByDate = new Map(visibleHistory.map((point) => [point.date, point]));

    // 각 날짜별 레짐 지속 일수를 계산
    const regimeDaysByDate = new Map<string, number>();
    let currentRegime: string | null = null;
    let streak = 0;
    if (data?.history) {
      data.history.forEach((point) => {
        if (!point.regime) {
          currentRegime = null;
          streak = 0;
          return;
        }
        if (point.regime === currentRegime) {
          streak += 1;
        } else {
          currentRegime = point.regime;
          streak = 1;
        }
        regimeDaysByDate.set(point.date, streak);
      });
    }

    chart.subscribeCrosshairMove((param) => {
      if (!param.point || !param.time) {
        tooltip.style.display = "none";
        return;
      }

      const date = String(param.time);
      const point = pointByDate.get(date);
      if (!point) {
        tooltip.style.display = "none";
        return;
      }

      const days = regimeDaysByDate.get(point.date) || 1;
      // 각 일자는 그날 기준 SuperTrend 전환 가격선을 자체적으로 갖는다(point.forecast).
      const fc = point.forecast;

      const getRegimeStatusText = (key: RegimeKey) => {
        if (point.regime === key) {
          return `<span style="color: #63e6be; font-weight: 700;">${days}일차</span>`;
        }
        if (fc) {
          return `<span style="color: #ffc078; font-weight: 700;">${regimeEntryText(fc, point.regime, key)}</span>`;
        }
        return `<span style="color: #94a3b8;">-</span>`;
      };

      const statusRows = fc
        ? `
        <div style="margin-top: 6px; border-top: 1px dashed rgba(255,255,255,0.25); padding-top: 6px; display: flex; flex-direction: column; gap: 3px;">
          <div style="font-weight: 700; color: #ffffff; margin-bottom: 2px; font-size: var(--fs-sm);">ST(SuperTrend) 전환 조건</div>
          <div style="display: flex; justify-content: space-between; gap: 15px;">
            <span>상승:</span> <strong>${getRegimeStatusText("accel_up")}</strong>
          </div>
          <div style="display: flex; justify-content: space-between; gap: 15px;">
            <span>하락:</span> <strong>${getRegimeStatusText("accel_down")}</strong>
          </div>
        </div>
      `
        : "";

      const regimeLabelText = point.regime ? REGIME_LABEL[point.regime] : "-";
      const regimeTextColor = point.regime ? REGIME_COLOR[point.regime] : "#ffffff";

      tooltip.innerHTML = `
        <div style="font-weight:700;margin-bottom:6px;color:#ffffff;font-size:var(--fs-sm);">${point.date}</div>
        <div style="display: flex; flex-direction: column; gap: 4px; font-size: var(--fs-sm); color: #e2e8f0;">
          <div>상태: <strong style="color: ${regimeTextColor}">${regimeLabelText} (${days}일차)</strong></div>
          <div>추세 점수: <strong style="color: #ffffff">${formatScore(point.trend_score)}</strong></div>
          ${statusRows}
        </div>
      `;
      tooltip.style.display = "block";

      const tooltipWidth = tooltip.offsetWidth || 180;
      const tooltipHeight = tooltip.offsetHeight || 100;
      const left =
        param.point.x + tooltipWidth + 16 > container.clientWidth
          ? param.point.x - tooltipWidth - 12
          : param.point.x + 12;
      const top =
        param.point.y + tooltipHeight + 16 > container.clientHeight
          ? param.point.y - tooltipHeight - 12
          : param.point.y + 12;
      tooltip.style.left = `${Math.max(8, left)}px`;
      tooltip.style.top = `${Math.max(8, top)}px`;
    });

    const redrawBands = () => {
      requestAnimationFrame(() => {
        renderRegimeBands(chart, overlay, visibleHistory);
      });
    };
    redrawBands();
    chart.timeScale().subscribeVisibleLogicalRangeChange(redrawBands);

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        chart.applyOptions({ width: entry.contentRect.width, height: entry.contentRect.height });
        redrawBands();
      }
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
      overlay.innerHTML = "";
      tooltip.style.display = "none";
    };
  }, [visibleHistory, showSuperTrend, adr]);

  return (
    <div
      style={{
        padding: compact ? "12px 14px" : "16px 20px",
        background: compact ? "#ffffff" : "#f8f9fa",
        height: "100%",
        boxSizing: "border-box",
      }}
    >
      {error ? (
        <div className="alert alert-danger mb-0">{error}</div>
      ) : loading && !data ? (
        <div style={{ color: "var(--text-muted)", padding: 20 }}>불러오는 중...</div>
      ) : visibleHistory.length < 2 ? (
        <div style={{ color: "var(--text-muted)", padding: 20 }}>표시할 데이터가 없습니다.</div>
      ) : (
        <div style={{ display: "flex", height: compact ? "100%" : "calc(100% - 32px)", minHeight: compact ? 0 : 300, flexDirection: "column" }}>
          {compact ? (() => {
            const cClose = latestPoint?.close ?? null;
            const cPrev = data?.history.at(-2)?.close ?? null;
            const cPct = cClose != null && cPrev != null && cPrev !== 0 ? (cClose / cPrev - 1) * 100 : null;
            // 추세 전환 조건 — 반대 레짐으로 바뀌는 목표가/변화율(SuperTrend 기준). 레짐은 2상태라 1건.
            const trans = forecastTransitions[0];
            const transColor = trans ? REGIME_COLOR[trans.next_regime] : "#64748b";
            // 현재 레짐 지속일수 — /market-trend 의 "상승/하락 N일째"와 동일 개념(현재 레짐 연속일).
            const regimeDays = (() => {
              const hist = data?.history ?? [];
              const cur = latestPoint?.regime;
              if (!cur) return null;
              let count = 0;
              for (let i = hist.length - 1; i >= 0; i -= 1) {
                if (hist[i].regime === cur) count += 1;
                else break;
              }
              return count;
            })();
            return (
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12, flexWrap: "wrap" }}>
                <strong style={{ fontSize: "var(--fs-2xl)", fontWeight: 800, letterSpacing: "-0.02em", color: "#0f172a" }}>{name}</strong>
                {cClose != null ? (
                  <span style={{ fontSize: "var(--fs-2xl)", fontWeight: 800, color: "#1e293b" }}>{formatNumber(cClose)}</span>
                ) : null}
                {cPct != null ? (
                  <span style={{ fontSize: "var(--fs-xl)", fontWeight: 800, color: cPct >= 0 ? "#d62828" : "#1971c2" }}>
                    {cPct >= 0 ? "+" : ""}
                    {cPct.toFixed(2)}%
                  </span>
                ) : null}
                <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", justifyContent: "flex-end" }}>
                  {trans && trans.change_pct != null ? (
                    <span
                      title={`${formatNumber(trans.target_price)}pt ${trans.mode === "drop_below" ? "아래로 내려가면" : "이상으로 마감하면"} ${REGIME_LABEL[trans.next_regime]} 추세로 전환`}
                      style={{
                        fontSize: "var(--fs-base)",
                        fontWeight: 800,
                        color: transColor,
                        whiteSpace: "nowrap",
                      }}
                    >
                      {trans.mode === "drop_below" ? "↓" : "↑"} {trans.change_pct >= 0 ? "+" : ""}
                      {trans.change_pct.toFixed(1)}%면 {REGIME_LABEL[trans.next_regime]}
                    </span>
                  ) : null}
                  {latestPoint?.regime ? (
                    <span
                      style={{
                        fontSize: "var(--fs-base)",
                        fontWeight: 800,
                        padding: "5px 15px",
                        borderRadius: 999,
                        color: "#fff",
                        background: REGIME_COLOR[latestPoint.regime],
                        boxShadow: `0 2px 8px ${REGIME_COLOR[latestPoint.regime]}45`,
                      }}
                    >
                      {latestPoint.regime === "accel_up" ? "상승" : "하락"}
                      {regimeDays != null ? ` ${regimeDays}일째` : ""}
                    </span>
                  ) : null}
                </div>
              </div>
            );
          })() : null}
          {!compact ? (
          <>
          {/* 추세 전환 조건(왼쪽) · 기간 수익률(오른쪽) — 좌우 반반. */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 16, alignItems: "stretch", marginBottom: 12 }}>
            <div style={{ flex: "1 1 320px", minWidth: 0 }}>
            {forecastTransitions.length > 0 && (
              <div
                style={{
                  margin: 0,
                  padding: "12px 16px",
                  borderRadius: "8px",
                  background: "rgba(245, 158, 11, 0.08)",
                  border: "1px solid rgba(245, 158, 11, 0.25)",
                  color: "#d97706",
                  fontSize: "var(--fs-sm)",
                  lineHeight: 1.5,
                  boxShadow: "0 2px 8px rgba(245, 158, 11, 0.03)",
                }}
              >
                <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: 6 }}>
                  <span style={{ fontSize: "var(--fs-lg)", lineHeight: 1 }}>⚠️</span>
                  <strong style={{ fontSize: "var(--fs-sm)" }}>
                    추세 전환 조건 <span style={{ fontWeight: 400, opacity: 0.85 }}>(SuperTrend 기준)</span>
                  </strong>
                </div>
                <ul style={{ margin: 0, paddingLeft: 20, overflowWrap: "anywhere", whiteSpace: "normal" }}>
                  {forecastTransitions.map((item, idx) => (
                    <li
                      key={idx}
                      style={{
                        marginBottom: idx === forecastTransitions.length - 1 ? 0 : 4,
                        color: REGIME_COLOR[item.next_regime],
                      }}
                    >
                      {data?.name} 지수가{" "}
                      <span style={{ fontWeight: 800, textDecoration: "underline" }}>
                        {formatNumber(item.target_price)}
                      </span>
                      pt
                      {item.mode === "drop_below" ? " 아래로 내려가면" : " 이상으로 마감하면"}
                      {" "}(현재 대비{" "}
                      <span style={{ fontWeight: 800 }}>
                        {item.change_pct! > 0 ? "+" : ""}
                        {item.change_pct!.toFixed(1)}%
                      </span>
                      ), 시장 상태가{" "}
                      <span style={{ fontWeight: 800 }}>{REGIME_LABEL[item.next_regime]}</span>
                      으로 변경될 것으로 예상됩니다.
                    </li>
                  ))}
                </ul>
              </div>
            )}
            </div>
            {periodReturns.length > 0 ? (
              <div style={{ flex: "1 1 320px", minWidth: 0 }}>
                <div style={{ fontSize: "var(--fs-sm)", fontWeight: 700, color: "#5f6b82", marginBottom: 4 }}>기간 수익률</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                  {periodReturns.map((r) => (
                    <div key={r.label} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6, fontSize: "var(--fs-sm)" }}>
                      <span style={{ fontWeight: 700 }}>{r.label}</span>
                      <span style={{ fontWeight: 800, color: r.pct == null ? "#94a3b8" : r.pct >= 0 ? "#d62828" : "#1971c2" }}>{formatPct(r.pct)}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
          {/* 최근 레짐 이력 뱃지 타임라인 (차트 밖 상단으로 배치하여 차트 가림 방지) */}
          {recentRegimeRanges.length > 0 && (
            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", marginBottom: 10, alignItems: "center", justifyContent: "flex-end" }}>
              <span style={{ fontSize: "var(--fs-sm)", fontWeight: 700, color: "#5f6b82" }}>최근 레짐 구간:</span>
              {recentRegimeRanges.slice().reverse().map((range, idx) => {
                const startTxt = formatShortMonthDay(range.startDate);
                const endTxt = range.isCurrent ? "현재" : formatShortMonthDay(range.endDate);
                return (
                  <span
                    key={idx}
                    style={{
                      padding: "2px 8px",
                      borderRadius: "10px",
                      fontSize: "var(--fs-sm)",
                      fontWeight: "700",
                      color: "#fff",
                      background: REGIME_COLOR[range.regime],
                      boxShadow: "0 1px 4px rgba(15, 23, 42, 0.12)",
                      display: "inline-flex",
                      alignItems: "center",
                    }}
                  >
                    {REGIME_LABEL[range.regime].replace(/^[^\s]+\s/, "")} {startTxt}~{endTxt}({range.days}일)
                  </span>
                );
              })}
            </div>
          )}

          <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 8, justifyContent: "space-between", alignItems: "center" }}>
            <div className="appSegmentedToggle" role="group" aria-label="시장지수 차트 기간">
              {CHART_RANGES.map((range) => (
                <button
                  key={range.key}
                  type="button"
                  className={`btn appSegmentedToggleButton ${rangeKey === range.key ? "is-active" : ""}`}
                  onClick={() => setRangeKey(range.key)}
                >
                  {range.label}
                </button>
              ))}
            </div>
            {adrSummary}
          </div>
          </>
          ) : null}
          <div style={{ position: "relative", width: "100%", minHeight: 220, flex: "1 1 auto" }}>
            <div
              ref={bandOverlayRef}
              style={{
                position: "absolute",
                inset: 0,
                zIndex: 0,
                overflow: "hidden",
                pointerEvents: "none",
              }}
            />
            <div
              ref={chartContainerRef}
              style={{ position: "absolute", inset: 0, zIndex: 1 }}
            />
            <div
              ref={tooltipRef}
              style={{
                display: "none",
                position: "absolute",
                zIndex: 3,
                minWidth: 230,
                whiteSpace: "nowrap",
                padding: "10px 13px",
                borderRadius: 6,
                background: "rgba(30, 41, 59, 0.95)",
                color: "#fff",
                fontSize: "var(--fs-sm)",
                lineHeight: 1.5,
                pointerEvents: "none",
                boxShadow: "0 8px 20px rgba(15, 23, 42, 0.18)",
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
