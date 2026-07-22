"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  ColorType,
  CrosshairMode,
  LineSeries,
  CandlestickSeries,
  HistogramSeries,
  createChart,
  createSeriesMarkers,
} from "lightweight-charts";
import type { IChartApi, LineData, CandlestickData, HistogramData, Time } from "lightweight-charts";


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
  offense_pct: number | null;
  defense_pct: number | null;
  error?: string;
};

type MarketTrendChartProps = {
  ticker: string;
  name: string;
  maType: string;
  maDays: number;
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
}: MarketTrendChartProps) {
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rangeKey, setRangeKey] = useState<ChartRangeKey>("6m");
  const showSuperTrend = true;
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const bandOverlayRef = useRef<HTMLDivElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    let alive = true;
    async function load() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch(
          `/api/market-trend/history?ticker=${encodeURIComponent(ticker)}`,
          { cache: "no-store" },
        );
        const payload = (await response.json()) as HistoryResponse;
        if (!response.ok) {
          throw new Error(payload.error ?? "히스토리를 불러오지 못했습니다.");
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
      lastValueVisible: true,
    });
    candleSeries.setData(buildCandleData(visibleHistory));

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

    chart.addSeries(LineSeries, {
      color: "#fa5252",
      lineWidth: 1,
      lineStyle: 2,
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
          <div style="font-weight: 700; color: #ffffff; margin-bottom: 2px; font-size: 11px;">ST(SuperTrend) 전환 조건</div>
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
        <div style="font-weight:700;margin-bottom:6px;color:#ffffff;font-size:12px;">${point.date}</div>
        <div style="display: flex; flex-direction: column; gap: 4px; font-size: 11px; color: #e2e8f0;">
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
  }, [visibleHistory, showSuperTrend]);

  return (
    <div
      style={{
        padding: "16px 20px",
        background: "#f8f9fa",
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
        <div style={{ display: "flex", height: "calc(100% - 32px)", minHeight: 300, flexDirection: "column" }}>
          <div style={{ marginBottom: 12 }}>
            {data?.offense_pct != null && data?.defense_pct != null ? (
              <>
                {/* 공격/방어 비율 — 기준 이동평균선(MA) 위면 공격 100, 아래면 12개월 최저까지의 거리로 방어. */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: 4, marginBottom: 6 }}>
                  <span style={{ fontSize: "0.72rem", fontWeight: 700, color: "#5f6b82" }}>
                    공격 / 방어 <span style={{ fontWeight: 400 }}>({maType}{maDays} 기준)</span>
                  </span>
                  <span style={{ fontSize: "0.82rem", fontWeight: 800 }}>
                    <span style={{ color: "#d62828" }}>공격 {data.offense_pct}%</span>
                    <span style={{ color: "#94a3b8" }}> · </span>
                    <span style={{ color: "#1971c2" }}>방어 {data.defense_pct}%</span>
                  </span>
                </div>
                <div
                  style={{ display: "flex", height: 34, borderRadius: 8, overflow: "hidden", border: "1px solid #e2e8f0", background: "#fff" }}
                  title={`${maType}${maDays} 괴리율 ${formatPct(latestPoint?.trend_pct)} · 공격 ${data.offense_pct}% / 방어 ${data.defense_pct}%`}
                >
                  {data.offense_pct > 0 ? (
                    <div style={{ width: `${data.offense_pct}%`, background: "#d62828", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.78rem", fontWeight: 800 }}>
                      {data.offense_pct >= 20 ? `공격 ${data.offense_pct}%` : ""}
                    </div>
                  ) : null}
                  {data.defense_pct > 0 ? (
                    <div style={{ width: `${data.defense_pct}%`, background: "#1971c2", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.78rem", fontWeight: 800 }}>
                      {data.defense_pct >= 20 ? `방어 ${data.defense_pct}%` : ""}
                    </div>
                  ) : null}
                </div>
                {/* 범례: 왼쪽=MA 위(공격 100) / 오른쪽=연최저(방어 100) */}
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", color: "#5f6b82", marginTop: 6 }}>
                  <span>MA 위 = 공격 100</span>
                  <span>연최저 = 방어 100</span>
                </div>
              </>
            ) : null}
          </div>
          {forecastTransitions.length > 0 && (
            <div
              style={{
                marginBottom: 12,
                padding: "12px 16px",
                borderRadius: "8px",
                background: "rgba(245, 158, 11, 0.08)",
                border: "1px solid rgba(245, 158, 11, 0.25)",
                color: "#d97706",
                fontSize: "0.82rem",
                lineHeight: 1.5,
                boxShadow: "0 2px 8px rgba(245, 158, 11, 0.03)",
              }}
            >
              <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: 6 }}>
                <span style={{ fontSize: "1.1rem", lineHeight: 1 }}>⚠️</span>
                <strong style={{ fontSize: "0.85rem" }}>
                  추세 전환 조건 <span style={{ fontWeight: 400, opacity: 0.85 }}>(SuperTrend 기준)</span>
                </strong>
              </div>
              <ul style={{ margin: 0, paddingLeft: 20 }}>
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
          {/* 최근 레짐 이력 뱃지 타임라인 (차트 밖 상단으로 배치하여 차트 가림 방지) */}
          {recentRegimeRanges.length > 0 && (
            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", marginBottom: 10, alignItems: "center", justifyContent: "flex-end" }}>
              <span style={{ fontSize: "0.78rem", fontWeight: 700, color: "#5f6b82" }}>최근 레짐 구간:</span>
              {recentRegimeRanges.slice().reverse().map((range, idx) => {
                const startTxt = formatShortMonthDay(range.startDate);
                const endTxt = range.isCurrent ? "현재" : formatShortMonthDay(range.endDate);
                return (
                  <span
                    key={idx}
                    style={{
                      padding: "2px 8px",
                      borderRadius: "10px",
                      fontSize: "0.72rem",
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
          </div>
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
                fontSize: "0.84rem",
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
