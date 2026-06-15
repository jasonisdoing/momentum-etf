"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  ColorType,
  CrosshairMode,
  LineSeries,
  createChart,
} from "lightweight-charts";
import type { IChartApi, LineData, Time } from "lightweight-charts";


type RegimeKey = "accel_up" | "decel_up" | "accel_down" | "decel_down";

type HistoryPoint = {
  date: string;
  close: number | null;
  ma: number | null;
  trend_pct: number | null;
  trend_score: number | null;
  regime: RegimeKey | null;
};

type HistoryResponse = {
  ticker: string;
  name: string;
  ma_type: string;
  ma_months: number;
  history: HistoryPoint[];
  trend_min_12m: number | null;
  trend_max_12m: number | null;
  error?: string;
};

type MarketTrendChartProps = {
  ticker: string;
  name: string;
  maType: string;
  maMonths: number;
};

type RegimeRange = {
  regime: RegimeKey;
  startDate: string;
  endDate: string;
  isCurrent: boolean;
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

// 3단계 통합: 상승(빨강) / 중립(녹색) / 하락(파랑). 중립은 sub-label 로 조정/진정 구분.
const REGIME_COLOR: Record<RegimeKey, string> = {
  accel_up: "#d62828",   // 빨강 — 상승
  decel_up: "#2f9e44",   // 녹색 — 중립(조정)
  decel_down: "#2f9e44", // 녹색 — 중립(진정)
  accel_down: "#1971c2", // 파랑 — 하락
};

const REGIME_LABEL: Record<RegimeKey, string> = {
  accel_up: "⬆️ 상승",
  decel_up: "➡️ 중립 (조정)",
  decel_down: "➡️ 중립 (진정)",
  accel_down: "⬇️ 하락",
};

function parseDateKey(date: string): Date {
  return new Date(`${date}T00:00:00`);
}

function formatKoreanAxisMonth(time: Time): string {
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

type GaugeData = {
  todayPct: number; // 오늘 추세%의 막대 내 위치 (0~100)
  zeroPct: number | null; // MA선(0%)의 위치 (범위가 0을 교차할 때만)
  trendMin: number;
  trendMax: number;
};

/**
 * 12개월 추세% 범위를 가로 막대로 표시.
 *   0% = 12개월 최저 추세 (trendMin) / 100% = 12개월 최고 추세 (trendMax)
 * 막대는 MA선(0)을 기준으로 아래(파랑)/위(빨강) 두 영역으로 나뉘고,
 * 오늘 핀의 색은 현재 레짐(slope 기반)으로 칠한다. (가속/감속 밴드는 그리지 않음 —
 * 레짐은 위치가 아니라 추세% 기울기로 결정되므로 1D 위치 밴드로 표현할 수 없다.)
 */
function computeGaugeData({
  trend,
  trendMin,
  trendMax,
}: {
  trend: number | null | undefined;
  trendMin: number | null | undefined;
  trendMax: number | null | undefined;
}): GaugeData | null {
  if (
    trend === null || trend === undefined || Number.isNaN(trend) ||
    trendMin === null || trendMin === undefined || Number.isNaN(trendMin) ||
    trendMax === null || trendMax === undefined || Number.isNaN(trendMax) ||
    trendMax <= trendMin
  ) {
    return null;
  }
  const project = (v: number) =>
    Math.max(0, Math.min(100, ((v - trendMin) / (trendMax - trendMin)) * 100));

  const zeroPct = trendMin < 0 && trendMax > 0 ? project(0) : null;

  return {
    todayPct: project(trend),
    zeroPct,
    trendMin,
    trendMax,
  };
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

/** 현재 + 최근 3개 레짐 구간을 차트 상단에 라벨로 표시 (4개). */
function renderRecentRegimeLabels(
  chart: IChartApi,
  overlay: HTMLDivElement,
  history: HistoryPoint[],
): void {
  overlay.innerHTML = "";
  const width = overlay.clientWidth;
  if (width <= 0) return;
  // 현재(가장 최근) + 직전 3개 = 4개. ranges[0] 이 현재.
  const ranges = buildRecentRegimeRanges(history, 4);

  // 현재(top)부터 1단계씩 내려가며 4단으로 배치.
  const ROW_HEIGHT = 22;
  const TOP_OFFSET = 4;
  ranges.forEach((range, idx) => {
    const start = chart.timeScale().timeToCoordinate(range.startDate as Time);
    const end = chart.timeScale().timeToCoordinate(range.endDate as Time);
    if (start === null && end === null) return;
    const left = Math.max(0, start ?? 0);
    const right = Math.min(width, end ?? width);
    // 1일치 구간(start === end)도 허용 — center 는 그 단일 좌표.
    if (right < left) return;
    const center = (left + right) / 2;

    const label = document.createElement("div");
    label.style.position = "absolute";
    label.style.top = `${TOP_OFFSET + idx * ROW_HEIGHT}px`;
    label.style.left = `${center}px`;
    label.style.transform = "translateX(-50%)";
    label.style.padding = "2px 8px";
    label.style.borderRadius = "10px";
    label.style.fontSize = "11px";
    label.style.fontWeight = "700";
    label.style.color = "#fff";
    label.style.background = REGIME_COLOR[range.regime];
    label.style.whiteSpace = "nowrap";
    label.style.pointerEvents = "none";
    label.style.boxShadow = "0 2px 6px rgba(15, 23, 42, 0.18)";

    const startTxt = formatShortMonthDay(range.startDate);
    const endTxt = range.isCurrent ? "현재" : formatShortMonthDay(range.endDate);
    label.textContent = `${REGIME_LABEL[range.regime].replace(/^[^\s]+\s/, "")} ${startTxt}~${endTxt}`;
    overlay.appendChild(label);

    // 라벨이 차트 경계를 넘으면 안쪽으로 클램프 (특히 현재 구간이 오른쪽 끝에 있을 때).
    const half = label.offsetWidth / 2;
    const minCenter = half + 4;
    const maxCenter = width - half - 4;
    const clamped = Math.max(minCenter, Math.min(maxCenter, center));
    if (clamped !== center) {
      label.style.left = `${clamped}px`;
    }
  });
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
  maMonths,
}: MarketTrendChartProps) {
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rangeKey, setRangeKey] = useState<ChartRangeKey>("ytd");
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const bandOverlayRef = useRef<HTMLDivElement | null>(null);
  const labelOverlayRef = useRef<HTMLDivElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    let alive = true;
    async function load() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch(
          `/api/market-trend/history?ticker=${encodeURIComponent(ticker)}&ma_type=${encodeURIComponent(maType)}&ma_months=${encodeURIComponent(String(maMonths))}`,
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
  }, [ticker, maType, maMonths]);

  const visibleHistory = useMemo(
    () => (data?.history ? filterHistoryByRange(data.history, rangeKey) : []),
    [data, rangeKey],
  );

  const latestPoint = data?.history.at(-1) ?? null;
  const gaugeData = computeGaugeData({
    trend: latestPoint?.trend_pct,
    trendMin: data?.trend_min_12m,
    trendMax: data?.trend_max_12m,
  });
  const gaugeLeft = gaugeData?.todayPct ?? null;
  const latestRegime = latestPoint?.regime ?? null;

  useEffect(() => {
    const container = chartContainerRef.current;
    const overlay = bandOverlayRef.current;
    const labelsOverlay = labelOverlayRef.current;
    const tooltip = tooltipRef.current;
    if (!container || !overlay || !tooltip || !labelsOverlay) return;

    chartRef.current?.remove();
    chartRef.current = null;
    tooltip.style.display = "none";
    overlay.innerHTML = "";
    labelsOverlay.innerHTML = "";

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
        tickMarkFormatter: (time: Time) => formatKoreanAxisMonth(time),
        rightOffset: 0,
        barSpacing: Math.max(6, Math.min(12, container.clientWidth / Math.max(visibleHistory.length, 1))),
      },
      handleScroll: true,
      handleScale: true,
    });
    chartRef.current = chart;

    const closeSeries = chart.addSeries(LineSeries, {
      color: "#1f2937",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
    });
    closeSeries.setData(buildLineData(visibleHistory, "close"));

    chart.addSeries(LineSeries, {
      color: "#fa5252",
      lineWidth: 1,
      lineStyle: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    }).setData(buildLineData(visibleHistory, "ma"));

    chart.timeScale().fitContent();

    const pointByDate = new Map(visibleHistory.map((point) => [point.date, point]));

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

      const regimeText = point.regime
        ? `<div style="margin-top:4px;color:${REGIME_COLOR[point.regime]};font-weight:700">${REGIME_LABEL[point.regime]}</div>`
        : "";
      tooltip.innerHTML = `
        <div style="font-weight:700;margin-bottom:2px">${point.date}</div>
        <div>종가: ${formatNumber(point.close)}</div>
        <div>MA: ${formatNumber(point.ma)}</div>
        <div>추세 점수: ${formatScore(point.trend_score)}</div>
        ${regimeText}
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
        renderRecentRegimeLabels(chart, labelsOverlay, visibleHistory);
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
      labelsOverlay.innerHTML = "";
      tooltip.style.display = "none";
    };
  }, [visibleHistory]);

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
        <div style={{ color: "#868e96", padding: 20 }}>불러오는 중...</div>
      ) : visibleHistory.length < 2 ? (
        <div style={{ color: "#868e96", padding: 20 }}>표시할 데이터가 없습니다.</div>
      ) : (
        <div style={{ display: "flex", height: "calc(100% - 32px)", minHeight: 300, flexDirection: "column" }}>
          <div style={{ marginBottom: 12 }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 8 }}>
              <div
                style={{
                  color: latestRegime ? REGIME_COLOR[latestRegime] : "#1f2937",
                  fontSize: "1.7rem",
                  fontWeight: 800,
                  lineHeight: 1,
                }}
              >
                {latestRegime ? REGIME_LABEL[latestRegime] : "레짐 없음"}
              </div>
            </div>
            {gaugeData ? (
              <>
                {/* 게이지 본체 — MA(0) 기준 아래(파랑)/위(빨강) 배경, 오늘 핀은 레짐색 */}
                <div
                  style={{
                    position: "relative",
                    height: 34,
                    marginTop: 18,
                    overflow: "visible",
                    borderRadius: 8,
                    border: "1px solid #e2e8f0",
                    background: "#fff",
                  }}
                >
                  {/* 방향 배경: MA 아래(파랑) / 위(빨강) */}
                  <div
                    style={{
                      position: "absolute",
                      inset: 0,
                      display: "flex",
                      borderRadius: 8,
                      overflow: "hidden",
                    }}
                  >
                    {gaugeData.zeroPct === null ? (
                      <div
                        style={{
                          width: "100%",
                          background: gaugeData.trendMin >= 0 ? "#d62828" : "#1971c2",
                          opacity: 0.14,
                        }}
                      />
                    ) : (
                      <>
                        <div style={{ width: `${gaugeData.zeroPct}%`, background: "#1971c2", opacity: 0.14 }} />
                        <div style={{ width: `${100 - gaugeData.zeroPct}%`, background: "#d62828", opacity: 0.14 }} />
                      </>
                    )}
                  </div>
                  {/* MA(0) marker */}
                  {gaugeData.zeroPct !== null ? (
                    <div
                      style={{
                        position: "absolute",
                        left: `${gaugeData.zeroPct}%`,
                        top: 0,
                        bottom: 0,
                        width: 0,
                        borderLeft: "1px dashed #475569",
                        transform: "translateX(-0.5px)",
                      }}
                      title="MA(0)"
                    >
                      <div
                        style={{
                          position: "absolute",
                          top: -16,
                          left: "50%",
                          transform: "translateX(-50%)",
                          fontSize: "0.68rem",
                          fontWeight: 700,
                          color: "#475569",
                          whiteSpace: "nowrap",
                        }}
                      >
                        MA
                      </div>
                    </div>
                  ) : null}
                  {/* 오늘 핀 — 색은 현재 레짐(slope 기반), 라벨은 추세% */}
                  {gaugeLeft !== null ? (
                    <div
                      style={{
                        position: "absolute",
                        left: `${gaugeLeft}%`,
                        top: "50%",
                        transform: "translate(-50%, -50%)",
                        padding: "2px 8px",
                        borderRadius: 999,
                        background: latestRegime ? REGIME_COLOR[latestRegime] : "#1f2937",
                        color: "#fff",
                        fontSize: "0.78rem",
                        fontWeight: 800,
                        boxShadow: "0 4px 12px rgba(15, 23, 42, 0.2)",
                      }}
                      title={`오늘 추세% (${latestRegime ? REGIME_LABEL[latestRegime] : "-"})`}
                    >
                      {formatPct(latestPoint?.trend_pct)}
                    </div>
                  ) : null}
                </div>
                {/* 범례: 최저 / MA / 최고 */}
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: "0.7rem",
                    color: "#5f6b82",
                    marginTop: 6,
                  }}
                >
                  <span>최저: {formatPct(gaugeData.trendMin)}</span>
                  <span>MA: 0%</span>
                  <span>최고: {formatPct(gaugeData.trendMax)}</span>
                </div>
              </>
            ) : null}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 8 }}>
            <div className="appSegmentedToggle" role="group" aria-label="시장지수 추세 차트 기간">
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
              ref={labelOverlayRef}
              style={{
                position: "absolute",
                inset: 0,
                zIndex: 2,
                overflow: "hidden",
                pointerEvents: "none",
              }}
            />
            <div
              ref={tooltipRef}
              style={{
                display: "none",
                position: "absolute",
                zIndex: 3,
                minWidth: 160,
                padding: "8px 10px",
                borderRadius: 6,
                background: "rgba(30, 41, 59, 0.95)",
                color: "#fff",
                fontSize: "0.82rem",
                lineHeight: 1.45,
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
