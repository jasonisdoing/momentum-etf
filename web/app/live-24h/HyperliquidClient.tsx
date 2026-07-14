"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { PageFrame } from "../components/PageFrame";

const REFRESH_MS = 10_000;

type Candle = { t: number; o: number; h: number; l: number; c: number };

type Quote = {
  symbol: string;
  name: string;
  type: "stock" | "index" | "toss";
  country: "kor" | "us";
  currency: "KRW" | "USD" | "POINT" | "FX";
  hyper_price: number | null;
  change_24h_pct: number | null;
  actual_price: number | null;
  actual_change_pct: number | null;
  diff_pct: number | null;
  session_open: boolean;
  price_data_open?: boolean;
  price_data_session?: "daymarket" | "premarket" | "regular" | "aftermarket" | "closed";
  candles?: Candle[];
  source_ticker?: string;
};

type HyperResponse = { quotes: Quote[]; usd_krw: number | null; error?: string };

type ComparisonSeries = {
  key: string;
  label: string;
  color: string;
  quote: Quote;
  priceMultiplier: number;
  visible: boolean;
};

type PriceDifference = { label: string; value: number | null };

type RepresentativeValue = {
  source: "국내시장" | "미국시장" | "Hyperliquid";
  price: number | null;
  changePct: number | null;
};

const TOSS_CANDLE_FRESH_MS = 30 * 60 * 1000;

function hasFreshTossCandle(quote: Quote | undefined): boolean {
  const latestCandle = quote?.candles?.at(-1);
  return Boolean(latestCandle && latestCandle.t >= Date.now() - TOSS_CANDLE_FRESH_MS);
}

function canUseTossAsChartSource(quote: Quote | undefined): boolean {
  return Boolean(
    quote?.price_data_open && quote.price_data_session !== "closed" && hasFreshTossCandle(quote),
  );
}

function selectRepresentativeValue(
  tossQuote: Quote,
  hyperliquidQuote: Quote,
  tossSource: "국내시장" | "미국시장",
  useToss: boolean,
): RepresentativeValue {
  if (useToss) {
    return { source: tossSource, price: tossQuote.hyper_price, changePct: tossQuote.diff_pct };
  }
  return {
    source: "Hyperliquid",
    price: hyperliquidQuote.hyper_price,
    changePct: hyperliquidQuote.change_24h_pct,
  };
}

function signColor(v: number | null | undefined): string {
  if (v === null || v === undefined || v === 0) return "#475569";
  return v > 0 ? "#dc2626" : "#1971c2";
}

function formatPrice(value: number | null, currency: "KRW" | "USD" | "POINT" | "FX"): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (currency === "KRW") return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
  if (currency === "FX") return `${new Intl.NumberFormat("ko-KR", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)}원`;
  if (currency === "POINT") return `${new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)}p`;
  return `$${new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)}`;
}

function formatPct(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
}

const THREE_HOURS_MS = 3 * 60 * 60 * 1000;
const KST_OFFSET_MS = 9 * 60 * 60 * 1000;
const KST_HOUR_FORMATTER = new Intl.DateTimeFormat("ko-KR", {
  timeZone: "Asia/Seoul",
  hour: "numeric",
  hour12: true,
});

function buildThreeHourTicks(startTime: number, endTime: number): number[] {
  const firstTick = Math.ceil((startTime + KST_OFFSET_MS) / THREE_HOURS_MS) * THREE_HOURS_MS - KST_OFFSET_MS;
  const ticks: number[] = [];
  for (let timestamp = firstTick; timestamp <= endTime; timestamp += THREE_HOURS_MS) {
    ticks.push(timestamp);
  }
  return ticks;
}

// 15분봉 기준 최근 N시간 변동률(%). 4N봉 전 대비. 데이터 부족 시 null.
function recentMove(candles: Candle[] | undefined, hours: number): number | null {
  if (!candles) return null;
  const idx = 4 * hours;
  if (candles.length <= idx) return null;
  const prev = candles[candles.length - 1 - idx]?.c;
  const cur = candles[candles.length - 1]?.c;
  if (!prev || !cur) return null;
  return (cur / prev - 1) * 100;
}

const SYMBOL_DISPLAY: Record<string, string> = {
  NQ_FUT: "토스 나스닥 100 선물",
  USDKRW: "토스 환율",
  SKHY_TOSS: "토스 SKHY",
  MU_TOSS: "토스 MU",
  SKHX_KR_TOSS: "000660",
  SMSN_KR_TOSS: "005930",
  VIX: "토스 VIX",
};

function displaySymbol(symbol: string, sourceTicker?: string): string {
  if (sourceTicker) return sourceTicker;
  return SYMBOL_DISPLAY[symbol.toUpperCase()] ?? symbol;
}

function getQuoteLink(symbol: string, sourceTicker?: string): string {
  const upper = symbol.toUpperCase();
  if (upper === "NQ_FUT") return "https://www.tossinvest.com/indices/RFU.NQc1";
  if (upper === "USDKRW") return "https://www.tossinvest.com/indices/exchange-rate";
  if (upper === "VIX") return "https://www.tossinvest.com/indices/RGI..VIX";
  if (upper === "SKHY_TOSS" || upper === "MU_TOSS") return "https://www.tossinvest.com";
  const map: Record<string, string> = {
    SMSN: "SAMSUNG",
    SKHX: "SKHYNIX",
  };
  const target = map[upper] || upper;
  return `https://app.hyperliquid.xyz/trade/xyz:${target}`;
}

function formatPriceLabel(value: number, currency: "KRW" | "USD" | "POINT" | "FX"): string {
  if (currency === "KRW") {
    return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}`;
  }
  if (currency === "FX") {
    return `${value.toFixed(1)}`;
  }
  if (currency === "POINT") {
    return `${value.toFixed(1)}p`;
  }
  return `$${value.toFixed(2)}`;
}

function ExtremumMarker({
  x,
  y,
  label,
  color,
  direction,
  chartWidth,
  chartHeight,
}: {
  x: number;
  y: number;
  label: string;
  color: string;
  direction: "high" | "low";
  chartWidth: number;
  chartHeight: number;
}) {
  const labelX = x;
  const labelY = direction === "high" ? Math.max(22, y - 34) : Math.min(chartHeight - 8, y + 50);
  const arrowY = direction === "high" ? y - 8 : y + 30;
  return (
    <g>
      <text x={labelX} y={labelY} textAnchor="middle" fill={color} fontSize="14" fontWeight="700">
        {label}
      </text>
      <text x={x} y={arrowY} textAnchor="middle" fill={color} fontSize="20" fontWeight="800">
        {direction === "high" ? "↓" : "↑"}
      </text>
    </g>
  );
}

function CandlestickChart({ candles, currency }: { candles: Candle[]; currency: "KRW" | "USD" | "POINT" | "FX" }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(450);
  const [height, setHeight] = useState(300);

  useEffect(() => {
    if (!containerRef.current) return;
    const updateHeight = () => {
      setHeight(Math.max(260, Math.min(620, Math.floor((window.innerHeight - 420) / 2))));
    };
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const newWidth = entry.contentRect.width;
        if (newWidth > 0) {
          setWidth(newWidth);
        }
      }
    });
    observer.observe(containerRef.current);
    updateHeight();
    window.addEventListener("resize", updateHeight);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", updateHeight);
    };
  }, []);

  const endTime = Date.now();
  const startTime = endTime - 24 * 60 * 60 * 1000;
  const visibleCandles = candles.filter((candle) => candle.t >= startTime && candle.t <= endTime);

  if (visibleCandles.length < 2) return null;

  const lows = visibleCandles.map((c) => c.l);
  const highs = visibleCandles.map((c) => c.h);
  const min = Math.min(...lows);
  const max = Math.max(...highs);
  const current = visibleCandles.at(-1)!.c;
  const highIndex = highs.indexOf(max);
  const lowIndex = lows.indexOf(min);
  const highDiffPct = (max / current - 1) * 100;
  const lowDiffPct = (min / current - 1) * 100;
  const range = max - min === 0 ? 1 : max - min;

  const chartWidth = width - 42;
  const chartHeight = height - 20;
  const paddingY = 60;

  const mapY = (val: number) => {
    return chartHeight - paddingY - ((val - min) / range) * (chartHeight - paddingY * 2);
  };
  const mapX = (timestamp: number) => ((timestamp - startTime) / (endTime - startTime)) * chartWidth;

  const candleWidth = Math.max(2, Math.min(7, chartWidth / 96 - 1));
  const timeTicks = buildThreeHourTicks(startTime, endTime);

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg width={width} height={height} style={{ overflow: "visible", marginTop: 5 }}>
        {/* 가로 점선 가이드라인 */}
        <line x1={0} y1={mapY(max)} x2={chartWidth} y2={mapY(max)} stroke="rgba(255,255,255,0.06)" strokeDasharray="3,3" />
        <line x1={0} y1={mapY(min)} x2={chartWidth} y2={mapY(min)} stroke="rgba(255,255,255,0.06)" strokeDasharray="3,3" />
        <line x1={0} y1={mapY((max + min) / 2)} x2={chartWidth} y2={mapY((max + min) / 2)} stroke="rgba(255,255,255,0.04)" strokeDasharray="3,3" />

        {/* Y축 가격 라벨 */}
        <text x={chartWidth + 5} y={mapY(max) + 4} fill="#94a3b8" fontSize="9.5" fontWeight="600">
          {formatPriceLabel(max, currency)}
        </text>
        <text x={chartWidth + 5} y={mapY((max + min) / 2) + 3} fill="#64748b" fontSize="9.5">
          {formatPriceLabel((max + min) / 2, currency)}
        </text>
        <text x={chartWidth + 5} y={mapY(min) + 3} fill="#94a3b8" fontSize="9.5" fontWeight="600">
          {formatPriceLabel(min, currency)}
        </text>

        {/* X축 시간 라벨 */}
        <line x1={0} y1={chartHeight} x2={chartWidth} y2={chartHeight} stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
        {timeTicks.map((timestamp) => {
          const x = mapX(timestamp);
          return (
            <g key={timestamp}>
              <line x1={x} y1={chartHeight} x2={x} y2={chartHeight + 4} stroke="#94a3b8" />
              <text x={x} y={height - 4} textAnchor="middle" fill="#64748b" fontSize="9.5">
                {KST_HOUR_FORMATTER.format(new Date(timestamp))}
              </text>
            </g>
          );
        })}

        {/* 캔들 그리기 */}
        {visibleCandles.map((c) => {
          const x = mapX(c.t);
          const yOpen = mapY(c.o);
          const yClose = mapY(c.c);
          const yHigh = mapY(c.h);
          const yLow = mapY(c.l);

          const isBull = c.c > c.o;
          const color = isBull ? "#ef4444" : "#3b82f6";

          const bodyY = Math.min(yOpen, yClose);
          const bodyHeight = Math.max(1.5, Math.abs(yOpen - yClose));

          return (
            <g key={c.t}>
              <line x1={x} y1={yHigh} x2={x} y2={yLow} stroke={color} strokeWidth="1" />
              <rect
                x={x - candleWidth / 2}
                y={bodyY}
                width={candleWidth}
                height={bodyHeight}
                fill={color}
                stroke={color}
                strokeWidth="0.5"
              />
            </g>
          );
        })}
        <ExtremumMarker
          x={mapX(visibleCandles[highIndex].t)}
          y={mapY(max)}
          label={`고점 ${formatPct(highDiffPct)}`}
          color="#dc2626"
          direction="high"
          chartWidth={chartWidth}
          chartHeight={chartHeight}
        />
        <ExtremumMarker
          x={mapX(visibleCandles[lowIndex].t)}
          y={mapY(min)}
          label={`저점 ${formatPct(lowDiffPct)}`}
          color="#1971c2"
          direction="low"
          chartWidth={chartWidth}
          chartHeight={chartHeight}
        />
      </svg>
    </div>
  );
}

function QuoteCard({ q, title, controls }: { q: Quote; title: string; controls?: ReactNode }) {
  const m1 = recentMove(q.candles, 1);
  const m3 = recentMove(q.candles, 3);
  return (
    <div className="card appCard" style={{ height: "100%" }}>
      <div className="card-body" style={{ padding: "0.7rem 1rem" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, marginBottom: 6 }}>
          <strong style={{ fontSize: "1rem" }}>{title}</strong>
          {controls}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
          {q.type !== "toss" ? (
            <img
              src="/static/HL%20symbol_mint%20green.svg"
              alt="Hyperliquid"
              width={18}
              height={18}
              style={{ display: "block", flex: "0 0 auto", objectFit: "contain" }}
            />
          ) : null}
          <span aria-label={q.country === "kor" ? "한국" : "미국"} style={{ fontSize: "0.9rem", lineHeight: 1 }}>
            {q.type === "toss" && !q.source_ticker ? "" : q.country === "kor" ? "🇰🇷" : "🇺🇸"}
          </span>
          <span style={{ fontSize: "1.05rem", fontWeight: 800 }}>{q.name}</span>
          <a
            href={getQuoteLink(q.symbol, q.source_ticker)}
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "#228be6", fontWeight: 600, textDecoration: "underline" }}
          >
            {displaySymbol(q.symbol, q.source_ticker)}
          </a>
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 7 }}>
            <span style={{ fontSize: "1.25rem", fontWeight: 800, color: signColor(q.diff_pct) }}>
              {formatPct(q.diff_pct)}
            </span>
            <span
              style={{
                fontSize: "0.74rem",
                fontWeight: 700,
                padding: "2px 8px",
                borderRadius: 999,
                whiteSpace: "nowrap",
                background: q.session_open ? "rgba(22, 163, 74, 0.12)" : "rgba(100, 116, 139, 0.16)",
                color: q.session_open ? "#16a34a" : "#475569",
              }}
            >
              {q.type === "toss" && !q.source_ticker
                ? "실시간"
                : q.type === "toss" && q.country === "kor" && !q.session_open
                  ? "휴장"
                  : q.session_open
                    ? "장중"
                    : "시간외"}
            </span>
          </div>
        </div>
        <div style={{ marginTop: 3 }}>
          <span style={{ fontSize: "1.55rem", fontWeight: 800 }}>{formatPrice(q.hyper_price, q.currency)}</span>
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            gap: 8,
            flexWrap: "wrap",
            marginTop: 2,
            fontSize: "0.9rem",
            color: "var(--text-muted)",
          }}
        >
          <span>
            1시간 <strong style={{ color: signColor(m1) }}>{formatPct(m1)}</strong>
            {", "}3시간 <strong style={{ color: signColor(m3) }}>{formatPct(m3)}</strong>
            {", "}24시간 <strong style={{ color: signColor(q.change_24h_pct) }}>{formatPct(q.change_24h_pct)}</strong>
          </span>
          <span style={{ opacity: 0.5 }}>·</span>
          <span>
            {q.type === "toss" ? "전일 기준" : "정규장 종가"} {formatPrice(q.actual_price, q.currency)}
            {q.type !== "toss" ? (
              <>
                {" "}
                <strong style={{ color: signColor(q.actual_change_pct) }}>{formatPct(q.actual_change_pct)}</strong>
              </>
            ) : null}
          </span>
        </div>
        <CandlestickChart candles={q.candles || []} currency={q.currency} />
      </div>
    </div>
  );
}

function calculatePriceDifference(left: number | null, right: number | null): number | null {
  if (left === null || right === null || right <= 0) return null;
  return (left / right - 1) * 100;
}

function ComparisonChart({
  series,
  candleSeriesKey,
  priorSeriesKey,
  currency,
}: {
  series: ComparisonSeries[];
  candleSeriesKey: string;
  priorSeriesKey?: string;
  currency: "KRW" | "USD";
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(450);
  const [height, setHeight] = useState(300);

  useEffect(() => {
    if (!containerRef.current) return;
    const updateHeight = () => {
      setHeight(Math.max(260, Math.min(620, Math.floor((window.innerHeight - 420) / 2))));
    };
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.width > 0) setWidth(entry.contentRect.width);
      }
    });
    observer.observe(containerRef.current);
    updateHeight();
    window.addEventListener("resize", updateHeight);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", updateHeight);
    };
  }, []);

  const endTime = Date.now();
  const startTime = endTime - 24 * 60 * 60 * 1000;
  const selectedSeries = series.find((item) => item.key === candleSeriesKey);
  const priorSeries = priorSeriesKey ? series.find((item) => item.key === priorSeriesKey) : undefined;
  const selectedCandles = (selectedSeries?.quote.candles ?? []).filter(
    (candle) => candle.t >= startTime && candle.t <= endTime,
  );
  const candleIntervalMs = 15 * 60 * 1000;
  const candleByTime = new Map<number, { candle: Candle; series: ComparisonSeries }>();
  const addCandles = (candles: Candle[], item: ComparisonSeries | undefined) => {
    if (!item) return;
    for (const candle of candles) {
      if (candle.t < startTime || candle.t > endTime) continue;
      const timestamp = Math.floor(candle.t / candleIntervalMs) * candleIntervalMs;
      candleByTime.set(timestamp, { candle: { ...candle, t: timestamp }, series: item });
    }
  };

  // 대체 소스로 전체 구간을 채운 뒤 주 소스를 덮어써, 거래시간 외 공백만 대체 소스로 연결한다.
  addCandles(priorSeries?.quote.candles ?? [], priorSeries);
  addCandles(selectedCandles, selectedSeries);
  const chartCandles = [...candleByTime.values()].sort((a, b) => a.candle.t - b.candle.t);
  const values = chartCandles.flatMap(({ candle, series: item }) => [
    candle.l * item.priceMultiplier,
    candle.h * item.priceMultiplier,
  ]);
  const chartWidth = width - 72;
  const chartHeight = height - 24;

  if (values.length < 2) {
    return (
      <div ref={containerRef} style={{ height, display: "grid", placeItems: "center", color: "var(--text-muted)" }}>
        표시할 거래시간 데이터가 없습니다.
      </div>
    );
  }

  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const padding = Math.max((rawMax - rawMin) * 0.3, rawMax * 0.001);
  const min = rawMin - padding;
  const max = rawMax + padding;
  const range = max - min || 1;
  const mapX = (timestamp: number) => ((timestamp - startTime) / (endTime - startTime)) * chartWidth;
  const mapY = (value: number) => chartHeight - ((value - min) / range) * chartHeight;
  const currentCandle = chartCandles.at(-1)!;
  const currentPrice = currentCandle.candle.c * currentCandle.series.priceMultiplier;
  const highCandle = chartCandles.reduce((highest, item) =>
    item.candle.h * item.series.priceMultiplier > highest.candle.h * highest.series.priceMultiplier ? item : highest,
  );
  const lowCandle = chartCandles.reduce((lowest, item) =>
    item.candle.l * item.series.priceMultiplier < lowest.candle.l * lowest.series.priceMultiplier ? item : lowest,
  );
  const highPrice = highCandle.candle.h * highCandle.series.priceMultiplier;
  const lowPrice = lowCandle.candle.l * lowCandle.series.priceMultiplier;

  const candleWidth = Math.max(2, Math.min(7, chartWidth / 96 - 1));
  const timeTicks = buildThreeHourTicks(startTime, endTime);

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg width={width} height={height} style={{ overflow: "visible", marginTop: 8 }}>
        {[min, (min + max) / 2, max].map((value) => (
          <g key={value}>
            <line x1={0} y1={mapY(value)} x2={chartWidth} y2={mapY(value)} stroke="#e2e8f0" strokeDasharray="3,3" />
            <text x={chartWidth + 6} y={mapY(value) + 4} fill="#64748b" fontSize="10">
              {formatPriceLabel(value, currency)}
            </text>
          </g>
        ))}
        {selectedSeries
          ? chartCandles.map(({ candle, series: item }) => {
            const open = candle.o * item.priceMultiplier;
            const high = candle.h * item.priceMultiplier;
            const low = candle.l * item.priceMultiplier;
            const close = candle.c * item.priceMultiplier;
            const x = mapX(candle.t);
            const color = close >= open ? "#ef4444" : "#2563eb";
            const bodyTop = mapY(Math.max(open, close));
            const bodyHeight = Math.max(1, Math.abs(mapY(open) - mapY(close)));
            return (
              <g key={`${item.key}-${candle.t}`}>
                <line x1={x} y1={mapY(high)} x2={x} y2={mapY(low)} stroke={color} strokeWidth="1" />
                <rect
                  x={x - candleWidth / 2}
                  y={bodyTop}
                  width={candleWidth}
                  height={bodyHeight}
                  fill={color}
                  stroke={color}
                  strokeWidth="0.5"
                />
              </g>
            );
          })
          : null}
        <ExtremumMarker
          x={mapX(highCandle.candle.t)}
          y={mapY(highPrice)}
          label={`고점 ${formatPct((highPrice / currentPrice - 1) * 100)}`}
          color="#dc2626"
          direction="high"
          chartWidth={chartWidth}
          chartHeight={chartHeight}
        />
        <ExtremumMarker
          x={mapX(lowCandle.candle.t)}
          y={mapY(lowPrice)}
          label={`저점 ${formatPct((lowPrice / currentPrice - 1) * 100)}`}
          color="#1971c2"
          direction="low"
          chartWidth={chartWidth}
          chartHeight={chartHeight}
        />
        <line x1={0} y1={chartHeight} x2={chartWidth} y2={chartHeight} stroke="#cbd5e1" />
        {timeTicks.map((timestamp) => {
          const x = mapX(timestamp);
          return (
            <g key={timestamp}>
              <line x1={x} y1={chartHeight} x2={x} y2={chartHeight + 4} stroke="#94a3b8" />
              <text x={x} y={height - 3} textAnchor="middle" fill="#64748b" fontSize="10">
                {KST_HOUR_FORMATTER.format(new Date(timestamp))}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function ComparisonCard({
  title,
  representative,
  candleSeriesKey,
  priorSeriesKey,
  series,
  differences,
  currency,
}: {
  title: string;
  representative: RepresentativeValue;
  candleSeriesKey: string;
  priorSeriesKey?: string;
  series: ComparisonSeries[];
  differences: PriceDifference[];
  currency: "KRW" | "USD";
}) {
  return (
    <div className="card appCard" style={{ height: "100%" }}>
      <div className="card-body" style={{ padding: "0.7rem 1rem" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 14, marginBottom: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <strong style={{ fontSize: "1.18rem" }}>{title}</strong>
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8, whiteSpace: "nowrap" }}>
            <strong style={{ fontSize: "1.18rem", color: signColor(representative.changePct) }}>
              {formatPct(representative.changePct)}
            </strong>
          </div>
        </div>
        <div className={`live24hPriceCardList ${series.length >= 3 ? "is-three" : "is-two"}`}>
          {series.map((item) => {
            const currentPrice = item.quote.hyper_price === null ? null : item.quote.hyper_price * item.priceMultiplier;
            const status =
              item.quote.type !== "toss"
                ? "24H"
                : item.quote.country === "kor"
                  ? item.quote.session_open
                    ? "본장"
                    : item.quote.price_data_session === "premarket"
                      ? "프리"
                      : item.quote.price_data_session === "aftermarket"
                        ? "애프터"
                        : "휴장"
                  : item.quote.price_data_session === "daymarket"
                    ? "데이"
                    : item.quote.price_data_session === "premarket"
                      ? "프리"
                      : item.quote.price_data_session === "regular"
                        ? "본장"
                        : item.quote.price_data_session === "aftermarket"
                          ? "애프터"
                          : "휴장";
            return (
              <span
                key={item.key}
                className="live24hPriceCard"
                style={{
                  border: item.key === candleSeriesKey ? `2px solid ${item.color}` : `1px solid ${item.color}33`,
                  borderRadius: 8,
                  background: `${item.color}0A`,
                  color: "var(--text-strong)",
                }}
              >
                {item.quote.type === "toss" ? (
                  <span aria-label={item.quote.country === "kor" ? "한국" : "미국"} style={{ lineHeight: 1 }}>
                    {item.quote.country === "kor" ? "🇰🇷" : "🇺🇸"}
                  </span>
                ) : (
                  <img
                    src="/static/HL%20symbol_mint%20green.svg"
                    alt="Hyperliquid"
                    width={16}
                    height={16}
                    style={{ display: "block" }}
                  />
                )}
                <strong className="live24hPriceCardName">{item.label}</strong>
                <strong className={`live24hPriceCardValue ${item.key === candleSeriesKey ? "is-active" : ""}`}>
                  {formatPrice(currentPrice, currency)}
                  {item.quote.type === "toss" && item.quote.diff_pct !== null ? (
                    <span style={{ color: signColor(item.quote.diff_pct) }}>({formatPct(item.quote.diff_pct)})</span>
                  ) : null}
                </strong>
                <small
                  className="live24hPriceCardSession"
                  style={{
                    borderRadius: 999,
                    background: item.visible ? "#dcfce7" : "#e2e8f0",
                    color: item.visible ? "#15803d" : "#334155",
                  }}
                >
                  {status}
                </small>
              </span>
            );
          })}
        </div>
        <div
          style={{
            display: "flex",
            gap: 18,
            flexWrap: "wrap",
            alignItems: "center",
            marginTop: 9,
            color: "#475569",
            fontSize: "1rem",
          }}
        >
          <strong style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>기준시장 대비</strong>
          {differences.map((difference) => (
            <span key={difference.label}>
              <strong>{difference.label}</strong>{" "}
              <strong style={{ color: signColor(difference.value), fontSize: "1.08rem" }}>{formatPct(difference.value)}</strong>
            </span>
          ))}
        </div>
        <ComparisonChart
          series={series}
          candleSeriesKey={candleSeriesKey}
          priorSeriesKey={priorSeriesKey}
          currency={currency}
        />
      </div>
    </div>
  );
}

export function HyperliquidClient() {
  const [quotes, setQuotes] = useState<Quote[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [macroSymbol, setMacroSymbol] = useState("NQ_FUT");
  const [usdKrw, setUsdKrw] = useState<number | null>(null);

  const load = useCallback(async (initial: boolean) => {
    try {
      if (initial) setLoading(true);
      const resp = await fetch("/api/live-24h", { cache: "no-store" });
      const payload = (await resp.json()) as HyperResponse;
      if (!resp.ok || payload.error) {
        throw new Error(payload.error ?? "24H 시세를 불러오지 못했습니다.");
      }
      setQuotes(payload.quotes ?? []);
      setUsdKrw(payload.usd_krw ?? null);
      setUpdatedAt(new Date().toLocaleTimeString("ko-KR"));
      setError(null);
    } catch (err) {
      if (initial) setError(err instanceof Error ? err.message : "24H 시세를 불러오지 못했습니다.");
    } finally {
      if (initial) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(true);
    const id = window.setInterval(() => void load(false), REFRESH_MS);
    return () => window.clearInterval(id);
  }, [load]);

  const titleRight = (
    <div className="appHeaderMetrics rankToolbarMeta">
      <div className="appHeaderMetric">
        <span>마지막 갱신:</span>
        <span className="appHeaderMetricValue">{updatedAt ?? "-"} (10초 마다) · 차트: 15분봉</span>
      </div>
    </div>
  );
  const quoteBySymbol = new Map(quotes.map((quote) => [quote.symbol, quote]));
  const macroQuote = quoteBySymbol.get(macroSymbol);
  const hynixKorQuote = quoteBySymbol.get("SKHX_KR_TOSS");
  const hynixAdrQuote = quoteBySymbol.get("SKHY_TOSS");
  const hynixHyperQuote = quoteBySymbol.get("SKHX");
  const micronTossQuote = quoteBySymbol.get("MU_TOSS");
  const micronHyperQuote = quoteBySymbol.get("MU");
  const samsungTossQuote = quoteBySymbol.get("SMSN_KR_TOSS");
  const samsungHyperQuote = quoteBySymbol.get("SMSN");
  const hynixTossActive = canUseTossAsChartSource(hynixKorQuote);
  const micronTossActive = canUseTossAsChartSource(micronTossQuote);
  const samsungTossActive = canUseTossAsChartSource(samsungTossQuote);
  const macroControls = (
    <div className="appSegmentedToggle" role="tablist" aria-label="시장 지표 선택">
      {[
        ["NQ_FUT", "나스닥"],
        ["USDKRW", "환율"],
        ["VIX", "VIX"],
      ].map(([symbol, label]) => (
        <button
          key={symbol}
          type="button"
          role="tab"
          aria-selected={macroSymbol === symbol}
          className={`appSegmentedToggleButton ${macroSymbol === symbol ? "is-active" : ""}`}
          onClick={() => setMacroSymbol(symbol)}
        >
          {label}
        </button>
      ))}
    </div>
  );
  const hynixSeries: ComparisonSeries[] =
    hynixKorQuote && hynixHyperQuote
      ? [
          { key: "000660", label: "한국", color: "#ef4444", quote: hynixKorQuote, priceMultiplier: 1, visible: hynixTossActive },
          ...(hynixAdrQuote && usdKrw
            ? [{ key: "SKHY", label: "미국", color: "#2563eb", quote: hynixAdrQuote, priceMultiplier: usdKrw * 10, visible: canUseTossAsChartSource(hynixAdrQuote) }]
            : []),
          { key: "SKHX", label: "Hyperliquid", color: "#10b981", quote: hynixHyperQuote, priceMultiplier: 1, visible: true },
        ]
      : [];
  const hynixKorPrice = hynixKorQuote?.hyper_price ?? null;
  const hynixAdrPrice = hynixAdrQuote?.hyper_price != null && usdKrw ? hynixAdrQuote.hyper_price * usdKrw * 10 : null;
  const hynixHyperPrice = hynixHyperQuote?.hyper_price ?? null;
  const micronSeries: ComparisonSeries[] =
    micronTossQuote && micronHyperQuote
      ? [
          { key: "MU_TOSS", label: "미국", color: "#2563eb", quote: micronTossQuote, priceMultiplier: 1, visible: micronTossActive },
          { key: "MU_HL", label: "Hyperliquid", color: "#10b981", quote: micronHyperQuote, priceMultiplier: 1, visible: true },
        ]
      : [];
  const samsungSeries: ComparisonSeries[] =
    samsungTossQuote && samsungHyperQuote
      ? [
          { key: "005930", label: "한국", color: "#ef4444", quote: samsungTossQuote, priceMultiplier: 1, visible: samsungTossActive },
          { key: "SMSN", label: "Hyperliquid", color: "#10b981", quote: samsungHyperQuote, priceMultiplier: 1, visible: true },
        ]
      : [];

  return (
    <PageFrame title="24H 시세" titleRight={titleRight}>
      <div className="appPageStack">
        {error ? <div className="alert alert-danger mb-0">{error}</div> : null}
        {loading && quotes.length === 0 ? (
          <div style={{ color: "var(--text-muted)", padding: 20 }}>불러오는 중…</div>
        ) : (
          <div className="row g-2" style={{ width: "100%" }}>
            <div className="col-12 col-lg-6">
              {macroQuote ? <QuoteCard q={macroQuote} title="시장 지표" controls={macroControls} /> : null}
            </div>
            <div className="col-12 col-lg-6">
              {hynixSeries.length && hynixKorQuote && hynixHyperQuote ? (
                <ComparisonCard
                  title="SK하이닉스"
                  representative={selectRepresentativeValue(hynixKorQuote, hynixHyperQuote, "국내시장", hynixTossActive)}
                  candleSeriesKey={hynixTossActive ? "000660" : "SKHX"}
                  priorSeriesKey={hynixTossActive ? "SKHX" : undefined}
                  series={hynixSeries}
                  currency="KRW"
                  differences={[
                    ...(hynixAdrPrice !== null
                      ? [{ label: "미국시장", value: calculatePriceDifference(hynixAdrPrice, hynixKorPrice) }]
                      : []),
                    { label: "Hyperliquid", value: calculatePriceDifference(hynixHyperPrice, hynixKorPrice) },
                  ]}
                />
              ) : null}
            </div>
            <div className="col-12 col-lg-6">
              {micronSeries.length && micronTossQuote && micronHyperQuote ? (
                <ComparisonCard
                  title="마이크론 비교"
                  representative={selectRepresentativeValue(micronTossQuote, micronHyperQuote, "미국시장", micronTossActive)}
                  candleSeriesKey={micronTossActive ? "MU_TOSS" : "MU_HL"}
                  priorSeriesKey={micronTossActive ? "MU_HL" : undefined}
                  series={micronSeries}
                  currency="USD"
                  differences={[
                    { label: "Hyperliquid", value: calculatePriceDifference(micronHyperQuote.hyper_price, micronTossQuote.hyper_price) },
                  ]}
                />
              ) : null}
            </div>
            <div className="col-12 col-lg-6">
              {samsungSeries.length && samsungTossQuote && samsungHyperQuote ? (
                <ComparisonCard
                  title="삼성전자 비교"
                  representative={selectRepresentativeValue(samsungTossQuote, samsungHyperQuote, "국내시장", samsungTossActive)}
                  candleSeriesKey={samsungTossActive ? "005930" : "SMSN"}
                  priorSeriesKey={samsungTossActive ? "SMSN" : undefined}
                  series={samsungSeries}
                  currency="KRW"
                  differences={[
                    { label: "Hyperliquid", value: calculatePriceDifference(samsungHyperQuote.hyper_price, samsungTossQuote.hyper_price) },
                  ]}
                />
              ) : null}
            </div>
          </div>
        )}
      </div>
    </PageFrame>
  );
}
