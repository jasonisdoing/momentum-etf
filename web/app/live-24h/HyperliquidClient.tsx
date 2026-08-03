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

function formatComparisonAxisPrice(value: number, currency: "KRW" | "USD"): string {
  if (currency === "KRW") {
    return `${Math.round(value / 10_000)}만원`;
  }
  return `$${Math.round(value)}`;
}

function buildRoundPriceTicks(min: number, max: number, targetCount = 7): number[] {
  const span = max - min;
  if (!Number.isFinite(span) || span <= 0) return [];

  const rawStep = span / Math.max(2, targetCount - 1);
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  const niceMultiplier = [1, 2, 2.5, 5, 10].find((candidate) => candidate >= normalized) ?? 10;
  const step = niceMultiplier * magnitude;
  const first = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let value = first; value <= max + step * 1e-9; value += step) {
    ticks.push(Number(value.toPrecision(12)));
  }
  return ticks;
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

// Hyperliquid 피드의 단일 봉 오류 틱을 보정한다. 오류는 고가/저가뿐 아니라 시가/종가에도 박힐 수 있어
// 각 봉의 o/h/l/c 를 **이웃 봉 종가의 중앙값(기준가)** 과 비교한다.
// 척도는 원시 봉 범위가 아니라 "각 봉이 기준가에서 벗어난 폭(extent)의 중앙값" 을 쓴다 — 이렇게 하면
// 종목 변동성/노이즈와 무관하게 '전형적 벗어남'을 잡아, 그 DETECT 배를 넘는 값만 오류로 본다.
// (전역 평균이 아니라 '이웃'과 비교해 추세 구간을 오탐하지 않는다.)
const _SANITIZE_DETECT = 4; // 전형 벗어남의 이 배수 초과 = 오류 틱
const _SANITIZE_PULL = 1.5; // 오류면 기준가 ± (전형 벗어남×이 배수) 로 눌러 정상 봉에 묻히게 한다
const _SANITIZE_WINDOW = 3; // 기준가 계산에 쓰는 좌우 이웃 개수

function sanitizeCandles(candles: Candle[]): Candle[] {
  const n = candles.length;
  if (n < 5) return candles;
  const refAt = (i: number): number => {
    const neigh: number[] = [];
    for (let j = Math.max(0, i - _SANITIZE_WINDOW); j <= Math.min(n - 1, i + _SANITIZE_WINDOW); j += 1) {
      if (j !== i) neigh.push(candles[j].c);
    }
    neigh.sort((a, b) => a - b);
    return neigh.length ? neigh[Math.floor(neigh.length / 2)] : candles[i].c;
  };
  const refs = candles.map((_, i) => refAt(i));
  // 각 봉이 기준가에서 벗어난 최대 폭. 대부분의 봉은 추세를 따라 작게 벗어나므로 그 중앙값이 '전형'.
  const extents = candles
    .map((c, i) => Math.max(Math.abs(c.h - refs[i]), Math.abs(c.l - refs[i]), Math.abs(c.o - refs[i]), Math.abs(c.c - refs[i])))
    .filter((e) => e > 0)
    .sort((a, b) => a - b);
  if (extents.length === 0) return candles;
  const medianExtent = extents[Math.floor(extents.length / 2)];
  if (!(medianExtent > 0)) return candles;
  const detect = medianExtent * _SANITIZE_DETECT;
  const pull = medianExtent * _SANITIZE_PULL;

  let changed = false;
  const out = candles.map((c, i) => {
    const ref = refs[i];
    const fix = (v: number) => (v > ref + detect ? ref + pull : v < ref - detect ? ref - pull : v);
    const o = fix(c.o);
    const cc = fix(c.c);
    const h = Math.max(o, cc, fix(c.h));
    const l = Math.min(o, cc, fix(c.l));
    if (o !== c.o || cc !== c.c || h !== c.h || l !== c.l) {
      changed = true;
      return { ...c, o, h, l, c: cc };
    }
    return c;
  });
  return changed ? out : candles;
}

function CandlestickChart({ candles, currency }: { candles: Candle[]; currency: "KRW" | "USD" | "POINT" | "FX" }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(450);
  const [height, setHeight] = useState(300);

  useEffect(() => {
    if (!containerRef.current) return;
    const updateHeight = () => {
      setHeight(Math.max(260, Math.min(620, Math.floor((window.innerHeight - 460) / 2))));
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
  const visibleCandles = sanitizeCandles(candles.filter((candle) => candle.t >= startTime && candle.t <= endTime));

  if (visibleCandles.length < 2) {
    return (
      <div ref={containerRef} style={{ height, display: "grid", placeItems: "center", color: "var(--text-muted)", marginTop: 12 }}>
        표시할 차트 데이터가 없습니다.
      </div>
    );
  }

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

  const chartWidth = width - 72;
  const chartHeight = height - 20;
  const paddingY = 60;

  const mapY = (val: number) => {
    return chartHeight - paddingY - ((val - min) / range) * (chartHeight - paddingY * 2);
  };
  const mapX = (timestamp: number) => ((timestamp - startTime) / (endTime - startTime)) * chartWidth;

  const candleWidth = Math.max(2, Math.min(7, chartWidth / 96 - 1));
  const timeTicks = buildThreeHourTicks(startTime, endTime);
  const priceTicks = buildRoundPriceTicks(min, max).filter((value) => Math.abs(mapY(value) - mapY(current)) > 18);
  const currentY = mapY(current);

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg width={width} height={height} style={{ overflow: "visible", marginTop: 5 }}>
        {priceTicks.map((value) => (
          <g key={value}>
            <line x1={0} y1={mapY(value)} x2={chartWidth} y2={mapY(value)} stroke="#d8e0ea" strokeDasharray="3,3" />
            <text x={chartWidth + 5} y={mapY(value) + 4} fill="#64748b" fontSize="9.5" fontWeight="600">
              {formatPriceLabel(value, currency)}
            </text>
          </g>
        ))}
        <line x1={0} y1={currentY} x2={chartWidth} y2={currentY} stroke="#334155" strokeWidth="1.4" strokeDasharray="5,3" />

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
        <rect x={chartWidth + 3} y={currentY - 10} width={62} height={20} rx={4} fill="#334155" />
        <text x={chartWidth + 34} y={currentY + 4} textAnchor="middle" fill="#ffffff" fontSize="10" fontWeight="800">
          {formatPriceLabel(current, currency)}
        </text>
      </svg>
    </div>
  );
}

function QuoteCard({ q, title, controls, compact = false }: { q: Quote; title: string; controls?: ReactNode; compact?: boolean }) {
  const m1 = recentMove(q.candles, 1);
  const m3 = recentMove(q.candles, 3);
  return (
    <div className="card appCard" style={{ height: "100%" }}>
      <div className="card-body" style={{ padding: "0.7rem 1rem" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, marginBottom: 6 }}>
          <strong style={{ fontSize: "var(--fs-base)" }}>{title}</strong>
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
          <span aria-label={q.country === "kor" ? "한국" : "미국"} style={{ fontSize: "var(--fs-base)", lineHeight: 1 }}>
            {q.type === "toss" && !q.source_ticker ? "" : q.country === "kor" ? "🇰🇷" : "🇺🇸"}
          </span>
          <span style={{ fontSize: "var(--fs-lg)", fontWeight: 800 }}>{q.name}</span>
          <a
            href={getQuoteLink(q.symbol, q.source_ticker)}
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "#228be6", fontWeight: 600, textDecoration: "underline" }}
          >
            {displaySymbol(q.symbol, q.source_ticker)}
          </a>
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 7 }}>
            <span style={{ fontSize: "var(--fs-xl)", fontWeight: 800, color: signColor(q.diff_pct) }}>
              {formatPct(q.diff_pct)}
            </span>
            <span
              style={{
                fontSize: "var(--fs-sm)",
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
          <span style={{ fontSize: "var(--fs-2xl)", fontWeight: 800 }}>{formatPrice(q.hyper_price, q.currency)}</span>
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            gap: 8,
            flexWrap: "wrap",
            marginTop: 2,
            fontSize: "var(--fs-base)",
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
        {compact ? null : <CandlestickChart candles={q.candles || []} currency={q.currency} />}
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
  overlayLineSeriesKey,
  currency,
  fixedHeight,
}: {
  series: ComparisonSeries[];
  candleSeriesKey: string;
  priorSeriesKey?: string;
  overlayLineSeriesKey?: string;
  currency: "KRW" | "USD";
  /** 홈 허브(한 화면 배치)용 고정 차트 높이 — 지정 시 뷰포트 기반 자동 높이를 끈다. */
  fixedHeight?: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(450);
  const [height, setHeight] = useState(fixedHeight ?? 300);

  useEffect(() => {
    if (!containerRef.current) return;
    const updateHeight = () => {
      if (fixedHeight !== undefined) {
        setHeight(fixedHeight);
        return;
      }
      setHeight(Math.max(260, Math.min(620, Math.floor((window.innerHeight - 460) / 2))));
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
  }, [fixedHeight]);

  const endTime = Date.now();
  const startTime = endTime - 24 * 60 * 60 * 1000;
  const selectedSeries = series.find((item) => item.key === candleSeriesKey);
  const priorSeries = priorSeriesKey ? series.find((item) => item.key === priorSeriesKey) : undefined;
  const overlaySeries = overlayLineSeriesKey
    ? series.find((item) => item.key === overlayLineSeriesKey)
    : undefined;
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
  // 병합 전에 각 소스 캔들에서 피드 오류 틱(스파이크)을 제거한다(시리즈별로 동질적이라 개별 처리).
  const priorCandles = (priorSeries?.quote.candles ?? [])
    .filter((candle) => candle.t >= startTime && candle.t <= endTime)
    .sort((a, b) => a.t - b.t);
  addCandles(sanitizeCandles(priorCandles), priorSeries);
  addCandles(sanitizeCandles([...selectedCandles].sort((a, b) => a.t - b.t)), selectedSeries);
  const chartCandles = [...candleByTime.values()].sort((a, b) => a.candle.t - b.candle.t);
  const overlayPointByTime = new Map<number, number>();
  for (const candle of overlaySeries?.quote.candles ?? []) {
    if (candle.t < startTime || candle.t > endTime) continue;
    const timestamp = Math.floor(candle.t / candleIntervalMs) * candleIntervalMs;
    overlayPointByTime.set(timestamp, candle.c * (overlaySeries?.priceMultiplier ?? 1));
  }
  const overlayPoints = [...overlayPointByTime.entries()]
    .map(([timestamp, value]) => ({ timestamp, value }))
    .sort((a, b) => a.timestamp - b.timestamp);
  const sourceBands: Array<{ key: string; start: number; end: number; color: string }> = [];
  for (const item of chartCandles) {
    const lastBand = sourceBands.at(-1);
    const candleEnd = Math.min(item.candle.t + candleIntervalMs, endTime);
    if (lastBand?.key === item.series.key && item.candle.t <= lastBand.end) {
      lastBand.end = candleEnd;
    } else {
      sourceBands.push({
        key: item.series.key,
        start: item.candle.t,
        end: candleEnd,
        color: item.series.color,
      });
    }
  }
  const overlayValues = overlayPoints.map((point) => point.value);
  const values = [
    ...chartCandles.flatMap(({ candle, series: item }) => [
      candle.l * item.priceMultiplier,
      candle.h * item.priceMultiplier,
    ]),
    ...overlayValues,
  ];
  const hasOverlayLine = overlayPoints.length >= 2;
  const plotLeft = 0;
  const chartWidth = width - (hasOverlayLine ? 94 : 72);
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
  const mapX = (timestamp: number) =>
    plotLeft + ((timestamp - startTime) / (endTime - startTime)) * (chartWidth - plotLeft);
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

  const candleWidth = Math.max(2, Math.min(7, (chartWidth - plotLeft) / 96 - 1));
  const timeTicks = buildThreeHourTicks(startTime, endTime);
  const priceTicks = buildRoundPriceTicks(min, max).filter(
    (value) => Math.abs(mapY(value) - mapY(currentPrice)) > 20,
  );
  const currentY = mapY(currentPrice);
  const overlayCurrentPoint = hasOverlayLine ? overlayPoints.at(-1) : undefined;
  const overlayCurrentY = overlayCurrentPoint ? mapY(overlayCurrentPoint.value) : null;
  const overlaySegments: Array<typeof overlayPoints> = [];
  for (const point of overlayPoints) {
    const segment = overlaySegments.at(-1);
    const previousPoint = segment?.at(-1);
    if (!segment || !previousPoint || point.timestamp - previousPoint.timestamp > candleIntervalMs) {
      overlaySegments.push([point]);
    } else {
      segment.push(point);
    }
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg width={width} height={height} style={{ overflow: "visible", marginTop: 8 }}>
        {sourceBands.map((band, index) => {
          const x = mapX(Math.max(band.start, startTime));
          const endX = mapX(Math.min(band.end, endTime));
          return (
            <rect
              key={`${band.key}-${band.start}-${index}`}
              x={x}
              y={0}
              width={Math.max(0, endX - x)}
              height={chartHeight}
              fill={band.color}
              fillOpacity={0.08}
            />
          );
        })}
        {priceTicks.map((value) => (
          <g key={`main-${value}`}>
            <line x1={plotLeft} y1={mapY(value)} x2={chartWidth} y2={mapY(value)} stroke="#e2e8f0" strokeDasharray="3,3" />
            <text x={chartWidth + 6} y={mapY(value) + 4} fill="#475569" fontSize="11.5" fontWeight="700">
              {formatComparisonAxisPrice(value, currency)}
            </text>
          </g>
        ))}
        <line x1={plotLeft} y1={currentY} x2={chartWidth} y2={currentY} stroke="#334155" strokeWidth="1.5" strokeDasharray="5,3" />
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
        {hasOverlayLine
          ? overlaySegments
              .filter((segment) => segment.length >= 2)
              .map((segment, index) => (
                <polyline
                  key={`overlay-line-${index}`}
                  points={segment.map((point) => `${mapX(point.timestamp)},${mapY(point.value)}`).join(" ")}
                  fill="none"
                  stroke={overlaySeries?.color}
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              ))
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
        <rect x={chartWidth + 4} y={currentY - 11} width={66} height={22} rx={4} fill="#334155" />
        <text x={chartWidth + 37} y={currentY + 4} textAnchor="middle" fill="#ffffff" fontSize="11.5" fontWeight="800">
          {formatComparisonAxisPrice(currentPrice, currency)}
        </text>
        {overlayCurrentPoint && overlayCurrentY !== null ? (
          <>
            <line
              x1={mapX(overlayCurrentPoint.timestamp)}
              y1={overlayCurrentY}
              x2={chartWidth}
              y2={overlayCurrentY}
              stroke={overlaySeries?.color}
              strokeWidth="1.5"
              strokeDasharray="5,3"
            />
            <rect x={chartWidth + 4} y={overlayCurrentY - 11} width={88} height={22} rx={4} fill={overlaySeries?.color} />
            <text
              x={chartWidth + 48}
              y={overlayCurrentY + 4}
              textAnchor="middle"
              fill="#ffffff"
              fontSize="11.5"
              fontWeight="800"
            >
              ADR {formatComparisonAxisPrice(overlayCurrentPoint.value, currency)}
            </text>
          </>
        ) : null}
        <line x1={plotLeft} y1={chartHeight} x2={chartWidth} y2={chartHeight} stroke="#cbd5e1" />
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
  overlayLineSeriesKey,
  series,
  differences,
  currency,
  compact = false,
}: {
  title: string;
  representative: RepresentativeValue;
  candleSeriesKey: string;
  priorSeriesKey?: string;
  overlayLineSeriesKey?: string;
  series: ComparisonSeries[];
  differences: PriceDifference[];
  currency: "KRW" | "USD";
  /** 홈 허브용 — 차트를 고정 높이(작게)로 그린다. */
  compact?: boolean;
}) {
  return (
    <div className="card appCard" style={{ height: "100%" }}>
      <div className="card-body" style={{ padding: "0.7rem 1rem" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 14, marginBottom: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <strong style={{ fontSize: "var(--fs-xl)" }}>{title}</strong>
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 6, whiteSpace: "nowrap" }}>
            <span style={{ fontSize: "var(--fs-sm)", color: "var(--text-muted)", fontWeight: 500 }}>
              {representative.source === "Hyperliquid"
                ? "하이퍼리퀴드"
                : representative.source === "국내시장"
                  ? "한국 거래소"
                  : "미국 거래소"}
            </span>
            <strong style={{ fontSize: "var(--fs-xl)", color: signColor(representative.changePct) }}>
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
            fontSize: "var(--fs-base)",
          }}
        >
          <strong style={{ color: "var(--text-muted)", fontSize: "var(--fs-base)" }}>기준시장 대비</strong>
          {differences.map((difference) => (
            <span key={difference.label}>
              <strong>{difference.label}</strong>{" "}
              <strong style={{ color: signColor(difference.value), fontSize: "var(--fs-lg)" }}>{formatPct(difference.value)}</strong>
            </span>
          ))}
        </div>
        <ComparisonChart
          series={series}
          candleSeriesKey={candleSeriesKey}
          priorSeriesKey={priorSeriesKey}
          overlayLineSeriesKey={overlayLineSeriesKey}
          currency={currency}
          fixedHeight={compact ? 185 : undefined}
        />
      </div>
    </div>
  );
}

// 24H 시세 보드 — 페이지(/live-24h)와 홈 허브가 공유하는 본문. compact=홈용(시세 카드 작게 3열).
export function Live24hBoard({ compact = false }: { compact?: boolean }) {
  const [quotes, setQuotes] = useState<Quote[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
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

  // compact(홈 허브): 시세 3장은 좌측 세로 스택(작게, 차트 없음), 비교 3장은 크게 — CSS 그리드로 배치.
  const rowClass = compact ? "hubLive24hRow" : "row g-2";
  const quoteColClass = compact ? "hubLive24hQuote" : "col-12 col-md-6 col-xxl-4";
  const comparisonColClass = compact ? "hubLive24hComparison" : "col-12 col-md-6 col-xxl-4";
  const metaLine = (
    <div className="d-flex justify-content-end text-secondary" style={{ fontSize: "var(--fs-sm)" }}>
      마지막 갱신: {updatedAt ?? "-"} (10초 마다) · 차트: 15분봉
    </div>
  );
  const quoteBySymbol = new Map(quotes.map((quote) => [quote.symbol, quote]));
  const nasdaqQuote = quoteBySymbol.get("NQ_FUT");
  const fxQuote = quoteBySymbol.get("USDKRW");
  const vixQuote = quoteBySymbol.get("VIX");
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
  // 카드 배치(is-three): 1줄=한국(전체), 2줄=[Hyperliquid | 미국] 반반 → 배열 순서를 그 순서로 둔다.
  const hynixSeries: ComparisonSeries[] =
    hynixKorQuote && hynixHyperQuote
      ? [
          { key: "000660", label: "한국", color: "#ef4444", quote: hynixKorQuote, priceMultiplier: 1, visible: hynixTossActive },
          { key: "SKHX", label: "Hyperliquid", color: "#10b981", quote: hynixHyperQuote, priceMultiplier: 1, visible: true },
          ...(hynixAdrQuote && usdKrw
            ? [{ key: "SKHY", label: "미국", color: "#2563eb", quote: hynixAdrQuote, priceMultiplier: usdKrw * 10, visible: canUseTossAsChartSource(hynixAdrQuote) }]
            : []),
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
    <div className="appPageStack">
      {metaLine}
      {error ? <div className="alert alert-danger mb-0">{error}</div> : null}
      {loading && quotes.length === 0 ? (
        <div style={{ color: "var(--text-muted)", padding: 20 }}>불러오는 중…</div>
      ) : (
        <div className={rowClass} style={{ width: "100%" }}>
          <div className={quoteColClass}>
            {nasdaqQuote ? <QuoteCard q={nasdaqQuote} title="나스닥" compact={compact} /> : null}
          </div>
          <div className={quoteColClass}>
            {fxQuote ? <QuoteCard q={fxQuote} title="환율" compact={compact} /> : null}
          </div>
          <div className={quoteColClass}>
            {vixQuote ? <QuoteCard q={vixQuote} title="VIX" compact={compact} /> : null}
          </div>
          <div className={comparisonColClass}>
              {hynixSeries.length && hynixKorQuote && hynixHyperQuote ? (
                <ComparisonCard
                  title="SK하이닉스"
                  representative={selectRepresentativeValue(hynixKorQuote, hynixHyperQuote, "국내시장", hynixTossActive)}
                  candleSeriesKey={hynixTossActive ? "000660" : "SKHX"}
                  priorSeriesKey={hynixTossActive ? "SKHX" : undefined}
                  series={hynixSeries}
                  currency="KRW"
                  compact={compact}
                  differences={[
                    ...(hynixAdrPrice !== null
                      ? [{ label: "미국시장", value: calculatePriceDifference(hynixAdrPrice, hynixKorPrice) }]
                      : []),
                    { label: "Hyperliquid", value: calculatePriceDifference(hynixHyperPrice, hynixKorPrice) },
                  ]}
                />
              ) : null}
            </div>
            <div className={comparisonColClass}>
              {micronSeries.length && micronTossQuote && micronHyperQuote ? (
                <ComparisonCard
                  title="마이크론"
                  representative={selectRepresentativeValue(micronTossQuote, micronHyperQuote, "미국시장", micronTossActive)}
                  candleSeriesKey={micronTossActive ? "MU_TOSS" : "MU_HL"}
                  priorSeriesKey={micronTossActive ? "MU_HL" : undefined}
                  series={micronSeries}
                  currency="USD"
                  compact={compact}
                  differences={[
                    { label: "Hyperliquid", value: calculatePriceDifference(micronHyperQuote.hyper_price, micronTossQuote.hyper_price) },
                  ]}
                />
              ) : null}
            </div>
            <div className={comparisonColClass}>
              {samsungSeries.length && samsungTossQuote && samsungHyperQuote ? (
                <ComparisonCard
                  title="삼성전자"
                  representative={selectRepresentativeValue(samsungTossQuote, samsungHyperQuote, "국내시장", samsungTossActive)}
                  candleSeriesKey={samsungTossActive ? "005930" : "SMSN"}
                  priorSeriesKey={samsungTossActive ? "SMSN" : undefined}
                  series={samsungSeries}
                  currency="KRW"
                  compact={compact}
                  differences={[
                    { label: "Hyperliquid", value: calculatePriceDifference(samsungHyperQuote.hyper_price, samsungTossQuote.hyper_price) },
                  ]}
                />
              ) : null}
            </div>
          </div>
        )}
    </div>
  );
}


// /live-24h 페이지 — 공용 보드를 PageFrame 으로 감싼다.
export function HyperliquidClient() {
  return (
    <PageFrame title="24H 시세">
      <Live24hBoard />
    </PageFrame>
  );
}