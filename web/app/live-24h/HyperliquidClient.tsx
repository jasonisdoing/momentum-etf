"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { PageFrame } from "../components/PageFrame";

const REFRESH_MS = 10_000;

type Candle = { o: number; h: number; l: number; c: number };

type Quote = {
  symbol: string;
  name: string;
  type: "stock" | "index";
  country: "kor" | "us";
  currency: "KRW" | "USD" | "POINT";
  hyper_price: number | null;
  change_24h_pct: number | null;
  actual_price: number | null;
  diff_pct: number | null;
  candles?: Candle[];
};

type HyperResponse = { quotes: Quote[]; usd_krw: number | null; error?: string };

function signColor(v: number | null | undefined): string {
  if (v === null || v === undefined || v === 0) return "#475569";
  return v > 0 ? "#dc2626" : "#1971c2";
}

function formatPrice(value: number | null, currency: "KRW" | "USD" | "POINT"): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (currency === "KRW") return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
  if (currency === "POINT") return `${new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)}p`;
  return `$${new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)}`;
}

function formatPct(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function getHyperliquidLink(symbol: string): string {
  const map: Record<string, string> = {
    SMSN: "SAMSUNG",
    SKHX: "SKHYNIX",
  };
  const target = map[symbol.toUpperCase()] || symbol;
  return `https://app.hyperliquid.xyz/trade/xyz:${target}`;
}

function formatPriceLabel(value: number, currency: "KRW" | "USD" | "POINT"): string {
  if (currency === "KRW") {
    return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}`;
  }
  if (currency === "POINT") {
    return `${value.toFixed(1)}p`;
  }
  return `$${value.toFixed(2)}`;
}

function CandlestickChart({ candles, currency }: { candles: Candle[]; currency: "KRW" | "USD" | "POINT" }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(450);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const newWidth = entry.contentRect.width;
        if (newWidth > 0) {
          setWidth(newWidth);
        }
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  if (!candles || candles.length < 2) return null;

  const lows = candles.map((c) => c.l);
  const highs = candles.map((c) => c.h);
  const min = Math.min(...lows);
  const max = Math.max(...highs);
  const range = max - min === 0 ? 1 : max - min;

  const height = 225;
  const chartWidth = width - 42;
  const chartHeight = height - 20;
  const paddingY = 8;

  const mapY = (val: number) => {
    return chartHeight - paddingY - ((val - min) / range) * (chartHeight - paddingY * 2);
  };

  const candleWidth = (chartWidth / candles.length) - 1.5;

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <svg width={width} height={height} style={{ overflow: "visible", marginTop: 12 }}>
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
        <text x={0} y={height - 4} fill="#64748b" fontSize="9.5">
          24시간 전
        </text>
        <text x={chartWidth / 2 - 22} y={height - 4} fill="#64748b" fontSize="9.5">
          12시간 전
        </text>
        <text x={chartWidth - 22} y={height - 4} fill="#94a3b8" fontSize="9.5" fontWeight="600">
          실시간
        </text>

        {/* 캔들 그리기 */}
        {candles.map((c, index) => {
          const x = index * (chartWidth / candles.length) + 0.75;
          const yOpen = mapY(c.o);
          const yClose = mapY(c.c);
          const yHigh = mapY(c.h);
          const yLow = mapY(c.l);

          const isBull = c.c > c.o;
          const color = isBull ? "#ef4444" : "#3b82f6";

          const bodyY = Math.min(yOpen, yClose);
          const bodyHeight = Math.max(1.5, Math.abs(yOpen - yClose));

          return (
            <g key={index}>
              <line x1={x + candleWidth / 2} y1={yHigh} x2={x + candleWidth / 2} y2={yLow} stroke={color} strokeWidth="1" />
              <rect
                x={x}
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
      </svg>
    </div>
  );
}

export function HyperliquidClient() {
  const [quotes, setQuotes] = useState<Quote[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);

  const load = useCallback(async (initial: boolean) => {
    try {
      if (initial) setLoading(true);
      const resp = await fetch("/api/live-24h", { cache: "no-store" });
      const payload = (await resp.json()) as HyperResponse;
      if (!resp.ok || payload.error) {
        throw new Error(payload.error ?? "24H 시세를 불러오지 못했습니다.");
      }
      setQuotes(payload.quotes ?? []);
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
        <span>갱신:</span>
        <span className="appHeaderMetricValue">{updatedAt ?? "-"} · 10초</span>
      </div>
    </div>
  );

  return (
    <PageFrame title="24H 시세" titleRight={titleRight}>
      <div className="appPageStack">
        {error ? <div className="alert alert-danger mb-0">{error}</div> : null}
        {loading && quotes.length === 0 ? (
          <div style={{ color: "#868e96", padding: 20 }}>불러오는 중…</div>
        ) : (
          <div className="row g-3" style={{ maxWidth: 1280, width: "100%" }}>
            {quotes.map((q) => (
              <div key={q.symbol} className="col-12 col-md-6">
                {/* Hyperliquid Card */}
                <div className="card appCard" style={{ height: "100%" }}>
                  <div className="card-body" style={{ padding: "1rem 1.25rem" }}>
                    <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 4 }}>
                      <span style={{ fontSize: "1.15rem", fontWeight: 800 }}>{q.name}</span>
                      <a
                        href={getHyperliquidLink(q.symbol)}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ color: "#228be6", fontWeight: 600, textDecoration: "underline" }}
                      >
                        {q.symbol}
                      </a>
                      <span style={{ marginLeft: "auto", fontSize: "0.95rem", color: "#94a3b8", fontWeight: 600 }}>
                        🌐 하이퍼리퀴드
                      </span>
                    </div>
                    <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap", marginTop: 6 }}>
                      <span style={{ fontSize: "1.35rem", fontWeight: 800 }}>
                        {formatPrice(q.hyper_price, q.currency)}
                      </span>
                      <span style={{ color: signColor(q.change_24h_pct), fontWeight: 700, fontSize: "0.85rem" }}>
                        {formatPct(q.change_24h_pct)}
                      </span>
                      <span style={{ color: "#94a3b8", fontSize: "0.92rem", marginLeft: "auto" }}>
                        실제 {formatPrice(q.actual_price, q.currency)} (
                        <strong style={{ color: signColor(q.diff_pct) }}>{formatPct(q.diff_pct)}</strong>)
                      </span>
                    </div>
                    <CandlestickChart candles={q.candles || []} currency={q.currency} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </PageFrame>
  );
}
