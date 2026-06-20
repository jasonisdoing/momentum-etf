"use client";

import { useCallback, useEffect, useState } from "react";

import { PageFrame } from "../components/PageFrame";

const REFRESH_MS = 10_000;

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

export function HyperliquidClient() {
  const [quotes, setQuotes] = useState<Quote[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);

  const load = useCallback(async (initial: boolean) => {
    try {
      if (initial) setLoading(true);
      const resp = await fetch("/api/hyperliquid", { cache: "no-store" });
      const payload = (await resp.json()) as HyperResponse;
      if (!resp.ok || payload.error) {
        throw new Error(payload.error ?? "Hyperliquid 시세를 불러오지 못했습니다.");
      }
      setQuotes(payload.quotes ?? []);
      setUpdatedAt(new Date().toLocaleTimeString("ko-KR"));
      setError(null);
    } catch (err) {
      if (initial) setError(err instanceof Error ? err.message : "Hyperliquid 시세를 불러오지 못했습니다.");
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
    <PageFrame title="하이퍼리퀴드" titleRight={titleRight}>
      <div className="appPageStack">
        {error ? <div className="alert alert-danger mb-0">{error}</div> : null}
        {loading && quotes.length === 0 ? (
          <div style={{ color: "#868e96", padding: 20 }}>불러오는 중…</div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 16, maxWidth: 480 }}>
            {quotes.map((q) => (
              <div
                key={q.symbol}
                className="card appCard"
              >
                <div className="card-body" style={{ padding: "1rem 1.25rem" }}>
                  <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 4 }}>
                    <span style={{ fontSize: "1.15rem", fontWeight: 800 }}>{q.name}</span>
                    <span style={{ color: "#64748b", fontWeight: 600 }}>{q.symbol}</span>
                    <span style={{ marginLeft: "auto", fontSize: "0.78rem", color: "#94a3b8" }}>
                      {`${q.country === "kor" ? "🇰🇷 한국" : "🇺🇸 미국"}${q.type === "index" ? "지수" : "주식"}`}
                    </span>
                  </div>
                  <div style={{ fontSize: "1.7rem", fontWeight: 800, lineHeight: 1.2, marginTop: 6 }}>
                    {formatPrice(q.hyper_price, q.currency)}
                  </div>
                  <div style={{ marginTop: 4, fontSize: "0.95rem" }}>
                    <span style={{ color: "#64748b" }}>24H </span>
                    <span style={{ color: signColor(q.change_24h_pct), fontWeight: 700 }}>
                      {formatPct(q.change_24h_pct)}
                    </span>
                  </div>
                  <div style={{ marginTop: 2, fontSize: "0.95rem" }}>
                    <span style={{ color: "#64748b" }}>실제가와 차이 </span>
                    <span style={{ color: signColor(q.diff_pct), fontWeight: 700 }}>{formatPct(q.diff_pct)}</span>
                    <span style={{ color: "#94a3b8", marginLeft: 6, fontSize: "0.85rem" }}>
                      (실제 {formatPrice(q.actual_price, q.currency)})
                    </span>
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
