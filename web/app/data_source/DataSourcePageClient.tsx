"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";

// ─── 타입 ────────────────────────────────────────────────────────────────────
type DataSourceRow = {
  category: string;
  country: string;
  provider: string;
  endpoint: string;
  usage: string;
  code_ref: string;
  note: string | null;
  // 호주 ETF 발행사별 행에만 붙는다(실제 캐시 집계).
  etf_count?: number;
  source?: string | null;
  source_label?: string;
  missing_count?: number;
  missing_tickers?: string[];
};

type DataSourcePayload = {
  sources: DataSourceRow[];
  au_holdings_order: string[];
  au_total: number;
  au_expense_ratio_count: number;
};

const COUNTRY_LABEL: Record<string, string> = {
  kor: "🇰🇷 한국",
  us: "🇺🇸 미국",
  au: "🇦🇺 호주",
  global: "🌐 글로벌",
  all: "🌐 전체",
};

// 수집 소스별 배지 색. 공식 소스일수록 진하게 — 신뢰도 차이를 한눈에 보이게 한다.
const SOURCE_TONE: Record<string, { bg: string; fg: string }> = {
  betashares_csv: { bg: "#dbeafe", fg: "#1e40af" },
  vanguard_au_api: { bg: "#dcfce7", fg: "#166534" },
  yfinance_holdings: { bg: "#fef3c7", fg: "#92400e" },
  naver_etf_component: { bg: "#dbeafe", fg: "#1e40af" },
};

const FALLBACK_TONE = { bg: "#fee2e2", fg: "#991b1b" };

function SourceBadge({ source, label }: { source: string | null; label: string | null }) {
  const tone = source ? (SOURCE_TONE[source] ?? FALLBACK_TONE) : FALLBACK_TONE;
  return (
    <span
      style={{
        display: "inline-block",
        padding: "0.15rem 0.5rem",
        borderRadius: "0.35rem",
        background: tone.bg,
        color: tone.fg,
        fontSize: "0.82rem",
        fontWeight: 700,
        whiteSpace: "nowrap",
      }}
    >
      {label ?? "수집 실패"}
    </span>
  );
}

export function DataSourcePageClient() {
  const [data, setData] = useState<DataSourcePayload | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/data-sources", { cache: "no-store" });
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? "데이터 소스를 불러오지 못했습니다.");
      }
      setData((await res.json()) as DataSourcePayload);
    } catch (e) {
      setError(e instanceof Error ? e.message : "오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // 카테고리 순서는 서버가 준 순서를 유지한다(시세 → 마스터 → 상세 → 지표).
  const groupedSources = useMemo(() => {
    if (!data) return [];
    const groups: { category: string; rows: DataSourceRow[] }[] = [];
    for (const row of data.sources) {
      const existing = groups.find((g) => g.category === row.category);
      if (existing) existing.rows.push(row);
      else groups.push({ category: row.category, rows: [row] });
    }
    return groups;
  }, [data]);

  const titleRight = data ? (
    <div className="appHeaderMetrics rankToolbarMeta">
      <div className="appHeaderMetric">
        <span>소스:</span>
        <span className="appHeaderMetricValue">{data.sources.length}개</span>
      </div>
      <div className="appHeaderMetric">
        <span>호주 ETF:</span>
        <span className="appHeaderMetricValue">{data.au_total}종</span>
      </div>
    </div>
  ) : null;

  return (
    <PageFrame title="데이터 소스" titleRight={titleRight}>
      {error ? (
        <div className="appBannerStack">
          <div className="bannerError alert alert-danger mb-0">{error}</div>
        </div>
      ) : null}

      {isLoading && !data ? (
        <section className="appSection">
          <div className="card appCard">
            <div style={{ padding: "3rem 1.5rem", textAlign: "center", color: "var(--text-muted)" }}>
              데이터 소스 불러오는 중...
            </div>
          </div>
        </section>
      ) : null}

      {data ? (
        <>
          {groupedSources.map((group) => (
            <section className="appSection" key={group.category}>
              <div className="card appCard">
                <div className="card-header">
                  <strong>{group.category}</strong>
                </div>
                <div style={{ overflowX: "auto" }}>
                  <table className="table table-sm" style={{ marginBottom: 0, minWidth: "52rem" }}>
                    <thead>
                      <tr>
                        <th style={{ whiteSpace: "nowrap" }}>시장</th>
                        <th style={{ whiteSpace: "nowrap" }}>제공처</th>
                        <th>용도</th>
                        <th>호출 대상</th>
                        <th style={{ whiteSpace: "nowrap" }}>코드 위치</th>
                      </tr>
                    </thead>
                    <tbody>
                      {group.rows.map((row) => (
                        <tr key={`${row.provider}-${row.endpoint}-${row.usage}`}>
                          <td style={{ whiteSpace: "nowrap" }}>{COUNTRY_LABEL[row.country] ?? row.country}</td>
                          <td style={{ whiteSpace: "nowrap", fontWeight: 700 }}>{row.provider}</td>
                          <td style={{ fontSize: "0.9rem" }}>
                            {/* 호주 발행사 행은 수집 경로를 배지로 강조한다(공식 소스 / yfinance 폴백 구분). */}
                            {row.source_label ? (
                              <div
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: "0.45rem",
                                  flexWrap: "wrap",
                                }}
                              >
                                <SourceBadge source={row.source ?? null} label={row.source_label} />
                                <span style={{ color: "var(--text-muted)" }}>ETF {row.etf_count}종</span>
                                {row.missing_count ? (
                                  <span
                                    style={{ color: "#991b1b", fontWeight: 700 }}
                                    title={(row.missing_tickers ?? []).join(", ")}
                                  >
                                    미수집 {row.missing_count}종
                                  </span>
                                ) : null}
                              </div>
                            ) : (
                              row.usage
                            )}
                            {row.note ? (
                              <div style={{ marginTop: "0.2rem", color: "var(--text-muted)", fontSize: "0.83rem" }}>
                                {row.note}
                              </div>
                            ) : null}
                            {row.missing_tickers?.length ? (
                              <div style={{ marginTop: "0.2rem", color: "#991b1b", fontSize: "0.83rem" }}>
                                미수집: <span className="appCodeText">{row.missing_tickers.join(", ")}</span>
                              </div>
                            ) : null}
                          </td>
                          <td className="appCodeText" style={{ fontSize: "0.82rem", wordBreak: "break-all" }}>
                            {row.endpoint}
                          </td>
                          <td
                            className="appCodeText"
                            style={{ fontSize: "0.8rem", color: "var(--text-muted)", whiteSpace: "nowrap" }}
                          >
                            {row.code_ref}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
          ))}
        </>
      ) : null}
    </PageFrame>
  );
}
