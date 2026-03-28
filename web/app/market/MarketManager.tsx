"use client";

import { useEffect, useMemo, useState } from "react";

type MarketRowItem = {
  ticker: string;
  name: string;
  listed_at: string;
  daily_change_pct: number | null;
  current_price: number | null;
  nav: number | null;
  deviation: number | null;
  return_3m_pct: number | null;
  prev_volume: number;
  market_cap: number;
};

type MarketResponse = {
  updated_at?: string | null;
  rows?: MarketRowItem[];
  error?: string;
};

const EXCLUSION_KEYWORD_GROUPS: Record<string, string[]> = {
  인버스: ["인버스"],
  "2X": ["2X"],
  레버리지: ["레버리지"],
  선물: ["선물"],
  "채권(모든종류)": ["채권", "미국채", "국채", "회사채", "단기채", "장기채"],
  혼합: ["혼합"],
  리츠: ["리츠"],
  합성: ["합성"],
  커버드콜: ["커버드콜"],
};

const DEFAULT_EXCLUDED_GROUPS = ["인버스", "2X", "레버리지", "선물", "채권(모든종류)", "혼합", "리츠"];

function formatKrwEok(value: number): string {
  return new Intl.NumberFormat("ko-KR").format(value);
}

function formatCount(value: number): string {
  return new Intl.NumberFormat("ko-KR").format(value);
}

function formatNullableNumber(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 2 }).format(value);
}

function formatPercent(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${value.toFixed(2)}%`;
}

function getSignedMetricClass(value: number | null): string | undefined {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return undefined;
  }
  if (value > 0) {
    return "metricPositive";
  }
  if (value < 0) {
    return "metricNegative";
  }
  return undefined;
}

function getDeviationClass(value: number | null): string | undefined {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return undefined;
  }
  if (value >= 2) {
    return "metricPositive metricStrong";
  }
  if (value <= -2) {
    return "metricNegative metricStrong";
  }
  return undefined;
}

function formatUpdatedAt(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("ko-KR", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function MarketManager() {
  const [rows, setRows] = useState<MarketRowItem[]>([]);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [minMarketCap, setMinMarketCap] = useState("");
  const [minPrevVolume, setMinPrevVolume] = useState("");
  const [excludedGroups, setExcludedGroups] = useState<string[]>(DEFAULT_EXCLUDED_GROUPS);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;

    async function load() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch("/api/market", { cache: "no-store" });
        const payload = (await response.json()) as MarketResponse;
        if (!response.ok) {
          throw new Error(payload.error ?? "ETF 마켓 데이터를 불러오지 못했습니다.");
        }

        if (!alive) {
          return;
        }

        setRows(payload.rows ?? []);
        setUpdatedAt(payload.updated_at ?? null);
      } catch (loadError) {
        if (alive) {
          setError(loadError instanceof Error ? loadError.message : "ETF 마켓 데이터를 불러오지 못했습니다.");
        }
      } finally {
        if (alive) {
          setLoading(false);
        }
      }
    }

    load();
    return () => {
      alive = false;
    };
  }, []);

  const filteredRows = useMemo(() => {
    const normalizedQuery = query.trim().toUpperCase();
    const expandedKeywords = excludedGroups.flatMap((group) => EXCLUSION_KEYWORD_GROUPS[group] ?? []);
    const marketCapFilter = Number(minMarketCap || 0);
    const volumeFilter = Number(minPrevVolume || 0);

    return rows.filter((row) => {
      if (
        normalizedQuery &&
        !row.ticker.toUpperCase().includes(normalizedQuery) &&
        !row.name.toUpperCase().includes(normalizedQuery)
      ) {
        return false;
      }

      if (expandedKeywords.some((keyword) => row.name.includes(keyword))) {
        return false;
      }

      if (marketCapFilter > 0 && row.market_cap < marketCapFilter) {
        return false;
      }

      if (volumeFilter > 0 && row.prev_volume < volumeFilter) {
        return false;
      }

      return true;
    }).sort((left, right) => {
      const leftValue = left.daily_change_pct ?? Number.NEGATIVE_INFINITY;
      const rightValue = right.daily_change_pct ?? Number.NEGATIVE_INFINITY;
      if (leftValue !== rightValue) {
        return rightValue - leftValue;
      }
      return left.ticker.localeCompare(right.ticker);
    });
  }, [excludedGroups, minMarketCap, minPrevVolume, query, rows]);

  function toggleGroup(group: string) {
    setExcludedGroups((current) =>
      current.includes(group) ? current.filter((item) => item !== group) : [...current, group],
    );
  }

  if (loading) {
    return (
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBody">
            <p>ETF 마켓 데이터를 불러오는 중...</p>
          </div>
        </div>
      </section>
    );
  }

  return (
    <div className="appPageStack">
      {error ? (
        <div className="appBannerStack">
          <div className="bannerError">{error}</div>
        </div>
      ) : null}
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBody">
          <div className="filterBar">
            <input
              className="field compactField"
              type="text"
              placeholder="티커 또는 종목명 검색"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            <input
              className="field compactField"
              type="number"
              placeholder="최소 시가총액(억)"
              value={minMarketCap}
              onChange={(event) => setMinMarketCap(event.target.value)}
            />
            <input
              className="field compactField"
              type="number"
              placeholder="최소 전일 거래량(주)"
              value={minPrevVolume}
              onChange={(event) => setMinPrevVolume(event.target.value)}
            />
          </div>

          <div className="pillRow">
            {Object.keys(EXCLUSION_KEYWORD_GROUPS).map((group) => {
              const isActive = excludedGroups.includes(group);
              return (
                <button
                  key={group}
                  type="button"
                  className={isActive ? "filterPill filterPillActive" : "filterPill"}
                  onClick={() => toggleGroup(group)}
                >
                  {group}
                </button>
              );
            })}
          </div>

          <div className="tableSummary">
            <span>총 {formatCount(filteredRows.length)}개</span>
            <span>전체 {formatCount(rows.length)}개</span>
            <span>KIS 마스터 갱신 {formatUpdatedAt(updatedAt)}</span>
            <span>기본 정렬 일간(%) 내림차순</span>
          </div>

          <div className="tableWrap">
            <table className="erpTable">
              <thead>
                <tr>
                  <th>티커</th>
                  <th>종목명</th>
                  <th>일간(%)</th>
                  <th>현재가</th>
                  <th>Nav</th>
                  <th>괴리율</th>
                  <th>3달(%)</th>
                  <th>상장일</th>
                  <th>전일거래량(주)</th>
                  <th>시가총액(억)</th>
                </tr>
              </thead>
              <tbody>
                {filteredRows.length === 0 ? (
                  <tr>
                    <td colSpan={10}>
                      <div className="tableEmpty">조건에 맞는 ETF가 없습니다.</div>
                    </td>
                  </tr>
                ) : (
                  filteredRows.map((row) => (
                    <tr key={row.ticker}>
                      <td>{row.ticker}</td>
                      <td>{row.name}</td>
                      <td className={getSignedMetricClass(row.daily_change_pct)}>{formatPercent(row.daily_change_pct)}</td>
                      <td>{formatNullableNumber(row.current_price)}</td>
                      <td>{formatNullableNumber(row.nav)}</td>
                      <td className={getDeviationClass(row.deviation)}>{formatPercent(row.deviation)}</td>
                      <td className={getSignedMetricClass(row.return_3m_pct)}>{formatPercent(row.return_3m_pct)}</td>
                      <td>{row.listed_at || "-"}</td>
                      <td>{formatCount(row.prev_volume)}</td>
                      <td>{formatKrwEok(row.market_cap)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          </div>
        </div>
      </section>
    </div>
  );
}
