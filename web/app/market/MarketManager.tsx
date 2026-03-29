"use client";

import { useEffect, useMemo, useState } from "react";
import { type GridColDef } from "@mui/x-data-grid";

import { AppDataGrid } from "../components/AppDataGrid";

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

type MarketGridRow = MarketRowItem & {
  id: string;
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
  const gridRows = useMemo<MarketGridRow[]>(
    () => filteredRows.map((row) => ({ ...row, id: row.ticker })),
    [filteredRows],
  );
  const columns = useMemo<GridColDef<MarketGridRow>[]>(
    () => [
      { field: "ticker", headerName: "티커", minWidth: 92, width: 92, renderCell: (params) => <span className="appCodeText">{params.row.ticker}</span> },
      { field: "name", headerName: "종목명", minWidth: 220, flex: 1 },
      {
        field: "daily_change_pct",
        headerName: "일간(%)",
        minWidth: 92,
        width: 92,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => <span className={getSignedMetricClass(params.row.daily_change_pct)}>{formatPercent(params.row.daily_change_pct)}</span>,
      },
      {
        field: "current_price",
        headerName: "현재가",
        minWidth: 108,
        width: 108,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatNullableNumber(params.row.current_price),
      },
      {
        field: "nav",
        headerName: "Nav",
        minWidth: 108,
        width: 108,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatNullableNumber(params.row.nav),
      },
      {
        field: "deviation",
        headerName: "괴리율",
        minWidth: 92,
        width: 92,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => <span className={getDeviationClass(params.row.deviation)}>{formatPercent(params.row.deviation)}</span>,
      },
      {
        field: "return_3m_pct",
        headerName: "3달(%)",
        minWidth: 92,
        width: 92,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => <span className={getSignedMetricClass(params.row.return_3m_pct)}>{formatPercent(params.row.return_3m_pct)}</span>,
      },
      { field: "listed_at", headerName: "상장일", minWidth: 112, width: 112 },
      {
        field: "prev_volume",
        headerName: "전일거래량(주)",
        minWidth: 120,
        width: 120,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatCount(params.row.prev_volume),
      },
      {
        field: "market_cap",
        headerName: "시가총액(억)",
        minWidth: 120,
        width: 120,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatKrwEok(params.row.market_cap),
      },
    ],
    [],
  );

  function toggleGroup(group: string) {
    setExcludedGroups((current) =>
      current.includes(group) ? current.filter((item) => item !== group) : [...current, group],
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

          <AppDataGrid rows={gridRows} columns={columns} loading={loading} minHeight="60vh" />
          </div>
        </div>
      </section>
    </div>
  );
}
