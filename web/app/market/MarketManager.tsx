"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { iconSetQuartzBold, themeQuartz } from "ag-grid-community";
import type { ColDef, RowClassParams } from "ag-grid-community";
import { useRouter } from "next/navigation";

import { AppAgGrid } from "../components/AppAgGrid";

type MarketRowItem = {
  ticker: string;
  ticker_pools: string;
  name: string;
  listed_at: string;
  daily_change_pct: number | null;
  current_price: number | null;
  nav: number | null;
  deviation: number | null;
  return_3m_pct: number | null;
  prev_volume: number;
  market_cap: number;
  is_held: boolean;
};

type MarketResponse = {
  updated_at?: string | null;
  rows?: MarketRowItem[];
  error?: string;
};

type MarketGridRow = MarketRowItem & {
  row_number: number;
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

const marketGridTheme = themeQuartz
  .withPart(iconSetQuartzBold)
  .withParams({
    accentColor: "#206bc4",
    backgroundColor: "#ffffff",
    foregroundColor: "#182433",
    headerBackgroundColor: "#f8fafc",
    headerTextColor: "#5b6778",
    spacing: 8,
    fontSize: 14,
    wrapperBorderRadius: 10,
    rowHeight: 38,
    headerHeight: 38,
    cellHorizontalPadding: 12,
    headerColumnBorder: true,
    headerColumnBorderHeight: "70%",
    columnBorder: true,
    oddRowBackgroundColor: "#fbfdff",
    headerCellHoverBackgroundColor: "#eef4fb",
    headerCellMovingBackgroundColor: "#e8f0fb",
    iconButtonHoverBackgroundColor: "#eef4fb",
    iconButtonHoverColor: "#206bc4",
    iconSize: 18,
  });

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

export function MarketManager({
  onHeaderSummaryChange,
}: {
  onHeaderSummaryChange?: (summary: { filteredCount: number; totalCount: number; updatedAt: string | null }) => void;
}) {
  const router = useRouter();
  const [rows, setRows] = useState<MarketRowItem[]>([]);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [minMarketCap, setMinMarketCap] = useState("300");
  const [minPrevVolume, setMinPrevVolume] = useState("50000");
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

    void load();
    return () => {
      alive = false;
    };
  }, []);

  const filteredRows = useMemo(() => {
    const normalizedQuery = query.trim().toUpperCase();
    const expandedKeywords = excludedGroups.flatMap((group) => EXCLUSION_KEYWORD_GROUPS[group] ?? []);
    const marketCapFilter = Number(minMarketCap || 0);
    const volumeFilter = Number(minPrevVolume || 0);

    return rows
      .filter((row) => {
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
      })
      .sort((left, right) => {
        const leftValue = left.daily_change_pct ?? Number.NEGATIVE_INFINITY;
        const rightValue = right.daily_change_pct ?? Number.NEGATIVE_INFINITY;
        if (leftValue !== rightValue) {
          return rightValue - leftValue;
        }
        return left.ticker.localeCompare(right.ticker);
      });
  }, [excludedGroups, minMarketCap, minPrevVolume, query, rows]);

  const gridRows = useMemo<MarketGridRow[]>(
    () => filteredRows.map((row, index) => ({ ...row, row_number: index + 1 })),
    [filteredRows],
  );

  useEffect(() => {
    onHeaderSummaryChange?.({
      filteredCount: filteredRows.length,
      totalCount: rows.length,
      updatedAt,
    });
  }, [filteredRows.length, onHeaderSummaryChange, rows.length, updatedAt]);

  const moveToTickerDetail = useCallback(
    (ticker: string) => {
      const normalizedTicker = String(ticker || "").trim().toUpperCase();
      if (!normalizedTicker) {
        return;
      }
      router.push(`/ticker?ticker=${encodeURIComponent(normalizedTicker)}`);
    },
    [router],
  );

  const columns = useMemo<ColDef<MarketGridRow>[]>(
    () => [
      { field: "row_number", headerName: "#", width: 72, maxWidth: 80 },
      {
        field: "ticker_pools",
        headerName: "종목풀",
        width: 108,
        maxWidth: 116,
        cellRenderer: (params: { value: string }) => String(params.value ?? "").trim() || "-",
      },
      {
        field: "ticker",
        headerName: "티커",
        width: 104,
        cellRenderer: (params: { value: string }) => {
          const value = String(params.value ?? "-");
          return (
            <button
              type="button"
              className="appCodeText"
              style={{ color: "inherit", textDecoration: "none", background: "none", border: "none", padding: 0 }}
              onClick={() => moveToTickerDetail(value)}
            >
              {value}
            </button>
          );
        },
      },
      {
        field: "name",
        headerName: "종목명",
        minWidth: 220,
        flex: 1,
        cellClass: "marketNameCell",
        cellRenderer: (params: { value: string | null | undefined }) => {
          const value = String(params.value ?? "-");
          return (
            <span className="marketNameMain" title={value}>
              {value}
            </span>
          );
        },
      },
      {
        field: "daily_change_pct",
        headerName: "일간(%)",
        width: 112,
        type: "rightAligned",
        sort: "desc",
        comparator: (a, b) => (a ?? Number.NEGATIVE_INFINITY) - (b ?? Number.NEGATIVE_INFINITY),
        cellRenderer: (params: { value: number | null }) => (
          <span className={getSignedMetricClass(params.value)}>{formatPercent(params.value)}</span>
        ),
      },
      {
        field: "current_price",
        headerName: "현재가",
        width: 110,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatNullableNumber(params.value),
      },
      {
        field: "nav",
        headerName: "Nav",
        width: 110,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => formatNullableNumber(params.value),
      },
      {
        field: "deviation",
        headerName: "괴리율",
        width: 96,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => (
          <span className={getDeviationClass(params.value)}>{formatPercent(params.value)}</span>
        ),
      },
      {
        field: "return_3m_pct",
        headerName: "3달(%)",
        width: 96,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null }) => (
          <span className={getSignedMetricClass(params.value)}>{formatPercent(params.value)}</span>
        ),
      },
      { field: "listed_at", headerName: "상장일", width: 112 },
      {
        field: "prev_volume",
        headerName: "전일거래량(주)",
        width: 128,
        type: "rightAligned",
        cellRenderer: (params: { value: number }) => formatCount(params.value),
      },
      {
        field: "market_cap",
        headerName: "시가총액(억)",
        width: 128,
        type: "rightAligned",
        cellRenderer: (params: { value: number }) => formatKrwEok(params.value),
      },
    ],
    [moveToTickerDetail],
  );

  function toggleGroup(group: string) {
    setExcludedGroups((current) =>
      current.includes(group) ? current.filter((item) => item !== group) : [...current, group],
    );
  }

  return (
    <div className="appPageStack appPageStackFill">
      {error ? (
        <div className="appBannerStack">
          <div className="bannerError">{error}</div>
        </div>
      ) : null}
      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <div className="appMainHeader marketMainHeader">
              <div className="appMainHeaderLeft marketMainHeaderLeft">
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">티커/종목명</span>
                  <input
                    className="field compactField"
                    type="text"
                    placeholder="티커 또는 종목명을 입력"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                  />
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">시가총액(억)</span>
                  <input
                    className="field compactField"
                    type="number"
                    placeholder="최소 시가총액"
                    value={minMarketCap}
                    onChange={(event) => setMinMarketCap(event.target.value)}
                  />
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">거래량(주)</span>
                  <input
                    className="field compactField"
                    type="number"
                    placeholder="최소 전일 거래량"
                    value={minPrevVolume}
                    onChange={(event) => setMinPrevVolume(event.target.value)}
                  />
                </label>
              </div>
            </div>
          </div>
          <div className="card-body appCardBodyTight appTableCardBodyFill">
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

            <div className="appGridFillWrap" style={{ minHeight: 0 }}>
              <AppAgGrid
                rowData={gridRows}
                columnDefs={columns}
                loading={loading}
                minHeight="100%"
                theme={marketGridTheme}
                getRowClass={(params: RowClassParams<MarketGridRow>) => (params.data?.is_held ? "appHeldRow" : "")}
                gridOptions={{
                  suppressMovableColumns: true,
                }}
              />
            </div>
          </div>
        </div>
      </section>

      <style jsx global>{`
        .marketNameCell {
          min-width: 0;
          overflow: hidden;
        }
      `}</style>

      <style jsx>{`
        .marketNameMain {
          display: block;
          width: 100%;
          min-width: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
      `}</style>
    </div>
  );
}
