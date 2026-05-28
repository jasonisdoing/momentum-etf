"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ColDef, RowClassParams, GridOptions } from "ag-grid-community";
import { PageFrame } from "../components/PageFrame";
import { AppAgGrid } from "../components/AppAgGrid";
import { ResponsiveFiltersSection } from "../components/ResponsiveFiltersSection";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { createAppGridTheme } from "../components/app-grid-theme";
import { readSessionTtlCache, writeSessionTtlCache } from "../../lib/session-ttl-cache";

// ─── 타입 ────────────────────────────────────────────────────────────────────
type HoldingCountryOption = {
  code: string;
  label: string;
};

type ComponentSource = {
  etf_ticker: string;
  etf_name: string;
  weight: number;
  current_price?: number | null;
  change_pct?: number | null;
  currency?: string;
  return_pct?: number | null;
  daily_profit_krw?: number | null;
  cumulative_profit_krw?: number | null;
  valuation_krw?: number | null;
};

type ComponentRow = {
  ticker: string;
  name: string;
  has_components?: boolean;
  total_weight: number;
  sources: ComponentSource[];
  current_price?: number | null;
  change_pct?: number | null;
  currency?: string;
  return_pct?: number | null;
  daily_profit_krw?: number | null;
  cumulative_profit_krw?: number | null;
  valuation_krw?: number | null;
};

type EtfDetail = {
  ticker: string;
  name: string;
  quantity: number;
  component_count: number;
  has_components: boolean;
};

type HoldingsComponentsData = {
  account_id: string;
  account_name: string;
  held_etf_count: number;
  components_total_count?: number;
  components_visible_limit?: number;
  components: ComponentRow[];
  etf_details: EtfDetail[];
};

type MainGridRow = ComponentRow & { rowType: "main" };
type DetailGridRow = { rowType: "detail"; parentTicker: string; sources: ComponentSource[] };
type GridRow = MainGridRow | DetailGridRow;

const HOLDING_COUNTRIES_CACHE_KEY = "momentum-etf:holdings-details:holding-countries";
const HOLDINGS_COMPONENTS_CACHE_KEY_PREFIX = "momentum-etf:holdings-details:holding-country:";
const HOLDING_COUNTRIES_CACHE_TTL_MS = 300_000;
const HOLDINGS_COMPONENTS_CACHE_TTL_MS = 30_000;
const REMEMBERED_HOLDING_COUNTRY_STORAGE_KEY = "momentum-etf:holdings-details:remembered-holding-country";

// ─── 유틸 ────────────────────────────────────────────────────────────────────
function formatWeight(w: number): string {
  return `${w.toFixed(2)}%`;
}

function formatPrice(val: number | null | undefined, currency?: string): string {
  if (val == null) return "-";
  const normalizedCurrency = (currency ?? "KRW").toUpperCase();
  if (normalizedCurrency === "KRW") {
    return `${Math.floor(val).toLocaleString()}원`;
  }
  const fractionDigits = normalizedCurrency === "JPY" ? 0 : 2;
  return `${val.toLocaleString(undefined, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  })} ${normalizedCurrency}`;
}

function formatSignedPercent(val: number | null | undefined): string {
  if (val == null) return "-";
  return `${val.toFixed(2)}%`;
}

function formatKrw(val: number | null | undefined): string {
  if (val == null) return "-";
  const rounded = Math.round(val);
  return `${rounded.toLocaleString()}원`;
}

function maskAmount(showAmounts: boolean, value: string): string {
  if (value === "-") return value;
  return showAmounts ? value : "••••";
}

function formatDisplayName(name: string | null | undefined): string {
  if (!name) return "-";
  const original = name.trim();
  const cleaned = original
    .replace(/\s*[\(\（][^\)\）]*[\)\）]\s*/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return cleaned || original;
}

function getSignedClass(val: number | null | undefined): string {
  if (val == null) return "";
  if (val > 0) return "metricPositive";
  if (val < 0) return "metricNegative";
  return "";
}

function formatSignedPercentWithPlus(val: number | null | undefined): string {
  if (val == null || Number.isNaN(val)) return "-";
  return `${val > 0 ? "+" : ""}${val.toFixed(2)}%`;
}

const BOX_VIEW_TOP_N = 100;

// 박스 좌측 strip: 변동률 절댓값에 따라 진한 빨강(양수) / 진한 파랑(음수). 5% 에서 최대 진하기.
function getChangeStripColor(changePct: number | null | undefined): string | undefined {
  if (changePct == null || Number.isNaN(changePct) || changePct === 0) return undefined;
  const intensity = Math.min(Math.abs(changePct) / 5, 1);
  const lightness = 60 - intensity * 30; // 60% (옅음) → 30% (진함)
  return changePct > 0
    ? `hsl(0, 75%, ${lightness}%)`
    : `hsl(216, 75%, ${lightness}%)`;
}

function HoldingsBoxView({ components }: { components: ComponentRow[] }) {
  // 비중 내림차순 상위 N개
  const topComponents = useMemo(() => {
    return [...components]
      .sort((a, b) => Number(b.total_weight ?? 0) - Number(a.total_weight ?? 0))
      .slice(0, BOX_VIEW_TOP_N);
  }, [components]);

  if (topComponents.length === 0) {
    return (
      <div style={{ padding: "1.5rem", color: "#64748b" }}>표시할 구성종목이 없습니다.</div>
    );
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
        gap: "0.5rem",
        padding: "0.75rem",
        overflowY: "auto",
        height: "100%",
        alignContent: "start",
      }}
    >
      {topComponents.map((row) => {
        const weight = Number(row.total_weight ?? 0);
        const changePct = row.change_pct;
        // 변동률 기반 배경 (양수 빨강 / 음수 파랑, 5%에서 alpha 최대치 0.18).
        let changeBg: string | undefined;
        if (changePct != null && !Number.isNaN(changePct) && changePct !== 0) {
          const alpha = Math.min(0.18, (Math.abs(changePct) / 5) * 0.18);
          changeBg =
            changePct > 0
              ? `rgba(239, 68, 68, ${alpha})`
              : `rgba(37, 99, 235, ${alpha})`;
        }
        const stripColor = getChangeStripColor(changePct);
        return (
          <div
            key={row.ticker}
            style={{
              position: "relative",
              overflow: "hidden",
              minHeight: "3.25rem",
              padding: "0.42rem 0.65rem",
              borderRight: "1px solid #d6deea",
              borderBottom: "1px solid #d6deea",
              borderRadius: "0.25rem",
              background: changeBg || "#fff",
              boxShadow: stripColor ? `inset 4px 0 0 0 ${stripColor}` : undefined,
            }}
          >
            {/* 윗줄: 종목명 (전체 너비) */}
            <div
              style={{
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
                color: "#1f2937",
                fontSize: "1rem",
                fontWeight: 800,
                lineHeight: 1.25,
              }}
            >
              {formatDisplayName(row.name) || row.ticker}
            </div>
            {/* 아랫줄: 티커 + 비중 + 변동률 */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: "0.5rem",
                marginTop: "0.15rem",
              }}
            >
              <div style={{ flex: "0 0 auto", color: "#9ca3af", fontSize: "0.85rem" }}>
                {row.ticker}
              </div>
              <span style={{ color: "#475569", fontWeight: 900, fontSize: "0.95rem" }}>
                {weight.toFixed(2)}%
              </span>
              <span
                className={getSignedClass(changePct)}
                style={{ fontSize: "0.95rem", fontWeight: 800 }}
              >
                {formatSignedPercentWithPlus(changePct)}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

const gridTheme = createAppGridTheme();

function isDetailRow(row: GridRow | undefined): row is DetailGridRow {
  return row?.rowType === "detail";
}

function readRememberedHoldingCountry(): string {
  if (typeof window === "undefined") return "";
  try {
    return window.localStorage.getItem(REMEMBERED_HOLDING_COUNTRY_STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

function writeRememberedHoldingCountry(code: string): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(REMEMBERED_HOLDING_COUNTRY_STORAGE_KEY, code);
  } catch {
    // ignore quota errors
  }
}

// ─── 컴포넌트 ─────────────────────────────────────────────────────────────────
export function HoldingsDetailsPageClient() {
  const [holdingCountries, setHoldingCountries] = useState<HoldingCountryOption[]>([]);
  const [selectedHoldingCountry, setSelectedHoldingCountry] = useState<string>(
    readRememberedHoldingCountry() || "",
  );
  const [data, setData] = useState<HoldingsComponentsData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // 뷰 모드: 리스트(기본 — 테이블) / 박스 (구성종목을 카드 박스로 나열)
  const [viewMode, setViewMode] = useState<"list" | "box">("list");
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);
  const [showAmounts, setShowAmounts] = useState(true);
  const requestSequenceRef = useRef(0);

  // 종목 국가 목록 로드 (미국/한국/호주/기타국가 고정 4개)
  useEffect(() => {
    async function fetchHoldingCountries() {
      try {
        const cached = readSessionTtlCache<HoldingCountryOption[]>(
          HOLDING_COUNTRIES_CACHE_KEY,
          HOLDING_COUNTRIES_CACHE_TTL_MS,
        );
        if (cached && cached.length > 0) {
          setHoldingCountries(cached);
          if (!selectedHoldingCountry) {
            const first = cached[0].code;
            setSelectedHoldingCountry(first);
            writeRememberedHoldingCountry(first);
          }
          return;
        }

        const res = await fetch("/api/holdings-components/holding-countries", { cache: "no-store" });
        if (!res.ok) throw new Error("종목 국가 목록을 불러오지 못했습니다.");
        const list = (await res.json()) as HoldingCountryOption[];
        writeSessionTtlCache(HOLDING_COUNTRIES_CACHE_KEY, list);
        setHoldingCountries(list);
        if (list.length > 0 && !selectedHoldingCountry) {
          const first = list[0].code;
          setSelectedHoldingCountry(first);
          writeRememberedHoldingCountry(first);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "오류가 발생했습니다.");
      }
    }
    void fetchHoldingCountries();
  }, []);

  // 선택된 종목 국가 데이터 로드
  const loadData = useCallback(async (countryCode: string) => {
    if (!countryCode) return;
    const requestSequence = requestSequenceRef.current + 1;
    requestSequenceRef.current = requestSequence;
    const cacheKey = `${HOLDINGS_COMPONENTS_CACHE_KEY_PREFIX}${countryCode}`;
    const cached = readSessionTtlCache<HoldingsComponentsData>(cacheKey, HOLDINGS_COMPONENTS_CACHE_TTL_MS);
    if (cached) {
      setError(null);
      setData(cached);
      setExpandedTicker(null);
      setIsLoading(false);
      return;
    }
    setIsLoading(true);
    setError(null);
    setData(null);
    setExpandedTicker(null);
    try {
      const res = await fetch(
        `/api/holdings-components/by-holding-country?country_code=${encodeURIComponent(countryCode)}`,
        { cache: "no-store" },
      );
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? "데이터를 불러오지 못했습니다.");
      }
      const result = (await res.json()) as HoldingsComponentsData;
      if (requestSequenceRef.current !== requestSequence) {
        return;
      }
      writeSessionTtlCache(cacheKey, result);
      setData(result);
    } catch (e) {
      if (requestSequenceRef.current !== requestSequence) {
        return;
      }
      setError(e instanceof Error ? e.message : "오류가 발생했습니다.");
    } finally {
      if (requestSequenceRef.current === requestSequence) {
        setIsLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    if (selectedHoldingCountry) void loadData(selectedHoldingCountry);
  }, [selectedHoldingCountry, loadData]);

  // 그리드 데이터 가공
  const gridRows = useMemo<GridRow[]>(() => {
    if (!data) return [];
    const result: GridRow[] = [];
    for (const comp of data.components) {
      result.push({ ...comp, rowType: "main" });
      if (expandedTicker === comp.ticker && comp.has_components === true && comp.sources.length > 0) {
        result.push({
          rowType: "detail",
          parentTicker: comp.ticker,
          sources: comp.sources,
        });
      }
    }
    return result;
  }, [data, expandedTicker]);

  const columnDefs = useMemo<ColDef<GridRow>[]>(() => [
    {
      headerName: "티커",
      field: "ticker",
      width: 100,
      cellRenderer: (params: { data?: GridRow; value?: string }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        return <TickerDetailLink ticker={String(params.value ?? "")} className="text-muted fw-semibold" />;
      },
    },
    {
      headerName: "종목명",
      field: "name",
      flex: 1,
      minWidth: 140,
      cellClass: "holdingsDetailsNameAgCell",
      cellRenderer: (params: { data?: GridRow; value?: string }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const mainRow = params.data as MainGridRow;
        const isExpanded = expandedTicker === mainRow.ticker;
        const hasSources = mainRow.has_components === true && mainRow.sources.length > 0;
        return (
          <div
            className={`d-flex align-items-center gap-2 holdingsDetailsNameCell ${hasSources ? "cursor-pointer" : ""}`}
            style={{ userSelect: "none" }}
          >
            {hasSources && (
              <span className="text-primary d-flex align-items-center" style={{ fontSize: "10px", transition: "transform 0.15s", transform: isExpanded ? "rotate(90deg)" : "none" }}>
                ▶
              </span>
            )}
            <span className="fw-bold text-dark holdingsDetailsNameText">{formatDisplayName(params.value)}</span>
          </div>
        );
      },
    },
    {
      headerName: "비중",
      field: "total_weight",
      width: 90,
      type: "rightAligned",
      cellRenderer: (params: { value?: number }) => (
        <span className="fw-bold" style={{ color: "#206bc4" }}>
          {params.value != null ? formatWeight(params.value) : "-"}
        </span>
      ),
    },
    {
      headerName: "현재가",
      field: "current_price",
      width: 110,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: number | null }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const row = params.data as MainGridRow;
        if (row.ticker === "-") return "-";
        return formatPrice(params.value, row.currency);
      },
    },
    {
      headerName: "일간(%)",
      field: "change_pct",
      width: 80,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: number | null }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const row = params.data as MainGridRow;
        if (row.ticker === "-") return "-";
        return <span className={getSignedClass(params.value)}>{formatSignedPercent(params.value)}</span>;
      },
    },
    {
      headerName: "금일 손익",
      field: "daily_profit_krw",
      width: 110,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: number | null }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        return <span className={showAmounts ? getSignedClass(params.value) : ""}>{maskAmount(showAmounts, formatKrw(params.value))}</span>;
      },
    },
    {
      headerName: "누적 손익",
      field: "cumulative_profit_krw",
      width: 110,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: number | null }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        return <span className={showAmounts ? getSignedClass(params.value) : ""}>{maskAmount(showAmounts, formatKrw(params.value))}</span>;
      },
    },
    {
      headerName: "평가금액",
      field: "valuation_krw",
      width: 140,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: number | null }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        return maskAmount(showAmounts, formatKrw(params.value));
      },
    },
  ], [expandedTicker, showAmounts]);

  // 상세 패널 렌더러 (부모 행과 컬럼/정렬 완벽 정밀 타격)
  const DetailRenderer = useCallback((params: { data?: GridRow }) => {
    if (!params.data || !isDetailRow(params.data)) return null;
    
    return (
      <div className="holdingsDetailNestedRows">
        {params.data.sources.map((src, idx) => (
          <div key={idx} className="holdingsDetailRow">
            {/* 1. 티커 (100px) - 좌측 패딩 12px, 폰트 13px 고정 */}
            <div className="hdColTicker fw-semibold text-muted" style={{ fontFamily: "monospace", fontSize: "13px" }}>
              <TickerDetailLink ticker={src.etf_ticker} />
            </div>
            
            {/* 2. 종목명 (flex:1) - 좌측 패딩 12px */}
            <div className="hdColName fw-bold text-dark">
              {formatDisplayName(src.etf_name)}
            </div>
            
            {/* 3. 비중 (90px) - 우측 패딩 12px */}
            <div className="hdColWeight fw-bold text-primary">
              {formatWeight(src.weight)}
            </div>
            
            {/* 4. 현재가 (110px) - 우측 패딩 12px */}
            <div className="hdColPrice">
              {formatPrice(src.current_price, src.currency)}
            </div>
            
            {/* 5. 일간(%) (100px) - 우측 패딩 12px */}
            <div className="hdColChange">
              <span className={getSignedClass(src.change_pct)}>
                {formatSignedPercent(src.change_pct)}
              </span>
            </div>

            <div className="hdColDailyProfit">
              <span className={showAmounts ? getSignedClass(src.daily_profit_krw) : ""}>
                {maskAmount(showAmounts, formatKrw(src.daily_profit_krw))}
              </span>
            </div>

            <div className="hdColCumulativeProfit">
              <span className={showAmounts ? getSignedClass(src.cumulative_profit_krw) : ""}>
                {maskAmount(showAmounts, formatKrw(src.cumulative_profit_krw))}
              </span>
            </div>

            <div className="hdColValuation">
              {maskAmount(showAmounts, formatKrw(src.valuation_krw))}
            </div>
          </div>
        ))}
      </div>
    );
  }, [showAmounts]);

  const gridOptions = useMemo<GridOptions<GridRow>>(() => ({
    getRowId: (params) => (isDetailRow(params.data) ? `detail:${params.data.parentTicker}` : params.data.ticker),
    isFullWidthRow: (params) => isDetailRow(params.rowNode.data),
    fullWidthCellRenderer: DetailRenderer,
    // 정밀 계산: Padding-top(8) + RowHeight(38) * N + Padding-bottom(8)
    getRowHeight: (params) => (isDetailRow(params.data) ? 16 + params.data.sources.length * 38 : 38),
    onCellClicked: (params) => {
      if (
        params.data &&
        !isDetailRow(params.data) &&
        params.colDef.field === "name" &&
        params.data.has_components === true
      ) {
        const ticker = (params.data as MainGridRow).ticker;
        setExpandedTicker((prev) => (prev === ticker ? null : ticker));
      }
    },
    overlayNoRowsTemplate: '<span class="ag-overlay-no-rows-center">데이터가 없습니다.</span>',
  }), [DetailRenderer]);

  // 헤더 우측 정보
  const componentsTotalCount = data?.components_total_count ?? data?.components.length ?? 0;
  const componentsVisibleLimit = data?.components_visible_limit ?? data?.components.length ?? 0;
  const componentsMetricText =
    componentsTotalCount > componentsVisibleLimit
      ? `${componentsTotalCount}개 중 상위 ${componentsVisibleLimit}개`
      : `${componentsTotalCount}개`;

  // 전체 구성종목 가중 평균 변동률과 추적 비중. 리스트/박스 뷰 모두 동일 값 사용.
  const averageStats = useMemo(() => {
    if (!data) return null;
    let tracked = 0;
    let weightedSum = 0;
    for (const row of data.components) {
      const w = Number(row.total_weight ?? 0);
      const c = row.change_pct;
      if (w <= 0) continue;
      if (c == null || Number.isNaN(c)) continue;
      tracked += w;
      weightedSum += w * c;
    }
    if (tracked <= 0) return null;
    return { avgPct: weightedSum / tracked, trackedWeight: tracked };
  }, [data]);

  const titleRight = data ? (
    <div className="appHeaderMetrics rankToolbarMeta">
      <div className="appHeaderMetric">
        <span>보유 ETF:</span>
        <span className="appHeaderMetricValue">{data.held_etf_count}개</span>
      </div>
      <div className="appHeaderMetric">
        <span>구성종목:</span>
        <span className="appHeaderMetricValue">{componentsMetricText}</span>
      </div>
      {averageStats ? (
        <div className="appHeaderMetric">
          <span>평균:</span>
          <span
            className="appHeaderMetricValue"
            style={{
              color:
                averageStats.avgPct > 0
                  ? "#d32f2f"
                  : averageStats.avgPct < 0
                  ? "#1d4ed8"
                  : undefined,
            }}
          >
            {formatSignedPercentWithPlus(averageStats.avgPct)}
          </span>
          <span style={{ color: "#94a3b8", marginLeft: "0.25rem", fontSize: "0.85rem" }}>
            (추적 {averageStats.trackedWeight.toFixed(0)}%)
          </span>
        </div>
      ) : null}
    </div>
  ) : null;

  return (
    <PageFrame title="보유종목 상세" fullHeight fullWidth titleRight={titleRight}>
      {error && (
        <div className="appBannerStack">
          <div className="bannerError alert alert-danger mb-0">{error}</div>
        </div>
      )}

      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <ResponsiveFiltersSection>
              <div className="appMainHeader">
                <div className="appMainHeaderLeft rankMainHeaderLeft">
                  <label className="appLabeledField">
                    <span className="appLabeledFieldLabel">종목 국가</span>
                    <select
                      className="form-select"
                      value={selectedHoldingCountry}
                      onChange={(e) => {
                        const nextCode = e.target.value;
                        setSelectedHoldingCountry(nextCode);
                        writeRememberedHoldingCountry(nextCode);
                      }}
                      disabled={holdingCountries.length === 0}
                    >
                      {holdingCountries.length === 0 ? (
                        <option value="">종목 국가 불러오는 중...</option>
                      ) : (
                        holdingCountries.map((c) => (
                          <option key={c.code} value={c.code}>
                            {c.label}
                          </option>
                        ))
                      )}
                    </select>
                  </label>
                  <label className="appLabeledField" style={{ minWidth: 0, width: "auto", flex: "0 0 auto" }}>
                    <span className="appLabeledFieldLabel">뷰</span>
                    <div className="appSegmentedToggle" role="group" aria-label="보기 형식 선택">
                      <button
                        type="button"
                        className={
                          viewMode === "list"
                            ? "btn appSegmentedToggleButton is-active"
                            : "btn appSegmentedToggleButton"
                        }
                        onClick={() => setViewMode("list")}
                      >
                        리스트
                      </button>
                      <button
                        type="button"
                        className={
                          viewMode === "box"
                            ? "btn appSegmentedToggleButton is-active"
                            : "btn appSegmentedToggleButton"
                        }
                        onClick={() => setViewMode("box")}
                      >
                        박스
                      </button>
                    </div>
                  </label>
                </div>
                <div className="appMainHeaderRight">
                  <button
                    type="button"
                    className={`btn btn-sm shadow-sm ${showAmounts ? "btn-outline-secondary" : "btn-dark"}`}
                    onClick={() => setShowAmounts((prev) => !prev)}
                  >
                    {showAmounts ? "금액 가리기" : "금액 보기"}
                  </button>
                </div>
              </div>
            </ResponsiveFiltersSection>
          </div>
          
          <div className="card-body appCardBodyTight appTableCardBodyFill holdingsDetailsBody">
            {viewMode === "box" ? (
              <HoldingsBoxView components={data?.components ?? []} />
            ) : (
            <div className="appGridFillWrap holdingsDetailsGridWrap">
              <AppAgGrid
                rowData={isLoading ? [] : gridRows}
                columnDefs={columnDefs}
                loading={isLoading}
                theme={gridTheme}
                minHeight="100%"
                gridOptions={gridOptions}
                getRowClass={(params: RowClassParams<GridRow>) => {
                  if (isDetailRow(params.data)) return "holdingsDetailFullRow";
                  return "";
                }}
              />
            </div>
            )}
          </div>
        </div>
      </section>

      <style jsx global>{`
        .holdingsDetailFullRow {
          background-color: #fbfcfe !important;
          border-bottom: 1px solid #e2e8f0;
        }
        .holdingsDetailNestedRows {
          background-color: #f1f5f9;
          padding: 8px 0;
          box-sizing: border-box;
        }
        .holdingsDetailRow {
          display: flex;
          align-items: center;
          height: 38px;
          border-bottom: 1px solid #e2e8f0;
          transition: background-color 0.15s;
          box-sizing: border-box;
        }
        .holdingsDetailRow:hover {
          background-color: #ffffff;
        }
        .holdingsDetailRow > div {
          height: 100%;
          display: flex;
          align-items: center;
          border-right: 1px solid #e2e8f0;
          box-sizing: border-box;
          font-size: 14px;
        }
        .holdingsDetailRow > div:last-child {
          border-right: none;
        }
        /* 각 컬럼 너비 및 패딩 정밀 동기화 (Ag-Grid 12px 기준) */
        .hdColTicker { width: 100px; padding-left: 12px; }
        .hdColName   { flex: 1; min-width: 0; padding-left: 12px; overflow: hidden; }
        .hdColWeight { width: 90px; padding-right: 12px; justify-content: flex-end; }
        .hdColPrice  { width: 110px; padding-right: 12px; justify-content: flex-end; }
        .hdColChange { width: 100px; padding-right: 12px; justify-content: flex-end; }
        .hdColDailyProfit { width: 140px; padding-right: 12px; justify-content: flex-end; }
        .hdColCumulativeProfit { width: 140px; padding-right: 12px; justify-content: flex-end; }
        .hdColValuation { width: 140px; padding-right: 12px; justify-content: flex-end; }
        .holdingsDetailsNameCell {
          width: 100%;
          min-width: 0;
          overflow: hidden;
        }
        .holdingsDetailsNameAgCell {
          overflow: hidden;
        }
        .holdingsDetailsNameAgCell .ag-cell-wrapper,
        .holdingsDetailsNameAgCell .ag-cell-value {
          min-width: 0;
          width: 100%;
          overflow: hidden;
        }
        .holdingsDetailsNameText {
          min-width: 0;
          max-width: 100%;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          display: inline-block;
        }
        .holdingsDetailsBody {
          display: flex;
          flex-direction: column;
          gap: 12px;
          min-height: 0;
        }
        .holdingsDetailsGridWrap {
          flex: 1 1 0;
          min-height: 280px;
        }
        .holdingsTreemapSection {
          flex: 0 0 48%;
          min-height: 300px;
          display: flex;
          flex-direction: column;
          gap: 8px;
          border: 1px solid #d8e2ef;
          border-radius: 10px;
          background: #ffffff;
          padding: 12px;
          box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        }
        .holdingsTreemapSectionCollapsed {
          flex-basis: auto;
          min-height: 0;
        }
        .holdingsTreemapHeader {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 12px;
          min-height: 28px;
        }
        .holdingsTreemapHeader h3 {
          margin: 0;
          font-size: 18px;
          font-weight: 800;
          color: #1f2937;
        }
        .holdingsTreemapHeaderActions {
          display: inline-flex;
          align-items: center;
          justify-content: flex-end;
          gap: 10px;
        }
        .holdingsTreemapHeader span {
          font-size: 13px;
          font-weight: 700;
          color: #718096;
        }
        .holdingsTreemapToggle {
          border: 0;
          background: transparent;
          color: #4a5568;
          cursor: pointer;
          font-size: 13px;
          font-weight: 800;
          line-height: 1;
          padding: 4px 0;
        }
        .holdingsTreemapToggle:hover {
          color: #1f2937;
        }
        .holdingsTreemapCanvas {
          position: relative;
          flex: 1 1 auto;
          height: 100%;
          min-height: 0;
          overflow: hidden;
          background: #eef2f7;
          border: 1px solid #d8e2ef;
        }
        .holdingsTreemapSvg {
          display: block;
          width: 100%;
          height: 100%;
        }
        .holdingsTreemapGauge {
          flex: 0 0 auto;
          padding: 2px 8px 0;
        }
        .holdingsTreemapGaugeBar {
          height: 14px;
          border-radius: 999px;
          background:
            linear-gradient(
              90deg,
              #3b82f6 0%,
              #2563eb 25%,
              #1e4f94 47%,
              #b8c0ce 50%,
              #a52a2a 53%,
              #c53030 75%,
              #ef4444 100%
            );
        }
        .holdingsTreemapGaugeLabels {
          display: flex;
          justify-content: space-between;
          margin-top: 6px;
          color: #8a94a6;
          font-size: 13px;
          font-weight: 800;
          line-height: 1;
        }
        .holdingsTreemapTicker {
          font-size: 16px;
          font-weight: 800;
          letter-spacing: 0;
          dominant-baseline: hanging;
        }
        .holdingsTreemapChange {
          font-size: 15px;
          font-weight: 800;
          dominant-baseline: hanging;
        }
        .holdingsTreemapWeight {
          font-size: 13px;
          font-weight: 800;
          dominant-baseline: text-after-edge;
        }
        .holdingsTreemapEmpty {
          flex: 1 1 auto;
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 260px;
          border: 1px dashed #cbd5e1;
          border-radius: 8px;
          color: #718096;
          font-weight: 700;
        }
        .hdColName {
          white-space: nowrap;
          text-overflow: ellipsis;
        }

        .cursor-pointer { cursor: pointer; }
      `}</style>
    </PageFrame>
  );
}
