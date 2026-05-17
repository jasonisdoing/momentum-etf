"use client";

import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ColDef, RowClassParams, GridOptions } from "ag-grid-community";
import { hierarchy, treemap, treemapSquarify } from "d3-hierarchy";
import type { HierarchyRectangularNode } from "d3-hierarchy";
import { PageFrame } from "../components/PageFrame";
import { AppAgGrid } from "../components/AppAgGrid";
import { ResponsiveFiltersSection } from "../components/ResponsiveFiltersSection";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { createAppGridTheme } from "../components/app-grid-theme";
import {
  readRememberedMomentumEtfAccountId,
  writeRememberedMomentumEtfAccountId,
} from "../components/account-selection";
import { readSessionTtlCache, writeSessionTtlCache } from "../../lib/session-ttl-cache";

// ─── 타입 ────────────────────────────────────────────────────────────────────
type AccountOption = {
  account_id: string;
  name: string;
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

type TreemapRect = {
  item: ComponentRow;
  x: number;
  y: number;
  width: number;
  height: number;
  normalizedWeight: number;
};

type TreemapDatum = {
  item?: ComponentRow;
  value?: number;
  children?: TreemapDatum[];
};

const HOLDINGS_COMPONENT_ACCOUNTS_CACHE_KEY = "momentum-etf:holdings-details:accounts";
const HOLDINGS_COMPONENTS_CACHE_KEY_PREFIX = "momentum-etf:holdings-details:data:";
const HOLDINGS_COMPONENT_ACCOUNTS_CACHE_TTL_MS = 300_000;
const HOLDINGS_COMPONENTS_CACHE_TTL_MS = 30_000;
const TREEMAP_FALLBACK_WIDTH = 1_200;
const TREEMAP_FALLBACK_HEIGHT = 360;
const TREEMAP_VISIBLE_LIMIT = 20;

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

function getTreemapLabel(item: ComponentRow): string {
  const ticker = item.ticker.trim();
  if (ticker === "-") return formatDisplayName(item.name);
  if (/^\d{6}$/.test(ticker)) return formatDisplayName(item.name);
  return ticker;
}

function getSignedClass(val: number | null | undefined): string {
  if (val == null) return "";
  if (val > 0) return "metricPositive";
  if (val < 0) return "metricNegative";
  return "";
}

function getTreemapColor(changePct: number | null | undefined): string {
  if (changePct == null || changePct === 0) return "#b8c0ce";
  const intensity = Math.min(Math.abs(changePct) / 6, 1);
  if (changePct > 0) {
    const lightness = 66 - intensity * 28;
    return `hsl(0 64% ${lightness}%)`;
  }
  const lightness = 66 - intensity * 26;
  return `hsl(216 78% ${lightness}%)`;
}

function getTreemapTextColor(changePct: number | null | undefined): string {
  if (changePct == null) return "#1f2937";
  if (changePct === 0) return "#1f2937";
  return "#ffffff";
}

function getTreemapWeight(item: ComponentRow): number {
  const weight = Number(item.total_weight);
  return Number.isFinite(weight) && weight > 0 ? weight : 0;
}

function buildTreemapRects(items: ComponentRow[], width: number, height: number): TreemapRect[] {
  const children = items
    .map((item) => ({ item, value: getTreemapWeight(item) }))
    .filter((entry) => entry.value > 0)
    .sort((a, b) => b.value - a.value)
    .slice(0, TREEMAP_VISIBLE_LIMIT);
  const totalWeight = children.reduce((acc, entry) => acc + entry.value, 0);
  if (totalWeight <= 0 || width <= 0 || height <= 0) return [];
  const normalizedChildren = children.map((entry) => ({
    item: entry.item,
    value: (entry.value / totalWeight) * 100,
  }));

  const root = hierarchy<TreemapDatum>({ children: normalizedChildren })
    .sum((datum) => datum.value ?? 0)
    .sort((a, b) => (b.value ?? 0) - (a.value ?? 0));

  treemap<TreemapDatum>()
    .tile(treemapSquarify.ratio(1.2))
    .size([width, height])
    .paddingInner(3)
    .round(true)(root);

  return (root as HierarchyRectangularNode<TreemapDatum>)
    .leaves()
    .map((node) => {
      const item = node.data.item;
      if (!item) return null;
      return {
        item,
        x: node.x0,
        y: node.y0,
        width: Math.max(0, node.x1 - node.x0),
        height: Math.max(0, node.y1 - node.y0),
        normalizedWeight: node.data.value ?? 0,
      };
    })
    .filter((rect): rect is TreemapRect => rect != null);
}

const gridTheme = createAppGridTheme();

function isDetailRow(row: GridRow | undefined): row is DetailGridRow {
  return row?.rowType === "detail";
}

function HoldingsTreemap({ components }: { components: ComponentRow[] }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const clipBaseId = React.useId().replace(/:/g, "");
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [isCollapsed, setIsCollapsed] = useState(false);
  const toggleLabel = isCollapsed ? "펼치기" : "접기";

  useLayoutEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const measure = () => {
      const rect = node.getBoundingClientRect();
      setSize({ width: rect.width, height: rect.height });
    };
    measure();
    const rafId = window.requestAnimationFrame(measure);
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      setSize({ width, height });
    });
    observer.observe(node);
    window.addEventListener("resize", measure);
    return () => {
      window.cancelAnimationFrame(rafId);
      window.removeEventListener("resize", measure);
      observer.disconnect();
    };
  }, []);

  const treemapSize = {
    width: size.width > 0 ? size.width : TREEMAP_FALLBACK_WIDTH,
    height: size.height > 0 ? size.height : TREEMAP_FALLBACK_HEIGHT,
  };
  const rects = useMemo(
    () => buildTreemapRects(components, treemapSize.width, treemapSize.height),
    [components, treemapSize.height, treemapSize.width],
  );

  if (components.length === 0) {
    return (
      <section className="holdingsTreemapSection">
        <div className="holdingsTreemapHeader">
          <h3>트리맵</h3>
        </div>
        <div className="holdingsTreemapEmpty">표시할 구성종목이 없습니다.</div>
      </section>
    );
  }

  return (
    <section className={isCollapsed ? "holdingsTreemapSection holdingsTreemapSectionCollapsed" : "holdingsTreemapSection"}>
      <div className="holdingsTreemapHeader">
        <h3>트리맵</h3>
        <div className="holdingsTreemapHeaderActions">
          <span>상위 20종목</span>
          <button
            type="button"
            className="holdingsTreemapToggle"
            aria-expanded={!isCollapsed}
            onClick={() => setIsCollapsed((prev) => !prev)}
          >
            {isCollapsed ? "▼" : "▲"} {toggleLabel}
          </button>
        </div>
      </div>
      {!isCollapsed && (
        <>
          <div ref={containerRef} className="holdingsTreemapCanvas">
            <svg
              className="holdingsTreemapSvg"
              viewBox={`0 0 ${treemapSize.width} ${treemapSize.height}`}
              preserveAspectRatio="none"
              role="img"
              aria-label="보유 구성종목 트리맵"
            >
              <defs>
                {rects.map((rect, index) => (
                  <clipPath key={`${rect.item.ticker}-${index}`} id={`${clipBaseId}-treemap-${index}`}>
                    <rect x={rect.x + 2} y={rect.y + 2} width={Math.max(0, rect.width - 4)} height={Math.max(0, rect.height - 4)} />
                  </clipPath>
                ))}
              </defs>
              {rects.map((rect, index) => {
                const item = rect.item;
                const showTicker = rect.width >= 44 && rect.height >= 24;
                const showChange = rect.width >= 70 && rect.height >= 44;
                const showWeight = rect.width >= 70 && rect.height >= 62;
                const textColor = getTreemapTextColor(item.change_pct);
                const subduedTextColor = item.change_pct == null || item.change_pct === 0 ? "#64748b" : "rgba(255, 255, 255, 0.72)";
                const title = `${formatDisplayName(item.name)} ${formatSignedPercent(item.change_pct)} ${formatWeight(rect.normalizedWeight)}`;
                const label = getTreemapLabel(item);
                return (
                  <g
                    key={item.ticker}
                    className="holdingsTreemapTileGroup"
                    clipPath={`url(#${clipBaseId}-treemap-${index})`}
                  >
                    <title>{title}</title>
                    <rect
                      x={rect.x + 1}
                      y={rect.y + 1}
                      width={Math.max(0, rect.width - 2)}
                      height={Math.max(0, rect.height - 2)}
                      fill={getTreemapColor(item.change_pct)}
                      stroke="#f8fafc"
                      strokeWidth={2}
                    />
                    {showTicker && (
                      <text
                        x={rect.x + 8}
                        y={rect.y + 20}
                        fill={textColor}
                        className="holdingsTreemapTicker"
                      >
                        {label}
                      </text>
                    )}
                    {showChange && (
                      <text
                        x={rect.x + 8}
                        y={rect.y + 40}
                        fill={textColor}
                        className="holdingsTreemapChange"
                      >
                        {formatSignedPercent(item.change_pct)}
                      </text>
                    )}
                    {showWeight && (
                      <text
                        x={rect.x + 8}
                        y={rect.y + rect.height - 10}
                        fill={subduedTextColor}
                        className="holdingsTreemapWeight"
                      >
                        {formatWeight(rect.normalizedWeight)}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
          <div className="holdingsTreemapGauge" aria-hidden="true">
            <div className="holdingsTreemapGaugeBar" />
            <div className="holdingsTreemapGaugeLabels">
              <span>-5%</span>
              <span>-3%</span>
              <span>-1%</span>
              <span>0%</span>
              <span>+1%</span>
              <span>+3%</span>
              <span>+5%</span>
            </div>
          </div>
        </>
      )}
    </section>
  );
}

// ─── 컴포넌트 ─────────────────────────────────────────────────────────────────
export function HoldingsDetailsPageClient() {
  const [accounts, setAccounts] = useState<AccountOption[]>([]);
  const [selectedAccount, setSelectedAccount] = useState<string>(
    readRememberedMomentumEtfAccountId() || "",
  );
  const [data, setData] = useState<HoldingsComponentsData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);
  const [showAmounts, setShowAmounts] = useState(true);
  const requestSequenceRef = useRef(0);

  // 계좌 목록 로드
  useEffect(() => {
    async function fetchAccounts() {
      try {
        const cached = readSessionTtlCache<AccountOption[]>(
          HOLDINGS_COMPONENT_ACCOUNTS_CACHE_KEY,
          HOLDINGS_COMPONENT_ACCOUNTS_CACHE_TTL_MS,
        );
        if (cached && cached.length > 0) {
          setAccounts(cached);
          if (!selectedAccount) {
            setSelectedAccount("TOTAL");
            writeRememberedMomentumEtfAccountId("TOTAL");
          }
          return;
        }

        const res = await fetch("/api/holdings-components/accounts", { cache: "no-store" });
        if (!res.ok) throw new Error("계좌 목록을 불러오지 못했습니다.");
        const list = (await res.json()) as AccountOption[];
        writeSessionTtlCache(HOLDINGS_COMPONENT_ACCOUNTS_CACHE_KEY, list);
        setAccounts(list);
        if (list.length > 0 && !selectedAccount) {
          setSelectedAccount("TOTAL");
          writeRememberedMomentumEtfAccountId("TOTAL");
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "오류가 발생했습니다.");
      }
    }
    void fetchAccounts();
  }, []);

  // 선택된 계좌 데이터 로드
  const loadData = useCallback(async (accountId: string) => {
    if (!accountId) return;
    const requestSequence = requestSequenceRef.current + 1;
    requestSequenceRef.current = requestSequence;
    const cacheKey = `${HOLDINGS_COMPONENTS_CACHE_KEY_PREFIX}${accountId}`;
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
      const res = await fetch(`/api/holdings-components?account_id=${encodeURIComponent(accountId)}`, {
        cache: "no-store",
      });
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
    if (selectedAccount) void loadData(selectedAccount);
  }, [selectedAccount, loadData]);

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
      width: 100,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: number | null }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        const row = params.data as MainGridRow;
        if (row.ticker === "-") return "-";
        return <span className={getSignedClass(params.value)}>{formatSignedPercent(params.value)}</span>;
      },
    },
    {
      headerName: "수익률",
      field: "return_pct",
      width: 100,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: number | null }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        return <span className={getSignedClass(params.value)}>{formatSignedPercent(params.value)}</span>;
      },
    },
    {
      headerName: "금일 손익",
      field: "daily_profit_krw",
      width: 140,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: number | null }) => {
        if (!params.data || isDetailRow(params.data)) return null;
        return <span className={showAmounts ? getSignedClass(params.value) : ""}>{maskAmount(showAmounts, formatKrw(params.value))}</span>;
      },
    },
    {
      headerName: "누적 손익",
      field: "cumulative_profit_krw",
      width: 140,
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

            <div className="hdColReturn">
              <span className={getSignedClass(src.return_pct)}>
                {formatSignedPercent(src.return_pct)}
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
                    <span className="appLabeledFieldLabel">계좌</span>
                    <select
                      className="form-select"
                      value={selectedAccount}
                      onChange={(e) => {
                        const nextId = e.target.value;
                        setSelectedAccount(nextId);
                        writeRememberedMomentumEtfAccountId(nextId);
                      }}
                      disabled={accounts.length === 0}
                    >
                      {accounts.length === 0 ? (
                        <option value="">계좌 불러오는 중...</option>
                      ) : (
                        <>
                          <option value="TOTAL">전체</option>
                          {accounts.map((acc) => (
                            <option key={acc.account_id} value={acc.account_id}>
                              {acc.name}
                            </option>
                          ))}
                        </>
                      )}
                    </select>
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
            <HoldingsTreemap components={data?.components ?? []} />
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
        .hdColReturn { width: 100px; padding-right: 12px; justify-content: flex-end; }
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
