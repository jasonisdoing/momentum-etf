"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ColDef, GridOptions, ValueFormatterParams } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";
import { createAppGridTheme } from "../components/app-grid-theme";
import { PageFrame } from "../components/PageFrame";
import { ResponsiveFiltersSection } from "../components/ResponsiveFiltersSection";
import { MarketTrendChart } from "./MarketTrendChart";

type MarketTrendItem = {
  name: string;
  ticker: string;
  price: number | null;
  change_pct: number | null;
  // 원본 추세 % (MA 괴리율 — 화면 미표시)
  trend_pct: number | null;
  // MA를 0점으로 두고 12개월 위/아래 괴리율로 정규화한 점수 (-100 ~ +100, 화면 표시용)
  trend_score: number | null;
  score_range_high: number | null;
  score_range_low: number | null;
  // 52주 전고점 대비 등락률 (현재가 ÷ 52주 최고 − 1) × 100, 0 이하
  pct_from_high: number | null;
  // 현재 레짐(백엔드 slope 기반) + 지속일수
  current_regime: RegimeKey | null;
  current_regime_days: number | null;
};

type MainRow = MarketTrendItem & { rowType: "main"; id: string };
type DetailRow = { rowType: "detail"; id: string; parentTicker: string; parentName: string };
type GridRow = MainRow | DetailRow;

function isDetailRow(row: GridRow | undefined): row is DetailRow {
  return !!row && row.rowType === "detail";
}

type MarketTrendResponse = {
  ma_type: string;
  ma_months: number;
  items: MarketTrendItem[];
  error?: string;
};

const gridTheme = createAppGridTheme();

function formatPrice(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 2 }).format(value);
}

function formatPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const rounded = Math.round(value);
  const sign = rounded > 0 ? "+" : "";
  return `${sign}${rounded}`;
}

function getSignedClass(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) return "";
  return value < 0 ? "metricNegative" : "metricPositive";
}

function renderSignedPercentCell(params: { value: number | null | undefined }) {
  return <span className={getSignedClass(params.value)}>{formatPct(params.value)}</span>;
}

function renderSignedScoreCell(params: { value: number | null | undefined }) {
  return <span className={getSignedClass(params.value)}>{formatScore(params.value)}</span>;
}

type RegimeKey = "accel_up" | "decel_up" | "accel_down" | "decel_down";

// 표시는 3단계(상승/중립/하락)로 통합. 중립은 sub-label 로 조정/진정 구분.
const REGIME_LABEL: Record<RegimeKey, string> = {
  accel_up: "⬆️ 상승",
  decel_up: "➡️ 중립 (조정)",
  decel_down: "➡️ 중립 (진정)",
  accel_down: "⬇️ 하락",
};

// 색상도 3색으로 통합 (중립은 둘 다 녹색).
const REGIME_COLORS: Record<RegimeKey, string> = {
  accel_up: "#d62828",   // 빨강
  decel_up: "#2f9e44",   // 녹색 (중립 - 조정)
  decel_down: "#2f9e44", // 녹색 (중립 - 진정)
  accel_down: "#1971c2", // 파랑
};

// 하단 설명도 3단계로.
const REGIME_DESCRIPTIONS: Array<{ key: RegimeKey; text: string }> = [
  { key: "accel_up", text: "⬆️ 상승: 가격이 MA 위에 있고, 추세 기울기가 상승(강화) 중인 국면입니다." },
  {
    key: "decel_up",
    text:
      "➡️ 중립: (조정) MA 위 + 추세 약화 / (진정) MA 아래 + 추세 회복.",
  },
  { key: "accel_down", text: "⬇️ 하락: 가격이 MA 아래에 있고, 추세 기울기가 하락(약화) 중인 위험 국면입니다." },
];

function renderRegimeCell(params: { data?: GridRow }) {
  const data = params.data;
  if (!data || isDetailRow(data)) return null;
  const key = data.current_regime;
  if (!key) return <span style={{ color: "#adb5bd" }}>-</span>;
  const fontWeight = key === "accel_up" || key === "accel_down" ? 700 : 500;
  return (
    <span style={{ color: REGIME_COLORS[key], fontWeight }}>
      {REGIME_LABEL[key]}
    </span>
  );
}

type MarketTrendClientProps = {
  defaultMaType: string;
  defaultMaMonths: number;
  // config.py 단일 소스 (page.tsx 가 /defaults 응답으로 전달)
  maTypes: string[];
  maMonthsMax: number;
  scoreAnchorPercentile: number;
};

export function MarketTrendClient({
  defaultMaType,
  defaultMaMonths,
  maTypes,
  maMonthsMax,
  scoreAnchorPercentile,
}: MarketTrendClientProps) {
  const [maType, setMaType] = useState<string>(defaultMaType);
  const [maMonths, setMaMonths] = useState<number>(defaultMaMonths);
  const [items, setItems] = useState<MarketTrendItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);
  // 페이지 첫 진입 시 최상단(코스피) 행을 한 번 자동 확장한다 (이후엔 사용자 제어).
  const didInitialExpandRef = useRef(false);

  useEffect(() => {
    let alive = true;
    async function load() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch(
          `/api/market-trend?ma_type=${encodeURIComponent(maType)}&ma_months=${encodeURIComponent(String(maMonths))}`,
          { cache: "no-store" },
        );
        const payload = (await response.json()) as MarketTrendResponse;
        if (!response.ok) {
          throw new Error(payload.error ?? "시장지수 추세 데이터를 불러오지 못했습니다.");
        }
        if (alive) {
          const loaded = payload.items ?? [];
          setItems(loaded);
          // 첫 로드 1회만 최상단 행 자동 확장 (MA 변경 재로드 때는 사용자 상태 유지).
          if (!didInitialExpandRef.current && loaded.length > 0) {
            didInitialExpandRef.current = true;
            setExpandedTicker(loaded[0].ticker);
          }
        }
      } catch (loadError) {
        if (alive)
          setError(
            loadError instanceof Error ? loadError.message : "시장지수 추세 데이터를 불러오지 못했습니다.",
          );
      } finally {
        if (alive) setLoading(false);
      }
    }
    load();
    return () => {
      alive = false;
    };
  }, [maType, maMonths]);

  const rowData = useMemo<GridRow[]>(() => {
    const result: GridRow[] = [];
    for (const item of items) {
      const mainRow: MainRow = { ...item, rowType: "main", id: item.ticker };
      result.push(mainRow);
      if (expandedTicker === item.ticker) {
        result.push({
          rowType: "detail",
          id: `${item.ticker}__detail`,
          parentTicker: item.ticker,
          parentName: item.name,
        });
      }
    }
    return result;
  }, [items, expandedTicker]);

  const columnDefs = useMemo<ColDef<GridRow>[]>(
    () => [
      {
        field: "name",
        headerName: "지수",
        flex: 1,
        minWidth: 95,
        sortable: true,
        cellRenderer: (params: { data?: GridRow; value?: string }) => {
          const data = params.data;
          if (!data || isDetailRow(data)) return "";
          const isExpanded = expandedTicker === data.ticker;
          return (
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
              <span style={{ fontSize: "0.8rem", color: "#868e96" }}>{isExpanded ? "▾" : "▸"}</span>
              <span>{params.value}</span>
            </span>
          );
        },
      },
      {
        field: "price",
        headerName: "현재가",
        flex: 0.9,
        minWidth: 90,
        sortable: true,
        type: "rightAligned",
        valueFormatter: (params: ValueFormatterParams<GridRow>) =>
          formatPrice(params.value as number | null | undefined),
      },
      {
        field: "change_pct",
        headerName: "일간(%)",
        flex: 0.7,
        minWidth: 75,
        sortable: true,
        type: "rightAligned",
        cellRenderer: renderSignedPercentCell,
      },
      {
        headerName: "추세",
        flex: 0.9,
        minWidth: 95,
        sortable: true,
        headerClass: "marketTrendRegimeHeader",
        cellStyle: {
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
        },
        valueGetter: (params) => {
          const data = params.data as GridRow | undefined;
          if (!data || isDetailRow(data)) return null;
          const key = data.current_regime;
          return key ? REGIME_LABEL[key] : null;
        },
        cellRenderer: renderRegimeCell,
      },
      {
        field: "current_regime_days",
        headerName: "기간",
        flex: 0.6,
        minWidth: 75,
        sortable: true,
        cellStyle: {
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
        },
        headerClass: "marketTrendRegimeHeader",
        cellRenderer: (params: { value?: number | null }) => {
          const d = params.value;
          if (d === null || d === undefined) return <span style={{ color: "#adb5bd" }}>-</span>;
          return <span style={{ color: "#1f2937" }}>{d}일째</span>;
        },
      },
      {
        field: "trend_score",
        headerName: "추세 점수",
        flex: 0.7,
        minWidth: 100,
        sortable: true,
        type: "rightAligned",
        cellRenderer: renderSignedScoreCell,
      },
      {
        field: "pct_from_high",
        headerName: "전고점 대비",
        flex: 0.8,
        minWidth: 110,
        sortable: true,
        type: "rightAligned",
        cellRenderer: renderSignedPercentCell,
      },
    ],
    [expandedTicker],
  );

  const detailHeight = 640;
  const gridOptions = useMemo<GridOptions<GridRow>>(
    () => ({
      isFullWidthRow: (params) => isDetailRow(params.rowNode.data ?? undefined),
      fullWidthCellRenderer: (params: { data?: GridRow }) => {
        const data = params.data;
        if (!data || !isDetailRow(data)) return null;
        return (
          <MarketTrendChart
            ticker={data.parentTicker}
            name={data.parentName}
            maType={maType}
            maMonths={maMonths}
          />
        );
      },
      getRowHeight: (params) => {
        if (isDetailRow(params.data ?? undefined)) return detailHeight;
        return undefined;
      },
      onCellClicked: (params) => {
        const data = params.data as GridRow | undefined;
        if (!data || isDetailRow(data)) return;
        if (params.colDef.field !== "name") return;
        const ticker = data.ticker;
        setExpandedTicker((current) => (current === ticker ? null : ticker));
      },
      domLayout: "autoHeight",
    }),
    [maType, maMonths],
  );

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>기준:</span>
          <span className="appHeaderMetricValue">
            MA: {maType} {maMonths}개월
          </span>
        </div>
      </div>
    ),
    [maType, maMonths],
  );

  return (
    <PageFrame title="시장지수 추세" fullWidth titleRight={titleRight}>
      <div className="appPageStack">
        <section className="appSection">
          <div className="card appCard">
            <div className="card-header">
              <ResponsiveFiltersSection>
                <div className="appMainHeader">
                  <div className="appMainHeaderLeft">
                    <label className="appLabeledField">
                      <span className="appLabeledFieldLabel">MA</span>
                      <div className="rankRuleFieldRow">
                        <select
                          className="form-select"
                          value={maType}
                          onChange={(event) => setMaType(event.target.value)}
                          disabled={loading}
                        >
                          {maTypes.map((option) => (
                            <option key={option} value={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                        <select
                          className="form-select"
                          value={String(maMonths)}
                          onChange={(event) => setMaMonths(Number(event.target.value))}
                          disabled={loading}
                        >
                          {Array.from({ length: maMonthsMax }, (_, index) => index + 1).map((month) => (
                            <option key={month} value={month}>
                              {month}개월
                            </option>
                          ))}
                        </select>
                      </div>
                    </label>
                  </div>
                </div>
              </ResponsiveFiltersSection>
            </div>
            <div className="card-body appCardBodyTight">
              {error ? <div className="alert alert-danger mb-2">{error}</div> : null}
              <AppAgGrid<GridRow>
                rowData={rowData}
                columnDefs={columnDefs}
                loading={loading}
                minHeight="auto"
                theme={gridTheme}
                getRowId={(params) => params.data.id}
                gridOptions={gridOptions}
              />
            </div>
          </div>
        </section>
        <section className="appSection">
          <div className="card appCard">
            <div className="card-body" style={{ fontSize: "1rem", lineHeight: 1.7 }}>
              <ul style={{ margin: 0, paddingLeft: "1.2rem" }}>
                {REGIME_DESCRIPTIONS.map(({ key, text }) => (
                  <li key={key} style={{ marginBottom: "2px", color: REGIME_COLORS[key] }}>
                    {text}
                  </li>
                ))}
              </ul>
              <hr style={{ margin: "12px 0", borderColor: "#e9ecef" }} />
              <ul
                style={{
                  margin: 0,
                  paddingLeft: "1.2rem",
                  fontSize: "0.9rem",
                  color: "#5f6b82",
                }}
              >
                <li>현재가: 최신 거래일 종가 (Yahoo Finance · 배당/분할 자동 조정).</li>
                <li>일간(%): (오늘 종가 ÷ 전일 종가 − 1) × 100.</li>
                <li>
                  추세 점수: 먼저 (종가 ÷ MA[{maType} {maMonths}개월] − 1) × 100 으로 원본 추세% 를
                  구한 뒤, MA와 같은 지점을 0점으로 둔 정규화 점수(−100 ~ +100). MA 위쪽은 최근 12개월
                  괴리율의 상위 {100 - scoreAnchorPercentile}%({scoreAnchorPercentile}퍼센타일)를 +100,
                  아래쪽은 하위 {100 - scoreAnchorPercentile}%({100 - scoreAnchorPercentile}퍼센타일)를 −100으로 환산합니다.
                  (단발 극단치 대신 상위/하위 {100 - scoreAnchorPercentile}% 구간을 천장·바닥으로 봅니다.)
                  12개월 내내 MA 위에 있으면 양수, 내내 아래에 있으면 음수입니다. <strong>수익률이 아닙니다.</strong>
                </li>
                <li>
                  레짐: 방향(MA 위/아래) × 가속/감속으로 상승·조정·진정·하락 4단계로 분류합니다.
                  가속/감속은 추세% 의 회귀 기울기로 판정하되 비대칭 창을 씁니다 — 강화는 짧은 창으로
                  빨리(저점 반등 포착), 약화는 긴 창으로 천천히. 기울기가 작은 구간(데드밴드)에서는
                  직전 상태를 유지해 잦은 라벨 변경(휩소)을 막습니다.
                </li>
              </ul>
            </div>
          </div>
        </section>
      </div>

      <style jsx global>{`
        .marketTrendRegimeHeader .ag-header-cell-label {
          justify-content: center;
        }
      `}</style>
    </PageFrame>
  );
}
