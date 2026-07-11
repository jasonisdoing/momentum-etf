"use client";

import { useEffect, useMemo, useState } from "react";
import type { ColDef, GridOptions, ValueFormatterParams } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";
import { createAppGridTheme } from "../components/app-grid-theme";
import { PageFrame } from "../components/PageFrame";
import { SystemPoolGrid } from "../components/SystemPoolGrid";
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
  // 현재 레짐(종가 vs SMA ± 버퍼) + 지속일수
  current_regime: RegimeKey | null;
  current_regime_days: number | null;
  days_since_last_up: number | null;
  days_since_last_neutral: number | null;
};

type MainRow = MarketTrendItem & { rowType: "main"; id: string };
type DetailRow = { rowType: "detail"; id: string; parentTicker: string; parentName: string };
type GridRow = MainRow | DetailRow;

function isDetailRow(row: GridRow | undefined): row is DetailRow {
  return !!row && row.rowType === "detail";
}

type MarketTrendResponse = {
  ma_days: number;
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

type RegimeKey = "accel_up" | "neutral" | "accel_down";

const REGIME_LABEL: Record<RegimeKey, string> = {
  accel_up: "⬆️ 상승",
  neutral: "➡️ 중립",
  accel_down: "⬇️ 하락",
};

const REGIME_COLORS: Record<RegimeKey, string> = {
  accel_up: "#d62828",   // 빨강
  neutral: "#2f9e44",    // 녹색 (중립)
  accel_down: "#1971c2", // 파랑
};

// 하단 설명도 3단계로.
const REGIME_DESCRIPTIONS: Array<{ key: RegimeKey; text: string }> = [
  { key: "accel_up", text: "⬆️ 상승: 슈퍼트렌드가 상승이고 종가도 MA(+버퍼) 위로 확인된 강세 국면입니다." },
  { key: "neutral", text: "➡️ 중립: 슈퍼트렌드와 가격이 엇갈리거나 가격이 MA 근처라 방향이 아직 확인되지 않은 대기 국면입니다." },
  { key: "accel_down", text: "⬇️ 하락: 슈퍼트렌드가 하락이고 종가도 MA(−버퍼) 아래로 밀린 위험 국면입니다." },
];

function renderRegimeCell(params: { data?: GridRow }) {
  const data = params.data;
  if (!data || isDetailRow(data)) return null;
  const key = data.current_regime;
  if (!key) return <span style={{ color: "var(--text-muted)" }}>-</span>;
  const fontWeight = key === "accel_up" || key === "accel_down" ? 700 : 500;
  return (
    <span style={{ color: REGIME_COLORS[key], fontWeight }}>
      {REGIME_LABEL[key]}
    </span>
  );
}

type MarketTrendClientProps = {
  // config.py 화면 고정값 (page.tsx 가 /defaults 응답으로 전달 — 표시 전용)
  maType: string;
  maDays: number;
  scoreAnchorPercentile: number;
};

export function MarketTrendClient({
  maType,
  maDays,
  scoreAnchorPercentile,
}: MarketTrendClientProps) {
  const [items, setItems] = useState<MarketTrendItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    async function load() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch("/api/market-trend", { cache: "no-store" });
        const payload = (await response.json()) as MarketTrendResponse;
        if (!response.ok) {
          throw new Error(payload.error ?? "시장지수 추세 데이터를 불러오지 못했습니다.");
        }
        if (alive) {
          setItems(payload.items ?? []);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
        flex: 0.8,
        minWidth: 85,
        sortable: true,
        cellRenderer: (params: { data?: GridRow; value?: string }) => {
          const data = params.data;
          if (!data || isDetailRow(data)) return "";
          const isExpanded = expandedTicker === data.ticker;
          return (
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>{isExpanded ? "▾" : "▸"}</span>
              <span>{params.value}</span>
            </span>
          );
        },
      },
      {
        field: "price",
        headerName: "현재가",
        flex: 0.6,
        minWidth: 78,
        sortable: true,
        type: "rightAligned",
        valueFormatter: (params: ValueFormatterParams<GridRow>) =>
          formatPrice(params.value as number | null | undefined),
      },
      {
        field: "change_pct",
        headerName: "일간(%)",
        flex: 0.5,
        minWidth: 66,
        sortable: true,
        type: "rightAligned",
        cellRenderer: renderSignedPercentCell,
      },
      {
        headerName: "추세",
        flex: 0.6,
        minWidth: 80,
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
        headerName: "기간(거래일)",
        flex: 1.4,
        minWidth: 240,
        sortable: true,
        cellStyle: {
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
        },
        headerClass: "marketTrendRegimeHeader",
        cellRenderer: (params: { value?: number | null; data?: MarketTrendItem }) => {
          const d = params.value;
          if (d === null || d === undefined) return <span style={{ color: "var(--text-muted)" }}>-</span>;
          const regime = params.data?.current_regime;
          if (regime === "accel_up") {
            return <span style={{ color: "var(--text-strong)" }}>상승 {d}일째</span>;
          }
          const sinceUp = params.data?.days_since_last_up;
          const upText = sinceUp !== null && sinceUp !== undefined ? `마지막 상승 후 ${sinceUp}일째` : "1년 내 상승 없음";
          if (regime === "accel_down") {
            const sinceNeutral = params.data?.days_since_last_neutral;
            const neutralText =
              sinceNeutral !== null && sinceNeutral !== undefined ? `, 마지막 중립 후 ${sinceNeutral}일째` : "";
            return <span style={{ color: "var(--text-strong)" }}>{upText}{neutralText}</span>;
          }
          return <span style={{ color: "var(--text-strong)" }}>{upText}</span>;
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

  const detailHeight = 768;
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
    [maDays],
  );

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>기준:</span>
          <span className="appHeaderMetricValue">
            {maType} {maDays}일
          </span>
        </div>
      </div>
    ),
    [maType, maDays],
  );

  return (
    <PageFrame title="시장지수 추세" fullWidth titleRight={titleRight}>
      <div className="appPageStack">
        <section className="appSection">
          <div className="card appCard">
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
                <li>일간(%): 전일 종가 대비 등락률.</li>
                <li>
                  추세 점수: 종가의 {maType}({maDays}일) 대비 괴리율을 −100~+100 으로 정규화한 값(0 = MA선).
                  최근 12개월 괴리율의 상위/하위 {100 - scoreAnchorPercentile}% 를 각각 천장(+100)·바닥(−100)으로 봅니다.
                  MA 위면 양수, 아래면 음수 — <strong>수익률이 아니라 MA 대비 위치입니다.</strong>
                </li>
                <li>
                  레짐: <strong>슈퍼트렌드(ST) 방향이 주 신호, {maType}({maDays}일)±버퍼(지수별)가 보조</strong>입니다.
                  ST ▲ 이고 종가가 MA+버퍼 위면 상승, ST ▼ 이고 종가가 MA−버퍼 아래면 하락, 그 외(ST·가격 불일치)는 중립.
                  MA는 방향을 뒤집지 못하고 '아직 확인 안 됨 → 중립'으로만 유보합니다. 버퍼는 지수별로 설정됩니다.
                </li>
              </ul>
            </div>
          </div>
        </section>
        <SystemPoolGrid />
      </div>

      <style jsx global>{`
        .marketTrendRegimeHeader .ag-header-cell-label {
          justify-content: center;
        }
      `}</style>
    </PageFrame>
  );
}
