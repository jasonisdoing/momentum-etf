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
  // MA를 0점으로 두고 12개월 위/아래 괴리율로 정규화한 점수 (-100 ~ +100, 참조용)
  trend_score: number | null;
  score_range_high: number | null;
  score_range_low: number | null;
  // 전고점 대비 등락률 (현재가 ÷ 최근 12개월 최고 − 1) × 100, 0 이하
  pct_from_high: number | null;
  // 전저점 대비 등락률 (현재가 ÷ 최근 12개월 최저 − 1) × 100, 0 이상
  pct_from_low: number | null;
  // 현재 레짐(SuperTrend 방향) + 지속일수
  current_regime: RegimeKey | null;
  current_regime_days: number | null;
  // 직전 레짐 구간 — 응답에는 계속 실려 오지만 ADR 컬럼으로 대체되어 지금은 표시하지 않는다.
  prev_regime: {
    regime: RegimeKey;
    start_date: string;
    end_date: string;
    days: number;
    start_truncated: boolean;
  } | null;
  days_since_last_up: number | null;
  days_since_last_neutral: number | null;
  // 현재 추세 구간 1일차 종가 대비 등락률. 상승 구간이어도 눌려 있으면 음수다.
  regime_change_pct: number | null;
  // ADR(등락비율) — 구성종목 데이터가 있는 한국 지수만 값이 있다.
  adr: number | null;
  adr_level: AdrLevel | null;
  adr_level_days: number | null;
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

function getSignedClass(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) return "";
  return value < 0 ? "metricNegative" : "metricPositive";
}

function renderSignedPercentCell(params: { value: number | null | undefined }) {
  return <span className={getSignedClass(params.value)}>{formatPct(params.value)}</span>;
}

type RegimeKey = "accel_up" | "accel_down";

const REGIME_LABEL: Record<RegimeKey, string> = {
  accel_up: "⬆️ 상승",
  accel_down: "⬇️ 하락",
};

const REGIME_COLORS: Record<RegimeKey, string> = {
  accel_up: "#d62828",   // 빨강
  accel_down: "#1971c2", // 파랑
};

/** ADR 4단계 — 백엔드 `classify_adr()` 과 같은 구분이다. */
type AdrLevel = "overbought" | "bullish" | "bearish" | "oversold";

const ADR_LEVEL_STYLE: Record<AdrLevel, { label: string; color: string; emoji: string }> = {
  // 과매수/과매도는 되돌림을 경계하는 구간이라 강세·약세와 색을 나눈다.
  overbought: { label: "과매수", color: "#d62828", emoji: "🔥" },
  bullish: { label: "강세", color: "#e8590c", emoji: "⬆️" },
  bearish: { label: "약세", color: "#1971c2", emoji: "⬇️" },
  oversold: { label: "과매도", color: "#0b7285", emoji: "🧊" },
};

const REGIME_DESCRIPTIONS: Array<{ key: RegimeKey; text: string }> = [
  { key: "accel_up", text: "⬆️ 상승: 가격이 SuperTrend 위에 위치한 강세 국면입니다." },
  { key: "accel_down", text: "⬇️ 하락: 가격이 SuperTrend 아래로 내려간 위험 국면입니다." },
];

type MarketTrendClientProps = {
  // config.py 화면 고정값 (page.tsx 가 /defaults 응답으로 전달 — 표시 전용)
  maDays: number;
  scoreAnchorPercentile: number;
  maType: string;
};

// 시장지수 추세 패널 — 페이지(/market-trend)와 홈 허브가 공유하는 본문(그리드 카드).
// compact(홈 허브): 핵심 컬럼(지수/일간/추세/기간)만 남긴 요약 그리드.
const COMPACT_TREND_HEADERS = ["지수", "현재가", "일간(%)", "추세", "ADR"];

export function MarketTrendPanel({
  maDays,
  maType,
  compact = false,
}: {
  maDays: number;
  maType: string;
  compact?: boolean;
}) {
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
              <span style={{ fontSize: "var(--fs-sm)", color: "var(--text-muted)" }}>{isExpanded ? "▾" : "▸"}</span>
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
        field: "current_regime_days",
        headerName: "추세",
        flex: 1.1,
        minWidth: 170,
        sortable: true,
        headerClass: "marketTrendRegimeHeader",
        cellStyle: {
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
        },
        headerTooltip:
          "SuperTrend 방향과 그 방향이 이어진 거래일 수. 괄호는 그 구간 1일차 종가 대비 등락률이다. " +
          "상승 구간이어도 1일차보다 눌려 있으면 음수가 나온다.",
        // ADR 컬럼과 같은 표기 — 색을 입힌 상태 + 지속일수.
        cellRenderer: (params: { data?: GridRow }) => {
          const data = params.data;
          if (!data || isDetailRow(data)) return null;
          const regime = data.current_regime;
          if (!regime) return <span style={{ color: "var(--text-muted)" }}>-</span>;

          // 상승은 현재 구간 일수, 하락은 마지막 상승 이후 경과일을 센다(원래 기간 컬럼과 같은 규칙).
          const days = regime === "accel_up" ? data.current_regime_days : data.days_since_last_up;
          const daysText = days != null ? `${days}일째` : regime === "accel_up" ? "" : "1년 내 상승 없음";
          const movePct = data.regime_change_pct;
          return (
            <span style={{ color: "var(--text-strong)" }} title={`${REGIME_LABEL[regime]} — 구간 1일차 종가 대비`}>
              <strong style={{ color: REGIME_COLORS[regime] }}>{REGIME_LABEL[regime]}</strong>
              {daysText ? ` ${daysText}` : ""}
              {movePct != null ? (
                // 추세 방향과 실제 등락이 어긋날 수 있어(상승 구간인데 눌림) 부호 색을 따로 준다.
                <span className={getSignedClass(movePct)}>{`(${formatPct(movePct)})`}</span>
              ) : null}
            </span>
          );
        },
      },
      {
        field: "adr",
        headerName: "ADR",
        flex: 1.1,
        minWidth: 170,
        sortable: true,
        cellStyle: {
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
        },
        headerClass: "marketTrendRegimeHeader",
        headerTooltip:
          "등락비율(Advance-Decline Ratio) — 20일 누적 상승종목수 ÷ 하락종목수 × 100. " +
          "지수는 대형주에 끌려가지만 ADR 은 얼마나 많은 종목이 함께 오르는지를 본다. " +
          "구성종목 데이터가 있는 코스피·코스닥만 표시된다.",
        cellRenderer: (params: { data?: GridRow }) => {
          const data = params.data;
          if (!data || isDetailRow(data)) return null;
          const level = data.adr_level;
          if (data.adr == null || !level) {
            return <span style={{ color: "var(--text-muted)" }}>-</span>;
          }
          const style = ADR_LEVEL_STYLE[level];
          return (
            <span style={{ color: "var(--text-strong)" }} title={`${style.label} — 20일 누적 ADR ${data.adr.toFixed(1)}`}>
              <strong style={{ color: style.color }}>
                {style.emoji} {style.label}
              </strong>
              {` ${data.adr.toFixed(1)}`}
              {data.adr_level_days ? ` · ${data.adr_level_days}일째` : ""}
            </span>
          );
        },
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
      {
        field: "pct_from_low",
        headerName: "전저점 대비",
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
  // ADR 판이 붙는 지수(코스피·코스닥)는 그 판 높이만큼 행을 늘린다.
  const adrPaneHeight = 200;
  // 어느 지수에 ADR 이 붙는지는 백엔드가 정한다(구성종목 데이터가 있는 시장만).
  // 행 높이는 응답을 기다리지 않고 정해야 해서 화면에도 같은 목록을 둔다.
  const ADR_INDEX_TICKERS = new Set(["^KS11", "^KQ11"]);
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
            maDays={maDays}
          />
        );
      },
      getRowHeight: (params) => {
        const data = params.data ?? undefined;
        if (isDetailRow(data)) {
          return ADR_INDEX_TICKERS.has(data.parentTicker) ? detailHeight + adrPaneHeight : detailHeight;
        }
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
    [maDays, maType],
  );

  return (
    <div className="card appCard">
      <div className="card-body appCardBodyTight">
        {error ? <div className="alert alert-danger mb-2">{error}</div> : null}
        <AppAgGrid<GridRow>
          rowData={rowData}
          columnDefs={compact ? columnDefs.filter((col) => COMPACT_TREND_HEADERS.includes(String(col.headerName ?? ""))) : columnDefs}
          loading={loading}
          minHeight="auto"
          theme={gridTheme}
          getRowId={(params) => params.data.id}
          gridOptions={gridOptions}
        />
      </div>
      <style jsx global>{`
        .marketTrendRegimeHeader .ag-header-cell-label {
          justify-content: center;
        }
        .regimeBtTable {
          width: 100%;
          border-collapse: collapse;
          font-size: var(--fs-base);
          white-space: nowrap;
        }
        .regimeBtTable th,
        .regimeBtTable td {
          padding: 5px 10px;
          text-align: right;
          border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        }
        .regimeBtTable th {
          color: var(--text-muted);
          font-weight: 600;
        }
        .regimeBtTable tr.regimeBtFixed {
          background: rgba(32, 107, 196, 0.06);
          font-weight: 700;
        }
        .regimeBtTable tr.regimeBtBest {
          background: rgba(47, 158, 68, 0.12);
          font-weight: 700;
        }
        .regimeBtSectionTitle {
          font-size: var(--fs-base);
          font-weight: 700;
          margin: 14px 0 6px;
        }
        .regimeBtSectionTitle:first-of-type {
          margin-top: 4px;
        }
        .regimeBtTable tr.regimeBtBh td {
          color: var(--text-muted);
          border-bottom: none;
        }
      `}</style>
    </div>
  );
}

export function MarketTrendClient({
  maDays,
  scoreAnchorPercentile: _scoreAnchorPercentile,
  maType,
}: MarketTrendClientProps) {
  return (
    <PageFrame title="시장지수 추세" fullWidth>
      <div className="appPageStack">
        <section className="appSection">
          <MarketTrendPanel maDays={maDays} maType={maType} />
        </section>
        <section className="appSection">
          <div className="card appCard">
            <div className="card-body" style={{ fontSize: "var(--fs-base)", lineHeight: 1.7 }}>
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
                  fontSize: "var(--fs-base)",
                  color: "#5f6b82",
                }}
              >
                <li>현재가: 최신 거래일 종가 (Yahoo Finance · 배당/분할 자동 조정).</li>
                <li>일간(%): 전일 종가 대비 등락률.</li>
                <li>
                  ADR(등락비율): 최근 20거래일 <strong>상승종목수 ÷ 하락종목수 × 100</strong> 입니다.
                  지수는 시가총액 큰 몇 종목에 끌려가지만 ADR 은 <strong>얼마나 많은 종목이 함께 오르는지</strong>를 봅니다.
                  120 이상 <strong>과매수</strong> · 100 이상 <strong>강세</strong> · 75 초과 <strong>약세</strong> · 75 이하 <strong>과매도</strong>
                  로 나누고, 그 단계가 이어진 거래일 수를 함께 표시합니다.
                  구성종목을 집계하는 <strong>코스피·코스닥</strong>만 값이 있습니다(대상: 시가총액 상위 200 · 150종목).
                </li>
                <li>
                  레짐: <strong>SuperTrend</strong> 기반으로 계산되며 상승과 하락 두 가지 상태만 존재합니다.
                </li>
              </ul>
            </div>
          </div>
        </section>
        <SystemPoolGrid />
      </div>
    </PageFrame>
  );
}
