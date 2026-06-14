"use client";

import { useEffect, useMemo, useState } from "react";
import type { ColDef, GridOptions, ValueFormatterParams } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";
import { createAppGridTheme } from "../components/app-grid-theme";
import { PageFrame } from "../components/PageFrame";
import { ResponsiveFiltersSection } from "../components/ResponsiveFiltersSection";
import { recommendedInvestPct } from "./allocation";
import { MarketTrendChart } from "./MarketTrendChart";

type MarketTrendItem = {
  name: string;
  ticker: string;
  price: number | null;
  change_pct: number | null;
  // 원본 추세 % (MA 괴리율 — 화면 미표시)
  trend_pct: number | null;
  trend_pct_w1: number | null;
  trend_pct_w2: number | null;
  trend_pct_w3: number | null;
  trend_pct_w4: number | null;
  // MA를 0점으로 두고 12개월 위/아래 괴리율로 정규화한 점수 (-100 ~ +100, 화면 표시용)
  trend_score: number | null;
  trend_score_w1: number | null;
  trend_score_w2: number | null;
  trend_score_w3: number | null;
  trend_score_w4: number | null;
  score_range_high: number | null;
  score_range_low: number | null;
  // 현재 레짐 지속일수 + 직전 3개 레짐 기간
  current_regime_days: number | null;
  prev_regime_1: RegimeRange | null;
  prev_regime_2: RegimeRange | null;
  prev_regime_3: RegimeRange | null;
};

type RegimeRange = {
  regime: RegimeKey;
  start_date: string;
  end_date: string;
  days: number;
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

/** 주간 레짐 셀 라벨. 중립은 sub-state 까지 명시 (좁은 셀에 맞춰 공백 제거). */
const REGIME_SHORT_LABEL: Record<RegimeKey, string> = {
  accel_up: "상승",
  decel_up: "중립(조정)",
  decel_down: "중립(진정)",
  accel_down: "하락",
};
/** "4월 8일" 같은 한국어 월/일 표시 (연도 생략). */
function formatKoreanDate(date: string | null | undefined): string {
  if (!date) return "-";
  const parts = date.split("-");
  if (parts.length !== 3) return date;
  const [, m, d] = parts;
  return `${Number(m)}월 ${Number(d)}일`;
}

/** 직전 레짐 기간 셀: "상승: 4월 8일~5월 15일 (28일)" 형태. */
function renderRegimeRangeCell(params: { value?: RegimeRange | null }) {
  const range = params.value ?? null;
  if (!range) return <span style={{ color: "#adb5bd" }}>-</span>;
  const label = REGIME_SHORT_LABEL[range.regime];
  const color = REGIME_COLORS[range.regime];
  return (
    <span style={{ color, fontWeight: 600, whiteSpace: "nowrap", fontSize: "0.85rem" }}>
      <strong>{label}</strong>
      <span style={{ fontWeight: 400, marginLeft: 6 }}>
        {formatKoreanDate(range.start_date)} ~ {formatKoreanDate(range.end_date)}
      </span>
    </span>
  );
}

function renderRegimeWeekCell(params: { value?: RegimeKey | null }) {
  const key = params.value ?? null;
  if (!key) return <span style={{ color: "#adb5bd" }}>-</span>;
  const fontWeight = key === "accel_up" || key === "accel_down" ? 700 : 500;
  // 중립 sub-state 는 글자가 길어 좁은 셀에서 잘리지 않도록 살짝 축소.
  const fontSize = key === "decel_up" || key === "decel_down" ? "0.82rem" : "0.9rem";
  return (
    <span style={{ color: REGIME_COLORS[key], fontWeight, fontSize, whiteSpace: "nowrap" }}>
      {REGIME_SHORT_LABEL[key]}
    </span>
  );
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
  { key: "accel_up", text: "⬆️ 상승: 가격이 MA 위에 있고, 추세가 최근 4주 평균보다 더 강한 국면입니다." },
  {
    key: "decel_up",
    text:
      "➡️ 중립: (조정) MA 위 + 추세 약화 / (진정) MA 아래 + 추세 회복.",
  },
  { key: "accel_down", text: "⬇️ 하락: 가격이 MA 아래에 있고, 추세가 최근 4주 평균보다 더 약해진 위험 국면입니다." },
];

/** 1·2·3·4주 전 추세% 평균. 4개 모두 유효해야 평균 반환. */
function averageWeeklyTrend(item: MarketTrendItem): number | null {
  const values = [item.trend_pct_w1, item.trend_pct_w2, item.trend_pct_w3, item.trend_pct_w4];
  const valid = values.filter((v): v is number => v !== null && v !== undefined && !Number.isNaN(v));
  if (valid.length < 4) return null;
  return (valid[0] + valid[1] + valid[2] + valid[3]) / 4;
}

/** 추세 vs 1·2·3·4주 평균 비교로 4단계 분류. delta = trend - avgPast. */
function classifyRegime(trend: number | null, avgPast: number | null): RegimeKey | null {
  if (trend === null || trend === undefined || Number.isNaN(trend)) return null;
  if (avgPast === null || avgPast === undefined || Number.isNaN(avgPast)) return null;
  const delta = trend - avgPast;
  if (trend >= 0) {
    return delta >= 0 ? "accel_up" : "decel_up";
  }
  return delta > 0 ? "decel_down" : "accel_down";
}

function renderRegimeCell(params: { data?: GridRow }) {
  const data = params.data;
  if (!data || isDetailRow(data)) return null;
  const key = classifyRegime(data.trend_pct, averageWeeklyTrend(data));
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
  allocNeutralInvest: number;
  allocUpSpan: number;
  allocDownSpan: number;
};

export function MarketTrendClient({
  defaultMaType,
  defaultMaMonths,
  maTypes,
  maMonthsMax,
  scoreAnchorPercentile,
  allocNeutralInvest,
  allocUpSpan,
  allocDownSpan,
}: MarketTrendClientProps) {
  const [maType, setMaType] = useState<string>(defaultMaType);
  const [maMonths, setMaMonths] = useState<number>(defaultMaMonths);
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
        const response = await fetch(
          `/api/market-trend?ma_type=${encodeURIComponent(maType)}&ma_months=${encodeURIComponent(String(maMonths))}`,
          { cache: "no-store" },
        );
        const payload = (await response.json()) as MarketTrendResponse;
        if (!response.ok) {
          throw new Error(payload.error ?? "시장지수 추세 데이터를 불러오지 못했습니다.");
        }
        if (alive) setItems(payload.items ?? []);
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
          const key = classifyRegime(data.trend_pct, averageWeeklyTrend(data));
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
        headerName: "권장 투자",
        flex: 0.7,
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
          return recommendedInvestPct(data.trend_score, allocNeutralInvest, allocUpSpan, allocDownSpan);
        },
        cellRenderer: (params: { value?: number | null }) => {
          const invest = params.value;
          if (invest === null || invest === undefined) return <span style={{ color: "#adb5bd" }}>-</span>;
          return <span style={{ color: "#1971c2", fontWeight: 700 }}>{invest.toFixed(0)}%</span>;
        },
      },
      ...([1, 2, 3] as const).map<ColDef<GridRow>>((slot) => ({
        field: `prev_regime_${slot}` as keyof MarketTrendItem,
        headerName: `최근${slot}`,
        flex: 1.5,
        minWidth: 190,
        sortable: false,
        cellDataType: false,
        cellStyle: {
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-start",
          textAlign: "left",
        },
        cellRenderer: renderRegimeRangeCell,
      })),
    ],
    [expandedTicker, allocNeutralInvest, allocUpSpan, allocDownSpan],
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
            allocNeutralInvest={allocNeutralInvest}
            allocUpSpan={allocUpSpan}
            allocDownSpan={allocDownSpan}
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
    [maType, maMonths, allocNeutralInvest, allocUpSpan, allocDownSpan],
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
                <li>1·2·3·4주: 같은 점수 정의를 해당 시점(N주 전 거래일)에 적용한 값.</li>
                <li>
                  권장 투자: 추세점수를 구간 선형으로 매핑한 보조 지표입니다. 점수 0(중립)=투자 {allocNeutralInvest}%,
                  +100={allocNeutralInvest + allocUpSpan}%, −100={allocNeutralInvest - allocDownSpan}% 를 앵커로,
                  점수 ≥ 0 이면 {allocNeutralInvest} + (점수/100)×{allocUpSpan}, &lt; 0 이면
                  {allocNeutralInvest} + (점수/100)×{allocDownSpan} (%). 현금 = 100 − 투자.{" "}
                  <strong>참고용이며 자동 매매가 아닙니다.</strong>
                </li>
                <li>
                  레짐: 추세 부호(MA 위/아래) × 1·2·3·4주 전 추세% 평균 대비 변화 방향(가속/감속) 으로
                  상승·조정·진정·하락 4단계로 분류합니다.
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
