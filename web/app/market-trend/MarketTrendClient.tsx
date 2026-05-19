"use client";

import { useEffect, useMemo, useState } from "react";
import type { ColDef, ValueFormatterParams } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";
import { createAppGridTheme } from "../components/app-grid-theme";
import { PageFrame } from "../components/PageFrame";
import { ResponsiveFiltersSection } from "../components/ResponsiveFiltersSection";

const MA_TYPE_OPTIONS = ["SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA", "ALMA"] as const;
const MA_MONTHS_MAX = 12;
const DEFAULT_MA_TYPE: (typeof MA_TYPE_OPTIONS)[number] = "ALMA";
const DEFAULT_MA_MONTHS = 4;

type CompareKey = "w1" | "m1" | "m3";
const COMPARE_OPTIONS: Array<{ value: CompareKey; label: string }> = [
  { value: "w1", label: "1주일전" },
  { value: "m1", label: "1달전" },
  { value: "m3", label: "3달전" },
];
const DEFAULT_COMPARE: CompareKey = "w1";

function getCompareValue(item: MarketTrendItem, key: CompareKey): number | null {
  switch (key) {
    case "w1":
      return item.trend_pct_w1;
    case "m1":
      return item.trend_pct_m1;
    case "m3":
      return item.trend_pct_m3;
    default:
      return null;
  }
}

type MarketTrendItem = {
  name: string;
  ticker: string;
  price: number | null;
  change_pct: number | null;
  trend_pct: number | null;
  trend_pct_w1: number | null;
  trend_pct_m1: number | null;
  trend_pct_m3: number | null;
};

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

function getSignedClass(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) return "";
  return value < 0 ? "metricNegative" : "metricPositive";
}

function renderSignedPercentCell(params: { value: number | null | undefined }) {
  return <span className={getSignedClass(params.value)}>{formatPct(params.value)}</span>;
}

type RegimeKey = "strong_up" | "weak_up" | "sideways" | "strong_down" | "weak_down";

const REGIME_LABEL: Record<RegimeKey, string> = {
  strong_up: "⬆️ 강한 상승",
  weak_up: "↗️ 상승 둔화",
  sideways: "➡️ 횡보",
  strong_down: "⬇️ 하락 심화",
  weak_down: "↘️ 하락 둔화",
};

// 레짐 별 텍스트 색상 (빨주노초파) — 셀과 하단 설명에서 동일하게 사용한다.
const REGIME_COLORS: Record<RegimeKey, string> = {
  strong_up: "#e03131",   // 빨강
  weak_up: "#f76707",     // 주황
  sideways: "#f59f00",    // 노랑(amber, 백색 배경 가독성 확보)
  strong_down: "#2f9e44", // 초록
  weak_down: "#1971c2",   // 파랑
};

const REGIME_DESCRIPTIONS: Array<{ key: RegimeKey; text: string }> = [
  { key: "strong_up", text: "⬆️ 강한 상승: 시장이 가속도를 붙여 올라가는 중입니다." },
  { key: "weak_up", text: "↗️ 상승 둔화: 여전히 상승세이나 고점 신호 또는 숨고르기 국면입니다." },
  { key: "sideways", text: "➡️ 횡보: 방향성 없이 에너지를 응축하거나 단기 소외된 구간입니다." },
  { key: "strong_down", text: "⬇️ 하락 심화: 낙폭이 커지며 투매가 나오거나 하락 탄력이 붙는 중입니다." },
  { key: "weak_down", text: "↘️ 하락 둔화: 하락세이나 바닥을 다지며 진정되는 중 (반등 가능성 타진)입니다." },
];

function classifyRegime(trend: number | null, compareValue: number | null): RegimeKey | null {
  if (trend === null || trend === undefined || Number.isNaN(trend)) return null;
  // 1순위: 횡보 (절대값 1% 이하)
  if (Math.abs(trend) <= 1.0) return "sideways";
  // 2/3순위 비교는 과거 시점 값 필요
  if (compareValue === null || compareValue === undefined || Number.isNaN(compareValue)) return null;
  if (trend > 0) {
    return trend > compareValue ? "strong_up" : "weak_up";
  }
  // trend < 0
  return trend < compareValue ? "strong_down" : "weak_down";
}

function makeRenderRegimeCell(compareKey: CompareKey) {
  return function RegimeCell(params: { data?: MarketTrendItem }) {
    const data = params.data;
    if (!data) return null;
    const key = classifyRegime(data.trend_pct, getCompareValue(data, compareKey));
    if (!key) return <span style={{ color: "#adb5bd" }}>-</span>;
    const fontWeight = key === "strong_up" || key === "strong_down" ? 700 : 500;
    return (
      <span style={{ color: REGIME_COLORS[key], fontWeight }}>
        {REGIME_LABEL[key]}
      </span>
    );
  };
}

export function MarketTrendClient() {
  const [maType, setMaType] = useState<string>(DEFAULT_MA_TYPE);
  const [maMonths, setMaMonths] = useState<number>(DEFAULT_MA_MONTHS);
  const [compareKey, setCompareKey] = useState<CompareKey>(DEFAULT_COMPARE);
  const [items, setItems] = useState<MarketTrendItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const columnDefs = useMemo<ColDef<MarketTrendItem>[]>(
    () => [
      {
        field: "name",
        headerName: "지수",
        flex: 1.2,
        minWidth: 140,
        sortable: true,
      },
      {
        field: "price",
        headerName: "현재가",
        flex: 1,
        minWidth: 120,
        sortable: true,
        type: "rightAligned",
        valueFormatter: (params: ValueFormatterParams<MarketTrendItem>) =>
          formatPrice(params.value as number | null | undefined),
      },
      {
        field: "change_pct",
        headerName: "일간(%)",
        flex: 1,
        minWidth: 100,
        sortable: true,
        type: "rightAligned",
        cellRenderer: renderSignedPercentCell,
      },
      {
        headerName: "레짐",
        flex: 1,
        minWidth: 110,
        sortable: true,
        headerClass: "marketTrendRegimeHeader",
        cellStyle: {
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
        },
        valueGetter: (params) => {
          const data = params.data as MarketTrendItem | undefined;
          if (!data) return null;
          const key = classifyRegime(data.trend_pct, getCompareValue(data, compareKey));
          return key ? REGIME_LABEL[key] : null;
        },
        cellRenderer: makeRenderRegimeCell(compareKey),
      },
      {
        field: "trend_pct",
        headerName: "추세",
        flex: 1,
        minWidth: 100,
        sortable: true,
        type: "rightAligned",
        cellRenderer: renderSignedPercentCell,
      },
      {
        field: "trend_pct_w1",
        headerName: "1주일전",
        flex: 1,
        minWidth: 100,
        sortable: true,
        type: "rightAligned",
        cellRenderer: renderSignedPercentCell,
      },
      {
        field: "trend_pct_m1",
        headerName: "1달전",
        flex: 1,
        minWidth: 100,
        sortable: true,
        type: "rightAligned",
        cellRenderer: renderSignedPercentCell,
      },
      {
        field: "trend_pct_m3",
        headerName: "3달전",
        flex: 1,
        minWidth: 100,
        sortable: true,
        type: "rightAligned",
        cellRenderer: renderSignedPercentCell,
      },
    ],
    [compareKey],
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
                          {MA_TYPE_OPTIONS.map((option) => (
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
                          {Array.from({ length: MA_MONTHS_MAX }, (_, index) => index + 1).map((month) => (
                            <option key={month} value={month}>
                              {month}개월
                            </option>
                          ))}
                        </select>
                      </div>
                    </label>
                    <label className="appLabeledField">
                      <span className="appLabeledFieldLabel">비교</span>
                      <select
                        className="form-select"
                        value={compareKey}
                        onChange={(event) => setCompareKey(event.target.value as CompareKey)}
                      >
                        {COMPARE_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                </div>
              </ResponsiveFiltersSection>
            </div>
            <div className="card-body appCardBodyTight">
              {error ? <div className="alert alert-danger mb-2">{error}</div> : null}
              <AppAgGrid<MarketTrendItem>
                rowData={items}
                columnDefs={columnDefs}
                loading={loading}
                minHeight="auto"
                theme={gridTheme}
                getRowId={(params) => params.data.ticker}
                gridOptions={{ domLayout: "autoHeight" }}
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
