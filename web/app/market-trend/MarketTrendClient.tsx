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
  // 현재 레짐(MA20/60 교차 + 확인일수) + 지속일수
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

const REGIME_DESCRIPTIONS: Array<{ key: RegimeKey; text: string }> = [
  { key: "accel_up", text: "⬆️ 상승: MA20이 MA60 위에 있고 종가가 MA20 위에서 확인된 강세 국면입니다." },
  { key: "neutral", text: "➡️ 중립: MA20이 MA60 위지만 종가가 MA20~MA60 사이에 있어 방향 확인을 기다리는 구간입니다." },
  { key: "accel_down", text: "⬇️ 하락: MA20이 MA60 아래이거나 종가가 MA60 아래로 밀린 위험 국면입니다." },
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

type RegimeCombo = {
  multiplier?: number;
  k?: number;
  buffer?: number;
  confirm_days?: number;
  up: number;
  neutral: number;
  down: number;
  flips: number;
  dwell: number;
  strat_return: number;
  strat_mdd: number;
  strat_sharpe: number;
  bh_return: number;
  bh_mdd: number;
  bh_sharpe: number;
};
type RegimeBacktestIndex = {
  ticker: string;
  name: string;
  confirm_days: number;
  ma_variants: RegimeCombo[];
  ma_periods: number[];
};
type RegimeBacktestResponse = {
  window_days?: number;
  months?: number;
  top_n?: number;
  indices?: RegimeBacktestIndex[];
  error?: string;
};

const BACKTEST_MONTH_OPTIONS = Array.from({ length: 36 }, (_, i) => i + 1);
const CASH_OPTIONS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100];

/** 백테스트 탭 — 선택 지수·N개월로 MA20/60 확인일수 후보를 비교한다(읽기 전용). */
function RegimeBacktestPanel() {
  const [indexOptions, setIndexOptions] = useState<{ ticker: string; name: string }[]>([]);
  const [selectedTicker, setSelectedTicker] = useState<string>("");
  const [months, setMonths] = useState<number>(12);
  const [upCash, setUpCash] = useState<number>(0);
  const [neutralCash, setNeutralCash] = useState<number>(15);
  const [downCash, setDownCash] = useState<number>(30);
  const [data, setData] = useState<RegimeBacktestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 지수 셀렉트 목록 로드 (/market-trend 지수 단일 소스)
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const resp = await fetch("/api/market-trend/indices", { cache: "no-store" });
        const payload = (await resp.json()) as { indices?: { ticker: string; name: string }[]; error?: string };
        if (!resp.ok || payload.error) throw new Error(payload.error ?? "지수 목록을 불러오지 못했습니다.");
        if (alive) {
          const list = payload.indices ?? [];
          setIndexOptions(list);
          if (list.length > 0) setSelectedTicker(list[0].ticker);
        }
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : "지수 목록을 불러오지 못했습니다.");
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const runBacktest = async () => {
    if (!selectedTicker) {
      setError("지수를 선택해주세요.");
      return;
    }
    try {
      setLoading(true);
      setError(null);
      const resp = await fetch(
        `/api/market-trend/regime-backtest?ticker=${encodeURIComponent(selectedTicker)}&months=${months}` +
          `&up_cash=${upCash}&neutral_cash=${neutralCash}&down_cash=${downCash}`,
        { cache: "no-store" },
      );
      const payload = (await resp.json()) as RegimeBacktestResponse;
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "레짐 백테스트를 불러오지 못했습니다.");
      setData(payload);
    } catch (e) {
      setError(e instanceof Error ? e.message : "레짐 백테스트를 불러오지 못했습니다.");
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const indices = data?.indices ?? [];

  return (
    <div className="appPageStack">
      <div className="card appCard">
        <div className="card-body">
          <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "flex-end" }}>
            <label className="appLabeledField" style={{ minWidth: 180 }}>
              <span className="appLabeledFieldLabel">지수</span>
              <select
                className="form-select form-select-sm"
                value={selectedTicker}
                onChange={(e) => setSelectedTicker(e.target.value)}
              >
                {indexOptions.length === 0 ? <option value="">불러오는 중…</option> : null}
                {indexOptions.map((opt) => (
                  <option key={opt.ticker} value={opt.ticker}>
                    {opt.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="appLabeledField" style={{ minWidth: 130 }}>
              <span className="appLabeledFieldLabel">기간(개월)</span>
              <select
                className="form-select form-select-sm"
                value={months}
                onChange={(e) => setMonths(Number(e.target.value))}
              >
                {BACKTEST_MONTH_OPTIONS.map((m) => (
                  <option key={m} value={m}>
                    최근 {m}개월
                  </option>
                ))}
              </select>
            </label>
            {([
              ["상승 현금 비중(%)", upCash, setUpCash],
              ["중립 현금 비중(%)", neutralCash, setNeutralCash],
              ["하락 현금 비중(%)", downCash, setDownCash],
            ] as const).map(([label, value, setter]) => (
              <label key={label} className="appLabeledField" style={{ minWidth: 130 }}>
                <span className="appLabeledFieldLabel">{label}</span>
                <select
                  className="form-select form-select-sm"
                  value={value}
                  onChange={(e) => setter(Number(e.target.value))}
                >
                  {CASH_OPTIONS.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>
            ))}
            <button type="button" className="btn btn-sm btn-primary" disabled={loading || !selectedTicker} onClick={() => void runBacktest()}>
              {loading ? "백테스트 중…" : "백테스트"}
            </button>
          </div>
          <p style={{ fontSize: "0.88rem", color: "var(--text-muted)", lineHeight: 1.6, margin: "12px 0 0" }}>
            <strong>MA20/60 교차</strong> 레짐에 <strong>확인 필터(0~5거래일)</strong>를 걸어 어느 N이 좋은지 비교합니다. 전략 = 레짐별 현금 비중(위 선택)을 적용해 전일 레짐을 다음날 수익에 반영.{" "}
            <strong>좋은 N을 골라 config에 직접 반영하세요.</strong>
          </p>
        </div>
      </div>
      {error ? <div className="alert alert-danger mb-0">{error}</div> : null}
      {!loading && !data ? (
        <div style={{ color: "var(--text-muted)", padding: 16 }}>지수와 기간을 고르고 백테스트를 눌러주세요.</div>
      ) : null}
      {indices.map((idx) => (
        <div className="card appCard" key={idx.ticker}>
          <div className="card-body">
            <h3 style={{ fontSize: "1rem", fontWeight: 800, marginBottom: 8 }}>
              {idx.name}{" "}
              <span style={{ color: "var(--text-muted)", fontWeight: 500, fontSize: "0.85rem" }}>
                ({idx.ticker}) · 현재 확인일수 {idx.confirm_days}거래일
              </span>
            </h3>
            <div style={{ overflowX: "auto" }}>
              <table className="regimeBtTable">
                <thead>
                  <tr>
                    <th style={{ textAlign: "left" }}>방식</th>
                    <th style={{ textAlign: "left" }}>기준</th>
                    <th>상승</th>
                    <th>중립</th>
                    <th>하락</th>
                    <th>휩소</th>
                    <th>유지일</th>
                    <th>전략수익</th>
                    <th>전략MDD</th>
                    <th>Sharpe</th>
                  </tr>
                </thead>
                <tbody>
                  {idx.ma_variants[0] ? (
                    <tr className="regimeBtBh">
                      <td style={{ textAlign: "left" }}>buy&amp;hold</td>
                      <td colSpan={6} />
                      <td className={getSignedClass(idx.ma_variants[0].bh_return)}>{formatPct(idx.ma_variants[0].bh_return)}</td>
                      <td className="metricNegative">{formatPct(idx.ma_variants[0].bh_mdd)}</td>
                      <td>{idx.ma_variants[0].bh_sharpe.toFixed(2)}</td>
                    </tr>
                  ) : null}
                  {idx.ma_variants.map((v) => {
                    const bestSharpe = Math.max(...idx.ma_variants.map((x) => x.strat_sharpe));
                    const isBest = v.strat_sharpe === bestSharpe;
                    return (
                      <tr key={`ma-${v.confirm_days}`} className={isBest ? "regimeBtBest" : ""}>
                        <td style={{ textAlign: "left" }}>
                          MA{idx.ma_periods?.[0] ?? 20}/{idx.ma_periods?.[1] ?? 60} 교차 + {v.confirm_days}일
                          {isBest ? " ⭐" : ""}
                        </td>
                        <td style={{ textAlign: "left", color: "var(--text-muted)", fontSize: "0.82rem" }}>
                          {v.confirm_days === 0 ? "즉시 전환" : `${v.confirm_days}거래일 연속 확인`}
                        </td>
                        <td>{v.up}</td>
                        <td>{v.neutral}</td>
                        <td>{v.down}</td>
                        <td>{v.flips}</td>
                        <td>{v.dwell.toFixed(1)}</td>
                        <td className={getSignedClass(v.strat_return)}>{formatPct(v.strat_return)}</td>
                        <td className="metricNegative">{formatPct(v.strat_mdd)}</td>
                        <td>{v.strat_sharpe.toFixed(2)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

type MarketTrendClientProps = {
  // config.py 화면 고정값 (page.tsx 가 /defaults 응답으로 전달 — 표시 전용)
  maDays: number;
  scoreAnchorPercentile: number;
};

export function MarketTrendClient({
  maDays,
  scoreAnchorPercentile,
}: MarketTrendClientProps) {
  const [items, setItems] = useState<MarketTrendItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);
  const [tab, setTab] = useState<"trend" | "backtest">("trend");

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
            SMA {maDays}일
          </span>
        </div>
      </div>
    ),
    [maDays],
  );

  return (
    <PageFrame title="시장지수 추세" fullWidth titleRight={titleRight}>
      <div className="appPageStack">
        <div className="appSegmentedToggle" role="tablist" aria-label="시장지수 추세 화면 선택">
          {([
            ["trend", "시장지수 추세"],
            ["backtest", "백테스트"],
          ] as const).map(([key, label]) => (
            <button
              key={key}
              type="button"
              role="tab"
              aria-selected={tab === key}
              className={`appSegmentedToggleButton ${tab === key ? "is-active" : ""}`}
              onClick={() => setTab(key)}
            >
              {label}
            </button>
          ))}
        </div>
        {tab === "backtest" ? (
          <RegimeBacktestPanel />
        ) : (
          <>
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
                  추세 점수: 종가의 SMA({maDays}일) 대비 괴리율을 −100~+100 으로 정규화한 값(0 = MA선).
                  최근 12개월 괴리율의 상위/하위 {100 - scoreAnchorPercentile}% 를 각각 천장(+100)·바닥(−100)으로 봅니다.
                  MA 위면 양수, 아래면 음수 — <strong>수익률이 아니라 MA 대비 위치입니다.</strong>
                </li>
                <li>
                  레짐: <strong>MA20/60 교차 + 지수별 확인일수</strong>입니다.
                  MA20 &lt; MA60이면 하락, MA20 ≥ MA60 이면서 종가가 MA20 위면 상승, 종가가 MA60 아래면 하락,
                  그 사이는 중립입니다. 새 레짐은 지수별 확인일수만큼 연속 확인된 뒤 확정됩니다.
                </li>
              </ul>
            </div>
          </div>
        </section>
        <SystemPoolGrid />
          </>
        )}
      </div>

      <style jsx global>{`
        .marketTrendRegimeHeader .ag-header-cell-label {
          justify-content: center;
        }
        .regimeBtTable {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.9rem;
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
          font-size: 0.92rem;
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
    </PageFrame>
  );
}
