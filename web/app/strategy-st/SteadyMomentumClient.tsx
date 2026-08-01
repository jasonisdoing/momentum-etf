"use client";

import type { ColDef } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppAgGrid } from "../components/AppAgGrid";
import { AppLoadingProgress, type LoadingProgress } from "../components/AppLoadingProgress";
import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";

const gridTheme = createAppGridTheme();

// 백테스트 기간 선택지 (최대 24개월 — 연간 재무 4개년 + 공시 90일 지연으로 커버되는 범위)
const BACKTEST_MONTH_OPTIONS = [1, 2, 3, 6, 9, 12, 24] as const;

// 선택 가능한 종목풀(1개 선택) — 백엔드 POOL_CONFIGS 와 동일해야 한다.
// divisor: 거래대금 입력 단위 → 통화 원단위 변환 (억원=1e8, $M=1e6)
// 로드 전 폴백 — 실제 표시는 백엔드가 주는 풀 설정의 공식 이름
const POOL_OPTIONS: readonly { id: string; label: string }[] = [
  { id: "kor", label: "코스피 개별주" },
  { id: "kor_kosdaq", label: "코스닥 개별주" },
  { id: "us", label: "나스닥 100 + S&P 100" },
  { id: "us_nasdaq", label: "나스닥 100" },
  { id: "us_snp", label: "S&P100" },
];

type Settings = {
  pool: string;
  lookback_months: number;
  top_n: number;
  slippage_pct: number;
  slope_filter: boolean;
};

type PickRow = {
  rank: number;
  is_reserve: boolean;
  streak_months: number | null;
  ticker: string;
  name: string;
  pool: string;
  return_6m_pct: number;
  rel_return_pct: number;
  win_label: string;
  slope_annual_pct: number;
  r_squared: number;
  momentum_score: number;
};

type PicksResult = {
  as_of: string;
  portfolio_month: string;
  rebalance_date: string;
  signal_date: string;
  universe_count: number;
  candidate_count: number;
  rows: PickRow[];
};

type BacktestMonthRow = {
  month: string;
  strategy_pct: number | null;
  benchmark_pct: number | null;
  reference_pct: number | null;
  excess_pp: number | null;
  holdings_count: number;
  turnover_pct: number | null;
  added: string[];
  removed: string[];
  is_pending?: boolean;
};

type BacktestResult = {
  start_date: string;
  end_date: string;
  months: number;
  strategy_total_pct: number;
  benchmark_total_pct: number;
  strategy_mdd_pct: number | null;
  benchmark_mdd_pct: number | null;
  strategy_sortino: number | null;
  benchmark_sortino: number | null;
  strategy_cagr_pct: number | null;
  benchmark_cagr_pct: number | null;
  reference_cagr_pct: number | null;
  benchmark_name: string;
  benchmark_ticker: string;
  reference_name: string | null;
  reference_total_pct: number | null;
  reference_mdd_pct: number | null;
  reference_sortino: number | null;
  monthly: BacktestMonthRow[];
};

type View = {
  settings: Settings;
  is_saved: boolean;
  pool_labels?: Record<string, string>;
  picks: PicksResult | null;
};

const cardStyle: React.CSSProperties = {
  padding: 14,
  display: "flex",
  flexDirection: "column",
  gap: 10,
};

const inputStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "4px 7px",
  fontSize: "0.86rem",
  width: 90,
  textAlign: "right",
};

const labelStyle: React.CSSProperties = {
  color: "var(--text-muted)",
  fontWeight: 700,
  fontSize: "0.82rem",
};

const thStyle: React.CSSProperties = { padding: "6px 8px", textAlign: "right" };
const tdStyle: React.CSSProperties = { padding: "6px 8px", textAlign: "right" };

function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return value.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatSigned(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

function signColor(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value) || value === 0) return "inherit";
  return value > 0 ? "var(--up-color, #d64545)" : "var(--down-color, #2f6fd0)";
}

export function SteadyMomentumClient() {
  const toast = useToast();
  const [view, setView] = useState<View | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [picking, setPicking] = useState(false);
  const [backtesting, setBacktesting] = useState(false);
  const [backtest, setBacktest] = useState<BacktestResult | null>(null);
  const [pickProgress, setPickProgress] = useState<LoadingProgress | null>(null);
  const [backtestProgress, setBacktestProgress] = useState<LoadingProgress | null>(null);
  const [backtestMonths, setBacktestMonths] = useState<number>(12);

  // 설정 입력 초안 (문자열로 보관해 입력 중 상태를 그대로 둔다)
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [draftPool, setDraftPool] = useState<string>("kor");
  const [draftSlopeFilter, setDraftSlopeFilter] = useState(true);

  const applyView = useCallback((data: View) => {
    setView(data);
    setDraftPool(data.settings.pool);
    setDraftSlopeFilter(data.settings.slope_filter);
    setDraft({
      lookback_months: String(data.settings.lookback_months),
      top_n: String(data.settings.top_n),
      slippage_pct: String(data.settings.slippage_pct),
    });
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await fetch("/api/strategy-st", { cache: "no-store" });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "설정을 불러오지 못했습니다.");
      applyView(payload as View);
    } catch (error) {
      // 백엔드 미기동이어도 화면은 기본값으로 렌더한다 (저장 시 재시도).
      applyView({
        settings: {
          pool: "kor",
          lookback_months: 6,
          top_n: 40,
          slippage_pct: 0.1,
          slope_filter: true,
        },
        is_saved: false,
        picks: null,
      });
      toast.warning(error instanceof Error ? error.message : "설정을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [applyView, toast]);

  useEffect(() => {
    void load();
  }, [load]);

  const saveSettings = useCallback(async () => {
    const lookback = Number(draft.lookback_months);
    const topN = Number(draft.top_n);
    const slippage = Number(draft.slippage_pct);
    if (![lookback, topN, slippage].every((v) => Number.isFinite(v))) {
      toast.error("설정 값이 올바르지 않습니다.");
      return;
    }
    setSaving(true);
    try {
      const resp = await fetch("/api/strategy-st", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          settings: {
            pool: draftPool,
            lookback_months: lookback,
            top_n: topN,
            slippage_pct: slippage,
            slope_filter: draftSlopeFilter,
          },
        }),
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "설정을 저장하지 못했습니다.");
      applyView(payload as View);
      toast.success("설정을 저장했습니다.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "설정을 저장하지 못했습니다.");
    } finally {
      setSaving(false);
    }
  }, [applyView, draft, draftPool, draftSlopeFilter, toast]);

  const runPicks = useCallback(async () => {
    setPicking(true);
    setPickProgress({ percent: 8, message: "후보 선정 중" });
    const timers: number[] = [];
    let interval: number | null = null;
    try {
      timers.push(
        window.setTimeout(() => {
          setPickProgress({ percent: 20, message: "모멘텀 순위 계산 중" });
          interval = window.setInterval(() => {
            setPickProgress((prev) => (prev ? { ...prev, percent: Math.min(90, prev.percent + 3) } : prev));
          }, 4000);
        }, 1200),
      );
      const resp = await fetch("/api/strategy-st/picks", { method: "POST" });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "선정에 실패했습니다.");
      setPickProgress({ percent: 100, message: "선정 결과 반영 중" });
      setView((prev) => (prev ? { ...prev, picks: payload as PicksResult } : prev));
      toast.success("선정을 갱신했습니다.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "선정에 실패했습니다.");
    } finally {
      timers.forEach((timer) => window.clearTimeout(timer));
      if (interval !== null) window.clearInterval(interval);
      setPicking(false);
      setPickProgress(null);
    }
  }, [toast]);

  const runBacktest = useCallback(async () => {
    setBacktesting(true);
    setBacktestProgress({ percent: 8, message: "리밸런싱 시점별 후보 계산 중" });
    const timers: number[] = [];
    let interval: number | null = null;
    try {
      timers.push(
        window.setTimeout(() => {
          setBacktestProgress({ percent: 20, message: "월별 시뮬레이션 중" });
          interval = window.setInterval(() => {
            setBacktestProgress((prev) => (prev ? { ...prev, percent: Math.min(90, prev.percent + 3) } : prev));
          }, 4000);
        }, 1200),
      );
      const resp = await fetch("/api/strategy-st/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ months: backtestMonths }),
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "백테스트에 실패했습니다.");
      setBacktestProgress({ percent: 100, message: "결과 반영 중" });
      setBacktest(payload as BacktestResult);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "백테스트에 실패했습니다.");
    } finally {
      timers.forEach((timer) => window.clearTimeout(timer));
      if (interval !== null) window.clearInterval(interval);
      setBacktesting(false);
      setBacktestProgress(null);
    }
  }, [backtestMonths, toast]);

  const pickColumns = useMemo<ColDef<PickRow>[]>(
    () => [
      { headerName: "순위", field: "rank", width: 72, type: "numericColumn" },
      {
        headerName: "연속",
        field: "streak_months",
        headerTooltip: "이번 포트폴리오까지 몇 달 연속 편입됐는지 (신규 = 이번 달 첫 편입, 최대 12개월 추적)",
        width: 84,
        valueFormatter: (p) =>
          p.value == null ? "-" : p.value <= 1 ? "신규" : p.value >= 12 ? "12개월+" : `${p.value}개월째`,
        cellStyle: (p) => ({
          color: p.value != null && p.value <= 1 && !p.data?.is_reserve ? "var(--up-color, #d64545)" : "inherit",
        }),
      },
      { headerName: "티커", field: "ticker", width: 92 },
      { headerName: "종목명", field: "name", flex: 1, minWidth: 140 },
      {
        headerName: "풀",
        field: "pool",
        width: 64,
        valueFormatter: (p) =>
          p.value === "kor_kosdaq" ? "KQ" : p.value === "us" ? "US" : "KS",
      },
      {
        headerName: `${view?.settings.lookback_months ?? 6}개월(%)`,
        field: "return_6m_pct",
        headerTooltip: "룩백 구간 수익률 — 룩백(개월) 설정을 따른다",
        width: 92,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value, 1),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
      {
        headerName: `상대${view?.settings.lookback_months ?? 6}개월(%)`,
        field: "rel_return_pct",
        headerTooltip: "룩백 구간의 벤치마크 대비 초과 수익률",
        width: 108,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value, 1),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
      {
        headerName: "월승률",
        field: "win_label",
        headerTooltip: "룩백 구간의 월별 상대수익 중 벤치마크를 이긴 달",
        width: 78,
      },
      {
        headerName: "상대기울기(%)",
        field: "slope_annual_pct",
        headerTooltip: "상대 가격선(종가÷벤치마크) 회귀의 연율화 기울기",
        width: 112,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value, 1),
      },
      {
        headerName: "R²",
        field: "r_squared",
        width: 82,
        type: "numericColumn",
        valueFormatter: (p) => (p.value != null ? p.value.toFixed(3) : "-"),
      },
      {
        headerName: "점수",
        field: "momentum_score",
        headerTooltip: "꾸준한 모멘텀 점수 = 연율화 상대기울기 × R²",
        width: 92,
        type: "numericColumn",
        valueFormatter: (p) => formatNumber(p.value, 1),
        cellStyle: () => ({ fontWeight: 700 }),
      },
    ],
    [view?.settings.lookback_months],
  );

  if (loading && !view) {
    return (
      <PageFrame title="Steady Momentum">
        <div style={{ color: "var(--text-muted)", padding: 20 }}>불러오는 중…</div>
      </PageFrame>
    );
  }
  if (!view) {
    return (
      <PageFrame title="Steady Momentum">
        <div style={{ color: "var(--text-muted)", padding: 20 }}>데이터가 없습니다.</div>
      </PageFrame>
    );
  }

  return (
    <PageFrame title="Steady Momentum">
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {/* ① 변수 설정 */}
        <div className="card appCard" style={cardStyle}>
          <div style={{ display: "flex", gap: 12, alignItems: "baseline", flexWrap: "wrap" }}>
            <span style={{ fontWeight: 700, fontSize: "0.95rem" }}>변수 설정</span>
            <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
              시장 대비 꾸준한 상대 모멘텀(상대기울기 × R²)으로 선정 · 우선주 제외
            </span>
            {!view.is_saved ? (
              <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>(미저장 — 기본값 표시 중)</span>
            ) : null}
          </div>
          <div style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap" }}>
            <span style={labelStyle}>종목풀</span>
            {POOL_OPTIONS.map((pool) => (
              <label key={pool.id} style={{ display: "flex", gap: 5, alignItems: "center", fontSize: "0.86rem" }}>
                <input
                  type="radio"
                  name="steadyMomentumPool"
                  checked={draftPool === pool.id}
                  onChange={() => setDraftPool(pool.id)}
                />
                {view.pool_labels?.[pool.id] ?? pool.label}
              </label>
            ))}
            <span style={labelStyle}>룩백(개월)</span>
            <input
              style={inputStyle}
              value={draft.lookback_months ?? ""}
              onChange={(e) => setDraft((d) => ({ ...d, lookback_months: e.target.value }))}
              inputMode="numeric"
            />
            <span style={labelStyle}>종목 수</span>
            <input
              style={inputStyle}
              value={draft.top_n ?? ""}
              onChange={(e) => setDraft((d) => ({ ...d, top_n: e.target.value }))}
              inputMode="numeric"
            />
            <span style={labelStyle}>슬리피지(%)</span>
            <input
              style={inputStyle}
              value={draft.slippage_pct ?? ""}
              onChange={(e) => setDraft((d) => ({ ...d, slippage_pct: e.target.value }))}
              inputMode="decimal"
            />
            <label style={{ display: "flex", gap: 5, alignItems: "center", fontSize: "0.84rem" }}>
              <input
                type="checkbox"
                checked={draftSlopeFilter}
                onChange={(e) => setDraftSlopeFilter(e.target.checked)}
              />
              시장 상대기울기 필터 (시장에 지는 추세 제외)
            </label>
            <button type="button" className="btn btn-sm btn-primary" onClick={() => void saveSettings()} disabled={saving}>
              {saving ? "저장 중…" : "저장"}
            </button>
          </div>

        </div>

        {/* ② 백테스트 */}
        <div className="card appCard" style={cardStyle}>
          <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ fontWeight: 700, fontSize: "0.95rem" }}>백테스트</span>
            <select
              style={{ ...inputStyle, width: 110, textAlign: "left" }}
              value={backtestMonths}
              onChange={(e) => setBacktestMonths(Number(e.target.value))}
            >
              {BACKTEST_MONTH_OPTIONS.map((m) => (
                <option key={m} value={m}>
                  {m}개월
                </option>
              ))}
            </select>
            <button type="button" className="btn btn-sm btn-dark" onClick={() => void runBacktest()} disabled={backtesting}>
              {backtesting ? "실행 중…" : "실행"}
            </button>
            <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
              월간 리밸런싱 · 현재 종목풀 기준(생존 편향 있음)
            </span>
          </div>
          {backtesting ? (
            <AppLoadingProgress title="백테스트 실행 중..." progress={backtestProgress} />
          ) : null}
          {backtest ? (
            <>
              <div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: "0.86rem" }}>
                <span>
                  {backtest.start_date} ~ {backtest.end_date}
                </span>
                <span>
                  전략{" "}
                  <b style={{ color: signColor(backtest.strategy_total_pct) }}>
                    {formatSigned(backtest.strategy_total_pct)}
                  </b>
                  <span style={{ color: "var(--text-muted)" }}>
                    {` (CAGR ${backtest.strategy_cagr_pct != null ? formatSigned(backtest.strategy_cagr_pct, 1) : "-"}`}
                    {backtest.strategy_mdd_pct != null ? ` · MDD ${backtest.strategy_mdd_pct.toFixed(1)}%` : ""}
                    {` · 소르티노 ${backtest.strategy_sortino != null ? backtest.strategy_sortino.toFixed(2) : "-"})`}
                  </span>
                </span>
                <span>
                  {backtest.benchmark_name}({backtest.benchmark_ticker}){" "}
                  <b style={{ color: signColor(backtest.benchmark_total_pct) }}>
                    {formatSigned(backtest.benchmark_total_pct)}
                  </b>
                  <span style={{ color: "var(--text-muted)" }}>
                    {` (CAGR ${backtest.benchmark_cagr_pct != null ? formatSigned(backtest.benchmark_cagr_pct, 1) : "-"}`}
                    {backtest.benchmark_mdd_pct != null ? ` · MDD ${backtest.benchmark_mdd_pct.toFixed(1)}%` : ""}
                    {` · 소르티노 ${backtest.benchmark_sortino != null ? backtest.benchmark_sortino.toFixed(2) : "-"})`}
                  </span>
                </span>
                {backtest.reference_name && backtest.reference_total_pct != null ? (
                  <span>
                    {backtest.reference_name}{" "}
                    <b style={{ color: signColor(backtest.reference_total_pct) }}>
                      {formatSigned(backtest.reference_total_pct)}
                    </b>
                    <span style={{ color: "var(--text-muted)" }}>
                      {` (CAGR ${backtest.reference_cagr_pct != null ? formatSigned(backtest.reference_cagr_pct, 1) : "-"}`}
                      {backtest.reference_mdd_pct != null ? ` · MDD ${backtest.reference_mdd_pct.toFixed(1)}%` : ""}
                      {` · 소르티노 ${backtest.reference_sortino != null ? backtest.reference_sortino.toFixed(2) : "-"})`}
                    </span>
                  </span>
                ) : null}
                <span>
                  초과{" "}
                  <b style={{ color: signColor(backtest.strategy_total_pct - backtest.benchmark_total_pct) }}>
                    {formatSigned(backtest.strategy_total_pct - backtest.benchmark_total_pct)}p
                  </b>
                </span>
              </div>
              <div style={{ overflowX: "auto" }}>
                <table style={{ minWidth: 980, borderCollapse: "collapse", fontSize: "0.84rem" }}>
                  <thead>
                    <tr style={{ color: "var(--text-muted)" }}>
                      <th style={{ ...thStyle, textAlign: "left" }}>월</th>
                      <th style={thStyle}>전략(%)</th>
                      <th style={thStyle}>{backtest.benchmark_ticker}(%)</th>
                      {backtest.reference_name ? <th style={thStyle}>{backtest.reference_name}(%)</th> : null}
                      <th style={thStyle}>초과(%p)</th>
                      <th style={thStyle}>종목 수</th>
                      <th style={thStyle}>교체율(%)</th>
                      <th style={{ ...thStyle, textAlign: "left" }}>편입</th>
                      <th style={{ ...thStyle, textAlign: "left" }}>편출</th>
                    </tr>
                  </thead>
                  <tbody>
                    {backtest.monthly.map((row) => (
                      <tr key={row.month} style={{ borderTop: "1px solid rgba(148,163,184,0.15)" }}>
                        <td style={{ ...tdStyle, textAlign: "left", fontWeight: 700 }}>
                          {row.month}
                          {row.is_pending ? (
                            <span style={{ marginLeft: 5, color: "var(--text-muted)", fontSize: "0.76rem", fontWeight: 400 }}>
                              예정
                            </span>
                          ) : null}
                        </td>
                        <td style={{ ...tdStyle, color: signColor(row.strategy_pct) }}>
                          {formatSigned(row.strategy_pct)}
                        </td>
                        <td style={{ ...tdStyle, color: signColor(row.benchmark_pct) }}>
                          {formatSigned(row.benchmark_pct)}
                        </td>
                        {backtest.reference_name ? (
                          <td style={{ ...tdStyle, color: signColor(row.reference_pct) }}>
                            {formatSigned(row.reference_pct)}
                          </td>
                        ) : null}
                        <td style={{ ...tdStyle, color: signColor(row.excess_pp), fontWeight: 700 }}>
                          {formatSigned(row.excess_pp)}
                        </td>
                        <td style={tdStyle}>{formatNumber(row.holdings_count)}</td>
                        <td style={tdStyle}>{row.turnover_pct == null ? "-" : formatNumber(row.turnover_pct)}</td>
                        <td style={{ ...tdStyle, textAlign: "left", color: "var(--up-color, #d64545)", maxWidth: 260 }}>
                          {row.added.length > 0 ? row.added.join(", ") : "-"}
                        </td>
                        <td style={{ ...tdStyle, textAlign: "left", color: "var(--down-color, #2f6fd0)", maxWidth: 260 }}>
                          {row.removed.length > 0 ? row.removed.join(", ") : "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <span style={{ color: "var(--text-muted)", fontSize: "0.84rem" }}>
              기간을 고르고 실행을 누르면 월별 성과가 표시됩니다.
            </span>
          )}
        </div>

        {/* ③ 현재 선정 종목 */}
        <div className="card appCard" style={cardStyle}>
          <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ fontWeight: 700, fontSize: "0.95rem" }}>현재 선정 종목</span>
            <button type="button" className="btn btn-sm btn-dark" onClick={() => void runPicks()} disabled={picking}>
              {picking ? "선정 중…" : "선정 실행"}
            </button>
            {view.picks ? (
              <span style={{ color: "var(--text-muted)", fontSize: "0.82rem" }}>
                <b style={{ color: "inherit" }}>{view.picks.portfolio_month} 포트폴리오</b> ·{" "}
                {view.picks.rebalance_date} 종가 교체 (판정 {view.picks.signal_date}) · 유니버스{" "}
                {view.picks.universe_count} → 후보 {view.picks.candidate_count} → 선정{" "}
                {view.picks.rows.filter((row) => !row.is_reserve).length}
                {view.picks.rows.some((row) => row.is_reserve)
                  ? ` (+차순위 ${view.picks.rows.filter((row) => row.is_reserve).length})`
                  : ""}{" "}
                · 다음 교체 전까지 결과가 바뀌지 않습니다
              </span>
            ) : (
              <span style={{ color: "var(--text-muted)", fontSize: "0.82rem" }}>
                선정 실행을 누르면 현재 기준 상위 종목이 표시됩니다.
              </span>
            )}
          </div>
          {picking ? (
            <AppLoadingProgress title="선정 실행 중..." progress={pickProgress} />
          ) : null}
          {view.picks ? (
            // autoHeight — 그리드가 행 수만큼만 높이를 차지해 하단 낭비가 없다.
            <div>
              <AppAgGrid<PickRow>
                rowData={view.picks.rows}
                columnDefs={pickColumns}
                loading={picking}
                theme={gridTheme}
                minHeight={0}
                height="auto"
                gridOptions={{ domLayout: "autoHeight" }}
                getRowClass={(p) => (p.data?.is_reserve ? "steadyReserveRow" : "")}
                getRowId={(p) => p.data.ticker}
              />
            </div>
          ) : null}
        </div>
      </div>
    </PageFrame>
  );
}
