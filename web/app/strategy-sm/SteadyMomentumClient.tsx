"use client";

import type { ColDef } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AppAgGrid } from "../components/AppAgGrid";
import { AppLoadingProgress, type LoadingProgress } from "../components/AppLoadingProgress";
import { PageFrame } from "../components/PageFrame";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";

const gridTheme = createAppGridTheme();

// 백테스트 기간 선택지 — 상한 24개월은 백엔드 MAX_BACKTEST_MONTHS 와 같아야 한다.
const BACKTEST_MONTH_OPTIONS = [1, 2, 3, 6, 9, 12, 24] as const;

// 종목풀 폴백 라벨 — 백엔드 미기동으로 pool_labels 를 못 받았을 때만 쓴다.
// 실제 표시 이름은 종목풀 설정 DB 가 단일 소스이며 백엔드 응답을 우선한다.
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

const DEFAULT_SETTINGS: Settings = {
  pool: "kor",
  lookback_months: 6,
  top_n: 40,
  slippage_pct: 0.1,
  slope_filter: true,
};

type PickRow = {
  rank: number;
  is_reserve: boolean;
  streak_months: number | null;
  ticker: string;
  name: string;
  return_lookback_pct: number;
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

const hintStyle: React.CSSProperties = { color: "var(--text-muted)", fontSize: "0.8rem" };
const numberInputStyle: React.CSSProperties = { width: 88, textAlign: "right" };

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

/** 성과 요약 한 덩어리 — 전략·벤치마크·참고 지수를 같은 형식으로 보여준다. */
function PerformanceSummary({
  label,
  totalPct,
  cagrPct,
  mddPct,
  sortino,
}: {
  label: string;
  totalPct: number | null;
  cagrPct: number | null;
  mddPct: number | null;
  sortino: number | null;
}) {
  return (
    <span>
      {label} <b style={{ color: signColor(totalPct) }}>{formatSigned(totalPct)}</b>
      <span style={hintStyle}>
        {` (CAGR ${cagrPct != null ? formatSigned(cagrPct, 1) : "-"}`}
        {mddPct != null ? ` · MDD ${mddPct.toFixed(1)}%` : ""}
        {` · 소르티노 ${sortino != null ? sortino.toFixed(2) : "-"})`}
      </span>
    </span>
  );
}

/**
 * 진행률을 일정 간격으로 90%까지 올린다. 서버가 단일 요청으로 응답해 실제 단계를
 * 알 수 없으므로, 실제 소요 시간(수 초)에 맞춘 완만한 램프만 보여준다.
 */
function startProgressRamp(
  setProgress: (updater: (prev: LoadingProgress | null) => LoadingProgress | null) => void,
): () => void {
  const timer = window.setInterval(() => {
    setProgress((prev) => (prev ? { ...prev, percent: Math.min(90, prev.percent + 6) } : prev));
  }, 400);
  return () => window.clearInterval(timer);
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
  const autoPickedRef = useRef(false);

  // 설정 입력 초안 (문자열로 보관해 입력 중 상태를 그대로 둔다)
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [draftPool, setDraftPool] = useState<string>(DEFAULT_SETTINGS.pool);
  const [draftSlopeFilter, setDraftSlopeFilter] = useState(DEFAULT_SETTINGS.slope_filter);

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

  const load = useCallback(async (): Promise<boolean> => {
    setLoading(true);
    try {
      const resp = await fetch("/api/strategy-sm", { cache: "no-store" });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "설정을 불러오지 못했습니다.");
      applyView(payload as View);
      return true;
    } catch (error) {
      // 백엔드 미기동이어도 화면은 기본값으로 렌더한다 (저장 시 재시도).
      applyView({ settings: DEFAULT_SETTINGS, is_saved: false, picks: null });
      toast.warning(error instanceof Error ? error.message : "설정을 불러오지 못했습니다.");
      return false;
    } finally {
      setLoading(false);
    }
  }, [applyView, toast]);

  const runPicks = useCallback(async () => {
    setPicking(true);
    setPickProgress({ percent: 10, message: "월 확정 포트폴리오 계산 중" });
    const stopRamp = startProgressRamp(setPickProgress);
    try {
      const resp = await fetch("/api/strategy-sm/picks", { method: "POST" });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "선정에 실패했습니다.");
      setPickProgress({ percent: 100, message: "선정 결과 반영 중" });
      setView((prev) => (prev ? { ...prev, picks: payload as PicksResult } : prev));
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "선정에 실패했습니다.");
    } finally {
      stopRamp();
      setPicking(false);
      setPickProgress(null);
    }
  }, [toast]);

  // 진입 시 저장된 설정으로 선정을 한 번 자동 실행한다 (가격 캐시 기반이라 수 초).
  useEffect(() => {
    void (async () => {
      const ok = await load();
      if (ok && !autoPickedRef.current) {
        autoPickedRef.current = true;
        await runPicks();
      }
    })();
  }, [load, runPicks]);

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
      const resp = await fetch("/api/strategy-sm", {
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
      // 설정이 바뀌면 이전 결과는 이 설정의 결과가 아니다. 선정과 백테스트를 함께 비운다.
      setBacktest(null);
      toast.success("설정을 저장했습니다. 선정과 백테스트를 다시 실행하세요.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "설정을 저장하지 못했습니다.");
    } finally {
      setSaving(false);
    }
  }, [applyView, draft, draftPool, draftSlopeFilter, toast]);

  const runBacktest = useCallback(async () => {
    setBacktesting(true);
    setBacktestProgress({ percent: 10, message: "월별 리밸런싱 시뮬레이션 중" });
    const stopRamp = startProgressRamp(setBacktestProgress);
    try {
      const resp = await fetch("/api/strategy-sm/backtest", {
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
      stopRamp();
      setBacktesting(false);
      setBacktestProgress(null);
    }
  }, [backtestMonths, toast]);

  // 저장하지 않은 입력이 있으면 실행 결과가 화면 값과 어긋난다 — 저장을 먼저 요구한다.
  const isDirty = useMemo(() => {
    if (!view) return false;
    const saved = view.settings;
    return (
      draftPool !== saved.pool ||
      draftSlopeFilter !== saved.slope_filter ||
      draft.lookback_months !== String(saved.lookback_months) ||
      draft.top_n !== String(saved.top_n) ||
      draft.slippage_pct !== String(saved.slippage_pct)
    );
  }, [draft, draftPool, draftSlopeFilter, view]);

  const lookbackMonths = view?.settings.lookback_months ?? DEFAULT_SETTINGS.lookback_months;

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
      {
        headerName: "티커",
        field: "ticker",
        width: 112,
        cellRenderer: (p: { value: string | null | undefined }) => <TickerDetailLink ticker={p.value} />,
      },
      { headerName: "종목명", field: "name", flex: 1, minWidth: 140 },
      {
        headerName: `${lookbackMonths}개월(%)`,
        field: "return_lookback_pct",
        headerTooltip: "룩백 구간 수익률 — 룩백(개월) 설정을 따른다",
        width: 92,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value, 1),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
      {
        headerName: `상대${lookbackMonths}개월(%)`,
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
        headerTooltip: "상대 가격선이 회귀선에 얼마나 잘 맞는지 (1에 가까울수록 매끄러운 추세)",
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
    [lookbackMonths],
  );

  const backtestColumns = useMemo<ColDef<BacktestMonthRow>[]>(() => {
    if (!backtest) return [];
    const columns: ColDef<BacktestMonthRow>[] = [
      {
        headerName: "월",
        field: "month",
        width: 116,
        valueFormatter: (p) => (p.data?.is_pending ? `${p.value} (예정)` : String(p.value ?? "")),
        cellStyle: () => ({ fontWeight: 700 }),
      },
      {
        headerName: "전략(%)",
        field: "strategy_pct",
        width: 92,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
      {
        headerName: `${backtest.benchmark_ticker}(%)`,
        headerTooltip: `벤치마크 ${backtest.benchmark_name}`,
        field: "benchmark_pct",
        width: 96,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
    ];
    if (backtest.reference_name) {
      columns.push({
        headerName: `${backtest.reference_name}(%)`,
        headerTooltip: "참고 지수 — 유사 컨셉 ETF (벤치마크가 아니며 선정에 관여하지 않는다)",
        field: "reference_pct",
        width: 104,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      });
    }
    columns.push(
      {
        headerName: "초과(%p)",
        field: "excess_pp",
        headerTooltip: "전략 − 벤치마크",
        width: 98,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value),
        cellStyle: (p) => ({ color: signColor(p.value), fontWeight: 700 }),
      },
      { headerName: "종목 수", field: "holdings_count", width: 88, type: "numericColumn" },
      {
        headerName: "교체율(%)",
        field: "turnover_pct",
        headerTooltip: "직전 달 대비 교체된 종목 비중",
        width: 100,
        type: "numericColumn",
        valueFormatter: (p) => (p.value == null ? "-" : formatNumber(p.value)),
      },
      {
        headerName: "편입",
        field: "added",
        flex: 1,
        minWidth: 160,
        valueFormatter: (p) => (p.value?.length ? p.value.join(", ") : "-"),
        cellStyle: () => ({ color: "var(--up-color, #d64545)" }),
      },
      {
        headerName: "편출",
        field: "removed",
        flex: 1,
        minWidth: 160,
        valueFormatter: (p) => (p.value?.length ? p.value.join(", ") : "-"),
        cellStyle: () => ({ color: "var(--down-color, #2f6fd0)" }),
      },
    );
    return columns;
  }, [backtest]);

  if (loading && !view) {
    return (
      <PageFrame title="Steady Momentum">
        <div style={{ ...hintStyle, padding: 20 }}>불러오는 중…</div>
      </PageFrame>
    );
  }
  if (!view) {
    return (
      <PageFrame title="Steady Momentum">
        <div style={{ ...hintStyle, padding: 20 }}>데이터가 없습니다.</div>
      </PageFrame>
    );
  }

  const selectedCount = view.picks?.rows.filter((row) => !row.is_reserve).length ?? 0;
  const reserveCount = view.picks?.rows.filter((row) => row.is_reserve).length ?? 0;

  return (
    <PageFrame title="Steady Momentum">
      <div className="appPageStack">
        {/* ① 변수 설정 */}
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">종목풀</span>
                  <select
                    className="form-select form-select-sm"
                    value={draftPool}
                    onChange={(e) => setDraftPool(e.target.value)}
                  >
                    {POOL_OPTIONS.map((pool) => (
                      <option key={pool.id} value={pool.id}>
                        {view.pool_labels?.[pool.id] ?? pool.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">룩백(개월)</span>
                  <input
                    className="form-control form-control-sm"
                    style={numberInputStyle}
                    value={draft.lookback_months ?? ""}
                    onChange={(e) => setDraft((d) => ({ ...d, lookback_months: e.target.value }))}
                    inputMode="numeric"
                  />
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">종목 수</span>
                  <input
                    className="form-control form-control-sm"
                    style={numberInputStyle}
                    value={draft.top_n ?? ""}
                    onChange={(e) => setDraft((d) => ({ ...d, top_n: e.target.value }))}
                    inputMode="numeric"
                  />
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">슬리피지(%)</span>
                  <input
                    className="form-control form-control-sm"
                    style={numberInputStyle}
                    value={draft.slippage_pct ?? ""}
                    onChange={(e) => setDraft((d) => ({ ...d, slippage_pct: e.target.value }))}
                    inputMode="decimal"
                  />
                </label>
                <label style={{ display: "flex", gap: 5, alignItems: "center", fontSize: "0.84rem" }}>
                  <input
                    type="checkbox"
                    checked={draftSlopeFilter}
                    onChange={(e) => setDraftSlopeFilter(e.target.checked)}
                  />
                  시장 상대기울기 필터
                </label>
              </div>
              <div className="appMainHeaderRight">
                {isDirty ? (
                  <span style={{ ...hintStyle, color: "var(--up-color, #d64545)", fontWeight: 700 }}>
                    저장하지 않은 변경
                  </span>
                ) : null}
                {!view.is_saved ? <span style={hintStyle}>(미저장 — 기본값 표시 중)</span> : null}
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  onClick={() => void saveSettings()}
                  // 아직 저장된 적이 없으면 기본값 그대로라도 저장할 수 있어야 한다.
                  disabled={saving || (!isDirty && view.is_saved)}
                >
                  {saving ? "저장 중…" : "저장"}
                </button>
              </div>
            </div>
            <div style={hintStyle}>
              시장 대비 꾸준한 상대 모멘텀(연율화 상대기울기 × R²)으로 선정 · 우선주와 고정 종목 제외 · 시장 상대기울기
              필터를 켜면 시장에 지는 추세를 후보에서 뺍니다
            </div>
          </div>
        </div>

        {/* ② 현재 선정 종목 */}
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <span style={{ fontWeight: 700, fontSize: "0.95rem" }}>현재 선정 종목</span>
                {view.picks ? (
                  <span style={{ ...hintStyle, fontSize: "0.82rem" }}>
                    <b style={{ color: "inherit" }}>{view.picks.portfolio_month} 포트폴리오</b> ·{" "}
                    {view.picks.rebalance_date} 종가 교체 (판정 {view.picks.signal_date}) · 유니버스{" "}
                    {view.picks.universe_count} → 후보 {view.picks.candidate_count} → 선정 {selectedCount}
                    {reserveCount > 0 ? ` (+차순위 ${reserveCount})` : ""} · 다음 교체 전까지 결과가 바뀌지 않습니다
                  </span>
                ) : (
                  <span style={{ ...hintStyle, fontSize: "0.82rem" }}>
                    선정 실행을 누르면 이번 달 확정 포트폴리오가 표시됩니다.
                  </span>
                )}
              </div>
              <div className="appMainHeaderRight">
                {isDirty ? <span style={hintStyle}>설정을 저장해야 실행할 수 있습니다</span> : null}
                <button
                  type="button"
                  className="btn btn-sm btn-dark"
                  onClick={() => void runPicks()}
                  disabled={picking || isDirty}
                >
                  {picking ? "선정 중…" : "선정 실행"}
                </button>
              </div>
            </div>
            {picking ? <AppLoadingProgress title="선정 실행 중..." progress={pickProgress} /> : null}
            {view.picks && !picking ? (
              // autoHeight — 그리드가 행 수만큼만 높이를 차지해 하단 낭비가 없다.
              <AppAgGrid<PickRow>
                rowData={view.picks.rows}
                columnDefs={pickColumns}
                theme={gridTheme}
                minHeight={0}
                height="auto"
                gridOptions={{ domLayout: "autoHeight" }}
                getRowClass={(p) => (p.data?.is_reserve ? "steadyReserveRow" : "")}
                getRowId={(p) => p.data.ticker}
              />
            ) : null}
          </div>
        </div>

        {/* ③ 백테스트 */}
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <span style={{ fontWeight: 700, fontSize: "0.95rem" }}>백테스트</span>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">기간</span>
                  <select
                    className="form-select form-select-sm"
                    style={{ width: 104 }}
                    value={backtestMonths}
                    onChange={(e) => setBacktestMonths(Number(e.target.value))}
                  >
                    {BACKTEST_MONTH_OPTIONS.map((m) => (
                      <option key={m} value={m}>
                        {m}개월
                      </option>
                    ))}
                  </select>
                </label>
                <span style={hintStyle}>월간 리밸런싱 · 현재 종목풀 기준(생존 편향 있음)</span>
              </div>
              <div className="appMainHeaderRight">
                {isDirty ? <span style={hintStyle}>설정을 저장해야 실행할 수 있습니다</span> : null}
                <button
                  type="button"
                  className="btn btn-sm btn-dark"
                  onClick={() => void runBacktest()}
                  disabled={backtesting || isDirty}
                >
                  {backtesting ? "실행 중…" : "실행"}
                </button>
              </div>
            </div>
            {backtesting ? <AppLoadingProgress title="백테스트 실행 중..." progress={backtestProgress} /> : null}
            {backtest && !backtesting ? (
              <>
                <div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: "0.86rem", padding: "2px 0 8px" }}>
                  <span>
                    {backtest.start_date} ~ {backtest.end_date}
                  </span>
                  <PerformanceSummary
                    label="전략"
                    totalPct={backtest.strategy_total_pct}
                    cagrPct={backtest.strategy_cagr_pct}
                    mddPct={backtest.strategy_mdd_pct}
                    sortino={backtest.strategy_sortino}
                  />
                  <PerformanceSummary
                    label={`${backtest.benchmark_name}(${backtest.benchmark_ticker})`}
                    totalPct={backtest.benchmark_total_pct}
                    cagrPct={backtest.benchmark_cagr_pct}
                    mddPct={backtest.benchmark_mdd_pct}
                    sortino={backtest.benchmark_sortino}
                  />
                  {backtest.reference_name && backtest.reference_total_pct != null ? (
                    <PerformanceSummary
                      label={`${backtest.reference_name} (참고)`}
                      totalPct={backtest.reference_total_pct}
                      cagrPct={backtest.reference_cagr_pct}
                      mddPct={backtest.reference_mdd_pct}
                      sortino={backtest.reference_sortino}
                    />
                  ) : null}
                  <span>
                    초과{" "}
                    <b style={{ color: signColor(backtest.strategy_total_pct - backtest.benchmark_total_pct) }}>
                      {formatSigned(backtest.strategy_total_pct - backtest.benchmark_total_pct)}p
                    </b>
                  </span>
                </div>
                <AppAgGrid<BacktestMonthRow>
                  rowData={backtest.monthly}
                  columnDefs={backtestColumns}
                  theme={gridTheme}
                  minHeight={0}
                  height="auto"
                  gridOptions={{ domLayout: "autoHeight" }}
                  getRowClass={(p) => (p.data?.is_pending ? "steadyPendingRow" : "")}
                  getRowId={(p) => p.data.month}
                />
              </>
            ) : !backtesting ? (
              <span style={{ ...hintStyle, fontSize: "0.84rem" }}>
                기간을 고르고 실행을 누르면 월별 성과가 표시됩니다.
              </span>
            ) : null}
          </div>
        </div>
      </div>
    </PageFrame>
  );
}
