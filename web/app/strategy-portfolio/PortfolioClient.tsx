"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { ColDef } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";
import { BacktestSummary } from "../components/BacktestSummary";
import { IconCheck } from "@tabler/icons-react";

import { GridToolbarButton } from "../components/GridToolbarButton";
import { useAddingTickerRow } from "../components/useAddingTickerRow";
import { MonthsSelect } from "../components/MonthsSelect";
import { NavTabs } from "../components/NavTabs";
import { PageFrame } from "../components/PageFrame";
import { StableInlineInput } from "../components/StableInlineInput";
import { StrategyNotes } from "../components/StrategyNotes";
import { StrategyHoldingCharts } from "../components/StrategyHoldingCharts";
import { type HoldingChartData } from "../components/HoldingChart";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { UnsavedChangesBadge } from "../components/UnsavedChangesBadge";
import { useToast } from "../components/ToastProvider";
import { readRememberedTickerType, writeRememberedTickerType } from "../components/account-selection";
import { createAppGridTheme } from "../components/app-grid-theme";
import { formatSignedPct, signColor, stockMemoColumn } from "@/lib/grid-cells";
import { renderStockNameCell } from "@/lib/name-highlight";
import { formatPoolLabel } from "@/lib/pool-label";
import { updateStockMemo } from "@/lib/stocks-store";

const gridTheme = createAppGridTheme();

// 종목 표 안쪽 탭 — 모멘텀·신고가·합성과 같은 구성. 차트는 담은 종목 수만큼 그리므로 열 때만 그린다.
const HOLDINGS_TABS = [
  { key: "list", label: "종목" },
  { key: "chart", label: "차트" },
] as const;
type HoldingsTab = (typeof HOLDINGS_TABS)[number]["key"];

/** 현금 행 — 종목 비중 합의 나머지다(`/asset-helper` 와 같은 규칙·같은 내부 티커). */
const CASH_TICKER = "__CASH__";

/** 접이식 전략 설명 — 다른 전략 화면과 같은 자리·같은 형식. */
const STRATEGY_NOTES = [
  {
    title: "구성",
    body:
      "종목풀에서 고른 종목에 목표 비중(%)을 직접 정해 그대로 들고 갑니다. 순위·이평선으로 " +
      "종목을 고르지 않으므로 교체·이탈이 없습니다. 현금 비중도 직접 정합니다.",
  },
  {
    title: "리밸런싱",
    body:
      "정한 주기마다 목표 비중으로 되돌립니다. 그 사이에는 시세대로 흘러가게 두고, " +
      "리밸런싱 기준을 넘긴 종목만 매매 지시가 나옵니다.",
  },
  {
    title: "합성 전략",
    body:
      "합성(A·B 슬리브)에서 모멘텀·신고가와 나란히 고를 수 있습니다. 슬리브 몫 안에서 " +
      "이 비중대로 나눕니다. 찾을 값이 없는 전략이라 튜닝은 없습니다.",
  },
];

/** 백테스트 기간 기본값 — 서버 응답(`constraints.default_backtest_months`)이 오기 전 초기 상태용. */
const DEFAULT_BACKTEST_MONTHS = 12;

/** 백테스트 보기 단위 — 모멘텀·신고가 화면과 같은 목록. */
const VIEW_MODES = [
  { key: "yearly", label: "연간" },
  { key: "monthly", label: "월간" },
  { key: "daily", label: "일간" },
];

type WeightRow = {
  ticker: string;
  name: string;
  memo?: string;
  /** 목표 비중(%) — 사용자가 직접 정한다. 현금 행도 같은 컬럼을 쓴다. */
  fixed_weight_pct: number | null;
  daily_change_pct: number | null;
  current_price: number | null;
  return_1m_pct: number | null;
  return_3m_pct: number | null;
  return_6m_pct: number | null;
  return_12m_pct: number | null;
  mdd_pct: number | null;
  sortino: number | null;
  /** 티커를 입력받는 중인 행 — 확인을 눌러야 확정된다(`/asset-helper` 와 같은 흐름). */
  is_adding?: boolean;
};

/** 종목풀 종목 + 표시 지표 — 백엔드 `universe_metrics()` 가 채운다. */
type UniverseRow = {
  ticker: string;
  name: string;
  /** 종목 메모 — 계좌가 아니라 종목에 붙는다(순위·자산 관리 화면과 같은 값). */
  memo: string;
  current_price: number | null;
  daily_change_pct: number | null;
  return_1m_pct: number | null;
  return_3m_pct: number | null;
  return_6m_pct: number | null;
  return_12m_pct: number | null;
  mdd_pct: number | null;
  sortino: number | null;
};

type PeriodRow = { period: string; strategy_pct: number; benchmark_pct: number };

/** 백엔드 응답 — `GET/PUT /api/strategy-portfolio`. */
type PoolOption = { ticker_type: string; name: string; icon?: string; order?: number | null };

type Settings = {
  pool: string;
  weights: { ticker: string; weight_pct: number }[];
  /** 현금 비중(%) — 사용자가 직접 정한다(`/asset-helper` 와 같은 규칙, 자동 흡수 없음). */
  cash_weight_pct: number;
  rebalance: string;
  band_pct: number;
};

type View = {
  settings: Settings;
  default_settings: Settings;
  settings_by_pool: Record<string, Partial<Settings>>;
  pool_options: PoolOption[];
  /** 이 풀에서 담을 수 있는 종목 + 표시 지표 — 티커 검증·이름·지표 컬럼이 여기서 온다. */
  universe: UniverseRow[];
  constraints: {
    rebalance_options: { value: string; label: string }[];
    band_pct_options: number[];
    month_options: number[];
    default_backtest_months: number;
    max_holdings: number;
  };
  error?: string;
};

type Backtest = {
  start_date: string;
  end_date: string;
  strategy_total_pct: number;
  strategy_cagr_pct: number;
  strategy_mdd_pct: number;
  strategy_sortino: number | null;
  benchmark_total_pct: number;
  benchmark_cagr_pct: number;
  benchmark_mdd_pct: number;
  benchmark_sortino: number | null;
  benchmark_name: string;
  rebalance_count: number;
  cash_weight_pct: number;
  trades: {
    date: string;
    ticker: string;
    side: string;
    reason: string;
    price: number;
    weight_before_pct: number;
    weight_after_pct: number;
  }[];
  daily: { date: string; strategy_pct: number; benchmark_pct: number }[];
  error?: string;
};

/** 퍼센트 셀 — `/asset-helper` 와 같은 표기(부호·색). */
function renderPctCell(params: { value?: number | null }) {
  const value = params.value;
  if (value == null || !Number.isFinite(value)) return <span>-</span>;
  return <span style={{ color: signColor(value) }}>{formatSignedPct(value, 2)}</span>;
}

function fmtNum(value: number | null | undefined, digits = 2): string {
  return value == null || !Number.isFinite(value) ? "-" : value.toFixed(digits);
}

export function PortfolioClient() {
  const toast = useToast();
  const [view, setView] = useState<View | null>(null);
  const [draft, setDraft] = useState<Settings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const [selected, setSelected] = useState<string[]>([]);
  // ── 차트 탭 (모멘텀·신고가·합성과 같은 구성 — 공용 StrategyHoldingCharts) ──
  const [holdingsTab, setHoldingsTab] = useState<HoldingsTab>("list");
  const [charts, setCharts] = useState<HoldingChartData[] | null>(null);
  const [chartsLoading, setChartsLoading] = useState(false);
  const [chartsError, setChartsError] = useState<string | null>(null);
  // 차트 기간(개월) — 백엔드 config.HOLDING_CHART_MONTHS 가 단일 소스. 응답에서 받아 문구에 쓴다.
  const [chartMonths, setChartMonths] = useState<number | null>(null);


  // 백테스트 — 기본 12개월(모멘텀·신고가 화면과 같다).
  const [backtestMonths, setBacktestMonths] = useState(DEFAULT_BACKTEST_MONTHS);
  const [viewMode, setViewMode] = useState("yearly");
  const [backtest, setBacktest] = useState<Backtest | null>(null);
  const [backtestError, setBacktestError] = useState<string | null>(null);
  const [backtesting, setBacktesting] = useState(false);

  // ── 로드 ────────────────────────────────────────────────────────────────
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        // 마지막으로 고른 풀은 브라우저에 기억한다(다른 화면들과 같은 공용 키).
        const remembered = readRememberedTickerType();
        const query = remembered ? `?pool=${encodeURIComponent(remembered)}` : "";
        const response = await fetch(`/api/strategy-portfolio${query}`, { cache: "no-store" });
        const payload = (await response.json()) as View;
        if (!response.ok) throw new Error(payload.error ?? "설정을 불러오지 못했습니다.");
        if (!alive) return;
        setView(payload);
        setDraft(payload.settings);
        setBacktestMonths(payload.constraints.default_backtest_months);
      } catch (loadError) {
        if (alive) setError(loadError instanceof Error ? loadError.message : "설정을 불러오지 못했습니다.");
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const loadPool = useCallback(async (nextPool: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/strategy-portfolio?pool=${encodeURIComponent(nextPool)}`, { cache: "no-store" });
      const payload = (await response.json()) as View;
      if (!response.ok) throw new Error(payload.error ?? "설정을 불러오지 못했습니다.");
      setView(payload);
      setDraft(payload.settings);
      // 풀이 바뀌면 전 풀의 백테스트 결과는 의미가 없다.
      setBacktest(null);
      setBacktestError(null);
      setSelected([]);
      writeRememberedTickerType(nextPool);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "설정을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, []);

  /** 종목 메모 저장 — 계좌가 아니라 종목에 붙는다(순위·자산 관리 화면과 같은 API). */
  const saveMemo = useCallback(
    async (ticker: string, memo: string) => {
      try {
        await updateStockMemo(ticker, memo);
        toast.success("메모 저장 완료");
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "메모 저장에 실패했습니다.");
      }
    },
    [toast],
  );

  const saveSettings = useCallback(async () => {
    if (!draft) return;
    setSaving(true);
    try {
      const response = await fetch("/api/strategy-portfolio", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ settings: draft }),
      });
      const payload = (await response.json()) as View;
      if (!response.ok) throw new Error(payload.error ?? "설정을 저장하지 못했습니다.");
      setView(payload);
      setDraft(payload.settings);
      // 설정이 바뀌면 결과가 달라진다 — 다른 전략 화면과 같은 규칙.
      setBacktest(null);
      setBacktestError(null);
      toast.success("포트폴리오 설정 저장 완료");
    } catch (saveError) {
      toast.error(saveError instanceof Error ? saveError.message : "설정을 저장하지 못했습니다.");
    } finally {
      setSaving(false);
    }
  }, [draft, toast]);

  const runBacktest = useCallback(async () => {
    if (!draft) return;
    setBacktesting(true);
    setBacktestError(null);
    try {
      const response = await fetch("/api/strategy-portfolio/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ settings: draft, months: backtestMonths }),
      });
      const payload = (await response.json()) as Backtest;
      if (!response.ok) throw new Error(payload.error ?? "백테스트에 실패했습니다.");
      setBacktest(payload);
    } catch (runError) {
      setBacktestError(runError instanceof Error ? runError.message : "백테스트에 실패했습니다.");
    } finally {
      setBacktesting(false);
    }
  }, [draft, backtestMonths]);

  // ── 파생 ────────────────────────────────────────────────────────────────
  const universeByTicker = useMemo(
    () => Object.fromEntries((view?.universe ?? []).map((item) => [item.ticker, item])),
    [view?.universe],
  );
  const nameByTicker = useMemo(
    () => Object.fromEntries((view?.universe ?? []).map((item) => [item.ticker, item.name])),
    [view?.universe],
  );

  const stockSum = useMemo(
    () => Math.round((draft?.weights ?? []).reduce((sum, row) => sum + (row.weight_pct || 0), 0) * 100) / 100,
    [draft?.weights],
  );
  // 현금은 **사용자가 정하는 값**이다 — 종목 합의 나머지로 자동 흡수하지 않는다.
  // 합이 100% 가 아니면 저장을 막고, 어디를 조정할지는 사용자가 정한다(`/asset-helper` 와 동일).
  const cashPct = draft?.cash_weight_pct ?? 0;
  const totalSum = Math.round((stockSum + cashPct) * 100) / 100;
  const weightOk = Math.abs(totalSum - 100) <= 0.01;
  const isDirty = useMemo(
    () => Boolean(view && draft) && JSON.stringify(view!.settings) !== JSON.stringify(draft),
    [view, draft],
  );

  const setWeights = (next: Settings["weights"]) => setDraft((previous) => (previous ? { ...previous, weights: next } : previous));

  const emptyMetrics = {
    daily_change_pct: null,
    current_price: null,
    return_1m_pct: null,
    return_3m_pct: null,
    return_6m_pct: null,
    return_12m_pct: null,
    mdd_pct: null,
    sortino: null,
  };

  /** 종목 추가 — `/assets`·`/asset-helper` 와 **같은 훅·같은 조회 API** 를 쓴다.
   *  화면마다 검증을 따로 만들면 호주 티커(`ASX:` 표기) 같은 규칙이 한쪽에만 반영된다. */
  const addResolve = useCallback(
    async (raw: string): Promise<{ ticker: string; name: string }> => {
      // 조회 범위를 이 종목풀로 제한한다 — 담을 수 있는 종목이 그 풀 것뿐이다.
      const pool = encodeURIComponent(draft?.pool ?? "");
      const resp = await fetch(`/api/ticker-resolve?ticker=${encodeURIComponent(raw)}&ticker_types=${pool}`);
      const data = (await resp.json()) as { ticker?: string; name?: string; error?: string; detail?: string };
      if (!resp.ok || data.error || !data.ticker || !data.name) {
        throw new Error(data.error ?? data.detail ?? "이 종목풀에 없는 종목입니다.");
      }
      if ((draft?.weights ?? []).some((row) => row.ticker === data.ticker)) {
        throw new Error(`이미 담은 종목입니다: ${data.ticker}`);
      }
      return { ticker: data.ticker, name: data.name };
    },
    [draft?.pool, draft?.weights],
  );
  const addOnValidated = useCallback(
    (resolved: { ticker: string; name: string }) => {
      setWeights([...(draft?.weights ?? []), { ticker: resolved.ticker, weight_pct: 0 }]);
      toast.success(`조회 성공: ${resolved.name}`);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- setWeights 는 draft 에서 파생된다
    [draft?.weights, toast],
  );
  const add = useAddingTickerRow({
    resolve: addResolve,
    onValidated: addOnValidated,
    onError: (message) => toast.error(message),
    normalize: (raw) => raw.trim().toUpperCase(),
    resetOnValidated: true,
  });

  /** 현금 고정행 — **표 맨 위**(`/asset-helper` 와 같은 순서). 비중은 직접 편집한다. */
  const cashRow: WeightRow = { ticker: CASH_TICKER, name: "현금", fixed_weight_pct: cashPct, ...emptyMetrics };
  const addingRow: WeightRow | null = add.addingRow
    ? { ticker: add.addingRow.ticker, name: add.addingRow.name, fixed_weight_pct: 0, is_adding: true, ...emptyMetrics }
    : null;
  const gridRows: WeightRow[] = [
    cashRow,
    ...(addingRow ? [addingRow] : []),
    ...(draft?.weights ?? []).map((row) => {
      const metrics = universeByTicker[row.ticker];
      return {
        ...emptyMetrics,
        // 지표는 종목풀 목록에서 온다 — 저장 설정에는 비중만 들어 있다.
        ...(metrics
          ? {
              current_price: metrics.current_price,
              daily_change_pct: metrics.daily_change_pct,
              return_1m_pct: metrics.return_1m_pct,
              return_3m_pct: metrics.return_3m_pct,
              return_6m_pct: metrics.return_6m_pct,
              return_12m_pct: metrics.return_12m_pct,
              mdd_pct: metrics.mdd_pct,
              sortino: metrics.sortino,
              memo: metrics.memo,
            }
          : {}),
        ticker: row.ticker,
        name: metrics?.name ?? row.ticker,
        fixed_weight_pct: row.weight_pct,
      };
    }),
  ];

  // 차트를 그릴 대상 — 담은 종목만. 현금 행과 추가 중인 행은 뺀다.
  const chartRows = useMemo(
    () => (draft?.weights ?? []).map((row) => row.ticker),
    [draft?.weights],
  );
  // 풀·구성이 바뀌면 이전 차트는 버린다.
  const chartKey = useMemo(
    () => `${draft?.pool ?? ""}|${chartRows.join(",")}`,
    [draft?.pool, chartRows],
  );
  useEffect(() => {
    setCharts(null);
    setChartsError(null);
  }, [chartKey]);
  // 차트 탭을 열 때만 받는다 — 담은 종목 수만큼 일봉을 실어 오므로 목록 탭에서는 낭비다.
  useEffect(() => {
    if (holdingsTab !== "chart" || !draft || charts || chartsLoading || chartsError) return;
    // 저장 중에는 초안이 아직 이전 풀 것이다 — 응답이 와서 종목 목록까지 바뀐 뒤에 받는다.
    if (saving) return;
    if (chartRows.length === 0) {
      setCharts([]);
      return;
    }
    setChartsLoading(true);
    void (async () => {
      try {
        const response = await fetch("/api/strategy-portfolio/charts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pool: draft.pool, tickers: chartRows }),
        });
        const payload = (await response.json()) as { charts?: HoldingChartData[]; months?: number; error?: string };
        if (!response.ok) throw new Error(payload.error ?? "차트를 불러오지 못했습니다.");
        setCharts(payload.charts ?? []);
        setChartMonths(payload.months ?? null);
      } catch (chartError) {
        const message = chartError instanceof Error ? chartError.message : "차트를 불러오지 못했습니다.";
        setChartsError(message);
        toast.error(message);
      } finally {
        setChartsLoading(false);
      }
    })();
  }, [holdingsTab, draft, charts, chartsLoading, chartsError, chartRows, saving, toast]);

  const columns = useMemo<ColDef<WeightRow>[]>(
    () => [
      {
        colId: "drag",
        headerName: "",
        width: 36,
        maxWidth: 36,
        pinned: "left",
        sortable: false,
        resizable: false,
        suppressMovable: true,
        // 현금 고정행·추가행은 드래그 불가. 확정된 종목만 순서 변경.
        rowDrag: (params) => Boolean(params.data && !params.data.is_adding && params.data.ticker !== CASH_TICKER),
        cellClass: "assetsDragCell",
        valueGetter: () => "",
      },
      {
        field: "ticker",
        headerName: "티커",
        minWidth: 110,
        width: 110,
        pinned: "left",
        cellRenderer: (params: { data?: WeightRow }) => {
          const row = params.data;
          if (!row) return "-";
          if (row.ticker === CASH_TICKER) return <span>-</span>;
          if (row.is_adding) {
            return (
              <StableInlineInput
                className="form-control form-control-sm assetsInlineInput assetsInlineInputTicker"
                placeholder="티커"
                initialValue={row.ticker}
                submitOnBlur={false}
                onChange={(value) => add.setTicker(value)}
                onSave={() => void add.validate()}
              />
            );
          }
          return <TickerDetailLink ticker={row.ticker} displayTicker={row.ticker} />;
        },
      },
      {
        field: "name",
        headerName: "종목명",
        minWidth: 220,
        flex: 1,
        cellRenderer: (params: { data?: WeightRow; value?: string }) => {
          const row = params.data;
          if (!row) return null;
          if (row.is_adding) {
            return (
              <div className="assetsNameLookup">
                <span className="assetsNameLookupStatus">
                  {add.addingRow?.isValidating ? "조회 중…" : (row.name || "티커를 입력한 뒤 확인하세요.")}
                </span>
                <button
                  type="button"
                  className="btn btn-outline-primary btn-sm assetsInlineButton d-inline-flex align-items-center gap-1"
                  disabled={add.addingRow?.isValidating}
                  onClick={() => void add.validate()}
                >
                  확인
                </button>
              </div>
            );
          }
          if (row.ticker === CASH_TICKER) return <span style={{ color: "var(--text-muted)" }}>현금</span>;
          return renderStockNameCell(params.value);
        },
      },
      // 종목 메모 — 전 화면 공용 컬럼(`@/lib/grid-cells`). 순위·자산 관리 화면과 같은 값이고
      // 저장 경로도 같다(셀을 벗어나면 바로 저장). 현금 행·추가행은 종목이 아니라 편집 불가.
      stockMemoColumn<WeightRow>({
        field: "memo",
        editable: (row) => Boolean(row && !row.is_adding && row.ticker !== CASH_TICKER),
        onSave: (row, memo) => void saveMemo(row.ticker, memo),
      }),
      { field: "daily_change_pct", headerName: "일간", minWidth: 88, width: 88, type: "rightAligned", cellRenderer: renderPctCell },
      {
        field: "current_price",
        headerName: "현재가",
        minWidth: 96,
        width: 96,
        type: "rightAligned",
        valueFormatter: (p) => (p.value == null ? "-" : new Intl.NumberFormat("ko-KR").format(p.value as number)),
      },
      { field: "return_1m_pct", headerName: "1달", minWidth: 84, width: 84, type: "rightAligned", cellRenderer: renderPctCell },
      { field: "return_3m_pct", headerName: "3달", minWidth: 84, width: 84, type: "rightAligned", cellRenderer: renderPctCell },
      { field: "return_6m_pct", headerName: "6달", minWidth: 84, width: 84, type: "rightAligned", cellRenderer: renderPctCell },
      { field: "return_12m_pct", headerName: "1년", minWidth: 84, width: 84, type: "rightAligned", cellRenderer: renderPctCell },
      { field: "mdd_pct", headerName: "MDD", minWidth: 84, width: 84, type: "rightAligned", cellRenderer: renderPctCell },
      {
        field: "sortino",
        headerName: "Sortino",
        minWidth: 90,
        width: 90,
        type: "rightAligned",
        valueFormatter: (p) => fmtNum(p.value as number | null),
      },
      {
        field: "fixed_weight_pct",
        headerName: "비중",
        minWidth: 92,
        width: 92,
        type: "rightAligned",
        // 현금 행도 편집한다 — 종목합 + 현금 = 100% 여야 저장된다(`/asset-helper` 와 동일).
        editable: (params) => Boolean(params.data && !params.data.is_adding),
        cellClass: (params) => (params.data?.is_adding ? undefined : "appEditableCell"),
        valueParser: (p) => Number(p.newValue),
        valueSetter: (p) => {
          const next = Number(p.newValue);
          if (!Number.isFinite(next) || next < 0 || next > 100) return false;
          if (p.data.ticker === CASH_TICKER) {
            setDraft((previous) => (previous ? { ...previous, cash_weight_pct: next } : previous));
            return true;
          }
          setWeights((draft?.weights ?? []).map((row) => (row.ticker === p.data.ticker ? { ...row, weight_pct: next } : row)));
          return true;
        },
        valueFormatter: (p) => (p.value == null ? "-" : `${Number(p.value).toFixed(2)}`),
      },
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [draft?.weights, nameByTicker, universeByTicker, add, saveMemo],
  );

  const periodRows = useMemo<PeriodRow[]>(() => {
    if (!backtest) return [];
    const keyLength = viewMode === "yearly" ? 4 : viewMode === "monthly" ? 7 : 10;
    // 일별 누적(%)을 구간별 수익률로 자른다 — 다른 전략 화면과 같은 방식.
    const byKey = new Map<string, { first: number; last: number }>();
    let previous: number | null = null;
    for (const row of backtest.daily) {
      const key = row.date.slice(0, keyLength);
      const base = previous ?? row.strategy_pct;
      const entry = byKey.get(key);
      if (!entry) byKey.set(key, { first: base, last: row.strategy_pct });
      else entry.last = row.strategy_pct;
      previous = row.strategy_pct;
    }
    const benchByKey = new Map<string, { first: number; last: number }>();
    let previousBench: number | null = null;
    for (const row of backtest.daily) {
      const key = row.date.slice(0, keyLength);
      const base = previousBench ?? row.benchmark_pct;
      const entry = benchByKey.get(key);
      if (!entry) benchByKey.set(key, { first: base, last: row.benchmark_pct });
      else entry.last = row.benchmark_pct;
      previousBench = row.benchmark_pct;
    }
    const step = (entry: { first: number; last: number }) =>
      Math.round(((1 + entry.last / 100) / (1 + entry.first / 100) - 1) * 10000) / 100;
    return [...byKey.entries()]
      .map(([period, entry]) => ({
        period,
        strategy_pct: step(entry),
        benchmark_pct: benchByKey.has(period) ? step(benchByKey.get(period)!) : 0,
      }))
      .reverse();
  }, [backtest, viewMode]);

  const periodColumns = useMemo<ColDef<PeriodRow>[]>(
    () => [
      {
        field: "period",
        headerName: viewMode === "yearly" ? "연도" : viewMode === "monthly" ? "월" : "일자",
        width: 140,
      },
      { field: "strategy_pct", headerName: "전략(%)", flex: 1, minWidth: 108, type: "rightAligned", cellRenderer: renderPctCell },
      { field: "benchmark_pct", headerName: "벤치마크(%)", flex: 1, minWidth: 108, type: "rightAligned", cellRenderer: renderPctCell },
    ],
    [viewMode],
  );

  const hintStyle = { fontSize: "var(--fs-sm)", color: "var(--text-muted)" } as const;

  if (loading && !view) {
    return (
      <PageFrame title="포트폴리오 전략" fullWidth>
        <div className="appPageStack">불러오는 중…</div>
      </PageFrame>
    );
  }
  if (!view || !draft) {
    return (
      <PageFrame title="포트폴리오 전략" fullWidth>
        <div className="alert alert-danger">{error ?? "설정을 불러오지 못했습니다."}</div>
      </PageFrame>
    );
  }

  return (
    <PageFrame title="포트폴리오 전략" fullWidth>
      <div className="appPageStack">
        {/* ① 설정 — 종목풀 · 리밸런싱 주기 · 리밸런싱 기준. 모멘텀·신고가 화면과 같은 자리. */}
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">종목풀</span>
                  <select
                    className="form-select form-select-sm"
                    value={draft.pool}
                    disabled={loading}
                    onChange={(event) => void loadPool(event.target.value)}
                  >
                    {view.pool_options.map((option) => (
                      <option key={option.ticker_type} value={option.ticker_type}>
                        {formatPoolLabel(option)}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">리밸런싱</span>
                  <select
                    className="form-select form-select-sm"
                    style={{ width: "auto" }}
                    value={draft.rebalance}
                    onChange={(event) => setDraft({ ...draft, rebalance: event.target.value })}
                  >
                    {view.constraints.rebalance_options.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">리밸런싱 기준</span>
                  <select
                    className="form-select form-select-sm"
                    style={{ width: "auto" }}
                    value={String(draft.band_pct)}
                    title="목표 비중과 이만큼(%) 벌어져야 되돌립니다"
                    onChange={(event) => setDraft({ ...draft, band_pct: Number(event.target.value) })}
                  >
                    {view.constraints.band_pct_options.map((option) => (
                      <option key={option} value={String(option)}>
                        {option}%
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="appMainHeaderRight">
                <UnsavedChangesBadge show={isDirty} />
                <button
                  type="button"
                  className="btn btn-success btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                  onClick={() => void saveSettings()}
                  disabled={saving || !isDirty || !weightOk}
                  title={weightOk ? undefined : "종목 비중 합이 100%를 넘습니다"}
                >
                  <IconCheck size={16} />
                  <span>{saving ? "저장 중…" : "저장"}</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* ② 목표 비중 — 컬럼·버튼은 `/asset-helper` 표준을 따른다. */}
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <span style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>목표 비중</span>
                <span style={hintStyle}>
                  종목 {draft.weights.length}/{view.constraints.max_holdings}개
                </span>
                <span
                  style={{
                    fontSize: "var(--fs-sm)",
                    fontWeight: 700,
                    color: weightOk ? "#2f9e44" : "var(--up-color, #d64545)",
                  }}
                  title="종목 + 현금 = 100% 여야 저장됩니다"
                >
                  합계 {totalSum.toFixed(2)}%{weightOk ? "" : " — 100% 여야 저장됩니다"}
                </span>
              </div>
              <div className="appMainHeaderRight">
                <GridToolbarButton
                  variant="add"
                  onClick={() => add.start()}
                  disabled={add.addingRow !== null || draft.weights.length >= view.constraints.max_holdings}
                />
                <GridToolbarButton
                  variant="delete"
                  onClick={() => {
                    setWeights(draft.weights.filter((row) => !selected.includes(row.ticker)));
                    setSelected([]);
                  }}
                  disabled={selected.length === 0}
                >
                  삭제
                </GridToolbarButton>
              </div>
            </div>
            <StrategyNotes items={STRATEGY_NOTES} />
            <NavTabs
              items={HOLDINGS_TABS}
              value={holdingsTab}
              onChange={setHoldingsTab}
              label="종목 보기"
              style={{ marginBottom: 12 }}
            />
            {holdingsTab === "chart" ? (
              <StrategyHoldingCharts
                charts={charts}
                loading={chartsLoading || saving}
                error={chartsError}
                emptyMessage="담은 종목이 없습니다."
                hint="포트폴리오는 진입·청산 판정이 없어 종목풀 설정의 단기·장기 이평선을 참고로 그립니다."
                months={chartMonths}
                chartProps={() => ({ strategyLabel: "포트폴리오" })}
              />
            ) : (
            <AppAgGrid<WeightRow>
              rowData={gridRows}
              columnDefs={columns}
              theme={gridTheme}
              minHeight={0}
              height="auto"
              gridOptions={{
                domLayout: "autoHeight",
                suppressMovableColumns: true,
                // AG Grid 최신 선택 API — `/asset-helper` 와 같은 형식.
                // 현금 고정행·추가행은 체크박스를 아예 그리지 않는다.
                rowSelection: {
                  mode: "multiRow",
                  checkboxes: (params) =>
                    Boolean(params.data && !params.data.is_adding && params.data.ticker !== CASH_TICKER),
                  headerCheckbox: true,
                  hideDisabledCheckboxes: true,
                  enableClickSelection: false,
                },
                rowDragManaged: true,
                selectionColumnDef: {
                  width: 52,
                  minWidth: 52,
                  maxWidth: 52,
                  pinned: "left",
                  sortable: false,
                  resizable: false,
                  headerName: "",
                  cellClass: "assetsSelectCell",
                },
                onSelectionChanged: (event) =>
                  setSelected(
                    event.api
                      .getSelectedRows()
                      .map((row) => (row as WeightRow).ticker)
                      .filter((ticker) => ticker && ticker !== CASH_TICKER),
                  ),
                // 드래그로 바꾼 순서를 그대로 저장한다 — 화면 표 순서가 곧 저장 순서다.
                onRowDragEnd: (event) => {
                  const ordered: Settings["weights"] = [];
                  event.api.forEachNodeAfterFilterAndSort((node) => {
                    const row = node.data as WeightRow | undefined;
                    if (!row || row.is_adding || row.ticker === CASH_TICKER) return;
                    const found = draft.weights.find((item) => item.ticker === row.ticker);
                    if (found) ordered.push(found);
                  });
                  if (ordered.length === draft.weights.length) setWeights(ordered);
                },
              }}
              getRowId={(p) => p.data.ticker}
            />
            )}
            {holdingsTab === "list" && draft.weights.length === 0 ? (
              <div style={{ ...hintStyle, padding: "16px 0", textAlign: "center" }}>
                ＋ 를 눌러 이 종목풀의 종목을 담고 비중을 정하세요.
              </div>
            ) : null}
          </div>
        </div>

        {/* ③ 백테스트 — 모멘텀·신고가 화면과 같은 구조. 찾을 값이 없어 튜닝은 두지 않는다. */}
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <span style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>백테스트</span>
              </div>
              <div className="appMainHeaderRight">
                {isDirty ? <span style={hintStyle}>설정을 저장해야 실행할 수 있습니다</span> : null}
                <MonthsSelect
                  value={backtestMonths}
                  options={view.constraints.month_options}
                  disabled={backtesting}
                  onChange={setBacktestMonths}
                />
                <button
                  type="button"
                  className="btn btn-sm btn-dark"
                  onClick={() => void runBacktest()}
                  disabled={backtesting || isDirty || draft.weights.length === 0}
                >
                  {backtesting ? "실행 중…" : "실행"}
                </button>
              </div>
            </div>
            {backtestError ? <div className="alert alert-danger">{backtestError}</div> : null}
            {backtesting ? (
              <div style={{ ...hintStyle, padding: "24px 0", textAlign: "center" }}>백테스트 실행 중…</div>
            ) : !backtest ? (
              <div style={{ ...hintStyle, padding: "24px 0", textAlign: "center" }}>실행을 누르면 결과가 표시됩니다.</div>
            ) : (
              <>
                <BacktestSummary
                  startDate={backtest.start_date}
                  endDate={backtest.end_date}
                  strategy={{
                    label: "전략",
                    totalPct: backtest.strategy_total_pct,
                    cagrPct: backtest.strategy_cagr_pct,
                    mddPct: backtest.strategy_mdd_pct,
                    sortino: backtest.strategy_sortino,
                  }}
                  benchmark={{
                    label: backtest.benchmark_name,
                    totalPct: backtest.benchmark_total_pct,
                    cagrPct: backtest.benchmark_cagr_pct,
                    mddPct: backtest.benchmark_mdd_pct,
                    sortino: backtest.benchmark_sortino,
                  }}
                />
                <div style={{ ...hintStyle, marginBottom: 10 }}>
                  리밸런싱 매매 {backtest.rebalance_count}건 · 현금 {backtest.cash_weight_pct}%
                </div>
                <NavTabs
                  items={VIEW_MODES}
                  value={viewMode}
                  onChange={setViewMode}
                  label="백테스트 보기 단위"
                  style={{ marginBottom: 10 }}
                />
                <AppAgGrid<PeriodRow>
                  rowData={periodRows}
                  columnDefs={periodColumns}
                  theme={gridTheme}
                  minHeight={0}
                  height="auto"
                  gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                />
              </>
            )}
          </div>
        </div>
      </div>
    </PageFrame>
  );
}
