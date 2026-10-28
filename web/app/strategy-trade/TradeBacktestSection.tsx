"use client";

import type { ColDef } from "ag-grid-community";
import { useCallback, useMemo, useState } from "react";

import { type BacktestDayRow } from "@/lib/backtest-periods";
import { formatSignedPct, signColor, stockNameColumn, tickerColumn } from "@/lib/grid-cells";

import { AppAgGrid } from "../components/AppAgGrid";
import { AppLoadingProgress, startProgressRamp, type LoadingProgress } from "../components/AppLoadingProgress";
import { BacktestPeriodTables } from "../components/BacktestPeriodTables";
import { BacktestSummary } from "../components/BacktestSummary";
import { BacktestTradeStats } from "../components/BacktestTradeStats";
import { MonthsSelect } from "../components/MonthsSelect";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";

const gridTheme = createAppGridTheme();

/** 체결 한 건 — 어느 종목의 몇 호 회차를 언제 사고 팔았는지. 안 판 회차는 exit_* 가 비어 보유중이다. */
type TradeRow = {
  ticker: string;
  name: string;
  round: number;
  quantity: number;
  entry_date: string;
  entry_price: number;
  exit_date: string | null;
  exit_price: number | null;
  return_pct: number | null;
  days: number | null;
  reason: string | null;
};

/** 백테스트 응답 — 종목풀의 전략 전체를 합친 한 벌의 결과. 모양은 다른 전략 화면과 같다. */
type TradeBacktest = {
  start_date: string;
  end_date: string;
  strategy_total_pct: number | null;
  strategy_cagr_pct: number | null;
  strategy_mdd_pct: number | null;
  strategy_mdd_from: string | null;
  strategy_mdd_to: string | null;
  strategy_sortino: number | null;
  benchmark_name: string;
  benchmark_ticker: string | null;
  benchmark_total_pct: number | null;
  benchmark_cagr_pct: number | null;
  benchmark_mdd_pct: number | null;
  benchmark_mdd_from: string | null;
  benchmark_mdd_to: string | null;
  benchmark_sortino: number | null;
  daily: BacktestDayRow[];
  trades: TradeRow[];
  trade_count: number;
  win_rate_pct: number | null;
  avg_win_pct: number | null;
  avg_loss_pct: number | null;
  reason_counts: Record<string, number>;
};

const hintStyle: React.CSSProperties = { color: "var(--text-muted)", fontSize: "var(--fs-sm)" };

function formatPrice(value: number | null | undefined): string {
  return value == null ? "-" : value.toLocaleString("ko-KR", { maximumFractionDigits: 0 });
}

/** 하단 백테스트 — 종목풀 하나에 켜 둔 사고팔기 전략 전체를 함께 굴린 결과를 기간별로 보여준다. */
export function TradeBacktestSection({
  pool,
  monthOptions,
  blockedReason,
}: {
  pool: string;
  monthOptions: number[];
  /** 실행할 수 없는 사유(저장 안 한 변경·전략 없음) — 있으면 실행 버튼이 꺼지고 문구가 툴팁으로 뜬다. */
  blockedReason: string | null;
}) {
  const toast = useToast();
  const [months, setMonths] = useState(12);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<LoadingProgress | null>(null);
  const [result, setResult] = useState<TradeBacktest | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    setProgress({ percent: 10, message: "종목풀의 사고팔기 전략을 함께 계산하는 중" });
    const stopRamp = startProgressRamp(setProgress);
    try {
      const response = await fetch(`/api/strategy-trade/backtest?pool=${encodeURIComponent(pool)}&months=${months}`, {
        cache: "no-store",
      });
      const payload = (await response.json()) as TradeBacktest & { error?: string };
      if (!response.ok || payload.error) throw new Error(payload.error ?? "백테스트를 불러오지 못했습니다.");
      setProgress({ percent: 100, message: "결과 반영 중" });
      setResult(payload);
    } catch (runError) {
      const message = runError instanceof Error ? runError.message : "백테스트를 불러오지 못했습니다.";
      setError(message);
      toast.error(message);
    } finally {
      stopRamp();
      setLoading(false);
      setProgress(null);
    }
  }, [months, pool, toast]);

  const tradeColumns = useMemo<ColDef<TradeRow>[]>(
    () => [
      tickerColumn<TradeRow>({ width: 96 }),
      stockNameColumn<TradeRow>({}),
      { headerName: "회차", field: "round", width: 76, type: "numericColumn", valueFormatter: (p) => `${p.value}호` },
      { headerName: "수량", field: "quantity", width: 84, type: "numericColumn", valueFormatter: (p) => formatPrice(p.value) },
      { headerName: "매수일", field: "entry_date", width: 116 },
      { headerName: "매수가", field: "entry_price", width: 100, type: "numericColumn", valueFormatter: (p) => formatPrice(p.value) },
      {
        headerName: "매도일",
        field: "exit_date",
        width: 116,
        valueFormatter: (p) => (p.value ? String(p.value) : "-"),
      },
      {
        headerName: "매도가",
        field: "exit_price",
        headerTooltip: "보유중 행은 현재가",
        width: 100,
        type: "numericColumn",
        valueFormatter: (p) => formatPrice(p.value),
      },
      {
        headerName: "수익률(%)",
        field: "return_pct",
        width: 110,
        type: "numericColumn",
        valueFormatter: (p) => (p.value == null ? "-" : formatSignedPct(p.value as number)),
        cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 700 }),
      },
      { headerName: "보유일", field: "days", width: 84, type: "numericColumn" },
      { headerName: "사유", field: "reason", width: 110 },
    ],
    [],
  );

  const benchmarkLabel = result
    ? result.benchmark_ticker
      ? `${result.benchmark_name}(${result.benchmark_ticker})`
      : result.benchmark_name
    : "";

  return (
    <section className="appSection">
      <div className="card appCard">
        <div className="card-header appCardHeader">
          <span style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>백테스트</span>
          <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <MonthsSelect value={months} options={monthOptions} disabled={loading} onChange={setMonths} />
            <button
              type="button"
              className="btn btn-sm btn-dark"
              disabled={loading || blockedReason !== null}
              title={blockedReason ?? "저장된 전략 설정으로 이 종목풀의 전략 전체를 함께 굴려 봅니다."}
              onClick={() => void run()}
            >
              {loading ? "실행 중…" : "실행"}
            </button>
          </span>
        </div>
        <div className="card-body appCardBodyTight">
          {error ? (
            <div className="alert alert-danger" style={{ marginBottom: 0 }}>
              {error}
            </div>
          ) : loading ? (
            <AppLoadingProgress title="백테스트 실행 중..." progress={progress} />
          ) : result ? (
            <>
              <BacktestSummary
                startDate={result.start_date}
                endDate={result.end_date}
                strategy={{
                  label: "전략",
                  totalPct: result.strategy_total_pct,
                  cagrPct: result.strategy_cagr_pct,
                  mddPct: result.strategy_mdd_pct,
                  mddFrom: result.strategy_mdd_from,
                  mddTo: result.strategy_mdd_to,
                  sortino: result.strategy_sortino,
                }}
                benchmark={{
                  label: benchmarkLabel,
                  totalPct: result.benchmark_total_pct,
                  cagrPct: result.benchmark_cagr_pct,
                  mddPct: result.benchmark_mdd_pct,
                  mddFrom: result.benchmark_mdd_from,
                  mddTo: result.benchmark_mdd_to,
                  sortino: result.benchmark_sortino,
                }}
              />
              <BacktestTradeStats stats={result} style={{ marginBottom: 12 }} />
              <BacktestPeriodTables
                daily={result.daily}
                benchmarkLabel={benchmarkLabel}
                extraTab={{
                  label: "체결",
                  content: (
                    <AppAgGrid<TradeRow>
                      rowData={result.trades}
                      columnDefs={tradeColumns}
                      theme={gridTheme}
                      minHeight="auto"
                      gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                      getRowClass={(p) => (p.data?.exit_date ? "" : "momentumPendingRow")}
                    />
                  ),
                }}
              />
            </>
          ) : (
            <div style={{ ...hintStyle, padding: "32px 0", textAlign: "center" }}>실행을 누르면 결과가 표시됩니다.</div>
          )}
        </div>
      </div>
    </section>
  );
}
