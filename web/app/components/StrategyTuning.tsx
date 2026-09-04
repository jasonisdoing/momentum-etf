"use client";

import type { ColDef } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AppAgGrid } from "./AppAgGrid";
import { MonthsSelect } from "./MonthsSelect";
import { AppLoadingProgress, type LoadingProgress } from "./AppLoadingProgress";
import { useToast } from "./ToastProvider";
import { createAppGridTheme } from "./app-grid-theme";
import { signColor } from "@/lib/grid-cells";

/** 축 값 — 숫자 외에 "off"/"on" 같은 상태값도 쓴다(모멘텀 주중 이탈). */
export type TuningValue = number | string | null;

/** 튜닝 축 하나 — 화면 설정 항목과 1:1. 값 목록은 저장 가능한 옵션과 같아야 한다.
 *  범위는 고정이다(값을 골라 빼는 기능은 두지 않는다 — 전부 돌린다). */
export type TuningAxis = {
  key: string;
  label: string;
  values: { value: TuningValue; label: string }[];
};

export type TuningRow = {
  params: Record<string, TuningValue>;
  total_pct: number;
  cagr_pct: number | null;
  mdd_pct: number;
  sortino: number | null;
  quarter_wins: number;
  trade_count?: number;
  win_rate_pct?: number;
  /** 이긴 거래의 평균 수익률(%) — 승률과 함께 봐야 손익비가 보인다. */
  avg_win_pct?: number | null;
  /** 진 거래의 평균 손실률(%) — 음수다. */
  avg_loss_pct?: number | null;
};

export type TuningResult = {
  months: number;
  rows: TuningRow[];
  quarter_count: number;
  axes: Record<string, { value: TuningValue; count: number; sortino: number; total_pct: number; mdd_pct: number }[]>;
  skipped?: string[];
};

const hint: React.CSSProperties = { color: "var(--text-muted)", fontSize: "var(--fs-sm)" };
const gridTheme = createAppGridTheme();

/** 그리드 행 — 조합 파라미터를 `p_<축>` 컬럼으로 펼치고 지표를 붙인다. */
type GridRow = {
  rank: number;
  params: Record<string, TuningValue>;
  total_pct: number;
  cagr_pct: number | null;
  mdd_pct: number;
  sortino: number | null;
  trade_count?: number;
  win_rate_pct?: number;
  avg_win_pct?: number | null;
  avg_loss_pct?: number | null;
  quarter_wins: number;
  is_current: boolean;
} & Record<string, unknown>;

function fmtPct(value: number | null | undefined, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

/** 축 범위 표기 — "5 ~ 10 (6개)" 처럼 시작·끝·개수를 한 형식으로. 값이 하나면 그 값만. */
function rangeLabel(axis: TuningAxis): string {
  const labels = axis.values.map((v) => v.label);
  if (labels.length === 1) return labels[0];
  return `${labels[0]} ~ ${labels[labels.length - 1]} (${labels.length}개)`;
}

/** 전략 튜닝 섹션 — 고정된 범위의 조합 전체를 백테스트해 비교한다 (항상 펼쳐져 있다).
 *
 *  모멘텀·신고가 화면이 같은 컴포넌트를 쓴다. 기간 셀렉트와 실행 버튼을 백테스트 카드와
 *  같은 자리(헤더 오른쪽)에 두고, 실행은 부모가 넘긴 `run` 으로 한다.
 *  결과 표의 컬럼 순서는 `axes` 순서(= 화면 설정 항목 순서)를 따른다.
 *  `current` 와 같은 조합 행은 굵게 표시해 지금 설정이 어디쯤인지 보여준다.
 *  진행도는 백테스트와 같은 램프(조합 수 × 조합당 예상 초)로 보여준다.
 *  결과는 이 컴포넌트 내부 state 라 부모가 비울 수 없다 — 종목풀처럼 결과의 전제가 바뀌는 값은
 *  호출부가 `key` 로 넘겨 재마운트시켜 비운다(백테스트 결과를 비우는 것과 같은 시점). */
export function StrategyTuning({
  axes,
  monthOptions,
  defaultMonths,
  fixedLabel,
  current,
  run,
  cancelRun,
  onApply,
  disabled,
  disabledHint,
}: {
  axes: TuningAxis[];
  /** 기간 선택지 — 백테스트와 같은 목록을 받는다. */
  monthOptions: number[];
  defaultMonths: number;
  /** 고정되는 나머지 설정 요약 — 예: "저장된 설정 기준 (종목풀 us)". */
  fixedLabel: string;
  current: Record<string, TuningValue>;
  /**
   * 튜닝 실행 — 부모가 자기 경로로 POST 하고 **Response 를 그대로** 돌려준다.
   * 응답은 SSE 스트림이라(진행 이벤트 + 결과 이벤트) 여기서 줄 단위로 읽는다.
   */
  run: (months: number, ranges: Record<string, TuningValue[]>, signal: AbortSignal) => Promise<Response>;
  /** 서버에서 현재 튜닝의 프로세스 풀까지 종료한다. */
  cancelRun: () => Promise<Response>;
  /** 행의 조합을 상단 설정에 넣고 저장까지 한다 — 부모가 자기 저장 흐름으로 처리한다. */
  onApply: (params: Record<string, TuningValue>) => Promise<void> | void;
  /** 실행 차단 — 부모가 **백테스트 버튼과 같은 조건**을 넘긴다(두 실행의 기준이 갈리면 안 된다). */
  disabled?: boolean;
  /** 차단 사유 — 백테스트 헤더와 같은 문구를 같은 자리에 보여준다. 없으면 표시하지 않는다. */
  disabledHint?: string;
}) {
  const toast = useToast();
  const [months, setMonths] = useState(defaultMonths);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<LoadingProgress | null>(null);
  const [result, setResult] = useState<TuningResult | null>(null);
  const [applying, setApplying] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const cancelRequestedRef = useRef(false);
  const mountedRef = useRef(true);

  useEffect(
    () => () => {
      mountedRef.current = false;
      abortRef.current?.abort();
    },
    [],
  );

  const apply = useCallback(
    async (params: Record<string, TuningValue>) => {
      setApplying(true);
      try {
        await onApply(params);
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "설정을 적용하지 못했습니다.");
      } finally {
        setApplying(false);
      }
    },
    [onApply, toast],
  );

  const comboCount = useMemo(() => axes.reduce((n, axis) => n * axis.values.length, 1), [axes]);
  // 결과가 도착하면 계산은 끝난 것이다. 비동기 정리 상태가 남더라도 화면을 잠그지 않는다.
  const busy = running && result == null;

  const cancel = useCallback(async () => {
    const controller = abortRef.current;
    if (!controller) return;
    cancelRequestedRef.current = true;
    setProgress((prev) => (prev ? { ...prev, message: "튜닝 중단 중…" } : prev));
    try {
      const response = await cancelRun();
      const payload = (await response.json().catch(() => ({}))) as { message?: string; error?: string };
      if (!response.ok) throw new Error(payload.error || `튜닝 중단에 실패했습니다. (${response.status})`);
      controller.abort();
      if (abortRef.current === controller) abortRef.current = null;
      if (mountedRef.current) {
        setRunning(false);
        setProgress(null);
      }
      toast.success(payload.message || "튜닝을 중단했습니다.");
    } catch (error) {
      cancelRequestedRef.current = false;
      setProgress((prev) => (prev ? { ...prev, message: "튜닝 중단 실패 — 백테스트 계속 실행 중" } : prev));
      toast.error(error instanceof Error ? error.message : "튜닝을 중단하지 못했습니다.");
    }
  }, [cancelRun, toast]);

  const execute = useCallback(async () => {
    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);
    setResult(null);
    // 첫 묶음이 끝날 때까지는 진행을 **올리지 않는다**. 예전에는 예상 시간에 맞춘 램프를
    // 돌렸는데, 예상이 빗나가면 90% 에 멈춰 있다가 실제 진행(1/12 = 8%)이 도착하는 순간
    // 바가 뒤로 갔다. 모르는 구간은 모른다고 두고, 무엇을 기다리는지만 정확히 적는다.
    setProgress({ percent: 0, message: `1/2 준비 중 — ${comboCount}조합의 가격·후보를 읽는 중` });

    try {
      const ranges = Object.fromEntries(axes.map((axis) => [axis.key, axis.values.map((v) => v.value)]));
      const response = await run(months, ranges, controller.signal);
      if (!response.ok) {
        const message = await response.text().catch(() => "");
        throw new Error(message.trim() || `튜닝 요청에 실패했습니다. (${response.status})`);
      }
      if (!response.body) throw new Error("튜닝 응답을 받지 못했습니다.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let payload: TuningResult | null = null;
      let cancelledMessage: string | null = null;

      // SSE — 이벤트 하나가 `data: {...}` 한 줄이다. 주석(`:` 시작)은 연결 확인용이라 버린다.
      const handleLine = (line: string) => {
        const text = line.trim();
        if (!text || text.startsWith(":")) return;
        if (!text.startsWith("data:")) return;
        const event = JSON.parse(text.slice("data:".length).trim()) as
          | { type: "progress"; phase?: "prepare" | "backtest" | "finalize"; done: number; total: number }
          | { type: "result"; payload: TuningResult }
          | { type: "notice" | "cancelled"; message: string }
          | { type: "error"; message: string };
        if (event.type === "progress") {
          const total = Math.max(1, event.total);
          const done = Math.max(0, Math.min(total, event.done));
          if (event.phase === "prepare") {
            setProgress({ percent: 0, message: `1/2 준비 중 — ${event.total}조합의 가격·후보를 읽는 중` });
          } else if (event.phase === "finalize") {
            setProgress({ percent: 98, message: `2/2 백테스트 완료 — ${event.total}조합 결과 집계 중` });
          } else {
            // 마지막 집계·전송이 남아 있어 백테스트 단계는 95% 를 넘기지 않는다.
            setProgress({
              percent: Math.min(95, (done / total) * 95),
              message:
                done === 0
                  ? `2/2 백테스트 중 — 첫 결과를 기다리는 중 (0/${event.total}조합)`
                  : `2/2 백테스트 중 — ${done}/${event.total}조합 완료`,
            });
          }
        } else if (event.type === "result") {
          payload = event.payload;
        } else if (event.type === "notice") {
          toast.warning(event.message);
        } else if (event.type === "cancelled") {
          cancelledMessage = event.message;
        } else {
          throw new Error(event.message);
        }
      };

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let cut = buffer.indexOf("\n");
        while (cut >= 0) {
          handleLine(buffer.slice(0, cut));
          buffer = buffer.slice(cut + 1);
          cut = buffer.indexOf("\n");
        }
      }
      handleLine(buffer);

      if (cancelledMessage) {
        if (!cancelRequestedRef.current) toast.warning(cancelledMessage);
        return;
      }
      if (!payload) throw new Error("튜닝 결과를 받지 못했습니다.");
      const resultPayload = payload as TuningResult;
      // 결과가 도착한 시점에 서버 작업은 끝났다. 그리드 렌더링을 기다리며 실행 상태를
      // 붙잡아 두면 완료 후에도 진행 바·중단 버튼·기간 잠금이 남는다.
      setResult(resultPayload);
      if (abortRef.current === controller) abortRef.current = null;
      setRunning(false);
      setProgress(null);
      const skipped = resultPayload.skipped;
      if (skipped?.length) toast.warning(`일부 조합을 건너뛰었습니다: ${skipped[0]}`);
    } catch (error) {
      if (!controller.signal.aborted) {
        toast.error(error instanceof Error ? error.message : "튜닝에 실패했습니다.");
      }
    } finally {
      cancelRequestedRef.current = false;
      const isCurrentRequest = abortRef.current === controller;
      if (isCurrentRequest) abortRef.current = null;
      // 중단 직후 새 튜닝이 시작됐다면 이전 요청의 정리가 새 진행 상태를 지우면 안 된다.
      if (mountedRef.current && isCurrentRequest) {
        setRunning(false);
        setProgress(null);
      }
    }
  }, [axes, comboCount, months, run, toast]);

  const labelOf = (axis: TuningAxis, value: TuningValue) =>
    axis.values.find((o) => o.value === value)?.label ?? String(value ?? "없음");
  const isCurrent = (row: TuningRow) => axes.every((axis) => row.params[axis.key] === current[axis.key]);

  const gridRows = useMemo<GridRow[]>(
    () =>
      (result?.rows ?? []).map((row, index) => ({
        rank: index + 1,
        params: row.params,
        ...Object.fromEntries(axes.map((axis) => [`p_${axis.key}`, labelOf(axis, row.params[axis.key])])),
        total_pct: row.total_pct,
        cagr_pct: row.cagr_pct,
        mdd_pct: row.mdd_pct,
        sortino: row.sortino,
        trade_count: row.trade_count,
        win_rate_pct: row.win_rate_pct,
        avg_win_pct: row.avg_win_pct,
        avg_loss_pct: row.avg_loss_pct,
        quarter_wins: row.quarter_wins,
        is_current: isCurrent(row),
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [result, axes, current],
  );

  const gridColumns = useMemo<ColDef<GridRow>[]>(() => {
    const num = (digits: number, signed = false, suffix = "") => (p: { value?: number | null }) =>
      p.value == null || !Number.isFinite(p.value) ? "-" : `${signed && p.value >= 0 ? "+" : ""}${p.value.toFixed(digits)}${suffix}`;
    const right: ColDef<GridRow> = { type: "rightAligned", sortable: true, resizable: true };
    const columns: ColDef<GridRow>[] = [
      { headerName: "#", field: "rank", width: 64, ...right },
      ...axes.map<ColDef<GridRow>>((axis) => ({
        headerName: axis.label,
        field: `p_${axis.key}`,
        width: 110,
        ...right,
        // 라벨이 아니라 원래 값으로 정렬한다(숫자 축은 숫자 순).
        comparator: (_a, _b, nodeA, nodeB) => {
          const va = nodeA.data?.params[axis.key];
          const vb = nodeB.data?.params[axis.key];
          const ia = axis.values.findIndex((o) => o.value === va);
          const ib = axis.values.findIndex((o) => o.value === vb);
          return ia - ib;
        },
      })),
      { headerName: `${result?.months ?? ""}개월 수익`, field: "total_pct", width: 120, valueFormatter: num(1, true, "%"), ...right },
      { headerName: "CAGR", field: "cagr_pct", width: 96, valueFormatter: num(1, true, "%"), ...right },
      { headerName: "MDD", field: "mdd_pct", width: 90, valueFormatter: num(1, false, "%"), ...right },
      { headerName: "소르티노", field: "sortino", width: 96, valueFormatter: num(2), ...right },
    ];
    columns.push(
      // 거래·승률은 두 전략 모두 항상 내려준다. 예전에는 첫 행에 값이 있을 때만 컬럼을 만들었는데,
      // 그러면 1등 조합이 거래 0건(승률 null)이면 나머지 행의 값까지 통째로 사라졌다.
      { headerName: "거래", field: "trade_count", width: 80, ...right },
      { headerName: "승률", field: "win_rate_pct", width: 84, valueFormatter: num(0, false, "%"), ...right },
      // 평균이익·평균손실 — 승률 옆에 둬야 손익비를 한눈에 견준다(승률이 높아도 손익비가
      // 나쁘면 못 쓰는 조합이다). 백엔드 공용 계산(utils/trade_stats.py)이 단일 소스다.
      {
        headerName: "평균이익",
        field: "avg_win_pct",
        width: 96,
        valueFormatter: num(1, true, "%"),
        ...right,
        cellStyle: (p) => ({ color: signColor(p.value as number) }),
      },
      {
        headerName: "평균손실",
        field: "avg_loss_pct",
        width: 96,
        valueFormatter: num(1, true, "%"),
        ...right,
        cellStyle: (p) => ({ color: signColor(p.value as number) }),
      },
      {
        headerName: "분기승수",
        field: "quarter_wins",
        width: 96,
        ...right,
        valueFormatter: (p) => `${p.value}/${result?.quarter_count ?? 0}`,
      },
      {
        headerName: "",
        field: "rank",
        width: 88,
        sortable: false,
        cellRenderer: (p: { data?: GridRow }) =>
          p.data ? (
            <button
              type="button"
              className="btn btn-sm btn-outline-dark"
              style={{ padding: "0 8px", lineHeight: 1.6 }}
              disabled={disabled || busy || applying || p.data.is_current}
              onClick={() => void apply(p.data!.params)}
            >
              {p.data.is_current ? "현재" : "적용"}
            </button>
          ) : null,
      },
    );
    return columns;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [axes, result, disabled, busy, applying]);

  return (
    <section className="appSection">
      <div className="card appCard">
        {/* 백테스트 섹션과 같은 헤더 구조(appMainHeader) — 셀렉트·버튼 크기가 그쪽과 맞춰진다. */}
        <div className="card-body" style={{ paddingBottom: 0 }}>
          <div className="appMainHeader">
            <div className="appMainHeaderLeft">
              <span style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>
                튜닝
                <span style={{ ...hint, fontWeight: 500, marginLeft: 8 }}>아래 범위의 조합을 전부 백테스트해 비교 · {comboCount}조합</span>
              </span>
            </div>
            <div className="appMainHeaderRight">
              {disabledHint && !running ? <span style={hint}>{disabledHint}</span> : null}
              <MonthsSelect value={months} options={monthOptions} disabled={busy} onChange={setMonths} />
              {busy ? (
                <button type="button" className="btn btn-sm btn-outline-danger" onClick={cancel}>
                  중단
                </button>
              ) : (
                <button type="button" className="btn btn-sm btn-dark" disabled={disabled} onClick={() => void execute()}>
                  실행
                </button>
              )}
            </div>
          </div>
        </div>
        <div className="card-body appCardBodyTight" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {/* 축별 범위 — 고정. 어디서 어디까지 돌리는지만 보여준다. */}
            <div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: "var(--fs-sm)" }}>
              {axes.map((axis) => (
                <span key={axis.key}>
                  <b>{axis.label}</b> {rangeLabel(axis)}
                </span>
              ))}
            </div>
            <div style={hint}>나머지 설정은 고정 — {fixedLabel}</div>

            {busy ? (
              <AppLoadingProgress title="튜닝 실행 중…" progress={progress} />
            ) : !result ? (
              <div style={{ ...hint, textAlign: "center", padding: "14px 0" }}>실행을 누르면 결과가 표시됩니다.</div>
            ) : (
              <>
                {/* 축별 평균 — 어느 값이 일관되게 나은지. */}
                <div style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: "var(--fs-sm)" }}>
                  <div style={{ fontWeight: 700 }}>항목별 평균 (소르티노 / 수익 / MDD)</div>
                  {axes.map((axis) => {
                    const entries = result.axes[axis.key] ?? [];
                    const best = Math.max(...entries.map((e) => e.sortino));
                    return (
                      <div key={axis.key} style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
                        <span style={{ fontWeight: 700, minWidth: 96 }}>{axis.label}</span>
                        {entries.map((entry) => (
                          <span key={String(entry.value)} style={{ fontWeight: entry.sortino === best ? 700 : 400 }}>
                            {labelOf(axis, entry.value)}: {entry.sortino.toFixed(2)} / {fmtPct(entry.total_pct, 0)} /{" "}
                            {entry.mdd_pct.toFixed(0)}%
                          </span>
                        ))}
                      </div>
                    );
                  })}
                </div>

                {/* 조합 그리드 — 컬럼 헤더로 직접 정렬할 수 있다. 현재 설정 행은 녹색(레버리지 튜닝과 동일), 맨 오른쪽 '적용'. */}
                <AppAgGrid<GridRow>
                  rowData={gridRows}
                  columnDefs={gridColumns}
                  theme={gridTheme}
                  minHeight={320}
                  height="34rem"
                  getRowClass={(p) => (p.data?.is_current ? "appHeldRow" : "")}
                  // 결과 행은 수백 개라 고정 높이에서 가상화한다. autoHeight 는 모든 셀을 한꺼번에
                  // 만들어 완료 화면과 중단 버튼을 수 초 동안 멈추게 했다.
                  gridOptions={{
                    suppressMovableColumns: true,
                    rowHeight: 34,
                  }}
                />
                <div style={hint}>
                  기본 소르티노 내림차순(헤더를 눌러 정렬) · CAGR = 기간 수익의 연환산 · 평균이익·평균손실 = 이긴/진
                  거래의 평균 손익률(청산분만) · 분기승수 = 그 분기에 조합들 중 상위 절반에 든 횟수(구간 일관성) ·
                  적용 = 그 조합을 상단 설정에 넣고 저장
                </div>
              </>
            )}
        </div>
      </div>
    </section>
  );
}
