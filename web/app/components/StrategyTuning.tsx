"use client";

import type { ColDef } from "ag-grid-community";
import { useCallback, useMemo, useState } from "react";

import { AppAgGrid } from "./AppAgGrid";
import { MonthsSelect } from "./MonthsSelect";
import { AppLoadingProgress, type LoadingProgress } from "./AppLoadingProgress";
import { useToast } from "./ToastProvider";
import { createAppGridTheme } from "./app-grid-theme";

/** 축 값 — 숫자 외에 "off"/"none" 같은 상태값도 쓴다(모멘텀 주중 손절선). */
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
 *  진행도는 백테스트와 같은 램프(조합 수 × 조합당 예상 초)로 보여준다. */
export function StrategyTuning({
  axes,
  monthOptions,
  defaultMonths,
  fixedLabel,
  current,
  secondsPerCombo,
  extraSeconds = 0,
  run,
  onApply,
  disabled,
}: {
  axes: TuningAxis[];
  /** 기간 선택지 — 백테스트와 같은 목록을 받는다. */
  monthOptions: number[];
  defaultMonths: number;
  /** 고정되는 나머지 설정 요약 — 예: "저장된 설정 기준 (종목풀 us)". */
  fixedLabel: string;
  current: Record<string, TuningValue>;
  /** 조합 하나당 예상 소요 초 — 진행도 램프 속도에 쓴다. */
  secondsPerCombo: number;
  /** 조합과 무관한 준비 시간(초) — 가격 로드·후보 계산 등. */
  extraSeconds?: number;
  run: (months: number, ranges: Record<string, TuningValue[]>) => Promise<TuningResult>;
  /** 행의 조합을 상단 설정에 넣고 저장까지 한다 — 부모가 자기 저장 흐름으로 처리한다. */
  onApply: (params: Record<string, TuningValue>) => Promise<void> | void;
  disabled?: boolean;
}) {
  const toast = useToast();
  const [months, setMonths] = useState(defaultMonths);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<LoadingProgress | null>(null);
  const [result, setResult] = useState<TuningResult | null>(null);
  const [applying, setApplying] = useState(false);

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
  // 램프 속도 기준 — 조합 수 × 조합당 초 + 준비 시간. 화면에는 예상 시간을 보여주지 않는다(잘 안 맞는다).
  const expectedSeconds = Math.max(10, Math.round(comboCount * secondsPerCombo + extraSeconds));

  const execute = useCallback(async () => {
    setRunning(true);
    setResult(null);
    setProgress({ percent: 0, message: `${comboCount}조합 백테스트 중` });
    // 서버가 단일 응답이라 실제 단계를 모른다 — 예상 시간에 맞춰 90%까지 올리는 램프
    // (내부는 소수로 누적, 표시는 AppLoadingProgress 가 정수로 반올림).
    const stepPercent = Math.max(0.1, 88 / expectedSeconds);
    const timer = window.setInterval(() => {
      setProgress((prev) => (prev ? { ...prev, percent: Math.min(90, prev.percent + stepPercent) } : prev));
    }, 1000);
    const stopRamp = () => window.clearInterval(timer);
    try {
      const ranges = Object.fromEntries(axes.map((axis) => [axis.key, axis.values.map((v) => v.value)]));
      const payload = await run(months, ranges);
      setProgress({ percent: 100, message: "결과 반영 중" });
      setResult(payload);
      if (payload.skipped?.length) toast.warning(`일부 조합을 건너뛰었습니다: ${payload.skipped[0]}`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "튜닝에 실패했습니다.");
    } finally {
      stopRamp();
      setRunning(false);
      setProgress(null);
    }
  }, [axes, comboCount, expectedSeconds, months, run, toast]);

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
    if (result?.rows[0]?.trade_count != null) columns.push({ headerName: "거래", field: "trade_count", width: 80, ...right });
    if (result?.rows[0]?.win_rate_pct != null)
      columns.push({ headerName: "승률", field: "win_rate_pct", width: 84, valueFormatter: num(0, false, "%"), ...right });
    columns.push(
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
              disabled={disabled || running || applying || p.data.is_current}
              onClick={() => void apply(p.data!.params)}
            >
              {p.data.is_current ? "현재" : "적용"}
            </button>
          ) : null,
      },
    );
    return columns;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [axes, result, disabled, running, applying]);

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
              <MonthsSelect value={months} options={monthOptions} disabled={running} onChange={setMonths} />
              <button type="button" className="btn btn-sm btn-dark" disabled={disabled || running} onClick={() => void execute()}>
                {running ? "실행 중…" : "실행"}
              </button>
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

            {running ? (
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
                  minHeight={0}
                  height="auto"
                  getRowClass={(p) => (p.data?.is_current ? "appHeldRow" : "")}
                  // 백테스트 표와 같이 행 수만큼 늘어난다 — 표 안 스크롤 없이 페이지 스크롤로 본다.
                  gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true, rowHeight: 34 }}
                />
                <div style={hint}>
                  기본 소르티노 내림차순(헤더를 눌러 정렬) · CAGR = 기간 수익의 연환산 · 분기승수 = 그 분기에 조합들 중 상위
                  절반에 든 횟수(구간 일관성) · 적용 = 그 조합을 상단 설정에 넣고 저장
                </div>
              </>
            )}
        </div>
      </div>
    </section>
  );
}
