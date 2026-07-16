"use client";

import { useCallback, useEffect, useState } from "react";

import { formatKstDateTime } from "@/lib/datetime";
import { useToast } from "../components/ToastProvider";

type Benchmark = { ticker?: string; name?: string };
type BtConfig = {
  SORT_METRIC?: string;
  BACKTEST_MONTHS?: number;
  BENCHMARK?: Benchmark;
  TOP_N_HOLD?: number[];
  MA_MONTHS?: number[];
};

type PoolEntry = {
  pool_id: string;
  name: string;
  config: BtConfig;
  live_top_n_hold?: number | null;
  updated_at?: string | null;
};
type ApiResponse = { pools?: PoolEntry[]; constraints?: Record<string, never>; error?: string };

const inputStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "4px 8px",
  fontSize: "0.88rem",
};
const labelStyle: React.CSSProperties = { color: "var(--text-muted)", fontWeight: 600, fontSize: "0.83rem", flexShrink: 0 };

function parseNums(text: string): number[] {
  return text.split(/[,\s]+/).map((t) => t.trim()).filter(Boolean).map(Number).filter((n) => Number.isFinite(n));
}

/** 풀 1개 백테스트 탐색공간 인라인 편집 행 (자체 저장). */
function PoolRow({ pool }: { pool: PoolEntry }) {
  const toast = useToast();
  const c = pool.config;
  const [sortMetric, setSortMetric] = useState((c.SORT_METRIC ?? "CAGR").toUpperCase());
  const [backtestMonths, setBacktestMonths] = useState(c.BACKTEST_MONTHS != null ? String(c.BACKTEST_MONTHS) : "");
  const [benchTicker, setBenchTicker] = useState(c.BENCHMARK?.ticker ?? "");
  const [benchName, setBenchName] = useState(c.BENCHMARK?.name ?? "");
  // 저장된 벤치마크가 있으면 잠금 상태로 시작 — [변경] 을 눌러야 편집 가능.
  const [benchEditing, setBenchEditing] = useState(!(c.BENCHMARK?.ticker && c.BENCHMARK?.name));
  // TOP_N_HOLD 초기값: 저장된 리스트가 없으면 종목풀 설정의 라이브 N 으로 시작.
  const [topNText, setTopNText] = useState(
    (c.TOP_N_HOLD ?? (pool.live_top_n_hold != null ? [pool.live_top_n_hold] : [])).join(", "),
  );
  const [maMonthsSet, setMaMonthsSet] = useState<Set<number>>(new Set(c.MA_MONTHS ?? [6, 12]));
  const [updatedAt, setUpdatedAt] = useState<string | null | undefined>(pool.updated_at);
  const [saving, setSaving] = useState(false);
  const [resolving, setResolving] = useState(false);

  // 백테스트 실행 및 상태 관련 훅
  const [backtestRunning, setBacktestRunning] = useState(false);
  const [queueStatus, setQueueStatus] = useState<string | null>(null);
  const [triggeredAt, setTriggeredAt] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState<string | null>(null);
  const [endedAt, setEndedAt] = useState<string | null>(null);

  const checkStatus = useCallback(async () => {
    try {
      const resp = await fetch(`/api/momentum-backtest/status?pool_id=${pool.pool_id}`, { cache: "no-store" });
      if (!resp.ok) return;
      const data = await resp.json();
      setBacktestRunning(Boolean(data.running));
      setQueueStatus(data.queue_status ?? null);
      setTriggeredAt(data.triggered_at ?? null);
      setStartedAt(data.started_at ?? null);
      setEndedAt(data.ended_at ?? null);
    } catch (err) {
      // 백그라운드 폴링 실패 무시
    }
  }, [pool.pool_id]);

  useEffect(() => {
    checkStatus();
    const id = window.setInterval(checkStatus, 3000);
    return () => window.clearInterval(id);
  }, [checkStatus]);

  const handleStartBacktest = async () => {
    if (backtestRunning) return;
    if (!window.confirm(`"${pool.name}" 풀의 백테스트를 시작할까요?`)) return;
    try {
      const resp = await fetch(`/api/momentum-backtest?pool_id=${pool.pool_id}`, { method: "POST" });
      const data = await resp.json();
      if (!resp.ok || data.error) {
        toast.error(data.error ?? "백테스트 시작 실패");
      } else {
        toast.success("백테스트 작업을 큐에 추가했습니다.");
        checkStatus();
      }
    } catch (err) {
      toast.error("백테스트 시작 실패");
    }
  };

  // 벤치마크 티커 → 종목명 조회(stock_meta). 이름은 수동 편집 불가(확인으로만 채움).
  const resolveBench = async () => {
    const t = benchTicker.trim();
    if (!t) {
      toast.error("티커를 입력해주세요.");
      return;
    }
    try {
      setResolving(true);
      const resp = await fetch(`/api/leverage-config/resolve?ticker=${encodeURIComponent(t)}`);
      const data = (await resp.json()) as { name?: string; error?: string };
      if (!resp.ok || data.error || !data.name) {
        toast.error(data.error ?? "종목명을 찾을 수 없습니다.");
        return;
      }
      setBenchName(data.name);
      setBenchEditing(false);
      toast.success(`${data.name}(${t}) 확인 완료`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "티커 조회 중 오류가 발생했습니다.");
    } finally {
      setResolving(false);
    }
  };

  const topNs = parseNums(topNText);
  const months = [...maMonthsSet].sort((a, b) => a - b);
  const combos = topNs.length * months.length;

  // 백테스트기간(Month) 셀렉트 옵션 — 저장된 비표준 값은 포함해 보존.
  const monthsBase = [6, 9, 12, 15, 18, 24];
  const curMonths = Math.trunc(Number(backtestMonths));
  const monthsOptions = curMonths >= 1 && !monthsBase.includes(curMonths)
    ? [...monthsBase, curMonths].sort((a, b) => a - b)
    : monthsBase;

  const save = async () => {
    const config: BtConfig = {
      SORT_METRIC: sortMetric,
      BACKTEST_MONTHS: Math.trunc(Number(backtestMonths)),
      BENCHMARK: { ticker: benchTicker.trim(), name: benchName.trim() },
      TOP_N_HOLD: topNs.map((n) => Math.trunc(n)),
      MA_MONTHS: months.map((n) => Math.trunc(n)),
    };
    try {
      setSaving(true);
      const resp = await fetch("/api/backtest-config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pool_id: pool.pool_id, config }),
      });
      const data = (await resp.json()) as { updated_at?: string | null; error?: string; detail?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? data.detail ?? "저장에 실패했습니다.");
      setUpdatedAt(data.updated_at);
      toast.success(`[백테스트] ${pool.pool_id} 저장 완료`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "저장에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  };

  const rowStyle: React.CSSProperties = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 6 };

  return (
    <div style={{ border: "1px solid rgba(148,163,184,0.25)", borderRadius: 8, padding: "10px 12px", marginBottom: 10 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
        <span style={{ fontWeight: 800 }}>{pool.name} <span style={{ color: "var(--text-muted)", fontWeight: 500 }}>({pool.pool_id})</span></span>
        <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>마지막 저장: {updatedAt ? formatKstDateTime(updatedAt) : "저장 이력 없음"}</span>
          <button
            type="button"
            className="btn btn-sm btn-dark"
            disabled={saving || combos === 0 || !benchTicker.trim() || !benchName.trim() || !(Math.trunc(Number(backtestMonths)) >= 1)}
            onClick={() => void save()}
          >
            {saving ? "저장 중…" : "저장"}
          </button>
        </span>
      </div>

      <div style={rowStyle}>
        <span style={labelStyle}>정렬기준</span>
        <select
          className="form-select form-select-sm"
          style={{ width: 100 }}
          value={sortMetric}
          onChange={(e) => setSortMetric(e.target.value)}
        >
          <option value="CAGR">CAGR</option>
          <option value="MDD">MDD</option>
        </select>
        <span style={{ ...labelStyle, marginLeft: 8 }}>백테스트기간(Month)</span>
        <select
          className="form-select form-select-sm"
          style={{ width: 74 }}
          value={backtestMonths}
          onChange={(e) => setBacktestMonths(e.target.value)}
        >
          {backtestMonths === "" && <option value="">선택</option>}
          {monthsOptions.map((m) => (
            <option key={m} value={String(m)}>
              {m}
            </option>
          ))}
        </select>
        <span style={{ ...labelStyle, marginLeft: 8 }}>벤치마크</span>
        {benchEditing ? (
          <>
            <input
              style={{ ...inputStyle, width: 110 }}
              placeholder="티커"
              value={benchTicker}
              onChange={(e) => {
                setBenchTicker(e.target.value);
                setBenchName(""); // 티커가 바뀌면 이름 불일치 방지를 위해 초기화
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void resolveBench();
                }
              }}
            />
            <button type="button" className="btn btn-sm btn-outline-secondary" disabled={resolving} onClick={() => void resolveBench()}>
              {resolving ? "조회 중…" : "조회"}
            </button>
            <input
              style={{ ...inputStyle, width: 180, background: "#f8fafc", color: "var(--text-muted)" }}
              placeholder="이름 (티커 입력 후 조회)"
              value={benchName}
              readOnly
            />
          </>
        ) : (
          <>
            <span style={{ fontSize: "0.88rem", fontWeight: 600 }}>
              {benchName} <span style={{ color: "var(--text-muted)", fontWeight: 500 }}>({benchTicker})</span>
            </span>
            <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => setBenchEditing(true)}>
              변경
            </button>
          </>
        )}
      </div>

      <div style={rowStyle}>
        <span style={{ ...labelStyle, width: 84 }}>보유 종목수</span>
        <input style={{ ...inputStyle, width: 190 }} placeholder="3, 4, 5, 6, 7, 8, 9, 10" value={topNText} onChange={(e) => setTopNText(e.target.value)} />
      </div>

      <div style={{ ...rowStyle, marginBottom: 0 }}>
        <span style={{ ...labelStyle, width: 84 }}>추세 개월</span>
        <div style={{ display: "flex", gap: 9, flexWrap: "wrap", alignItems: "center" }}>
          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24].map((m) => (
            <label key={m} style={{ display: "flex", alignItems: "center", gap: 3, fontSize: "0.83rem", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={maMonthsSet.has(m)}
                onChange={() => {
                  setMaMonthsSet((prev) => {
                    const next = new Set(prev);
                    if (next.has(m)) next.delete(m);
                    else next.add(m);
                    return next;
                  });
                }}
              />
              {m}
            </label>
          ))}
        </div>
        <span style={{ marginLeft: "auto", fontSize: "0.82rem", color: combos > 0 ? "#475569" : "#dc2626" }}>조합수 <b>{combos.toLocaleString()}</b></span>
      </div>

      {/* 백테스트 실행 및 상태 조회 영역 */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 12, paddingTop: 12, borderTop: "1px dashed rgba(148,163,184,0.2)", fontSize: "0.83rem" }}>
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          style={{
            padding: "4px 12px",
            backgroundColor: backtestRunning ? "#64748b" : "#475569",
            color: "#ffffff",
            border: "none",
            borderRadius: 6,
            cursor: backtestRunning ? "not-allowed" : "pointer",
            fontWeight: 600,
          }}
          disabled={backtestRunning}
          onClick={handleStartBacktest}
        >
          {backtestRunning ? "백테스트 진행 중" : "백테스트 시작"}
        </button>

        <div style={{ display: "flex", gap: 12, color: "var(--text-muted)", alignItems: "center" }}>
          <span>
            상태: <b style={{ color: queueStatus === "running" ? "#2563eb" : queueStatus === "pending" ? "#d97706" : "#1e293b" }}>
              {queueStatus === "running"
                ? "실행 중"
                : queueStatus === "pending"
                ? "대기 중"
                : queueStatus === "done"
                ? "완료"
                : queueStatus === "failed"
                ? "실패"
                : "시작 대기"}
            </b>
          </span>
          {startedAt && (
            <span>시작: {formatKstDateTime(startedAt)}</span>
          )}
          {endedAt && (
            <span>종료: {formatKstDateTime(endedAt)}</span>
          )}
        </div>
      </div>
    </div>
  );
}

export function BacktestConfigSection() {
  const toast = useToast();
  const [pools, setPools] = useState<PoolEntry[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const resp = await fetch("/api/backtest-config", { cache: "no-store" });
      const data = (await resp.json()) as ApiResponse;
      if (!resp.ok || data.error) throw new Error(data.error ?? "백테스트 설정을 불러오지 못했습니다.");
      setPools(data.pools ?? []);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "백테스트 설정을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="card appCard" style={{ marginTop: 16 }}>
      <div className="card-body">
        <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>백테스트 탐색 공간</h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginBottom: 12 }}>
          모멘텀-백테스트(`python backtest/run.py`)가 풀별로 전수 탐색하는 값입니다. (라이브 적용값과 별개 — TOP_N_HOLD는 위 종목풀 설정에서 관리)
        </p>

        {loading ? (
          <div style={{ color: "var(--text-muted)", padding: 12 }}>불러오는 중…</div>
        ) : pools.length === 0 ? (
          <div style={{ color: "var(--text-muted)", padding: 12 }}>등록된 백테스트 설정이 없습니다.</div>
        ) : (
          pools.map((p) => <PoolRow key={p.pool_id} pool={p} />)
        )}
      </div>
    </div>
  );
}
