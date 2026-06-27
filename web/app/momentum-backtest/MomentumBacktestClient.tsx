"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";

type BacktestStatus = {
  queue_status?: string | null;
  running?: boolean;
  exit_code?: number | null;
  error?: string | null;
  triggered_at?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  files?: string[];
  selected_file?: string | null;
  log_text?: string;
  log_file?: string | null;
  done?: boolean;
  progress_pct?: number | null;
  completed?: number | null;
  total?: number | null;
};

type FileEntry = { file: string; pool: string; date: string; order: number };

/** `<order>_<pool>-backtest_<date>.log` → {pool, date, order}. */
function parseFile(name: string): FileEntry | null {
  const m = name.match(/^(.*)-backtest_(\d{4}-\d{2}-\d{2})\.log$/);
  if (!m) return null;
  const prefix = m[1];
  const om = prefix.match(/^(\d+)_(.*)$/);
  return {
    file: name,
    pool: om ? om[2] : prefix,
    date: m[2],
    order: om ? parseInt(om[1], 10) : 999,
  };
}

export function MomentumBacktestClient() {
  const toast = useToast();
  const [bt, setBt] = useState<BacktestStatus | null>(null);
  const [files, setFiles] = useState<string[]>([]);
  const [selectedPool, setSelectedPool] = useState<string>("");
  const [selectedDate, setSelectedDate] = useState<string>("");
  const [triggering, setTriggering] = useState(false);
  const [mounted, setMounted] = useState(false);

  const refetch = useCallback(async (f: string | null) => {
    try {
      const qs = f ? `?file=${encodeURIComponent(f)}` : "";
      const resp = await fetch(`/api/momentum-backtest/status${qs}`, { cache: "no-store" });
      const data = (await resp.json()) as BacktestStatus & { error?: string };
      if (resp.ok && !data.error) {
        setBt(data);
        if (Array.isArray(data.files)) setFiles(data.files);
      }
    } catch {
      /* 폴링 실패는 무시 */
    }
  }, []);

  const entries = useMemo(
    () => files.map(parseFile).filter((e): e is FileEntry => e !== null),
    [files],
  );
  const pools = useMemo(() => {
    const order = new Map<string, number>();
    for (const e of entries) if (!order.has(e.pool)) order.set(e.pool, e.order);
    return [...order.entries()].sort((a, b) => a[1] - b[1]).map(([p]) => p);
  }, [entries]);
  const datesFor = useCallback(
    (pool: string) => [...new Set(entries.filter((e) => e.pool === pool).map((e) => e.date))].sort((a, b) => b.localeCompare(a)),
    [entries],
  );
  const viewFile = useMemo(
    () => entries.find((e) => e.pool === selectedPool && e.date === selectedDate)?.file ?? null,
    [entries, selectedPool, selectedDate],
  );

  useEffect(() => {
    setMounted(true);
    void refetch(null);
  }, [refetch]);

  // 파일 목록이 바뀌면 선택을 유효하게 정규화(없으면 최신 파일의 풀/날짜로).
  useEffect(() => {
    if (entries.length === 0) return;
    const pool = selectedPool && pools.includes(selectedPool) ? selectedPool : entries[0].pool;
    const dates = datesFor(pool);
    const date = selectedDate && dates.includes(selectedDate) ? selectedDate : dates[0];
    if (pool !== selectedPool) setSelectedPool(pool);
    if (date !== selectedDate) setSelectedDate(date ?? "");
  }, [entries, pools, datesFor, selectedPool, selectedDate]);

  // 선택 파일이 정해지면 그 내용을 불러온다.
  useEffect(() => {
    if (viewFile) void refetch(viewFile);
  }, [viewFile, refetch]);

  const running = !!bt?.running;
  const isLatest = files.length > 0 && viewFile === files[0];

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => void refetch(viewFile), 2500);
    return () => clearInterval(id);
  }, [running, viewFile, refetch]);

  const onPoolChange = (p: string) => {
    setSelectedPool(p);
    setSelectedDate(datesFor(p)[0] ?? "");
  };

  const start = async () => {
    setTriggering(true);
    try {
      const resp = await fetch("/api/momentum-backtest", { method: "POST" });
      const data = (await resp.json()) as { enqueued?: boolean; reason?: string; error?: string };
      if (!resp.ok || data.error) {
        toast.error(data.error ?? "백테스트를 시작하지 못했습니다.");
        return;
      }
      toast.success(data.enqueued ? "백테스트를 시작했습니다. 워커가 순서대로 실행합니다." : (data.reason ?? "이미 진행 중입니다."));
      // 최신(새 실행) 따라가도록 선택 초기화 → 정규화가 최신 파일로 맞춤
      setSelectedPool("");
      setSelectedDate("");
      await refetch(null);
    } finally {
      setTriggering(false);
    }
  };

  if (!mounted) {
    return (
      <PageFrame title="모멘텀 백테스트">
        <div className="appPageStack" style={{ maxWidth: 1000 }}>
          <div style={{ color: "#868e96", padding: 20 }}>불러오는 중…</div>
        </div>
      </PageFrame>
    );
  }

  const pct = bt?.progress_pct ?? null;
  const statusLabel = bt?.queue_status === "running" ? "실행 중"
    : bt?.queue_status === "pending" ? "대기 중(워커 시작 대기)"
    : bt?.queue_status === "done" ? (bt?.exit_code === 0 ? "완료" : `실패 (exit ${bt?.exit_code})`)
    : bt?.queue_status === "failed" ? "실패"
    : "대기";
  const selectStyle: React.CSSProperties = { border: "1px solid rgba(148,163,184,0.4)", borderRadius: 6, padding: "5px 8px", fontSize: "0.85rem" };

  return (
    <PageFrame title="모멘텀 백테스트">
      <div className="appPageStack" style={{ maxWidth: 1000 }}>
        <div className="card appCard">
          <div className="card-body">
            <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 8 }}>백테스트 실행</h2>
            <p style={{ color: "#94a3b8", fontSize: "0.85rem", marginBottom: 12 }}>
              탐색공간은 모멘텀-설정에서 편집합니다. 버튼을 누르면 배치 큐에 등록되고 워커가 모든 종목풀을 전수 탐색합니다(시간이 걸립니다). 결과는 풀별 로그 파일로 저장됩니다.
            </p>
            <button type="button" className="btn btn-dark" disabled={triggering || running} onClick={() => void start()} style={{ minWidth: 200 }}>
              {running ? "백테스트 진행 중…" : triggering ? "시작 중…" : "백테스트 시작"}
            </button>
            <div style={{ marginTop: 12, fontSize: "0.85rem", color: "#475569" }}>
              상태: <b>{statusLabel}</b>
              {bt?.started_at ? <span style={{ color: "#94a3b8", marginLeft: 12 }}>시작: {new Date(bt.started_at).toLocaleString("ko-KR")}</span> : null}
              {bt?.ended_at ? <span style={{ color: "#94a3b8", marginLeft: 12 }}>종료: {new Date(bt.ended_at).toLocaleString("ko-KR")}</span> : null}
            </div>
          </div>
        </div>

        <div className="card appCard">
          <div className="card-body">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8, marginBottom: 12 }}>
              <h2 style={{ fontSize: "1.05rem", fontWeight: 800, margin: 0 }}>진행도 / 결과</h2>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ color: "#64748b", fontSize: "0.85rem", fontWeight: 600 }}>종목풀</span>
                <select style={selectStyle} value={selectedPool} onChange={(e) => onPoolChange(e.target.value)} disabled={pools.length === 0}>
                  {pools.length === 0 ? <option value="">결과 없음</option> : null}
                  {pools.map((p) => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
                <span style={{ color: "#64748b", fontSize: "0.85rem", fontWeight: 600 }}>날짜</span>
                <select style={selectStyle} value={selectedDate} onChange={(e) => setSelectedDate(e.target.value)} disabled={!selectedPool}>
                  {datesFor(selectedPool).map((d, i) => (
                    <option key={d} value={d}>{d}{i === 0 ? " (최신)" : ""}</option>
                  ))}
                </select>
                {running && isLatest ? <span style={{ color: "#2563eb", fontSize: "0.8rem" }}>● 실시간</span> : null}
              </div>
            </div>
            {pct != null ? (
              <div style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", color: "#475569", marginBottom: 4 }}>
                  <span>{bt?.done ? "완료" : "진행률"}{bt?.completed != null && bt?.total != null ? ` (${bt.completed}/${bt.total})` : ""}</span>
                  <span>{pct.toFixed(1)}%</span>
                </div>
                <div style={{ height: 10, background: "rgba(148,163,184,0.25)", borderRadius: 6, overflow: "hidden" }}>
                  <div style={{ width: `${Math.min(100, Math.max(0, pct))}%`, height: "100%", background: bt?.done ? "#16a34a" : "#2563eb", transition: "width 0.4s" }} />
                </div>
              </div>
            ) : null}
            {bt?.error ? <div className="alert alert-danger">{bt.error}</div> : null}
            {bt?.log_text ? (
              <pre style={{ background: "#0f172a", color: "#e2e8f0", padding: 14, borderRadius: 8, fontSize: "0.78rem", lineHeight: 1.5, whiteSpace: "pre", overflowX: "auto" }}>
                {bt.log_text}
              </pre>
            ) : (
              <div style={{ color: "#94a3b8", padding: 12 }}>
                {pools.length === 0 ? "아직 백테스트 결과가 없습니다. \"백테스트 시작\"을 눌러 실행하세요." : "선택한 결과가 비어 있습니다."}
              </div>
            )}
          </div>
        </div>
      </div>
    </PageFrame>
  );
}
