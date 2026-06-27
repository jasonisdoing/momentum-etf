"use client";

import { useCallback, useEffect, useState } from "react";

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

export function MomentumBacktestClient() {
  const toast = useToast();
  const [bt, setBt] = useState<BacktestStatus | null>(null);
  const [files, setFiles] = useState<string[]>([]);
  const [viewFile, setViewFile] = useState<string | null>(null); // null = 최신 따라가기
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

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    void refetch(viewFile);
  }, [viewFile, refetch]);

  const isLatest = viewFile === null || (files.length > 0 && viewFile === files[0]);

  useEffect(() => {
    if (!bt?.running || !isLatest) return;
    const id = setInterval(() => void refetch(viewFile), 2500);
    return () => clearInterval(id);
  }, [bt?.running, isLatest, viewFile, refetch]);

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
      setViewFile(null);
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

  const running = !!bt?.running;
  const pct = bt?.progress_pct ?? null;
  const statusLabel = bt?.queue_status === "running" ? "실행 중"
    : bt?.queue_status === "pending" ? "대기 중(워커 시작 대기)"
    : bt?.queue_status === "done" ? (bt?.exit_code === 0 ? "완료" : `실패 (exit ${bt?.exit_code})`)
    : bt?.queue_status === "failed" ? "실패"
    : "대기";
  const selectedFile = viewFile ?? bt?.selected_file ?? "";

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
                <span style={{ color: "#64748b", fontSize: "0.85rem", fontWeight: 600 }}>결과 파일</span>
                <select
                  style={{ border: "1px solid rgba(148,163,184,0.4)", borderRadius: 6, padding: "5px 8px", fontSize: "0.85rem", maxWidth: 320 }}
                  value={selectedFile}
                  onChange={(e) => setViewFile(e.target.value || null)}
                  disabled={files.length === 0}
                >
                  {files.length === 0 ? <option value="">결과 없음</option> : null}
                  {files.map((f) => (
                    <option key={f} value={f}>{f}{f === files[0] ? " (최신)" : ""}</option>
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
                {files.length === 0 ? "아직 백테스트 결과가 없습니다. \"백테스트 시작\"을 눌러 실행하세요." : "선택한 파일이 비어 있습니다."}
              </div>
            )}
          </div>
        </div>
      </div>
    </PageFrame>
  );
}
