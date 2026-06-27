"use client";

import { useCallback, useEffect, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";

type TuneStatus = {
  queue_status?: string | null;
  running?: boolean;
  exit_code?: number | null;
  error?: string | null;
  triggered_at?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  log_text?: string;
  log_file?: string | null;
  done?: boolean;
  progress_pct?: number | null;
  completed?: number | null;
  total?: number | null;
};

export function LeverageTuneClient() {
  const toast = useToast();
  const [tune, setTune] = useState<TuneStatus | null>(null);
  const [triggering, setTriggering] = useState(false);
  const [mounted, setMounted] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const resp = await fetch("/api/leverage-tune/status?profile=switch", { cache: "no-store" });
      const data = (await resp.json()) as TuneStatus & { error?: string };
      if (resp.ok && !data.error) setTune(data);
    } catch {
      /* 폴링 실패는 무시(다음 주기에 재시도) */
    }
  }, []);

  useEffect(() => {
    setMounted(true);
    void fetchStatus();
  }, [fetchStatus]);

  useEffect(() => {
    if (!tune?.running) return;
    const id = setInterval(() => void fetchStatus(), 2500);
    return () => clearInterval(id);
  }, [tune?.running, fetchStatus]);

  const startTune = async () => {
    setTriggering(true);
    try {
      const resp = await fetch("/api/leverage-tune?profile=switch", { method: "POST" });
      const data = (await resp.json()) as { enqueued?: boolean; reason?: string; error?: string };
      if (!resp.ok || data.error) {
        toast.error(data.error ?? "튜닝을 시작하지 못했습니다.");
        return;
      }
      toast.success(data.enqueued ? "튜닝을 시작했습니다. 워커가 순서대로 실행합니다." : (data.reason ?? "이미 진행 중입니다."));
      await fetchStatus();
    } finally {
      setTriggering(false);
    }
  };

  if (!mounted) {
    return (
      <PageFrame title="레버리지 튜닝">
        <div className="appPageStack" style={{ maxWidth: 1000 }}>
          <div style={{ color: "#868e96", padding: 20 }}>불러오는 중…</div>
        </div>
      </PageFrame>
    );
  }

  const running = !!tune?.running;
  const pct = tune?.progress_pct ?? null;
  const statusLabel = tune?.queue_status === "running" ? "실행 중"
    : tune?.queue_status === "pending" ? "대기 중(워커 시작 대기)"
    : tune?.queue_status === "done" ? (tune?.exit_code === 0 ? "완료" : `실패 (exit ${tune?.exit_code})`)
    : tune?.queue_status === "failed" ? "실패"
    : "대기";

  return (
    <PageFrame title="레버리지 튜닝">
      <div className="appPageStack" style={{ maxWidth: 1000 }}>
        <div className="card appCard">
          <div className="card-body">
            <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 8 }}>튜닝 실행</h2>
            <p style={{ color: "#94a3b8", fontSize: "0.85rem", marginBottom: 12 }}>
              후보군·범위는 레버리지-설정에서 편집합니다. 버튼을 누르면 배치 큐에 등록되고 워커가 전수 탐색을 실행하며, 결과는 전략 설정(공격/방어/컷오프)에 자동 반영됩니다.
            </p>
            <button type="button" className="btn btn-dark" disabled={triggering || running} onClick={() => void startTune()} style={{ minWidth: 200 }}>
              {running ? "튜닝 진행 중…" : triggering ? "시작 중…" : "튜닝 시작"}
            </button>
            <div style={{ marginTop: 12, fontSize: "0.85rem", color: "#475569" }}>
              상태: <b>{statusLabel}</b>
              {tune?.started_at ? <span style={{ color: "#94a3b8", marginLeft: 12 }}>시작: {new Date(tune.started_at).toLocaleString("ko-KR")}</span> : null}
              {tune?.ended_at ? <span style={{ color: "#94a3b8", marginLeft: 12 }}>종료: {new Date(tune.ended_at).toLocaleString("ko-KR")}</span> : null}
            </div>
          </div>
        </div>

        <div className="card appCard">
          <div className="card-body">
            <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 12 }}>진행도 / 결과</h2>
            {pct != null ? (
              <div style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", color: "#475569", marginBottom: 4 }}>
                  <span>{tune?.done ? "완료" : "진행률"}{tune?.completed != null && tune?.total != null ? ` (${tune.completed}/${tune.total})` : ""}</span>
                  <span>{pct.toFixed(1)}%</span>
                </div>
                <div style={{ height: 10, background: "rgba(148,163,184,0.25)", borderRadius: 6, overflow: "hidden" }}>
                  <div style={{ width: `${Math.min(100, Math.max(0, pct))}%`, height: "100%", background: tune?.done ? "#16a34a" : "#2563eb", transition: "width 0.4s" }} />
                </div>
              </div>
            ) : null}
            {tune?.error ? <div className="alert alert-danger">{tune.error}</div> : null}
            {tune?.log_text ? (
              <pre style={{ background: "#0f172a", color: "#e2e8f0", padding: 14, borderRadius: 8, fontSize: "0.78rem", lineHeight: 1.5, whiteSpace: "pre", overflowX: "auto" }}>
                {tune.log_text}
              </pre>
            ) : (
              <div style={{ color: "#94a3b8", padding: 12 }}>아직 튜닝 결과가 없습니다. "튜닝 시작"을 눌러 실행하세요.</div>
            )}
            {tune?.log_file ? <div style={{ color: "#94a3b8", fontSize: "0.78rem", marginTop: 6 }}>로그: {tune.log_file}</div> : null}
          </div>
        </div>
      </div>
    </PageFrame>
  );
}
