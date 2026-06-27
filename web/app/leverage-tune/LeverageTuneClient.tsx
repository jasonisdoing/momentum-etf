"use client";

import { useCallback, useEffect, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import {
  CandidateKey,
  inputStyle,
  rangeValues,
  TuningConfig,
  useLeverageConfig,
} from "../leverage-settings/useLeverageConfig";

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

function FieldRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", gap: 12, alignItems: "center", padding: "7px 0", borderBottom: "1px solid rgba(148,163,184,0.15)" }}>
      <span style={{ width: 120, flexShrink: 0, color: "#64748b", fontWeight: 600 }}>{label}</span>
      <div style={{ flex: 1, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>{children}</div>
    </div>
  );
}

export function LeverageTuneClient() {
  const { config, setConfig, error, setError, loading, saving, mounted, toast, saveConfigDirect, fetchTickerName } =
    useLeverageConfig();

  const [tune, setTune] = useState<TuneStatus | null>(null);
  const [triggering, setTriggering] = useState(false);

  // ── 튜닝 탐색 공간 편집 ──
  const setTuning = (updater: (t: TuningConfig) => TuningConfig) =>
    setConfig((c) => (c ? { ...c, tuning: updater(c.tuning ?? {}) } : c));

  const addCandidate = (listKey: CandidateKey) =>
    setTuning((t) => ({ ...t, [listKey]: [...(t[listKey] ?? []), { ticker: "", name: "" }] }));

  const setCandidate = (listKey: CandidateKey, i: number, value: string) =>
    setTuning((t) => {
      const list = [...(t[listKey] ?? [])];
      list[i] = { ...list[i], ticker: value };
      return { ...t, [listKey]: list };
    });

  const removeCandidate = useCallback(async (listKey: CandidateKey, i: number) => {
    if (!config) return;
    const list = (config.tuning?.[listKey] ?? []).filter((_, idx) => idx !== i);
    const updated = { ...config, tuning: { ...(config.tuning ?? {}), [listKey]: list } };
    const success = await saveConfigDirect(updated);
    if (success) toast.success("[튜닝 후보] 종목 삭제 및 저장 완료");
  }, [config, saveConfigDirect, toast]);

  const setRange = (key: "buy_cutoff_range" | "sell_cutoff_range", field: "min" | "max" | "step", raw: string) =>
    setTuning((t) => ({ ...t, [key]: { ...(t[key] ?? {}), [field]: raw === "" ? undefined : Number(raw) } }));

  const resolveCandidate = useCallback(async (listKey: CandidateKey, value: string, index: number) => {
    setError(null);
    const cleanTicker = value.trim();
    if (!cleanTicker || !config) return;

    const list = config.tuning?.[listKey] ?? [];
    const dup = list.some((c, idx) => c.ticker.trim().toUpperCase() === cleanTicker.toUpperCase() && idx !== index && c.name);
    if (dup) {
      toast.error("이미 등록된 후보 종목입니다.");
      return;
    }

    let name: string;
    if (cleanTicker.toUpperCase() === "CASH") {
      name = "현금";
    } else {
      const resolved = await fetchTickerName(cleanTicker);
      if (resolved.error) {
        toast.error(resolved.error);
        return;
      }
      name = resolved.name;
    }

    const next = [...list];
    next[index] = { ticker: cleanTicker, name };
    const updated = { ...config, tuning: { ...(config.tuning ?? {}), [listKey]: next } };
    setConfig(updated);
    const success = await saveConfigDirect(updated);
    if (success) {
      const label = listKey === "offense_candidates" ? "공격 후보" : "방어 후보";
      toast.success(`[${label}] ${name}(${cleanTicker}) 추가 및 저장 완료`);
    }
  }, [config, fetchTickerName, saveConfigDirect, setConfig, setError, toast]);

  const saveRanges = async () => {
    if (!config) return;
    const success = await saveConfigDirect(config);
    if (success) toast.success("튜닝 탐색 공간 저장 완료");
  };

  // ── 튜닝 실행/상태 폴링 ──
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

  const candidateList = (listKey: CandidateKey, label: string) => {
    const list = config?.tuning?.[listKey] ?? [];
    return (
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
          <span style={{ fontWeight: 700, color: "#334155" }}>{label} <span style={{ color: "#94a3b8", fontWeight: 500 }}>({list.length})</span></span>
          <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => addCandidate(listKey)}>+ 추가</button>
        </div>
        {list.map((c, i) => {
          const isExisting = !!c.name;
          return (
            <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", padding: "4px 0", flexWrap: "wrap" }}>
              <input
                style={{ ...inputStyle, width: 110, backgroundColor: isExisting ? "#f8fafc" : undefined, color: isExisting ? "#64748b" : undefined, cursor: isExisting ? "not-allowed" : undefined }}
                placeholder="티커"
                value={c.ticker ?? ""}
                onChange={(e) => setCandidate(listKey, i, e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); void resolveCandidate(listKey, e.currentTarget.value, i); } }}
                readOnly={isExisting}
              />
              {!isExisting && (
                <button
                  type="button"
                  className="btn btn-sm btn-outline-secondary"
                  style={{ padding: "4px 8px", fontSize: "0.82rem" }}
                  onClick={() => void resolveCandidate(listKey, c.ticker ?? "", i)}
                >
                  확인
                </button>
              )}
              <input
                style={{ ...inputStyle, flex: 1, minWidth: 120, backgroundColor: "#f8fafc", color: "#64748b", cursor: "not-allowed" }}
                placeholder="이름 (티커 입력 후 확인)"
                value={c.name ?? ""}
                readOnly
              />
              <button type="button" className="btn btn-sm btn-outline-danger" onClick={() => removeCandidate(listKey, i)}>삭제</button>
            </div>
          );
        })}
      </div>
    );
  };

  const rangeEditor = (key: "buy_cutoff_range" | "sell_cutoff_range", label: string) => {
    const r = config?.tuning?.[key];
    return (
      <FieldRow label={label}>
        {(["min", "max", "step"] as const).map((f) => (
          <span key={f} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ color: "#94a3b8", fontSize: "0.8rem" }}>{f}</span>
            <input type="number" step="0.1" style={{ ...inputStyle, width: 70 }} value={r?.[f] ?? ""} onChange={(e) => setRange(key, f, e.target.value)} />
          </span>
        ))}
      </FieldRow>
    );
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

  const t = config?.tuning;
  const buys = rangeValues(t?.buy_cutoff_range);
  const sells = rangeValues(t?.sell_cutoff_range);
  const pairs = buys.reduce((acc, b) => acc + sells.filter((s) => b < s).length, 0);
  const combos = pairs * (t?.offense_candidates?.length ?? 0) * (t?.defense_candidates?.length ?? 0);

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
        {error ? <div className="alert alert-danger mb-3">{error}</div> : null}
        {loading && !config ? (
          <div style={{ color: "#868e96", padding: 20 }}>불러오는 중…</div>
        ) : config ? (
          <>
            <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
              {/* 왼쪽: 튜닝 탐색 공간 */}
              <div style={{ flex: "1 1 480px", display: "flex", flexDirection: "column", gap: 16 }}>
                <div className="card appCard">
                  <div className="card-body">
                    <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>튜닝 탐색 공간</h2>
                    <p style={{ color: "#94a3b8", fontSize: "0.85rem", marginBottom: 12 }}>
                      후보군·범위를 전수 탐색해 최적값을 전략 설정에 기록합니다. (벤치마크는 후보군에서 자동 파생)
                    </p>
                    {candidateList("offense_candidates", "공격 후보")}
                    <div style={{ height: 12 }} />
                    {candidateList("defense_candidates", "방어 후보 (CASH = 현금)")}
                    <div style={{ height: 12, borderTop: "1px solid rgba(148,163,184,0.2)", marginTop: 4 }} />
                    {rangeEditor("buy_cutoff_range", "매수컷 범위(%)")}
                    {rangeEditor("sell_cutoff_range", "매도컷 범위(%)")}
                    <div style={{ marginTop: 8, fontSize: "0.85rem", color: combos > 0 ? "#475569" : "#dc2626" }}>
                      유효 조합수: <b>{combos.toLocaleString()}</b>개
                      <span style={{ color: "#94a3b8" }}> (매수컷&lt;매도컷 {pairs}쌍 × 공격 {t?.offense_candidates?.length ?? 0} × 방어 {t?.defense_candidates?.length ?? 0})</span>
                    </div>
                    <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
                      <button type="button" className="btn btn-outline-dark" disabled={saving} onClick={saveRanges}>
                        {saving ? "저장 중…" : "범위 저장"}
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* 오른쪽: 튜닝 실행 */}
              <div style={{ flex: "1 1 320px", display: "flex", flexDirection: "column", gap: 16 }}>
                <div className="card appCard" style={{ width: "100%" }}>
                  <div className="card-body">
                    <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 8 }}>튜닝 실행</h2>
                    <p style={{ color: "#94a3b8", fontSize: "0.85rem", marginBottom: 12 }}>
                      버튼을 누르면 배치 큐에 등록되고 워커가 전수 탐색을 실행합니다. 결과는 전략 설정(공격/방어/컷오프)에 자동 반영됩니다.
                    </p>
                    <button type="button" className="btn btn-dark" disabled={triggering || running} onClick={() => void startTune()} style={{ width: "100%" }}>
                      {running ? "튜닝 진행 중…" : triggering ? "시작 중…" : "튜닝 시작"}
                    </button>
                    <div style={{ marginTop: 12, fontSize: "0.85rem", color: "#475569" }}>
                      상태: <b>{statusLabel}</b>
                      {tune?.started_at ? <div style={{ color: "#94a3b8", marginTop: 4 }}>시작: {new Date(tune.started_at).toLocaleString("ko-KR")}</div> : null}
                      {tune?.ended_at ? <div style={{ color: "#94a3b8" }}>종료: {new Date(tune.ended_at).toLocaleString("ko-KR")}</div> : null}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* 아래: 진행도 + 결과 */}
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
                  <pre style={{ background: "#0f172a", color: "#e2e8f0", padding: 14, borderRadius: 8, fontSize: "0.78rem", lineHeight: 1.5, maxHeight: 460, overflow: "auto", whiteSpace: "pre-wrap" }}>
                    {tune.log_text}
                  </pre>
                ) : (
                  <div style={{ color: "#94a3b8", padding: 12 }}>아직 튜닝 결과가 없습니다. "튜닝 시작"을 눌러 실행하세요.</div>
                )}
                {tune?.log_file ? <div style={{ color: "#cbd5e1", fontSize: "0.78rem", marginTop: 6 }}>로그: {tune.log_file}</div> : null}
              </div>
            </div>
          </>
        ) : null}
      </div>
    </PageFrame>
  );
}
