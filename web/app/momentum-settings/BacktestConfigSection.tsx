"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { useToast } from "../components/ToastProvider";

type Benchmark = { ticker?: string; name?: string };
type BtConfig = {
  BENCHMARK?: Benchmark;
  HOLDING_BONUS_SCORE?: number[];
  MA_TYPE?: string[];
  MA_MONTHS?: number[];
  RSI_LIMIT?: number[];
};
type PoolEntry = { pool_id: string; name: string; config: BtConfig };
type ApiResponse = { pools?: PoolEntry[]; constraints?: { ma_types?: string[] }; error?: string };

const inputStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "5px 8px",
  fontSize: "0.9rem",
};

function parseNums(text: string): number[] {
  return text
    .split(/[,\s]+/)
    .map((t) => t.trim())
    .filter(Boolean)
    .map(Number)
    .filter((n) => Number.isFinite(n));
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", gap: 12, alignItems: "center", padding: "7px 0", borderBottom: "1px solid rgba(148,163,184,0.15)" }}>
      <span style={{ width: 150, flexShrink: 0, color: "#64748b", fontWeight: 600 }}>{label}</span>
      <div style={{ flex: 1, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>{children}</div>
    </div>
  );
}

export function BacktestConfigSection() {
  const toast = useToast();
  const [pools, setPools] = useState<PoolEntry[]>([]);
  const [maTypes, setMaTypes] = useState<string[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // 선택 풀의 편집 버퍼
  const [benchTicker, setBenchTicker] = useState("");
  const [benchName, setBenchName] = useState("");
  const [bonusText, setBonusText] = useState("");
  const [monthsText, setMonthsText] = useState("");
  const [rsiText, setRsiText] = useState("");
  const [maSet, setMaSet] = useState<Set<string>>(new Set());

  const fillBuffers = useCallback((cfg: BtConfig) => {
    setBenchTicker(cfg.BENCHMARK?.ticker ?? "");
    setBenchName(cfg.BENCHMARK?.name ?? "");
    setBonusText((cfg.HOLDING_BONUS_SCORE ?? []).join(", "));
    setMonthsText((cfg.MA_MONTHS ?? []).join(", "));
    setRsiText((cfg.RSI_LIMIT ?? []).join(", "));
    setMaSet(new Set((cfg.MA_TYPE ?? []).map((m) => m.toUpperCase())));
  }, []);

  const selectPool = useCallback((poolId: string, list: PoolEntry[]) => {
    setSelected(poolId);
    const entry = list.find((p) => p.pool_id === poolId);
    if (entry) fillBuffers(entry.config);
  }, [fillBuffers]);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const resp = await fetch("/api/backtest-config", { cache: "no-store" });
      const data = (await resp.json()) as ApiResponse;
      if (!resp.ok || data.error) throw new Error(data.error ?? "백테스트 설정을 불러오지 못했습니다.");
      const list = data.pools ?? [];
      setPools(list);
      setMaTypes(data.constraints?.ma_types ?? []);
      if (list.length > 0) selectPool(selected && list.some((p) => p.pool_id === selected) ? selected : list[0].pool_id, list);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "백테스트 설정을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [selected, selectPool, toast]);

  useEffect(() => {
    void load();
    // 최초 1회만 로드
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggleMa = (t: string) =>
    setMaSet((prev) => {
      const next = new Set(prev);
      if (next.has(t)) next.delete(t);
      else next.add(t);
      return next;
    });

  const bonus = useMemo(() => parseNums(bonusText), [bonusText]);
  const months = useMemo(() => parseNums(monthsText), [monthsText]);
  const rsi = useMemo(() => parseNums(rsiText), [rsiText]);
  const combos = bonus.length * maSet.size * months.length * rsi.length;

  const save = async () => {
    const config: BtConfig = {
      BENCHMARK: { ticker: benchTicker.trim(), name: benchName.trim() },
      HOLDING_BONUS_SCORE: bonus,
      MA_TYPE: [...maSet],
      MA_MONTHS: months.map((n) => Math.trunc(n)),
      RSI_LIMIT: rsi,
    };
    try {
      setSaving(true);
      const resp = await fetch("/api/backtest-config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pool_id: selected, config }),
      });
      const data = (await resp.json()) as { config?: BtConfig; error?: string; detail?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? data.detail ?? "저장에 실패했습니다.");
      setPools((prev) => prev.map((p) => (p.pool_id === selected && data.config ? { ...p, config: data.config } : p)));
      toast.success(`[백테스트] ${selected} 탐색공간 저장 완료`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "저장에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="card appCard" style={{ marginTop: 16 }}>
      <div className="card-body">
        <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>백테스트 탐색 공간</h2>
        <p style={{ color: "#94a3b8", fontSize: "0.85rem", marginBottom: 12 }}>
          모멘텀-백테스트(`python backtest/run.py`)가 풀별로 전수 탐색하는 값입니다. (라이브 적용값과 별개 — TOP_N_HOLD는 위 종목풀 설정에서 관리)
        </p>

        {loading ? (
          <div style={{ color: "#868e96", padding: 12 }}>불러오는 중…</div>
        ) : pools.length === 0 ? (
          <div style={{ color: "#94a3b8", padding: 12 }}>등록된 백테스트 설정이 없습니다.</div>
        ) : (
          <>
            <Row label="종목풀">
              <select style={inputStyle} value={selected} onChange={(e) => selectPool(e.target.value, pools)}>
                {pools.map((p) => (
                  <option key={p.pool_id} value={p.pool_id}>{p.name} ({p.pool_id})</option>
                ))}
              </select>
            </Row>
            <Row label="벤치마크">
              <input style={{ ...inputStyle, width: 120 }} placeholder="티커" value={benchTicker} onChange={(e) => setBenchTicker(e.target.value)} />
              <input style={{ ...inputStyle, flex: 1, minWidth: 160 }} placeholder="이름" value={benchName} onChange={(e) => setBenchName(e.target.value)} />
            </Row>
            <Row label="보유보너스 점수">
              <input style={{ ...inputStyle, flex: 1 }} placeholder="예: 0, 10" value={bonusText} onChange={(e) => setBonusText(e.target.value)} />
            </Row>
            <Row label="MA 타입">
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                {maTypes.map((t) => (
                  <label key={t} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: "0.85rem", cursor: "pointer" }}>
                    <input type="checkbox" checked={maSet.has(t)} onChange={() => toggleMa(t)} />
                    {t}
                  </label>
                ))}
              </div>
            </Row>
            <Row label="MA 개월">
              <input style={{ ...inputStyle, flex: 1 }} placeholder="예: 3, 6, 9, 12" value={monthsText} onChange={(e) => setMonthsText(e.target.value)} />
            </Row>
            <Row label="RSI 상한">
              <input style={{ ...inputStyle, flex: 1 }} placeholder="예: 100" value={rsiText} onChange={(e) => setRsiText(e.target.value)} />
            </Row>

            <div style={{ marginTop: 10, fontSize: "0.85rem", color: combos > 0 ? "#475569" : "#dc2626" }}>
              조합수: <b>{combos.toLocaleString()}</b>개
              <span style={{ color: "#94a3b8" }}> (보너스 {bonus.length} × MA타입 {maSet.size} × MA개월 {months.length} × RSI {rsi.length} × TOP_N_HOLD 1)</span>
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
              <button type="button" className="btn btn-dark" disabled={saving || combos === 0} onClick={() => void save()}>
                {saving ? "저장 중…" : "저장"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
