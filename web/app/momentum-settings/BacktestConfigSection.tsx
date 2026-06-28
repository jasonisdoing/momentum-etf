"use client";

import { useCallback, useEffect, useState } from "react";

import { formatKstDateTime } from "@/lib/datetime";
import { useToast } from "../components/ToastProvider";

type Benchmark = { ticker?: string; name?: string };
type BtConfig = {
  BENCHMARK?: Benchmark;
  HOLDING_BONUS_SCORE?: number[];
  ATH_BONUS?: number[];
  MA_TYPE?: string[];
  MA_MONTHS?: number[];
  RSI_LIMIT?: number[];
};
type PoolEntry = { pool_id: string; name: string; config: BtConfig; updated_at?: string | null };
type ApiResponse = { pools?: PoolEntry[]; constraints?: { ma_types?: string[] }; error?: string };

const inputStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "4px 8px",
  fontSize: "0.88rem",
};
const labelStyle: React.CSSProperties = { color: "#64748b", fontWeight: 600, fontSize: "0.83rem", flexShrink: 0 };

function parseNums(text: string): number[] {
  return text.split(/[,\s]+/).map((t) => t.trim()).filter(Boolean).map(Number).filter((n) => Number.isFinite(n));
}

/** 풀 1개 백테스트 탐색공간 인라인 편집 행 (자체 저장). */
function PoolRow({ pool, maTypes }: { pool: PoolEntry; maTypes: string[] }) {
  const toast = useToast();
  const c = pool.config;
  const [benchTicker, setBenchTicker] = useState(c.BENCHMARK?.ticker ?? "");
  const [benchName, setBenchName] = useState(c.BENCHMARK?.name ?? "");
  const [bonusText, setBonusText] = useState((c.HOLDING_BONUS_SCORE ?? []).join(", "));
  const [athText, setAthText] = useState((c.ATH_BONUS ?? []).join(", "));
  const [monthsText, setMonthsText] = useState((c.MA_MONTHS ?? []).join(", "));
  const [rsiText, setRsiText] = useState((c.RSI_LIMIT ?? []).join(", "));
  const [maSet, setMaSet] = useState<Set<string>>(new Set((c.MA_TYPE ?? []).map((m) => m.toUpperCase())));
  const [updatedAt, setUpdatedAt] = useState<string | null | undefined>(pool.updated_at);
  const [saving, setSaving] = useState(false);

  const bonus = parseNums(bonusText);
  const ath = parseNums(athText);
  const months = parseNums(monthsText);
  const rsi = parseNums(rsiText);
  const combos = bonus.length * Math.max(1, ath.length) * maSet.size * months.length * rsi.length;

  const toggleMa = (t: string) =>
    setMaSet((prev) => {
      const next = new Set(prev);
      if (next.has(t)) next.delete(t);
      else next.add(t);
      return next;
    });

  const save = async () => {
    const config: BtConfig = {
      BENCHMARK: { ticker: benchTicker.trim(), name: benchName.trim() },
      HOLDING_BONUS_SCORE: bonus,
      ATH_BONUS: ath,
      MA_TYPE: [...maSet],
      MA_MONTHS: months.map((n) => Math.trunc(n)),
      RSI_LIMIT: rsi,
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
        <span style={{ fontWeight: 800 }}>{pool.name} <span style={{ color: "#94a3b8", fontWeight: 500 }}>({pool.pool_id})</span></span>
        <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ color: "#94a3b8", fontSize: "0.8rem" }}>마지막 저장: {updatedAt ? formatKstDateTime(updatedAt) : "저장 이력 없음"}</span>
          <button type="button" className="btn btn-sm btn-dark" disabled={saving || combos === 0} onClick={() => void save()}>
            {saving ? "저장 중…" : "저장"}
          </button>
        </span>
      </div>

      <div style={rowStyle}>
        <span style={{ ...labelStyle, width: 56 }}>벤치마크</span>
        <input style={{ ...inputStyle, width: 110 }} placeholder="티커" value={benchTicker} onChange={(e) => setBenchTicker(e.target.value)} />
        <input style={{ ...inputStyle, flex: 1, minWidth: 140 }} placeholder="이름" value={benchName} onChange={(e) => setBenchName(e.target.value)} />
      </div>

      <div style={rowStyle}>
        <span style={{ ...labelStyle, width: 56 }}>보유보너스</span>
        <input style={{ ...inputStyle, width: 110 }} placeholder="0, 10" value={bonusText} onChange={(e) => setBonusText(e.target.value)} />
        <span style={{ ...labelStyle, marginLeft: 8 }}>ATH 보너스</span>
        <input style={{ ...inputStyle, width: 130 }} placeholder="0, 10, 20, 30" value={athText} onChange={(e) => setAthText(e.target.value)} />
      </div>

      <div style={rowStyle}>
        <span style={{ ...labelStyle, width: 56 }}>MA 타입</span>
        <div style={{ display: "flex", gap: 9, flexWrap: "wrap" }}>
          {maTypes.map((t) => (
            <label key={t} style={{ display: "flex", alignItems: "center", gap: 3, fontSize: "0.83rem", cursor: "pointer" }}>
              <input type="checkbox" checked={maSet.has(t)} onChange={() => toggleMa(t)} />
              {t}
            </label>
          ))}
        </div>
      </div>

      <div style={{ ...rowStyle, marginBottom: 0 }}>
        <span style={{ ...labelStyle, width: 56 }}>MA 개월</span>
        <input style={{ ...inputStyle, width: 130 }} placeholder="3, 6, 9, 12" value={monthsText} onChange={(e) => setMonthsText(e.target.value)} />
        <span style={{ ...labelStyle, marginLeft: 8 }}>RSI 상한</span>
        <input style={{ ...inputStyle, width: 150 }} placeholder="80, 90, 100" value={rsiText} onChange={(e) => setRsiText(e.target.value)} />
        <span style={{ marginLeft: "auto", fontSize: "0.82rem", color: combos > 0 ? "#475569" : "#dc2626" }}>조합수 <b>{combos.toLocaleString()}</b></span>
      </div>
    </div>
  );
}

export function BacktestConfigSection() {
  const toast = useToast();
  const [pools, setPools] = useState<PoolEntry[]>([]);
  const [maTypes, setMaTypes] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const resp = await fetch("/api/backtest-config", { cache: "no-store" });
      const data = (await resp.json()) as ApiResponse;
      if (!resp.ok || data.error) throw new Error(data.error ?? "백테스트 설정을 불러오지 못했습니다.");
      setPools(data.pools ?? []);
      setMaTypes(data.constraints?.ma_types ?? []);
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
        <p style={{ color: "#94a3b8", fontSize: "0.85rem", marginBottom: 12 }}>
          모멘텀-백테스트(`python backtest/run.py`)가 풀별로 전수 탐색하는 값입니다. (라이브 적용값과 별개 — TOP_N_HOLD는 위 종목풀 설정에서 관리)
        </p>

        {loading ? (
          <div style={{ color: "#868e96", padding: 12 }}>불러오는 중…</div>
        ) : pools.length === 0 ? (
          <div style={{ color: "#94a3b8", padding: 12 }}>등록된 백테스트 설정이 없습니다.</div>
        ) : (
          pools.map((p) => <PoolRow key={p.pool_id} pool={p} maTypes={maTypes} />)
        )}
      </div>
    </div>
  );
}
