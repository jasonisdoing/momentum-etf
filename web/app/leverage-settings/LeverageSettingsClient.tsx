"use client";

import { useCallback, useEffect, useState } from "react";

import { PageFrame } from "../components/PageFrame";

type AssetRef = { ticker: string; name?: string };

type LeverageConfig = {
  backtested_date?: string;
  strategy?: string;
  market?: string;
  months_range?: number;
  start_date?: string;
  signal?: AssetRef;
  offense?: AssetRef;
  defense?: AssetRef;
  slippage?: number;
  drawdown_buy_cutoff?: number;
  drawdown_sell_cutoff?: number;
  benchmarks?: AssetRef[];
  [key: string]: unknown;
};

type LeverageState = {
  date?: string;
  target?: string;
  target_name?: string;
  updated_at?: string;
};

type LeverageResponse = {
  profile: string;
  config: LeverageConfig;
  state: LeverageState;
  error?: string;
};

const inputStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "5px 8px",
  fontSize: "0.9rem",
};

function FieldRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", gap: 12, alignItems: "center", padding: "7px 0", borderBottom: "1px solid rgba(148,163,184,0.15)" }}>
      <span style={{ width: 120, flexShrink: 0, color: "#64748b", fontWeight: 600 }}>{label}</span>
      <div style={{ flex: 1, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>{children}</div>
    </div>
  );
}

function ReadRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: "flex", gap: 12, padding: "8px 0", borderBottom: "1px solid rgba(148,163,184,0.15)" }}>
      <span style={{ width: 120, flexShrink: 0, color: "#64748b", fontWeight: 600 }}>{label}</span>
      <span style={{ fontWeight: 600 }}>{value}</span>
    </div>
  );
}

export function LeverageSettingsClient() {
  const [config, setConfig] = useState<LeverageConfig | null>(null);
  const [state, setState] = useState<LeverageState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const resp = await fetch("/api/leverage-config?profile=switch", { cache: "no-store" });
      const payload = (await resp.json()) as LeverageResponse;
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "레버리지 설정을 불러오지 못했습니다.");
      setConfig(payload.config);
      setState(payload.state);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "레버리지 설정을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const setField = (key: keyof LeverageConfig, value: unknown) =>
    setConfig((c) => (c ? { ...c, [key]: value } : c));

  const setAsset = (key: "signal" | "offense" | "defense", field: "ticker" | "name", value: string) =>
    setConfig((c) => (c ? { ...c, [key]: { ...(c[key] ?? { ticker: "" }), [field]: value } } : c));

  const setBenchmark = (i: number, field: "ticker" | "name", value: string) =>
    setConfig((c) => {
      if (!c) return c;
      const bms = [...(c.benchmarks ?? [])];
      bms[i] = { ...bms[i], [field]: value };
      return { ...c, benchmarks: bms };
    });

  const addBenchmark = () =>
    setConfig((c) => (c ? { ...c, benchmarks: [...(c.benchmarks ?? []), { ticker: "", name: "" }] } : c));

  const removeBenchmark = (i: number) =>
    setConfig((c) => (c ? { ...c, benchmarks: (c.benchmarks ?? []).filter((_, idx) => idx !== i) } : c));

  const numField = (key: keyof LeverageConfig, raw: string) =>
    setField(key, raw === "" ? undefined : Number(raw));

  const save = async () => {
    if (!config) return;
    try {
      setSaving(true);
      setError(null);
      setNotice(null);
      const resp = await fetch("/api/leverage-config?profile=switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config }),
      });
      const payload = (await resp.json()) as LeverageResponse;
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "저장에 실패했습니다.");
      setConfig(payload.config);
      setState(payload.state);
      setNotice("저장되었습니다.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "저장에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  };

  const asset = (key: "signal" | "offense" | "defense", label: string) => (
    <FieldRow label={label}>
      <input
        style={{ ...inputStyle, width: 110 }}
        placeholder="티커"
        value={config?.[key]?.ticker ?? ""}
        onChange={(e) => setAsset(key, "ticker", e.target.value)}
      />
      <input
        style={{ ...inputStyle, flex: 1, minWidth: 140 }}
        placeholder="이름"
        value={config?.[key]?.name ?? ""}
        onChange={(e) => setAsset(key, "name", e.target.value)}
      />
    </FieldRow>
  );

  return (
    <PageFrame title="레버리지 설정">
      <div className="appPageStack" style={{ maxWidth: 720 }}>
        {error ? <div className="alert alert-danger mb-0">{error}</div> : null}
        {notice ? <div className="alert alert-success mb-0">{notice}</div> : null}
        {loading && !config ? (
          <div style={{ color: "#868e96", padding: 20 }}>불러오는 중…</div>
        ) : config ? (
          <>
            <div className="card appCard">
              <div className="card-body">
                <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 12 }}>전략 설정 (switch)</h2>
                <FieldRow label="시장">
                  <select style={inputStyle} value={config.market ?? "kor"} onChange={(e) => setField("market", e.target.value)}>
                    <option value="kor">🇰🇷 한국</option>
                    <option value="us">🇺🇸 미국</option>
                  </select>
                </FieldRow>
                {asset("signal", "시그널 지수")}
                {asset("offense", "공격 자산")}
                {asset("defense", "방어 자산")}
                <FieldRow label="매수 컷(%)">
                  <input type="number" step="0.1" style={{ ...inputStyle, width: 100 }} value={config.drawdown_buy_cutoff ?? ""} onChange={(e) => numField("drawdown_buy_cutoff", e.target.value)} />
                </FieldRow>
                <FieldRow label="매도 컷(%)">
                  <input type="number" step="0.1" style={{ ...inputStyle, width: 100 }} value={config.drawdown_sell_cutoff ?? ""} onChange={(e) => numField("drawdown_sell_cutoff", e.target.value)} />
                </FieldRow>
                <FieldRow label="슬리피지(%)">
                  <input type="number" step="0.1" style={{ ...inputStyle, width: 100 }} value={config.slippage ?? ""} onChange={(e) => numField("slippage", e.target.value)} />
                </FieldRow>
                <FieldRow label="기간(개월)">
                  <input type="number" step="1" style={{ ...inputStyle, width: 100 }} value={config.months_range ?? ""} onChange={(e) => numField("months_range", e.target.value)} />
                  {config.start_date ? <span style={{ color: "#94a3b8", fontSize: "0.82rem" }}>start_date: {config.start_date}</span> : null}
                </FieldRow>
                <ReadRow label="최근 튜닝일" value={config.backtested_date ?? "-"} />
              </div>
            </div>

            <div className="card appCard">
              <div className="card-body">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <h2 style={{ fontSize: "1.05rem", fontWeight: 800, margin: 0 }}>벤치마크</h2>
                  <button type="button" className="btn btn-sm btn-outline-secondary" onClick={addBenchmark}>+ 추가</button>
                </div>
                {(config.benchmarks ?? []).map((b, i) => (
                  <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", padding: "5px 0" }}>
                    <input style={{ ...inputStyle, width: 110 }} placeholder="티커" value={b.ticker ?? ""} onChange={(e) => setBenchmark(i, "ticker", e.target.value)} />
                    <input style={{ ...inputStyle, flex: 1 }} placeholder="이름" value={b.name ?? ""} onChange={(e) => setBenchmark(i, "name", e.target.value)} />
                    <button type="button" className="btn btn-sm btn-outline-danger" onClick={() => removeBenchmark(i)}>삭제</button>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ display: "flex", gap: 8 }}>
              <button type="button" className="btn btn-dark" disabled={saving} onClick={save}>
                {saving ? "저장 중…" : "저장"}
              </button>
              <button type="button" className="btn btn-outline-secondary" disabled={saving} onClick={() => void load()}>
                되돌리기
              </button>
            </div>

            <div className="card appCard">
              <div className="card-body">
                <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>직전 추천 상태</h2>
                <p style={{ color: "#94a3b8", fontSize: "0.85rem", marginBottom: 12 }}>추천 배치가 자동 갱신합니다 (읽기 전용).</p>
                <ReadRow label="확정일" value={state?.date ?? "-"} />
                <ReadRow label="보유 종목" value={state?.target_name ? `${state.target_name} (${state.target ?? ""})` : "-"} />
                <ReadRow label="갱신 시각" value={state?.updated_at ?? "-"} />
              </div>
            </div>
          </>
        ) : null}
      </div>
    </PageFrame>
  );
}
