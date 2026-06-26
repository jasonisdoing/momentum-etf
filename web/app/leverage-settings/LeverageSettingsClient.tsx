"use client";

import { useCallback, useEffect, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";

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
  holding_days?: number;
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
  const [mounted, setMounted] = useState(false);
  const toast = useToast();

  const saveConfigDirect = useCallback(async (updatedConfig: LeverageConfig): Promise<boolean> => {
    try {
      setSaving(true);
      setError(null);
      setNotice(null);
      const resp = await fetch("/api/leverage-config?profile=switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: updatedConfig }),
      });
      const payload = (await resp.json()) as LeverageResponse;
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "저장에 실패했습니다.");
      setConfig(payload.config);
      setState(payload.state);
      return true;
    } catch (err) {
      const msg = err instanceof Error ? err.message : "저장에 실패했습니다.";
      toast.error(msg);
      return false;
    } finally {
      setSaving(false);
    }
  }, []);

  const fetchTickerName = useCallback(async (ticker: string): Promise<{ name: string; error?: string }> => {
    const clean = ticker.trim();
    if (!clean) return { name: "", error: "티커를 입력해주세요." };

    try {
      const resp = await fetch(`/api/leverage-config/resolve?ticker=${encodeURIComponent(clean)}`);
      const data = (await resp.json()) as { name?: string; error?: string };
      if (!resp.ok || data.error) {
        return { name: "", error: data.error ?? "티커 조회에 실패했습니다." };
      }
      return { name: data.name ?? "", error: data.name ? undefined : "존재하지 않는 티커입니다." };
    } catch (err) {
      return { name: "", error: err instanceof Error ? err.message : "티커 조회 중 오류가 발생했습니다." };
    }
  }, []);

  const setAsset = useCallback((key: "signal" | "offense" | "defense", field: "ticker" | "name", value: string) =>
    setConfig((c) => (c ? { ...c, [key]: { ...(c[key] ?? { ticker: "" }), [field]: value } } : c)), []);

  const setBenchmark = useCallback((i: number, field: "ticker" | "name", value: string) =>
    setConfig((c) => {
      if (!c) return c;
      const bms = [...(c.benchmarks ?? [])];
      bms[i] = { ...bms[i], [field]: value };
      return { ...c, benchmarks: bms };
    }), []);

  const resolveTickerName = useCallback(async (
    type: "signal" | "offense" | "defense" | "benchmark",
    value: string,
    index?: number
  ) => {
    setError(null);
    setNotice(null);
    const cleanTicker = value.trim();
    if (!cleanTicker) return;

    // 벤치마크 내 중복 검사
    if (type === "benchmark" && config?.benchmarks) {
      const isDuplicate = config.benchmarks.some(
        (b, idx) => b.ticker.trim().toUpperCase() === cleanTicker.toUpperCase() && idx !== index && b.name
      );
      if (isDuplicate) {
        toast.error("이미 등록된 벤치마크 종목입니다.");
        return;
      }
    }

    const { name, error: fetchError } = await fetchTickerName(cleanTicker);

    if (fetchError) {
      toast.error(fetchError);
      return;
    }

    if (type === "benchmark" && index !== undefined) {
      if (!config) return;
      const bms = [...(config.benchmarks ?? [])];
      bms[index] = { ticker: cleanTicker, name };
      const updated = { ...config, benchmarks: bms };

      setConfig(updated);

      const success = await saveConfigDirect(updated);
      if (success) {
        toast.success(`[벤치마크] ${name}(${cleanTicker}) 추가 및 저장 완료`);
      }
    } else if (type !== "benchmark") {
      setAsset(type, "name", name);
      toast.success(`[추천] ${name}(${cleanTicker}) 확인 완료`);
    }
  }, [config, fetchTickerName, saveConfigDirect, setAsset, toast]);

  const handleKeyDown = useCallback((
    e: React.KeyboardEvent<HTMLInputElement>,
    type: "signal" | "offense" | "defense" | "benchmark",
    value: string,
    index?: number
  ) => {
    if (e.key === "Enter") {
      e.preventDefault();
      void resolveTickerName(type, value, index);
    }
  }, [resolveTickerName]);

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
    setMounted(true);
    void load();
  }, [load]);

  const setField = (key: keyof LeverageConfig, value: unknown) =>
    setConfig((c) => (c ? { ...c, [key]: value } : c));

  const addBenchmark = () =>
    setConfig((c) => (c ? { ...c, benchmarks: [...(c.benchmarks ?? []), { ticker: "", name: "" }] } : c));

  const removeBenchmark = useCallback(async (i: number) => {
    if (!config) return;
    const bms = (config.benchmarks ?? []).filter((_, idx) => idx !== i);
    const updated = { ...config, benchmarks: bms };
    const success = await saveConfigDirect(updated);
    if (success) {
      toast.success("[벤치마크] 종목 삭제 및 저장 완료");
    }
  }, [config, saveConfigDirect, toast]);

  const numField = (key: keyof LeverageConfig, raw: string) =>
    setField(key, raw === "" ? undefined : Number(raw));

  const save = async () => {
    if (!config) return;
    const success = await saveConfigDirect(config);
    if (success) {
      toast.success("전략 설정 저장 완료");
    }
  };

  const asset = (key: "signal" | "offense" | "defense", label: string) => (
    <FieldRow label={label}>
      <div style={{ display: "flex", gap: 6, alignItems: "center", flex: 1, flexWrap: "wrap" }}>
        <input
          style={{ ...inputStyle, width: 110 }}
          placeholder="티커"
          value={config?.[key]?.ticker ?? ""}
          onChange={(e) => setAsset(key, "ticker", e.target.value)}
          onKeyDown={(e) => handleKeyDown(e, key, e.currentTarget.value)}
        />
        <button
          type="button"
          className="btn btn-sm btn-outline-secondary"
          style={{ padding: "4px 8px", fontSize: "0.82rem" }}
          onClick={() => void resolveTickerName(key, config?.[key]?.ticker ?? "")}
        >
          확인
        </button>
        <input
          style={{ ...inputStyle, flex: 1, minWidth: 140, backgroundColor: "#f8fafc", color: "#64748b", cursor: "not-allowed" }}
          placeholder="이름 (티커 입력 후 확인 클릭)"
          value={config?.[key]?.name ?? ""}
          readOnly
        />
      </div>
    </FieldRow>
  );

  if (!mounted) {
    return (
      <PageFrame title="레버리지 설정">
        <div className="appPageStack" style={{ maxWidth: 1000 }}>
          <div style={{ color: "#868e96", padding: 20 }}>불러오는 중…</div>
        </div>
      </PageFrame>
    );
  }

  return (
    <PageFrame title="레버리지 설정">
      <div className="appPageStack" style={{ maxWidth: 1000 }}>
        {error ? <div className="alert alert-danger mb-3">{error}</div> : null}
        {notice ? <div className="alert alert-success mb-3">{notice}</div> : null}
        {loading && !config ? (
          <div style={{ color: "#868e96", padding: 20 }}>불러오는 중…</div>
        ) : config ? (
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
            {/* 왼쪽 컬럼: 전략 설정 및 벤치마크 */}
            <div style={{ flex: "1 1 480px", display: "flex", flexDirection: "column", gap: 16 }}>
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
                  <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
                    <button type="button" className="btn btn-dark" disabled={saving} onClick={save}>
                      {saving ? "저장 중…" : "저장"}
                    </button>
                  </div>
                </div>
              </div>

              <div className="card appCard">
                <div className="card-body">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                    <h2 style={{ fontSize: "1.05rem", fontWeight: 800, margin: 0 }}>벤치마크</h2>
                    <button type="button" className="btn btn-sm btn-outline-secondary" onClick={addBenchmark}>+ 추가</button>
                  </div>
                  {(config.benchmarks ?? []).map((b, i) => {
                    const isExisting = !!b.name;
                    return (
                      <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", padding: "5px 0", flexWrap: "wrap" }}>
                        <input
                          style={{
                            ...inputStyle,
                            width: 110,
                            backgroundColor: isExisting ? "#f8fafc" : undefined,
                            color: isExisting ? "#64748b" : undefined,
                            cursor: isExisting ? "not-allowed" : undefined,
                          }}
                          placeholder="티커"
                          value={b.ticker ?? ""}
                          onChange={(e) => setBenchmark(i, "ticker", e.target.value)}
                          onKeyDown={(e) => handleKeyDown(e, "benchmark", e.currentTarget.value, i)}
                          readOnly={isExisting}
                        />
                        {!isExisting && (
                          <button
                            type="button"
                            className="btn btn-sm btn-outline-secondary"
                            style={{ padding: "4px 8px", fontSize: "0.82rem" }}
                            onClick={() => void resolveTickerName("benchmark", b.ticker ?? "", i)}
                          >
                            확인
                          </button>
                        )}
                        <input
                          style={{ ...inputStyle, flex: 1, backgroundColor: "#f8fafc", color: "#64748b", cursor: "not-allowed" }}
                          placeholder="이름 (티커 입력 후 확인 클릭)"
                          value={b.name ?? ""}
                          readOnly
                        />
                        <button type="button" className="btn btn-sm btn-outline-danger" onClick={() => removeBenchmark(i)}>삭제</button>
                      </div>
                    );
                  })}
                </div>
              </div>

            </div>

            {/* 오른쪽 컬럼: 직전 추천 상태 */}
            <div style={{ flex: "1 1 320px", display: "flex", flexDirection: "column", gap: 16 }}>
              <div className="card appCard" style={{ width: "100%" }}>
                <div className="card-body">
                  <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>직전 추천 상태</h2>
                  <p style={{ color: "#94a3b8", fontSize: "0.85rem", marginBottom: 12 }}>추천 배치가 자동 갱신합니다 (읽기 전용).</p>
                  <ReadRow label="확정일" value={state?.date ?? "-"} />
                  <ReadRow label="보유 종목" value={state?.target_name ? `${state.target_name} (${state.target ?? ""})` : "-"} />
                  <ReadRow label="보유일" value={state?.holding_days !== undefined ? `${state.holding_days}거래일째` : "-"} />
                  <ReadRow label="갱신 시각" value={state?.updated_at ?? "-"} />
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </PageFrame>
  );
}
