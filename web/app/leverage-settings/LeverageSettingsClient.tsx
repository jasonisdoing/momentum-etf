"use client";

import { useCallback } from "react";

import { PageFrame } from "../components/PageFrame";
import { CandidateKey, inputStyle, LeverageConfig, rangeValues, TuningConfig, useLeverageConfig } from "./useLeverageConfig";

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
  const { config, setConfig, state, error, setError, loading, saving, mounted, toast, saveConfigDirect, fetchTickerName } =
    useLeverageConfig();

  // ── 전략 설정 ──
  const setAsset = useCallback((key: "signal" | "offense" | "defense", field: "ticker" | "name", value: string) =>
    setConfig((c) => (c ? { ...c, [key]: { ...(c[key] ?? { ticker: "" }), [field]: value } } : c)), [setConfig]);

  const setField = (key: keyof LeverageConfig, value: unknown) =>
    setConfig((c) => (c ? { ...c, [key]: value } : c));

  const numField = (key: keyof LeverageConfig, raw: string) =>
    setField(key, raw === "" ? undefined : Number(raw));

  const resolveAsset = useCallback(async (key: "signal" | "offense" | "defense", value: string) => {
    setError(null);
    const cleanTicker = value.trim();
    if (!cleanTicker) return;
    const { name, error: fetchError } = await fetchTickerName(cleanTicker);
    if (fetchError) {
      toast.error(fetchError);
      return;
    }
    setAsset(key, "name", name);
    toast.success(`[추천] ${name}(${cleanTicker}) 확인 완료`);
  }, [fetchTickerName, setAsset, setError, toast]);

  const save = async () => {
    if (!config) return;
    const success = await saveConfigDirect(config);
    if (success) toast.success("전략 설정 저장 완료");
  };

  const asset = (key: "signal" | "offense" | "defense", label: string) => (
    <FieldRow label={label}>
      <div style={{ display: "flex", gap: 6, alignItems: "center", flex: 1, flexWrap: "wrap" }}>
        <input
          style={{ ...inputStyle, width: 110 }}
          placeholder="티커"
          value={config?.[key]?.ticker ?? ""}
          onChange={(e) => setAsset(key, "ticker", e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); void resolveAsset(key, e.currentTarget.value); } }}
        />
        <button
          type="button"
          className="btn btn-sm btn-outline-secondary"
          style={{ padding: "4px 8px", fontSize: "0.82rem" }}
          onClick={() => void resolveAsset(key, config?.[key]?.ticker ?? "")}
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

  // ── 튜닝 탐색 공간 ──
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
      <PageFrame title="레버리지 설정">
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

  return (
    <PageFrame title="레버리지 설정">
      <div className="appPageStack" style={{ maxWidth: 1000 }}>
        {error ? <div className="alert alert-danger mb-3">{error}</div> : null}
        {loading && !config ? (
          <div style={{ color: "#868e96", padding: 20 }}>불러오는 중…</div>
        ) : config ? (
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
            {/* 왼쪽: 전략 설정 + 튜닝 탐색 공간 */}
            <div style={{ flex: "1 1 480px", minWidth: 0, display: "flex", flexDirection: "column", gap: 16 }}>
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
                  <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>튜닝 탐색 공간</h2>
                  <p style={{ color: "#94a3b8", fontSize: "0.85rem", marginBottom: 12 }}>
                    레버리지-튜닝에서 전수 탐색하는 후보군·범위입니다. (벤치마크는 후보군에서 자동 파생)
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
                    <button type="button" className="btn btn-outline-dark" disabled={saving} onClick={save}>
                      {saving ? "저장 중…" : "범위 저장"}
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* 오른쪽: 직전 추천 상태 */}
            <div style={{ flex: "1 1 320px", minWidth: 0, display: "flex", flexDirection: "column", gap: 16 }}>
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
