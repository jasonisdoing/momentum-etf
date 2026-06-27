"use client";

import { useCallback } from "react";

import { PageFrame } from "../components/PageFrame";
import { inputStyle, LeverageConfig, useLeverageConfig } from "./useLeverageConfig";

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

  const setAsset = useCallback((key: "signal" | "offense" | "defense", field: "ticker" | "name", value: string) =>
    setConfig((c) => (c ? { ...c, [key]: { ...(c[key] ?? { ticker: "" }), [field]: value } } : c)), [setConfig]);

  const setField = (key: keyof LeverageConfig, value: unknown) =>
    setConfig((c) => (c ? { ...c, [key]: value } : c));

  const numField = (key: keyof LeverageConfig, raw: string) =>
    setField(key, raw === "" ? undefined : Number(raw));

  const resolveTickerName = useCallback(async (key: "signal" | "offense" | "defense", value: string) => {
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

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>, key: "signal" | "offense" | "defense", value: string) => {
    if (e.key === "Enter") {
      e.preventDefault();
      void resolveTickerName(key, value);
    }
  };

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
        {loading && !config ? (
          <div style={{ color: "#868e96", padding: 20 }}>불러오는 중…</div>
        ) : config ? (
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
            {/* 왼쪽: 전략 설정 */}
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
            </div>

            {/* 오른쪽: 직전 추천 상태 */}
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
