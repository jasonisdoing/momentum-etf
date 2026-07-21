"use client";

import { useCallback, useEffect, useState } from "react";

import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";

type AlarmAccount = {
  account_id: string;
  name: string;
  icon: string;
  order: number;
  ma20_enabled: boolean;
  ma20_ma_days: number;
  stoploss_enabled: boolean;
  stoploss_threshold_pct: number;
};
type AlarmView = {
  ma_days_options: number[];
  stoploss_pct_options: number[];
  ma_type: string;
  accounts: AlarmAccount[];
  error?: string;
};

const selectStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "3px 6px",
  fontSize: "0.85rem",
};

export function AlarmsClient() {
  const toast = useToast();
  const [view, setView] = useState<AlarmView | null>(null);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);
  const [sending, setSending] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => setMounted(true), []);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await fetch("/api/alarms", { cache: "no-store" });
      const payload = (await resp.json()) as AlarmView;
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "불러오지 못했습니다.");
      setView(payload);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    void load();
  }, [load]);

  const saveAccount = async (account_id: string, alarm_type: "ma20" | "stoploss", enabled: boolean, value: number) => {
    setBusy(`${account_id}-${alarm_type}`);
    try {
      const resp = await fetch("/api/alarms/account", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account_id, alarm_type, enabled, value }),
      });
      const payload = (await resp.json()) as AlarmView;
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "저장에 실패했습니다.");
      setView(payload);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "저장에 실패했습니다.");
      await load();
    } finally {
      setBusy(null);
    }
  };

  const sendManual = async () => {
    setSending(true);
    try {
      const resp = await fetch("/api/alarms/send", { method: "POST" });
      const payload = (await resp.json()) as { sent?: boolean; reason?: string; accounts?: number; error?: string };
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "발송에 실패했습니다.");
      if (payload.sent) toast.success(`슬랙 발송 완료 (${payload.accounts ?? 0}개 계좌)`);
      else toast.error(payload.reason ?? "발송 대상이 없습니다.");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "발송에 실패했습니다.");
    } finally {
      setSending(false);
    }
  };

  const accountRow = (a: AlarmAccount, alarm_type: "ma20" | "stoploss") => {
    const enabled = alarm_type === "ma20" ? a.ma20_enabled : a.stoploss_enabled;
    const value = alarm_type === "ma20" ? a.ma20_ma_days : a.stoploss_threshold_pct;
    const options = alarm_type === "ma20" ? view?.ma_days_options ?? [] : view?.stoploss_pct_options ?? [];
    const isBusy = busy === `${a.account_id}-${alarm_type}`;
    return (
      <div key={a.account_id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 0", borderBottom: "1px solid rgba(148,163,184,0.15)" }}>
        <div className="form-check form-switch" style={{ paddingLeft: "2.6em", marginBottom: 0, flexShrink: 0 }}>
          <input
            className="form-check-input"
            type="checkbox"
            role="switch"
            style={{ width: "2.2em", height: "1.2em" }}
            checked={enabled}
            disabled={isBusy}
            onChange={(e) => void saveAccount(a.account_id, alarm_type, e.target.checked, value)}
          />
        </div>
        <span style={{ flex: 1, minWidth: 0, fontWeight: 600, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={`${a.name} (${a.account_id})`}>
          {a.order}. {a.icon ? `${a.icon} ` : ""}{a.name}
        </span>
        <select
          style={selectStyle}
          value={value}
          disabled={isBusy}
          onChange={(e) => void saveAccount(a.account_id, alarm_type, enabled, Number(e.target.value))}
        >
          {options.map((o) => (
            <option key={o} value={o}>{alarm_type === "ma20" ? `${view?.ma_type ?? ""} ${o}일` : `${o}%`}</option>
          ))}
        </select>
      </div>
    );
  };

  if (!mounted) return null;

  const accounts = view ? [...view.accounts].sort((a, b) => a.order - b.order) : [];

  return (
    <PageFrame title="알람">
      <div className="appPageStack" style={{ maxWidth: 640, display: "flex", flexDirection: "column", gap: 16 }}>
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button type="button" className="btn btn-sm btn-dark" disabled={sending} onClick={() => void sendManual()}>
            {sending ? "발송 중…" : "슬랙 테스트"}
          </button>
        </div>

        {/* 이동선 이탈 */}
        <div className="card appCard">
          <div className="card-body">
            <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>📉 이동선 이탈 알림</h2>
            <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginBottom: 12 }}>
              계좌별로 켜고, 계좌마다 다른 기준 이평선을 선택할 수 있습니다. 보유 종목의 종가가 그 이평선 아래면 알립니다.
            </p>
            {loading || !view ? <div style={{ color: "var(--text-muted)" }}>불러오는 중…</div> : (
              <div style={{ display: "flex", flexDirection: "column" }}>{accounts.map((a) => accountRow(a, "ma20"))}</div>
            )}
          </div>
        </div>

        {/* 손절 */}
        <div className="card appCard">
          <div className="card-body">
            <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>🛑 손절 알림</h2>
            <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginBottom: 12 }}>
              계좌별로 켜고, 계좌마다 다른 손절 기준을 선택할 수 있습니다. 보유 종목의 수익률이 그 기준 이하면 알립니다.
            </p>
            {loading || !view ? <div style={{ color: "var(--text-muted)" }}>불러오는 중…</div> : (
              <div style={{ display: "flex", flexDirection: "column" }}>{accounts.map((a) => accountRow(a, "stoploss"))}</div>
            )}
          </div>
        </div>
      </div>
    </PageFrame>
  );
}
