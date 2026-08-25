"use client";

import { useCallback, useEffect, useState } from "react";

import { MaDaysSelect } from "../components/MaDaysSelect";
import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";

type AlarmAccount = {
  account_id: string;
  name: string;
  icon: string;
  order: number;
  country_code: string;
  ma_enabled: boolean;
  ma_short_days: number;
  ma_long_days: number;
  ma_icon: string;
  stoploss_enabled: boolean;
  stoploss_threshold_pct: number;
  stoploss_icon: string;
};
type AlarmView = {
  /** 이평선 선택지 — 국가별(계좌의 country_code 로 고른다). */
  ma_options_by_country: Record<string, { short_ma_options: number[]; long_ma_options: number[] }>;
  stoploss_pct_options: number[];
  accounts: AlarmAccount[];
  error?: string;
};

type AlarmType = "ma" | "stoploss";

const selectStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "3px 6px",
  fontSize: "var(--fs-sm)",
};

export function AlarmsClient() {
  const toast = useToast();
  const [view, setView] = useState<AlarmView | null>(null);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);
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

  const saveAccount = async (
    account_id: string,
    alarm_type: AlarmType,
    enabled: boolean,
    values: Record<string, number>,
    icon?: string,
  ) => {
    setBusy(`${account_id}-${alarm_type}`);
    try {
      const resp = await fetch("/api/alarms/account", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account_id, alarm_type, enabled, values, ...(icon !== undefined ? { icon } : {}) }),
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

  const accountRow = (a: AlarmAccount, alarm_type: AlarmType) => {
    const isMa = alarm_type === "ma";
    const enabled = isMa ? a.ma_enabled : a.stoploss_enabled;
    const icon = isMa ? a.ma_icon : a.stoploss_icon;
    const isBusy = busy === `${a.account_id}-${alarm_type}`;
    // 이동선은 단기·장기 두 기준을 함께 저장한다 — 한쪽만 바꿔도 나머지를 같이 보낸다.
    const currentValues: Record<string, number> = isMa
      ? { short_days: a.ma_short_days, long_days: a.ma_long_days }
      : { threshold_pct: a.stoploss_threshold_pct };

    // 다른 화면과 같은 공용 이평선 셀렉트 — 표기는 "20일"로 통일(단기/장기는 툴팁).
    const daysSelect = (key: "short_days" | "long_days", label: string) => (
      <MaDaysSelect
        title={`${label} 이평선`}
        value={currentValues[key]}
        options={view?.ma_options_by_country[a.country_code]?.[key === "short_days" ? "short_ma_options" : "long_ma_options"]}
        disabled={isBusy}
        onChange={(days) => void saveAccount(a.account_id, alarm_type, enabled, { ...currentValues, [key]: days })}
      />
    );

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
            onChange={(e) => void saveAccount(a.account_id, alarm_type, e.target.checked, currentValues)}
          />
        </div>
        <span style={{ flex: 1, minWidth: 0, fontWeight: 600, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={`${a.name} (${a.account_id})`}>
          {a.order}. {a.icon ? `${a.icon} ` : ""}{a.name}
        </span>
        {isMa ? (
          <>
            {daysSelect("short_days", "단기")}
            {daysSelect("long_days", "장기")}
          </>
        ) : (
          <select
            className="form-select form-select-sm"
            style={{ width: 96 }}
            value={a.stoploss_threshold_pct}
            disabled={isBusy}
            onChange={(e) =>
              void saveAccount(a.account_id, alarm_type, enabled, { threshold_pct: Number(e.target.value) })
            }
          >
            {(view?.stoploss_pct_options ?? []).map((o) => (
              <option key={o} value={o}>{`${o}%`}</option>
            ))}
            {/* 선택지가 바뀌어 저장값이 목록 밖이면 숨기지 않고 그대로 보여줘 사용자가 바꾸게 한다 */}
            {!(view?.stoploss_pct_options ?? []).includes(a.stoploss_threshold_pct) && (
              <option value={a.stoploss_threshold_pct}>{`${a.stoploss_threshold_pct}% (선택지 밖)`}</option>
            )}
          </select>
        )}
        <input
          type="text"
          key={`${a.account_id}-${alarm_type}-icon-${icon}`}
          style={{ ...selectStyle, width: 52, textAlign: "center" }}
          defaultValue={icon}
          maxLength={8}
          disabled={isBusy}
          title="자산 화면 종목명에 붙는 배지 아이콘(비우면 표시 안 함)"
          onBlur={(e) => {
            const next = e.target.value.trim();
            if (next !== icon) void saveAccount(a.account_id, alarm_type, enabled, currentValues, next);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.currentTarget.blur();
          }}
        />
      </div>
    );
  };

  if (!mounted) return null;

  const accounts = view ? [...view.accounts].sort((a, b) => a.order - b.order) : [];

  return (
    <PageFrame title="알람">
      <div className="appPageStack" style={{ maxWidth: 640, display: "flex", flexDirection: "column", gap: 16 }}>
        {/* 이동선 이탈 */}
        <div className="card appCard">
          <div className="card-body">
            <h2 style={{ fontSize: "var(--fs-lg)", fontWeight: 800, marginBottom: 4 }}>📉 이동선 이탈 알림</h2>
            <p style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", marginBottom: 12 }}>
              계좌별로 켜고, 단기·장기 이평선을 각각 고를 수 있습니다. 보유 종목의 종가가 <strong>둘 중 하나라도</strong> 아래면
              알립니다(종목풀 순위 화면의 회색 처리와 같은 기준). 맨 오른쪽 아이콘은 자산 관리·자산 헬퍼 종목명에
              붙는 배지입니다(비우면 표시 안 함).
            </p>
            {loading || !view ? <div style={{ color: "var(--text-muted)" }}>불러오는 중…</div> : (
              <div style={{ display: "flex", flexDirection: "column" }}>{accounts.map((a) => accountRow(a, "ma"))}</div>
            )}
          </div>
        </div>

        {/* 손절 */}
        <div className="card appCard">
          <div className="card-body">
            <h2 style={{ fontSize: "var(--fs-lg)", fontWeight: 800, marginBottom: 4 }}>🛑 손절 알림</h2>
            <p style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", marginBottom: 12 }}>
              계좌별로 켜고, 계좌마다 다른 손절 기준을 선택할 수 있습니다. 보유 종목의 수익률이 그 기준 이하면 알립니다.
              맨 오른쪽 아이콘은 자산 관리·자산 헬퍼 종목명에 붙는 배지입니다(비우면 표시 안 함).
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
