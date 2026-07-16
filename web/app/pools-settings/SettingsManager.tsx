"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { formatKstDateTime } from "@/lib/datetime";
import { useToast } from "../components/ToastProvider";

const EDITABLE_KEYS = [
  "TOP_N_HOLD",
  "SHORT_MA_DAYS",
  "MAIN_MA_DAYS",
] as const;

type EditableKey = (typeof EDITABLE_KEYS)[number];

const KEY_LABELS: Record<EditableKey, string> = {
  TOP_N_HOLD: "보유 종목수",
  SHORT_MA_DAYS: "단기 이평선",
  MAIN_MA_DAYS: "메인 이평선",
};

const KEY_WIDTHS: Record<EditableKey, number> = {
  TOP_N_HOLD: 90,
  SHORT_MA_DAYS: 104,
  MAIN_MA_DAYS: 104,
};

const DEFAULT_MA_DAY_OPTIONS = [5, 10, 20, 40, 60, 120, 240];

type SettingField = { value: string | number | null };
type SettingsMap = Record<EditableKey, SettingField>;

type PoolEntry = {
  pool_id?: string;
  ticker_type?: string;
  name: string;
  icon?: string;
  order?: number;
  settings: SettingsMap;
  updated_at?: string;
};

type PoolSettingsResponse = {
  pools: PoolEntry[];
  constraints: { ma_day_options: number[]; editable_keys: string[] };
  error?: string;
};

/** 한 행의 편집 중인 값 (모두 문자열로 보관, 저장 시 파싱). */
type RowDraft = Record<EditableKey, string>;

function toDraft(settings: SettingsMap): RowDraft {
  return EDITABLE_KEYS.reduce((acc, key) => {
    const v = settings[key]?.value;
    acc[key] = v === null || v === undefined ? "" : String(v);
    return acc;
  }, {} as RowDraft);
}

export function SettingsManager() {
  const toast = useToast();
  const [data, setData] = useState<PoolSettingsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, RowDraft>>({});
  const [savingId, setSavingId] = useState<string | null>(null);

  const rows = useMemo(() => {
    if (!data) return [] as { id: string; entry: PoolEntry }[];
    return [...data.pools]
      .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
      .map((p) => ({ id: p.ticker_type ?? "", entry: p }));
  }, [data]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch("/api/pool-settings", { cache: "no-store" });
      const payload = (await resp.json()) as PoolSettingsResponse;
      if (!resp.ok || payload.error) {
        throw new Error(payload.error ?? "설정을 불러오지 못했습니다.");
      }
      setData(payload);
      const nextDrafts: Record<string, RowDraft> = {};
      payload.pools.forEach((p) => {
        if (p.ticker_type) nextDrafts[p.ticker_type] = toDraft(p.settings);
      });
      setDrafts(nextDrafts);
    } catch (err) {
      setError(err instanceof Error ? err.message : "설정을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const updateDraft = useCallback((id: string, key: EditableKey, value: string) => {
    setDrafts((prev) => ({ ...prev, [id]: { ...prev[id], [key]: value } }));
  }, []);

  const isDirty = useCallback(
    (id: string, settings: SettingsMap) => {
      const draft = drafts[id];
      if (!draft) return false;
      return EDITABLE_KEYS.some((key) => {
        const orig = settings[key]?.value;
        const origStr = orig === null || orig === undefined ? "" : String(orig);
        return draft[key] !== origStr;
      });
    },
    [drafts],
  );

  const handleSave = useCallback(
    async (id: string) => {
      const draft = drafts[id];
      if (!draft) return;
      const values: Record<string, string> = {};
      EDITABLE_KEYS.forEach((key) => {
        values[key] = draft[key];
      });
      setSavingId(id);
      try {
        const resp = await fetch("/api/pool-settings", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pool_id: id, values }),
        });
        const payload = await resp.json();
        if (!resp.ok || payload.error) {
          throw new Error(payload.error ?? payload.detail ?? "저장에 실패했습니다.");
        }
        toast.success("설정을 저장했습니다.");
        await load();
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "저장에 실패했습니다.");
      } finally {
        setSavingId(null);
      }
    },
    [drafts, load, toast],
  );

  if (loading && !data) {
    return <div className="appPageStack">불러오는 중…</div>;
  }
  if (error) {
    return (
      <div className="appBannerStack">
        <div className="bannerError alert alert-danger mb-0">{error}</div>
      </div>
    );
  }
  if (!data) return null;
  const maDayOptions = data.constraints.ma_day_options?.length ? data.constraints.ma_day_options : DEFAULT_MA_DAY_OPTIONS;

  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBodyTight">
            <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>종목풀 설정</h2>
            <p className="tableFooterMeta" style={{ marginBottom: 12, color: "var(--text-muted)", fontSize: "0.85rem" }}>
              종목풀의 구조(이름/순서/국가 등)는 pools.json 이 유지하고, 아래 값은 DB 에서 저장·수정합니다.
            </p>
            <div style={{ overflowX: "auto" }}>
              <table className="table table-sm appSettingsTable" style={{ minWidth: 880 }}>
                <thead>
                  <tr>
                    <th style={{ width: 140, whiteSpace: "nowrap" }}>종목풀</th>
                    {EDITABLE_KEYS.map((key) => (
                      <th key={key} style={{ textAlign: "right", whiteSpace: "nowrap", width: KEY_WIDTHS[key], minWidth: KEY_WIDTHS[key] }}>
                        {KEY_LABELS[key]}
                      </th>
                    ))}
                    <th style={{ textAlign: "center", minWidth: 80 }}>저장</th>
                    <th style={{ textAlign: "left", minWidth: 160 }}>마지막 저장</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(({ id, entry }) => {
                    const draft = drafts[id] ?? toDraft(entry.settings);
                    const dirty = isDirty(id, entry.settings);

                    return (
                      <tr key={id}>
                        <td style={{ whiteSpace: "nowrap" }}>
                          {entry.icon ? `${entry.icon} ` : ""}
                          {entry.name}
                        </td>
                        {EDITABLE_KEYS.map((key) => (
                          <td key={key} style={{ textAlign: "right" }}>
                            {key === "SHORT_MA_DAYS" || key === "MAIN_MA_DAYS" ? (
                              <select
                                className="form-select form-select-sm"
                                style={{ width: KEY_WIDTHS[key], marginLeft: "auto" }}
                                value={draft[key]}
                                onChange={(e) => updateDraft(id, key, e.target.value)}
                              >
                                {maDayOptions.map((day) => (
                                  <option key={day} value={String(day)}>
                                    {day}일
                                  </option>
                                ))}
                              </select>
                            ) : (
                              <input
                                type="number"
                                className="form-control form-control-sm"
                                style={{ textAlign: "right", width: KEY_WIDTHS[key], marginLeft: "auto" }}
                                value={draft[key]}
                                min={1}
                                max={key === "TOP_N_HOLD" ? 100 : undefined}
                                onChange={(e) => updateDraft(id, key, e.target.value)}
                              />
                            )}
                          </td>
                        ))}
                        <td style={{ textAlign: "center" }}>
                          <button
                            type="button"
                            className="btn btn-sm btn-primary"
                            disabled={!dirty || savingId === id}
                            onClick={() => handleSave(id)}
                          >
                            {savingId === id ? "저장 중…" : "저장"}
                          </button>
                        </td>
                        <td style={{ textAlign: "left", fontSize: "0.82rem", color: "var(--text-muted)", verticalAlign: "middle" }}>
                          {entry.updated_at ? (
                            <span style={{ fontWeight: 600, color: "#475569" }}>
                              {formatKstDateTime(entry.updated_at)}
                            </span>
                          ) : (
                            <span style={{ color: "var(--text-muted)" }}>기록 없음</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
