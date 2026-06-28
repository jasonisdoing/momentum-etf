"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { formatKstDateTime } from "@/lib/datetime";
import { useToast } from "../components/ToastProvider";

const EDITABLE_KEYS = [
  "TOP_N_HOLD",
  "HOLDING_BONUS_SCORE",
  "ATH_BONUS",
  "MA_TYPE",
  "MA_MONTHS",
  "RSI_LIMIT",
] as const;

type EditableKey = (typeof EDITABLE_KEYS)[number];

const KEY_LABELS: Record<EditableKey, string> = {
  TOP_N_HOLD: "보유 종목수",
  HOLDING_BONUS_SCORE: "보유 보너스",
  MA_TYPE: "MA 타입",
  MA_MONTHS: "MA 개월",
  RSI_LIMIT: "RSI 상한",
  ATH_BONUS: "ATH 보너스",
};

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
  save_method?: string;
};

type PoolSettingsResponse = {
  all: PoolEntry;
  pools: PoolEntry[];
  constraints: { ma_types: string[]; ma_months_max: number; editable_keys: string[] };
  error?: string;
};

/** 한 행의 편집 중인 값 (모두 문자열로 보관, 저장 시 파싱). */
type RowDraft = Record<EditableKey, string>;

/** 보유/ATH 보너스 셀렉트 옵션 — /momentum-pools 와 동일한 0~50 (5단위). 현재값이 비표준이면 포함해 보존. */
function bonusOptions(current: string): number[] {
  const base = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
  const cur = Number(current);
  if (Number.isFinite(cur) && !base.includes(cur)) {
    return [...base, cur].sort((a, b) => a - b);
  }
  return base;
}

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
    const all = { id: data.all.pool_id ?? "__all__", entry: data.all };
    const pools = [...data.pools]
      .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
      .map((p) => ({ id: p.ticker_type ?? "", entry: p }));
    return [all, ...pools];
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
      nextDrafts[payload.all.pool_id ?? "__all__"] = toDraft(payload.all.settings);
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

  const maTypes = data.constraints.ma_types;
  const monthsMax = data.constraints.ma_months_max;

  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBodyTight">
            <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>종목풀 설정</h2>
            <p className="tableFooterMeta" style={{ marginBottom: 12, color: "#94a3b8", fontSize: "0.85rem" }}>
              종목풀의 구조(이름/순서/국가 등)는 pools.json 이 유지하고, 아래 6개 값은 DB 에서 저장·수정합니다.
            </p>
            <div style={{ overflowX: "auto" }}>
              <table className="table table-sm appSettingsTable" style={{ minWidth: 720 }}>
                <thead>
                  <tr>
                    <th style={{ minWidth: 160 }}>종목풀</th>
                    {EDITABLE_KEYS.map((key) => (
                      <th key={key} style={{ textAlign: "right", minWidth: 96 }}>
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
                    const isAll = id === "__all__";
                    return (
                      <tr key={id} style={isAll ? { background: "#f8fafc", fontWeight: 600 } : undefined}>
                        <td>
                          {entry.icon ? `${entry.icon} ` : ""}
                          {entry.name}
                        </td>
                        {EDITABLE_KEYS.map((key) => (
                          <td key={key} style={{ textAlign: "right" }}>
                            {key === "MA_TYPE" ? (
                              <select
                                className="form-select form-select-sm"
                                value={draft[key]}
                                onChange={(e) => updateDraft(id, key, e.target.value)}
                              >
                                {maTypes.map((t) => (
                                  <option key={t} value={t}>
                                    {t}
                                  </option>
                                ))}
                              </select>
                            ) : key === "HOLDING_BONUS_SCORE" || key === "ATH_BONUS" ? (
                              <select
                                className="form-select form-select-sm"
                                value={draft[key]}
                                onChange={(e) => updateDraft(id, key, e.target.value)}
                              >
                                {bonusOptions(draft[key]).map((score) => (
                                  <option key={score} value={String(score)}>
                                    {score}
                                  </option>
                                ))}
                              </select>
                            ) : (
                              <input
                                type="number"
                                className="form-control form-control-sm"
                                style={{ textAlign: "right" }}
                                value={draft[key]}
                                min={1}
                                max={key === "MA_MONTHS" ? monthsMax : key === "RSI_LIMIT" || key === "TOP_N_HOLD" ? 100 : undefined}
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
                        <td style={{ textAlign: "left", fontSize: "0.82rem", color: "#64748b", verticalAlign: "middle" }}>
                          {entry.updated_at ? (
                            <div style={{ display: "flex", flexDirection: "column", lineHeight: 1.25 }}>
                              <span style={{ fontWeight: 600, color: "#475569" }}>
                                {formatKstDateTime(entry.updated_at)}
                              </span>
                              <span style={{ fontSize: "0.72rem", color: "#94a3b8" }}>
                                방식: {entry.save_method || "미지정"}
                              </span>
                            </div>
                          ) : (
                            <span style={{ color: "#cbd5e1" }}>기록 없음</span>
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
