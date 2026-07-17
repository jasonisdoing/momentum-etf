"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { formatKstDateTime } from "@/lib/datetime";
import { useToast } from "../components/ToastProvider";

/** 숫자 셀렉트/입력으로 편집하는 키. */
const NUMERIC_KEYS = ["TOP_N_HOLD", "SHORT_MA_DAYS", "LONG_MA_DAYS", "SLOPE_DAYS"] as const;

/** 화면 표시 순서 = 헤더 순서. 셀도 반드시 이 순서로 그려야 한다. */
const EDITABLE_KEYS = [...NUMERIC_KEYS, "BENCHMARK"] as const;

type NumericKey = (typeof NUMERIC_KEYS)[number];
type EditableKey = (typeof EDITABLE_KEYS)[number];

const KEY_LABELS: Record<EditableKey, string> = {
  TOP_N_HOLD: "보유 종목수",
  SHORT_MA_DAYS: "단기 이평선",
  LONG_MA_DAYS: "장기 이평선",
  SLOPE_DAYS: "기울기 일수",
  BENCHMARK: "벤치마크",
};

const DEFAULT_MA_DAY_OPTIONS = [5, 10, 20, 40, 60, 120, 240];
const DEFAULT_SLOPE_DAY_OPTIONS = [1, 2, 3, 5, 10, 20, 40, 60];
const COUNTRY_OPTIONS = ["kor", "us", "au"] as const;
const CURRENCY_OPTIONS = ["KRW", "USD", "AUD"] as const;

/** 계좌 설정(account_settings.benchmark)과 같은 형태. 미설정이면 null. */
type Benchmark = { ticker?: string; name?: string };

type SettingField = { value: string | number | Benchmark | null };
type SettingsMap = Record<EditableKey, SettingField>;

type PoolEntry = {
  ticker_type: string;
  name: string;
  icon?: string;
  order?: number;
  country_code?: string;
  currency?: string;
  settings: SettingsMap;
  updated_at?: string;
};

type PoolSettingsResponse = {
  pools: PoolEntry[];
  constraints: { ma_day_options: number[]; slope_day_options?: number[]; editable_keys: string[] };
  error?: string;
};

type PoolDraft = {
  ticker_type: string;
  name: string;
  icon: string;
  order: string;
  country_code: string;
  currency: string;
  // 벤치마크는 {ticker, name} 이라 초안에서는 두 값으로 나눠 든다. 이름은 조회로만 채운다.
  benchmarkTicker: string;
  benchmarkName: string;
} & Record<NumericKey, string>;

const EMPTY_DRAFT: PoolDraft = {
  ticker_type: "",
  name: "",
  icon: "",
  order: "",
  country_code: "kor",
  currency: "KRW",
  TOP_N_HOLD: "10",
  SHORT_MA_DAYS: "10",
  LONG_MA_DAYS: "20",
  SLOPE_DAYS: "5",
  benchmarkTicker: "",
  benchmarkName: "",
};

function toBenchmark(field: SettingField | undefined): Benchmark {
  const value = field?.value;
  return value && typeof value === "object" ? (value as Benchmark) : {};
}

const inputStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.45)",
  borderRadius: 6,
  padding: "4px 8px",
  fontSize: "0.86rem",
  minHeight: 30,
};

function toDraft(pool: PoolEntry): PoolDraft {
  return {
    ticker_type: pool.ticker_type ?? "",
    name: pool.name ?? "",
    icon: pool.icon ?? "",
    order: pool.order === null || pool.order === undefined ? "" : String(pool.order),
    country_code: pool.country_code ?? "kor",
    currency: pool.currency ?? "KRW",
    TOP_N_HOLD: String(pool.settings.TOP_N_HOLD?.value ?? ""),
    SHORT_MA_DAYS: String(pool.settings.SHORT_MA_DAYS?.value ?? ""),
    LONG_MA_DAYS: String(pool.settings.LONG_MA_DAYS?.value ?? ""),
    SLOPE_DAYS: String(pool.settings.SLOPE_DAYS?.value ?? ""),
    benchmarkTicker: toBenchmark(pool.settings.BENCHMARK).ticker ?? "",
    benchmarkName: toBenchmark(pool.settings.BENCHMARK).name ?? "",
  };
}

function draftToValues(draft: PoolDraft) {
  return {
    ticker_type: draft.ticker_type.trim().toLowerCase(),
    name: draft.name.trim(),
    icon: draft.icon.trim(),
    order: Number(draft.order),
    country_code: draft.country_code,
    currency: draft.currency,
    TOP_N_HOLD: Number(draft.TOP_N_HOLD),
    SHORT_MA_DAYS: Number(draft.SHORT_MA_DAYS),
    LONG_MA_DAYS: Number(draft.LONG_MA_DAYS),
    SLOPE_DAYS: Number(draft.SLOPE_DAYS),
    // 티커/이름이 모두 비면 미설정(null). 하나만 있으면 백엔드가 거부한다.
    BENCHMARK: draft.benchmarkTicker.trim()
      ? { ticker: draft.benchmarkTicker.trim().toUpperCase(), name: draft.benchmarkName.trim() }
      : null,
  };
}

function isDirty(draft: PoolDraft, original: PoolEntry) {
  const baseline = toDraft(original);
  return Object.keys(baseline).some((key) => draft[key as keyof PoolDraft] !== baseline[key as keyof PoolDraft]);
}

function SelectField({
  value,
  options,
  width,
  onChange,
}: {
  value: string;
  options: readonly string[] | number[];
  width: number;
  onChange: (value: string) => void;
}) {
  return (
    <select className="form-select form-select-sm" style={{ width }} value={value} onChange={(event) => onChange(event.target.value)}>
      {options.map((option) => (
        <option key={String(option)} value={String(option)}>
          {String(option)}
        </option>
      ))}
    </select>
  );
}

/** 벤치마크 등록/변경 — 계좌 설정(`/account-settings`)과 같은 방식.
 *
 * 티커를 입력해 조회하면 종목명이 채워지고 확정된다. 이름은 조회로만 채우므로
 * 존재하지 않는 종목이 저장될 수 없다. 티커를 비우면 미설정.
 */
function BenchmarkField({
  ticker,
  name,
  onChange,
}: {
  ticker: string;
  name: string;
  onChange: (key: keyof PoolDraft, value: string) => void;
}) {
  const toast = useToast();
  const [editing, setEditing] = useState(!(ticker && name));
  const [resolving, setResolving] = useState(false);

  const resolve = async () => {
    const target = ticker.trim();
    if (!target) {
      toast.error("티커를 입력해주세요.");
      return;
    }
    try {
      setResolving(true);
      const resp = await fetch(`/api/leverage-config/resolve?ticker=${encodeURIComponent(target)}`);
      const payload = (await resp.json()) as { name?: string; error?: string };
      if (!resp.ok || payload.error || !payload.name) {
        toast.error(payload.error ?? "종목명을 찾을 수 없습니다.");
        return;
      }
      onChange("benchmarkName", payload.name);
      setEditing(false);
      toast.success(`${payload.name}(${target}) 확인 완료`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "티커 조회 중 오류가 발생했습니다.");
    } finally {
      setResolving(false);
    }
  };

  if (!editing) {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ fontSize: "0.86rem", fontWeight: 600, whiteSpace: "nowrap" }}>
          {name} <span style={{ color: "var(--text-muted)", fontWeight: 500 }}>({ticker})</span>
        </span>
        <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => setEditing(true)}>
          변경
        </button>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <input
        style={{ ...inputStyle, width: 96 }}
        placeholder="티커"
        title="비우면 미설정"
        value={ticker}
        onChange={(event) => {
          onChange("benchmarkTicker", event.target.value);
          onChange("benchmarkName", "");
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            void resolve();
          }
        }}
      />
      <button type="button" className="btn btn-sm btn-outline-secondary" disabled={resolving} onClick={() => void resolve()}>
        {resolving ? "조회 중…" : "조회"}
      </button>
    </div>
  );
}

export function SettingsManager() {
  const toast = useToast();
  const [data, setData] = useState<PoolSettingsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, PoolDraft>>({});
  const [newDraft, setNewDraft] = useState<PoolDraft>(EMPTY_DRAFT);
  const [savingId, setSavingId] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const rows = useMemo(() => {
    if (!data) return [] as PoolEntry[];
    return [...data.pools].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
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
      const nextDrafts: Record<string, PoolDraft> = {};
      payload.pools.forEach((pool) => {
        nextDrafts[pool.ticker_type] = toDraft(pool);
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

  const updateDraft = useCallback((id: string, key: keyof PoolDraft, value: string) => {
    setDrafts((prev) => ({ ...prev, [id]: { ...prev[id], [key]: value } }));
  }, []);

  const updateNewDraft = useCallback((key: keyof PoolDraft, value: string) => {
    setNewDraft((prev) => ({ ...prev, [key]: value }));
  }, []);

  const handleCreate = useCallback(async () => {
    setCreating(true);
    try {
      const resp = await fetch("/api/pool-settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ values: draftToValues(newDraft) }),
      });
      const payload = await resp.json();
      if (!resp.ok || payload.error) {
        throw new Error(payload.error ?? payload.detail ?? "종목풀 생성에 실패했습니다.");
      }
      toast.success("종목풀을 추가했습니다.");
      setNewDraft(EMPTY_DRAFT);
      await load();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "종목풀 생성에 실패했습니다.");
    } finally {
      setCreating(false);
    }
  }, [load, newDraft, toast]);

  const handleSave = useCallback(
    async (pool: PoolEntry) => {
      const draft = drafts[pool.ticker_type];
      if (!draft) return;
      setSavingId(pool.ticker_type);
      try {
        const resp = await fetch("/api/pool-settings", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pool_id: pool.ticker_type, values: draftToValues(draft) }),
        });
        const payload = await resp.json();
        if (!resp.ok || payload.error) {
          throw new Error(payload.error ?? payload.detail ?? "저장에 실패했습니다.");
        }
        toast.success("종목풀을 저장했습니다.");
        await load();
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "저장에 실패했습니다.");
      } finally {
        setSavingId(null);
      }
    },
    [drafts, load, toast],
  );

  const handleDelete = useCallback(
    async (pool: PoolEntry) => {
      const message = [
        `'${pool.name}' (${pool.ticker_type}) 종목풀을 하드 삭제합니다.`,
        "계좌에 연결된 종목풀은 서버에서 삭제를 차단합니다.",
        "삭제 시 이 종목풀에 등록된 종목 메타도 함께 제거됩니다.",
        "계속할까요?",
      ].join("\n");
      if (!window.confirm(message)) return;

      setDeletingId(pool.ticker_type);
      try {
        const resp = await fetch("/api/pool-settings", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pool_id: pool.ticker_type }),
        });
        const payload = await resp.json();
        if (!resp.ok || payload.error) {
          throw new Error(payload.error ?? payload.detail ?? "삭제에 실패했습니다.");
        }
        const deletedStocks = payload.deleted?.deleted_stocks ?? 0;
        toast.success(`종목풀을 삭제했습니다. 제거된 종목: ${deletedStocks}개`);
        await load();
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "삭제에 실패했습니다.");
      } finally {
        setDeletingId(null);
      }
    },
    [load, toast],
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
  const slopeDayOptions = data.constraints.slope_day_options?.length
    ? data.constraints.slope_day_options
    : DEFAULT_SLOPE_DAY_OPTIONS;

  const renderDraftCells = (draft: PoolDraft, onChange: (key: keyof PoolDraft, value: string) => void, options?: { idReadonly?: boolean }) => (
    <>
      <td>
        <input
          style={{ ...inputStyle, width: 96, background: options?.idReadonly ? "#f8fafc" : undefined }}
          value={draft.ticker_type}
          readOnly={options?.idReadonly}
          onChange={(event) => onChange("ticker_type", event.target.value)}
        />
      </td>
      <td>
        <input style={{ ...inputStyle, width: 160 }} value={draft.name} onChange={(event) => onChange("name", event.target.value)} />
      </td>
      <td>
        <input style={{ ...inputStyle, width: 56 }} value={draft.icon} onChange={(event) => onChange("icon", event.target.value)} />
      </td>
      <td>
        <input
          type="number"
          style={{ ...inputStyle, width: 64, textAlign: "right" }}
          value={draft.order}
          onChange={(event) => onChange("order", event.target.value)}
        />
      </td>
      <td>
        <SelectField value={draft.country_code} options={COUNTRY_OPTIONS} width={76} onChange={(value) => onChange("country_code", value)} />
      </td>
      <td>
        <SelectField value={draft.currency} options={CURRENCY_OPTIONS} width={82} onChange={(value) => onChange("currency", value)} />
      </td>
      <td>
        <input
          type="number"
          min={1}
          max={100}
          style={{ ...inputStyle, width: 84, textAlign: "right" }}
          value={draft.TOP_N_HOLD}
          onChange={(event) => onChange("TOP_N_HOLD", event.target.value)}
        />
      </td>
      <td>
        <SelectField value={draft.SHORT_MA_DAYS} options={maDayOptions} width={90} onChange={(value) => onChange("SHORT_MA_DAYS", value)} />
      </td>
      <td>
        <SelectField value={draft.LONG_MA_DAYS} options={maDayOptions} width={90} onChange={(value) => onChange("LONG_MA_DAYS", value)} />
      </td>
      <td>
        <SelectField value={draft.SLOPE_DAYS} options={slopeDayOptions} width={90} onChange={(value) => onChange("SLOPE_DAYS", value)} />
      </td>
      <td>
        <BenchmarkField ticker={draft.benchmarkTicker} name={draft.benchmarkName} onChange={onChange} />
      </td>
    </>
  );

  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBodyTight">
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "flex-start", marginBottom: 10 }}>
              <div>
                <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>종목풀 설정</h2>
                <p className="tableFooterMeta" style={{ marginBottom: 0, color: "var(--text-muted)", fontSize: "0.85rem" }}>
                  종목풀 구조와 이평선 설정은 DB(pool_settings)가 단일 소스입니다. ticker_type 은 생성 후 변경할 수 없습니다.
                </p>
              </div>
              <button type="button" className="btn btn-sm btn-outline-secondary" disabled={loading} onClick={() => void load()}>
                새로고침
              </button>
            </div>

            <div style={{ overflowX: "auto" }}>
              <table className="table table-sm appSettingsTable" style={{ minWidth: 1120 }}>
                <thead>
                  <tr>
                    <th style={{ width: 104 }}>ID</th>
                    <th style={{ width: 168 }}>종목풀명</th>
                    <th style={{ width: 64 }}>아이콘</th>
                    <th style={{ width: 72, textAlign: "right" }}>순서</th>
                    <th style={{ width: 84 }}>국가</th>
                    <th style={{ width: 90 }}>통화</th>
                    {EDITABLE_KEYS.map((key) => (
                      <th key={key} style={{ whiteSpace: "nowrap", width: 98 }}>
                        {KEY_LABELS[key]}
                      </th>
                    ))}
                    <th style={{ textAlign: "center", width: 86 }}>저장</th>
                    <th style={{ textAlign: "center", width: 86 }}>삭제</th>
                    <th style={{ textAlign: "left", minWidth: 150 }}>마지막 저장</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ background: "#f8fafc" }}>
                    {renderDraftCells(newDraft, updateNewDraft)}
                    <td style={{ textAlign: "center" }}>
                      <button
                        type="button"
                        className="btn btn-sm btn-primary"
                        disabled={creating || !newDraft.ticker_type.trim() || !newDraft.name.trim()}
                        onClick={() => void handleCreate()}
                      >
                        {creating ? "추가 중…" : "추가"}
                      </button>
                    </td>
                    <td style={{ textAlign: "center" }}>
                      <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => setNewDraft(EMPTY_DRAFT)}>
                        초기화
                      </button>
                    </td>
                    <td style={{ color: "var(--text-muted)", fontSize: "0.82rem" }}>신규 종목풀</td>
                  </tr>
                  {rows.map((pool) => {
                    const draft = drafts[pool.ticker_type] ?? toDraft(pool);
                    const dirty = isDirty(draft, pool);
                    return (
                      <tr key={pool.ticker_type}>
                        {renderDraftCells(draft, (key, value) => updateDraft(pool.ticker_type, key, value), { idReadonly: true })}
                        <td style={{ textAlign: "center" }}>
                          <button
                            type="button"
                            className="btn btn-sm btn-primary"
                            disabled={!dirty || savingId === pool.ticker_type}
                            onClick={() => void handleSave(pool)}
                          >
                            {savingId === pool.ticker_type ? "저장 중…" : "저장"}
                          </button>
                        </td>
                        <td style={{ textAlign: "center" }}>
                          <button
                            type="button"
                            className="btn btn-sm btn-outline-danger"
                            disabled={deletingId === pool.ticker_type}
                            onClick={() => void handleDelete(pool)}
                          >
                            {deletingId === pool.ticker_type ? "삭제 중…" : "삭제"}
                          </button>
                        </td>
                        <td style={{ textAlign: "left", fontSize: "0.82rem", color: "var(--text-muted)", verticalAlign: "middle" }}>
                          {pool.updated_at ? (
                            <span style={{ fontWeight: 600, color: "#475569" }}>{formatKstDateTime(pool.updated_at)}</span>
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
