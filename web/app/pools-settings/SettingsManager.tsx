"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";

import { formatKstDateTime } from "@/lib/datetime";
import { useToast } from "../components/ToastProvider";
import { AppModal } from "../components/AppModal";

/** 숫자 셀렉트/입력으로 편집하는 키. */
const NUMERIC_KEYS = ["TOP_N_HOLD", "SHORT_MA_DAYS", "LONG_MA_DAYS", "SLOPE_DAYS", "BUY_SLIPPAGE_PCT", "SELL_SLIPPAGE_PCT"] as const;

/** 화면 표시 순서 = 헤더 순서. 셀도 반드시 이 순서로 그려야 한다. */
const EDITABLE_KEYS = [...NUMERIC_KEYS, "BENCHMARK", "MARKET_REGIME_INDEX"] as const;

type NumericKey = (typeof NUMERIC_KEYS)[number];
type EditableKey = (typeof EDITABLE_KEYS)[number];

const KEY_LABELS: Record<EditableKey, string> = {
  TOP_N_HOLD: "보유 종목수",
  SHORT_MA_DAYS: "단기 이평선",
  LONG_MA_DAYS: "장기 이평선",
  SLOPE_DAYS: "기울기 일수",
  BUY_SLIPPAGE_PCT: "매수 슬리피지(%)",
  SELL_SLIPPAGE_PCT: "매도 슬리피지(%)",
  BENCHMARK: "벤치마크",
  MARKET_REGIME_INDEX: "시장 레짐",
};

const DEFAULT_MA_DAY_OPTIONS = [5, 10, 20, 40, 60, 120, 240];
const DEFAULT_SLOPE_DAY_OPTIONS = [1, 2, 3, 5, 10, 20, 40, 60];
const DEFAULT_SLIPPAGE_PCT_OPTIONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];
const COUNTRY_OPTIONS = ["kor", "us", "au"] as const;
const CURRENCY_OPTIONS = ["KRW", "USD", "AUD"] as const;

/** 계좌 설정(account_settings.benchmark)과 같은 형태. 미설정이면 null. */
type Benchmark = { ticker?: string; name?: string };
type MarketIndexOption = { ticker: string; name: string };

type SettingField = { value: string | number | Benchmark | MarketIndexOption | null };
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
  constraints: {
    ma_day_options: number[];
    slope_day_options?: number[];
    slippage_pct_options?: number[];
    market_indices?: MarketIndexOption[];
    editable_keys: string[];
  };
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
  marketRegimeTicker: string;
  marketRegimeName: string;
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
  BUY_SLIPPAGE_PCT: "0.25",
  SELL_SLIPPAGE_PCT: "0.25",
  benchmarkTicker: "",
  benchmarkName: "",
  marketRegimeTicker: "",
  marketRegimeName: "",
};

function toBenchmark(field: SettingField | undefined): Benchmark {
  const value = field?.value;
  return value && typeof value === "object" ? (value as Benchmark) : {};
}

function toMarketRegimeIndex(field: SettingField | undefined): MarketIndexOption | null {
  const value = field?.value;
  if (!value || typeof value !== "object") return null;
  const ticker = String((value as MarketIndexOption).ticker ?? "").trim();
  const name = String((value as MarketIndexOption).name ?? "").trim();
  return ticker ? { ticker, name } : null;
}

const inputStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.45)",
  borderRadius: 6,
  padding: "4px 8px",
  fontSize: "var(--fs-sm)",
  minHeight: 30,
};
const labelStyle: React.CSSProperties = { color: "var(--text-muted)", fontWeight: 700, fontSize: "var(--fs-sm)", flexShrink: 0 };

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
    BUY_SLIPPAGE_PCT: String(pool.settings.BUY_SLIPPAGE_PCT?.value ?? ""),
    SELL_SLIPPAGE_PCT: String(pool.settings.SELL_SLIPPAGE_PCT?.value ?? ""),
    benchmarkTicker: toBenchmark(pool.settings.BENCHMARK).ticker ?? "",
    benchmarkName: toBenchmark(pool.settings.BENCHMARK).name ?? "",
    marketRegimeTicker: toMarketRegimeIndex(pool.settings.MARKET_REGIME_INDEX)?.ticker ?? "",
    marketRegimeName: toMarketRegimeIndex(pool.settings.MARKET_REGIME_INDEX)?.name ?? "",
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
    BUY_SLIPPAGE_PCT: Number(draft.BUY_SLIPPAGE_PCT),
    SELL_SLIPPAGE_PCT: Number(draft.SELL_SLIPPAGE_PCT),
    // 티커/이름이 모두 비면 미설정(null). 하나만 있으면 백엔드가 거부한다.
    BENCHMARK: draft.benchmarkTicker.trim()
      ? { ticker: draft.benchmarkTicker.trim().toUpperCase(), name: draft.benchmarkName.trim() }
      : null,
    MARKET_REGIME_INDEX: draft.marketRegimeTicker.trim()
      ? { ticker: draft.marketRegimeTicker.trim(), name: draft.marketRegimeName.trim() }
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
        <span style={{ fontSize: "var(--fs-sm)", fontWeight: 600, whiteSpace: "nowrap" }}>
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

export function SettingsManager({ onSummaryChange }: { onSummaryChange?: (totalCount: number) => void }) {
  const toast = useToast();
  const [data, setData] = useState<PoolSettingsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const [error, setError] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, PoolDraft>>({});
  const [newDraft, setNewDraft] = useState<PoolDraft>(EMPTY_DRAFT);
  const [savingId, setSavingId] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [isCreatingNew, setIsCreatingNew] = useState(false);

  const rows = useMemo(() => {
    if (!data?.pools) return [] as PoolEntry[];
    return [...data.pools].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
  }, [data]);

  useEffect(() => {
    onSummaryChange?.(rows.length);
  }, [rows.length, onSummaryChange]);

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
      setIsCreatingNew(false);
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
  const slippageOptions = data.constraints.slippage_pct_options?.length
    ? data.constraints.slippage_pct_options
    : DEFAULT_SLIPPAGE_PCT_OPTIONS;
  const marketIndices = data.constraints.market_indices ?? [];

  const rowStyle: React.CSSProperties = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 8 };
  const inputLabelStyle: React.CSSProperties = { display: "flex", alignItems: "center", gap: 6, minWidth: 160 };

  const renderField = (
    label: string,
    node: React.ReactNode,
    options: { minWidth?: number; labelWidth?: number } = {},
  ) => (
    <label style={{ ...inputLabelStyle, minWidth: options.minWidth ?? 160 }}>
      <span style={{ ...labelStyle, width: options.labelWidth ?? 72 }}>{label}</span>
      {node}
    </label>
  );

  const renderDraftFormFields = (
    draft: PoolDraft,
    onChange: (key: keyof PoolDraft, value: string) => void,
    idReadonly?: boolean,
  ) => (
    <>
      <div style={rowStyle}>
        {renderField(
          "ID",
          <input
            style={{ ...inputStyle, width: 110, background: idReadonly ? "#f8fafc" : undefined }}
            value={draft.ticker_type}
            readOnly={idReadonly}
            onChange={(event) => onChange("ticker_type", event.target.value)}
          />,
          { minWidth: 178, labelWidth: 44 },
        )}
        {renderField(
          "이름",
          <input style={{ ...inputStyle, width: 220 }} value={draft.name} onChange={(event) => onChange("name", event.target.value)} />,
          { minWidth: 292, labelWidth: 44 },
        )}
        {renderField(
          "아이콘",
          <input style={{ ...inputStyle, width: 56 }} value={draft.icon} onChange={(event) => onChange("icon", event.target.value)} />,
          { minWidth: 126, labelWidth: 56 },
        )}
        {renderField(
          "순서",
          <input
            type="number"
            style={{ ...inputStyle, width: 62, textAlign: "right" }}
            value={draft.order}
            onChange={(event) => onChange("order", event.target.value)}
          />,
          { minWidth: 126, labelWidth: 44 },
        )}
      </div>

      <div style={rowStyle}>
        {renderField(
          "국가",
          <SelectField value={draft.country_code} options={COUNTRY_OPTIONS} width={82} onChange={(value) => onChange("country_code", value)} />,
          { minWidth: 144, labelWidth: 44 },
        )}
        {renderField(
          "통화",
          <SelectField value={draft.currency} options={CURRENCY_OPTIONS} width={88} onChange={(value) => onChange("currency", value)} />,
          { minWidth: 150, labelWidth: 44 },
        )}
        {renderField(
          "보유 종목수",
          <input
            type="number"
            min={1}
            max={100}
            style={{ ...inputStyle, width: 76, textAlign: "right" }}
            value={draft.TOP_N_HOLD}
            onChange={(event) => onChange("TOP_N_HOLD", event.target.value)}
          />,
          { minWidth: 164, labelWidth: 82 },
        )}
        {renderField(
          "단기",
          <SelectField value={draft.SHORT_MA_DAYS} options={maDayOptions} width={82} onChange={(value) => onChange("SHORT_MA_DAYS", value)} />,
          { minWidth: 144, labelWidth: 44 },
        )}
        {renderField(
          "장기",
          <SelectField value={draft.LONG_MA_DAYS} options={maDayOptions} width={82} onChange={(value) => onChange("LONG_MA_DAYS", value)} />,
          { minWidth: 144, labelWidth: 44 },
        )}
      </div>

      <div style={rowStyle}>
        {renderField(
          "기울기",
          <SelectField value={draft.SLOPE_DAYS} options={slopeDayOptions} width={82} onChange={(value) => onChange("SLOPE_DAYS", value)} />,
          { minWidth: 154, labelWidth: 56 },
        )}
        {renderField(
          "매수 슬리피지",
          <SelectField value={draft.BUY_SLIPPAGE_PCT} options={slippageOptions} width={90} onChange={(value) => onChange("BUY_SLIPPAGE_PCT", value)} />,
          { minWidth: 196, labelWidth: 94 },
        )}
        {renderField(
          "매도 슬리피지",
          <SelectField value={draft.SELL_SLIPPAGE_PCT} options={slippageOptions} width={90} onChange={(value) => onChange("SELL_SLIPPAGE_PCT", value)} />,
          { minWidth: 196, labelWidth: 94 },
        )}
        <span style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>% 단위</span>
      </div>

      <div style={{ ...rowStyle, marginBottom: 0 }}>
        <span style={{ ...labelStyle, width: 72 }}>벤치마크</span>
        <BenchmarkField ticker={draft.benchmarkTicker} name={draft.benchmarkName} onChange={onChange} />
        {renderField(
          "시장 레짐",
          <select
            className="form-select form-select-sm"
            style={{ width: 170 }}
            value={draft.marketRegimeTicker}
            onChange={(event) => {
              const ticker = event.target.value;
              const selected = marketIndices.find((item) => item.ticker === ticker);
              onChange("marketRegimeTicker", ticker);
              onChange("marketRegimeName", selected?.name ?? "");
            }}
          >
            <option value="">미설정</option>
            {marketIndices.map((item) => (
              <option key={item.ticker} value={item.ticker}>
                {item.name}
              </option>
            ))}
          </select>,
          { minWidth: 260, labelWidth: 72 },
        )}
      </div>
    </>
  );

  const renderDraftCard = ({
    title,
    subtitle,
    draft,
    onChange,
    idReadonly,
    updatedAt,
    primaryButton,
    secondaryButton,
  }: {
    title: string;
    subtitle: string;
    draft: PoolDraft;
    onChange: (key: keyof PoolDraft, value: string) => void;
    idReadonly?: boolean;
    updatedAt?: string;
    primaryButton: React.ReactNode;
    secondaryButton: React.ReactNode;
  }) => (
    <div style={{ border: "1px solid rgba(148,163,184,0.25)", borderRadius: 10, padding: "10px 12px", background: "#fff" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10, flexWrap: "wrap", marginBottom: 8 }}>
        <div>
          <div style={{ fontWeight: 850, fontSize: "var(--fs-base)" }}>{title}</div>
          <div style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>{subtitle}</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          {updatedAt ? <span style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>{formatKstDateTime(updatedAt)}</span> : null}
          {primaryButton}
          {secondaryButton}
        </div>
      </div>
      {renderDraftFormFields(draft, onChange, idReadonly)}
    </div>
  );

  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <div>
                  <h2 style={{ fontSize: "var(--fs-lg)", fontWeight: 800, margin: 0 }}>종목풀 설정</h2>
                  <div className="tableFooterMeta" style={{ margin: 0, color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
                    종목풀 구조와 이평선 설정은 DB(pool_settings)가 단일 소스입니다.
                  </div>
                </div>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button type="button" className="btn btn-sm btn-primary" onClick={() => setIsCreatingNew(!isCreatingNew)}>
                  등록
                </button>
                <button type="button" className="btn btn-sm btn-outline-secondary" disabled={loading} onClick={() => void load()}>
                  새로고침
                </button>
              </div>
            </div>
          </div>

          <div className="card-body appCardBodyTight appTableCardBodyFill" style={{ overflowY: "auto", padding: 12 }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(max(520px, calc(50% - 5px)), 1fr))", gap: 10 }}>
              {rows.map((pool) => {
                const draft = drafts[pool.ticker_type] ?? toDraft(pool);
                const dirty = isDirty(draft, pool);
                return (
                  <React.Fragment key={pool.ticker_type}>
                    {renderDraftCard({
                  title: `${pool.icon ?? ""} ${pool.name}`.trim(),
                  subtitle: pool.ticker_type,
                  draft,
                  onChange: (key, value) => updateDraft(pool.ticker_type, key, value),
                  idReadonly: true,
                  updatedAt: pool.updated_at,
                  primaryButton: (
                    <button
                      type="button"
                      className="btn btn-sm btn-primary"
                      disabled={!dirty || savingId === pool.ticker_type}
                      onClick={() => void handleSave(pool)}
                    >
                      {savingId === pool.ticker_type ? "저장 중…" : "저장"}
                    </button>
                  ),
                  secondaryButton: (
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-danger"
                      disabled={deletingId === pool.ticker_type}
                      onClick={() => void handleDelete(pool)}
                    >
                      {deletingId === pool.ticker_type ? "삭제 중…" : "삭제"}
                    </button>
                  ),
                    })}
                  </React.Fragment>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      <AppModal
        open={isCreatingNew}
        title="신규 종목풀"
        subtitle="ticker_type 은 생성 후 변경할 수 없습니다."
        onClose={() => {
          setNewDraft(EMPTY_DRAFT);
          setIsCreatingNew(false);
        }}
        size="xl"
        footer={
          <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, width: "100%" }}>
            <button
              type="button"
              className="btn btn-outline-secondary"
              onClick={() => {
                setNewDraft(EMPTY_DRAFT);
                setIsCreatingNew(false);
              }}
            >
              취소
            </button>
            <button
              type="button"
              className="btn btn-primary"
              disabled={creating || !newDraft.ticker_type.trim() || !newDraft.name.trim()}
              onClick={() => void handleCreate()}
            >
              {creating ? "추가 중…" : "추가"}
            </button>
          </div>
        }
      >
        <div style={{ display: "grid", gap: 8 }}>
          {renderDraftFormFields(newDraft, updateNewDraft)}
        </div>
      </AppModal>
    </div>
  );
}
