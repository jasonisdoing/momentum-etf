"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import type { ColDef, ICellRendererParams } from "ag-grid-community";

import { formatKstDateTime } from "@/lib/datetime";
import { AppAgGrid } from "../components/AppAgGrid";
import { createAppGridTheme } from "../components/app-grid-theme";
import { LastSavedCell } from "../components/LastSavedCell";
import { MaDaysSelect, type MaOptionsPayload } from "../components/MaDaysSelect";
import { UnsavedChangesBadge } from "../components/UnsavedChangesBadge";
import { useToast } from "../components/ToastProvider";
import { AppModal } from "../components/AppModal";
import { ensureAsxPrefix } from "../components/TickerDetailLink";

/** 숫자 셀렉트/입력으로 편집하는 키. */
const NUMERIC_KEYS = [
  "TOP_N_HOLD",
  "SHORT_MA_DAYS",
  "LONG_MA_DAYS",
  "BUY_SLIPPAGE_PCT",
  "SELL_SLIPPAGE_PCT",
  "STOPLOSS_THRESHOLD_PCT",
] as const;

/** 화면 표시 순서 = 헤더 순서. 셀도 반드시 이 순서로 그려야 한다. */
const EDITABLE_KEYS = [...NUMERIC_KEYS, "BENCHMARK", "MARKET_REGIME_INDEX"] as const;

type NumericKey = (typeof NUMERIC_KEYS)[number];
type EditableKey = (typeof EDITABLE_KEYS)[number];

const KEY_LABELS: Record<EditableKey, string> = {
  TOP_N_HOLD: "보유종목 수",
  SHORT_MA_DAYS: "단기 이평선",
  LONG_MA_DAYS: "장기 이평선",
  BUY_SLIPPAGE_PCT: "매수 슬리피지(%)",
  SELL_SLIPPAGE_PCT: "매도 슬리피지(%)",
  STOPLOSS_THRESHOLD_PCT: "손절 기준(%)",
  BENCHMARK: "벤치마크",
  MARKET_REGIME_INDEX: "ADR 기준",
};

const DEFAULT_SLIPPAGE_PCT_OPTIONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];
// 손절 기준(%) — 백엔드 pool_settings_store.STOPLOSS_PCT_OPTIONS 가 단일 소스. 응답이 없을 때만 쓰는 대체값.
const DEFAULT_STOPLOSS_PCT_OPTIONS = [-7, -10];
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
  // 풀 성격(stock/etf) — 미설정이면 null. 전략 SM 등이 섹터·업종 UI 노출에 쓴다.
  pool_kind?: string | null;
  /** 그 풀에 담긴 종목 수 — 현황이라 편집 대상이 아니다(설정 초안에 넣지 않는다). */
  stock_count?: number;
  settings: SettingsMap;
  updated_at?: string;
  /** 저장 경로 — "사용자"(사람이 화면에서 저장) / "모멘텀 전략"(전략 화면·튜닝 적용). */
  save_method?: string;
};

/** 전략 하나의 12개월 백테스트 요약 — 버튼을 눌러야 채워진다. */
type StrategyBacktest = {
  cagr_pct: number | null;
  mdd_pct: number | null;
  sortino: number | null;
  /** 그 전략의 저장 설정이 없어 돌릴 수 없었던 경우. */
  no_settings?: boolean;
  /** 계산해 저장한 시각. 설정이 바뀌면 서버가 지운다. */
  updated_at?: string | null;
};

/** 종목풀 → 전략 → 결과. 값이 없으면 아직 안 돌린 것이다. */
type BacktestByPool = Record<string, Partial<Record<StrategyKey, StrategyBacktest>>>;

/** 소수 자리 맞춘 숫자 — 값이 없으면 빈칸(설정이 없거나 못 돌린 전략). */
function fmt(value: number | null, digits: number): string {
  return value == null ? "-" : value.toFixed(digits);
}

const STRATEGY_COLUMNS: { key: StrategyKey; label: string }[] = [
  { key: "momentum", label: "모멘텀" },
  { key: "new_high", label: "신고가" },
  { key: "portfolio", label: "포트폴리오" },
];

type StrategyKey = "momentum" | "new_high" | "portfolio";

type PoolSettingsResponse = {
  pools: PoolEntry[];
  constraints: {
    /** 이평선 선택지 — 국가별(풀의 country_code 로 고른다). 백엔드 utils/ma_options 가 단일 소스. */
    ma_options_by_country: Record<string, MaOptionsPayload>;
    top_n_hold_options: number[];
    slippage_pct_options?: number[];
    stoploss_pct_options?: number[];
    market_indices?: MarketIndexOption[];
    editable_keys: string[];
  };
  /** 전략별 12개월 백테스트 — 저장된 값. 설정이 바뀌면 서버가 지운다. */
  backtests?: BacktestByPool;
  error?: string;
};

type PoolDraft = {
  ticker_type: string;
  name: string;
  icon: string;
  order: string;
  country_code: string;
  currency: string;
  // 풀 성격 — "stock"(개별주) / "etf" / ""(미설정, 구 문서 하위 호환).
  pool_kind: string;
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
  pool_kind: "etf",
  TOP_N_HOLD: "5",
  // 이평선은 국가마다 선택지가 달라 여기서 못 정한다 — `withDefaultMaDays` 가 채운다.
  SHORT_MA_DAYS: "",
  LONG_MA_DAYS: "",
  // 슬리피지는 보수적으로 — 낙관적인 값으로 열면 백테스트가 실제보다 좋게 나온다.
  BUY_SLIPPAGE_PCT: "0.5",
  SELL_SLIPPAGE_PCT: "0.5",
  STOPLOSS_THRESHOLD_PCT: "-10",
  benchmarkTicker: "",
  benchmarkName: "",
  marketRegimeTicker: "",
  marketRegimeName: "",
};

/** 이평선 기본값을 그 국가 선택지 중 **가장 작은 값**으로 채운다.
 *
 *  선택지에 없는 값(예전 기본값 10/20)을 초안에 두면 셀렉트가 빈 채로 열리고, 사용자가
 *  건드리지 않으면 저장할 수 없는 값이 그대로 남는다. 목록은 국가마다 달라 상수로 못 둔다. */
function withDefaultMaDays(
  draft: PoolDraft,
  optionsByCountry: Record<string, { short_ma_options: number[]; long_ma_options: number[] }>,
): PoolDraft {
  const options = optionsByCountry[draft.country_code];
  if (!options) return draft;
  const smallest = (days: number[]) => (days.length ? String(Math.min(...days)) : "");
  return {
    ...draft,
    SHORT_MA_DAYS: options.short_ma_options.includes(Number(draft.SHORT_MA_DAYS))
      ? draft.SHORT_MA_DAYS
      : smallest(options.short_ma_options),
    LONG_MA_DAYS: options.long_ma_options.includes(Number(draft.LONG_MA_DAYS))
      ? draft.LONG_MA_DAYS
      : smallest(options.long_ma_options),
  };
}

/** 그리드 행 — 편집 중인 초안 그대로에 표시용 필드를 얹는다. */
type PoolGridRow = PoolDraft & {
  __dirty: boolean;
  __updatedAt?: string;
  __saveMethod?: string;
  __stockCount?: number;
};

// 셀렉트 에디터가 들어가는 행이라 기본(34px)보다 조금 높인다.
const poolSettingsGridTheme = createAppGridTheme({ rowHeight: 38 });

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
    pool_kind: pool.pool_kind ?? "",
    TOP_N_HOLD: String(pool.settings.TOP_N_HOLD?.value ?? ""),
    SHORT_MA_DAYS: String(pool.settings.SHORT_MA_DAYS?.value ?? ""),
    LONG_MA_DAYS: String(pool.settings.LONG_MA_DAYS?.value ?? ""),
    BUY_SLIPPAGE_PCT: String(pool.settings.BUY_SLIPPAGE_PCT?.value ?? ""),
    SELL_SLIPPAGE_PCT: String(pool.settings.SELL_SLIPPAGE_PCT?.value ?? ""),
    STOPLOSS_THRESHOLD_PCT: String(pool.settings.STOPLOSS_THRESHOLD_PCT?.value ?? ""),
    benchmarkTicker: toBenchmark(pool.settings.BENCHMARK).ticker ?? "",
    benchmarkName: toBenchmark(pool.settings.BENCHMARK).name ?? "",
    marketRegimeTicker: toMarketRegimeIndex(pool.settings.MARKET_REGIME_INDEX)?.ticker ?? "",
    marketRegimeName: toMarketRegimeIndex(pool.settings.MARKET_REGIME_INDEX)?.name ?? "",
  };
}

function draftToValues(draft: PoolDraft) {
  const benchmarkTicker =
    draft.country_code === "au"
      ? ensureAsxPrefix(draft.benchmarkTicker)
      : draft.benchmarkTicker.trim().toUpperCase();
  return {
    ticker_type: draft.ticker_type.trim().toLowerCase(),
    name: draft.name.trim(),
    icon: draft.icon.trim(),
    order: Number(draft.order),
    country_code: draft.country_code,
    currency: draft.currency,
    // 빈 값(미설정)은 보내지 않아 기존 상태를 유지한다 — 토글은 항상 stock/etf 를 보낸다.
    ...(draft.pool_kind ? { pool_kind: draft.pool_kind } : {}),
    TOP_N_HOLD: Number(draft.TOP_N_HOLD),
    SHORT_MA_DAYS: Number(draft.SHORT_MA_DAYS),
    LONG_MA_DAYS: Number(draft.LONG_MA_DAYS),
    BUY_SLIPPAGE_PCT: Number(draft.BUY_SLIPPAGE_PCT),
    SELL_SLIPPAGE_PCT: Number(draft.SELL_SLIPPAGE_PCT),
    STOPLOSS_THRESHOLD_PCT: Number(draft.STOPLOSS_THRESHOLD_PCT),
    // 티커/이름이 모두 비면 미설정(null). 하나만 있으면 백엔드가 거부한다.
    BENCHMARK: benchmarkTicker
      ? { ticker: benchmarkTicker, name: draft.benchmarkName.trim() }
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
  countryCode,
  tickerTypes,
  onChange,
}: {
  ticker: string;
  name: string;
  countryCode: string;
  tickerTypes: string[];
  onChange: (key: keyof PoolDraft, value: string) => void;
}) {
  const toast = useToast();
  const [editing, setEditing] = useState(!(ticker && name));
  const [resolving, setResolving] = useState(false);

  const resolve = async () => {
    const country = countryCode.trim().toLowerCase();
    const target = country === "au" ? ensureAsxPrefix(ticker) : ticker.trim().toUpperCase();
    if (!target) {
      toast.error("티커를 입력해주세요.");
      return;
    }
    if (tickerTypes.length === 0) {
      toast.error(`국가(${country})에 등록된 종목풀이 없습니다.`);
      return;
    }
    try {
      setResolving(true);
      const resp = await fetch(
        `/api/ticker-resolve?ticker=${encodeURIComponent(target)}&ticker_types=${encodeURIComponent(tickerTypes.join(","))}`,
      );
      const payload = (await resp.json()) as {
        ticker?: string;
        name?: string;
        country_code?: string;
        error?: string;
      };
      if (!resp.ok || payload.error || !payload.ticker || !payload.name) {
        toast.error(payload.error ?? "종목명을 찾을 수 없습니다.");
        return;
      }
      if (String(payload.country_code ?? "").toLowerCase() !== country) {
        toast.error(`국가(${country}) 종목만 벤치마크로 등록할 수 있습니다.`);
        return;
      }
      onChange("benchmarkTicker", payload.ticker);
      onChange("benchmarkName", payload.name);
      setEditing(false);
      toast.success(`${payload.name}(${payload.ticker}) 확인 완료`);
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
  // 전략별 12개월 백테스트 — 버튼을 눌렀을 때만 계산한다(풀×전략이라 한 번에 다 돌리면 느리다).
  const [backtests, setBacktests] = useState<BacktestByPool>({});
  const [backtestRunning, setBacktestRunning] = useState<string | null>(null);
  // 진행 상황 — {끝난 수, 전체}. 버튼에 「12/33 백테스트 진행중」으로 보여준다.
  const [backtestProgress, setBacktestProgress] = useState<{ done: number; total: number } | null>(null);
  // 전략 컬럼은 평소에 숨긴다 — 값이 길어(「565.6% · -32.1% · 3.95」) 늘 펼쳐 두면
  // 설정 컬럼들이 밀린다. 「백테스트」를 누르면 그때 펼친다.
  const [showBacktests, setShowBacktests] = useState(false);
  // 이평선 선택지 — 백엔드가 단일 소스다. 새 풀 초안의 기본값도 여기서 고른다.
  const maOptionsByCountry = useMemo(() => data?.constraints.ma_options_by_country ?? {}, [data]);
  const [drafts, setDrafts] = useState<Record<string, PoolDraft>>({});
  const [newDraft, setNewDraft] = useState<PoolDraft>(EMPTY_DRAFT);
  const [savingId, setSavingId] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [isCreatingNew, setIsCreatingNew] = useState(false);
  // 삭제는 체크박스로 고른 행을 상단 버튼으로 한 번에 처리한다.
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  // 벤치마크는 티커 조회가 필요한 2단계라 셀에서 편집하지 않고 이 모달로 뺀다.
  const [benchmarkTargetId, setBenchmarkTargetId] = useState<string | null>(null);

  const rows = useMemo(() => {
    if (!data?.pools) return [] as PoolEntry[];
    return [...data.pools].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
  }, [data]);

  // 그리드 행 = 초안 그대로 + 변경 여부. 초안을 그리므로 저장 전 값이 화면에 남는다.
  // AppAgGrid 가 rowData 가 바뀔 때마다 행을 다시 그리므로, 실제로 바뀔 때만 새 배열을 만든다.
  const gridRows = useMemo<PoolGridRow[]>(
    () =>
      rows.map((pool) => {
        const draft = drafts[pool.ticker_type] ?? toDraft(pool);
        return {
          ...draft,
          __dirty: isDirty(draft, pool),
          __updatedAt: pool.updated_at,
          __saveMethod: pool.save_method,
          __stockCount: pool.stock_count,
        };
      }),
    [drafts, rows],
  );
  const dirtyCount = gridRows.filter((row) => row.__dirty).length;
  const tickerTypesForCountry = useCallback(
    (countryCode: string) =>
      rows
        .filter((pool) => String(pool.country_code ?? "").toLowerCase() === countryCode.trim().toLowerCase())
        .map((pool) => pool.ticker_type),
    [rows],
  );

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
      setBacktests(payload.backtests ?? {});
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

  /** 모든 종목풀 × 전략을 **한 건씩 순서대로** 돌린다.
   *
   *  한꺼번에 보내면 서버가 백테스트 수십 개를 동시에 돌려 느려지고, 어디까지 됐는지도
   *  안 보인다. 한 건이 끝날 때마다 화면에 채워 진행 상황이 그대로 드러나게 한다.
   *  설정이 없는 조합은 서버가 빈 결과를 돌려주므로 칸이 비어 있는다. */
  const runAllBacktests = useCallback(async () => {
    setShowBacktests(true);
    const pools = rows.map((row) => row.ticker_type).filter(Boolean);
    const total = pools.length * STRATEGY_COLUMNS.length;
    let done = 0;
    setBacktestProgress({ done: 0, total });
    let failed = 0;
    for (const pool of pools) {
      for (const { key } of STRATEGY_COLUMNS) {
        setBacktestRunning(`${pool}:${key}`);
        try {
          const resp = await fetch(`/api/pool-settings/backtest?pool=${encodeURIComponent(pool)}&strategy=${key}`);
          const payload = await resp.json();
          if (!resp.ok || payload.error) {
            throw new Error(payload.error ?? "백테스트에 실패했습니다.");
          }
          setBacktests((prev) => ({ ...prev, [pool]: { ...prev[pool], [key]: payload.result } }));
        } catch {
          failed += 1;
        }
        done += 1;
        setBacktestProgress({ done, total });
      }
    }
    setBacktestRunning(null);
    setBacktestProgress(null);
    if (failed > 0) {
      toast.error(`백테스트 ${failed}건이 실패했습니다.`);
    } else {
      toast.success("백테스트를 마쳤습니다.");
    }
  }, [rows, toast]);

  const updateNewDraft = useCallback(
    (key: keyof PoolDraft, value: string) => {
      // 국가를 바꾸면 이평선 선택지가 통째로 바뀐다 — 목록 밖 값이 남지 않게 다시 채운다.
      setNewDraft((prev) => withDefaultMaDays({ ...prev, [key]: value }, maOptionsByCountry));
    },
    [maOptionsByCountry],
  );

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
      setNewDraft(withDefaultMaDays(EMPTY_DRAFT, maOptionsByCountry));
      setIsCreatingNew(false);
      await load();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "종목풀 생성에 실패했습니다.");
    } finally {
      setCreating(false);
    }
  }, [load, maOptionsByCountry, newDraft, toast]);

  /** 변경된 행만 모아 한 번에 저장한다 — 상단 저장 버튼 1개가 전부를 처리한다. */
  const handleSaveAll = useCallback(async () => {
    const targets = rows.filter((pool) => {
      const draft = drafts[pool.ticker_type];
      return draft && isDirty(draft, pool);
    });
    if (targets.length === 0) return;

    setSavingId("__all__");
    const failed: string[] = [];
    try {
      for (const pool of targets) {
        try {
          const resp = await fetch("/api/pool-settings", {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ pool_id: pool.ticker_type, values: draftToValues(drafts[pool.ticker_type]) }),
          });
          const payload = await resp.json();
          if (!resp.ok || payload.error) {
            throw new Error(payload.error ?? payload.detail ?? "저장에 실패했습니다.");
          }
        } catch (err) {
          failed.push(`${pool.ticker_type}: ${err instanceof Error ? err.message : "저장 실패"}`);
        }
      }
      const savedCount = targets.length - failed.length;
      if (savedCount > 0) toast.success(`종목풀 ${savedCount}개를 저장했습니다.`);
      if (failed.length > 0) toast.error(`저장 실패 ${failed.length}건 — ${failed.join(" / ")}`);
      await load();
    } finally {
      setSavingId(null);
    }
  }, [drafts, load, rows, toast]);

  /** 체크한 행을 한 번에 삭제한다. 계좌에 연결된 풀은 서버가 막는다. */
  const handleDeleteSelected = useCallback(async () => {
    const targets = rows.filter((pool) => selectedIds.includes(pool.ticker_type));
    if (targets.length === 0) return;

    const message = [
      `종목풀 ${targets.length}개를 하드 삭제합니다.`,
      targets.map((pool) => `  • ${pool.name} (${pool.ticker_type})`).join("\n"),
      "",
      "계좌에 연결된 종목풀은 서버에서 삭제를 차단합니다.",
      "삭제 시 이 종목풀에 등록된 종목 메타도 함께 제거됩니다.",
      "계속할까요?",
    ].join("\n");
    if (!window.confirm(message)) return;

    setDeletingId("__selected__");
    const failed: string[] = [];
    let deletedStocks = 0;
    try {
      for (const pool of targets) {
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
          deletedStocks += payload.deleted?.deleted_stocks ?? 0;
        } catch (err) {
          failed.push(`${pool.ticker_type}: ${err instanceof Error ? err.message : "삭제 실패"}`);
        }
      }
      const okCount = targets.length - failed.length;
      if (okCount > 0) toast.success(`종목풀 ${okCount}개를 삭제했습니다. 제거된 종목: ${deletedStocks}개`);
      if (failed.length > 0) toast.error(`삭제 실패 ${failed.length}건 — ${failed.join(" / ")}`);
      setSelectedIds([]);
      await load();
    } finally {
      setDeletingId(null);
    }
  }, [load, rows, selectedIds, toast]);

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
  const slippageOptions = data.constraints.slippage_pct_options?.length
    ? data.constraints.slippage_pct_options
    : DEFAULT_SLIPPAGE_PCT_OPTIONS;
  const stoplossOptions = data.constraints.stoploss_pct_options?.length
    ? data.constraints.stoploss_pct_options
    : DEFAULT_STOPLOSS_PCT_OPTIONS;
  const marketIndices = data.constraints.market_indices ?? [];
  /** 셀렉트 편집 컬럼 — 목록 밖 저장값도 후보에 남겨 빈 셀렉트가 되지 않게 한다. */
  const selectCol = (
    field: keyof PoolDraft & ColDef<PoolGridRow>["field"],
    headerName: string,
    width: number,
    values: (row: PoolGridRow) => (string | number)[],
    extra?: Partial<ColDef<PoolGridRow>>,
  ): ColDef<PoolGridRow> => ({
    field,
    headerName,
    width,
    editable: true,
    cellEditor: "agSelectCellEditor",
    cellEditorParams: (params: { data: PoolGridRow }) => {
      const list = values(params.data).map(String);
      const current = String(params.data[field] ?? "");
      return { values: current && !list.includes(current) ? [current, ...list] : list };
    },
    ...extra,
  });

  /** 숫자 편집 컬럼 — 초안은 문자열로 들지만 그리드에는 숫자로 넘긴다.
   *
   * 문자열인 채로 두면 AG Grid 가 셀 타입을 `text` 로 추론해 `agNumberCellEditor` 가
   * 호환되지 않는 것으로 판정돼 **편집 자체가 되지 않는다**. valueGetter/valueSetter 로
   * 그리드 쪽만 숫자로 바꿔 준다. (`field` 대신 `colId` 를 쓰므로 onCellValueChanged 는
   * 이 컬럼을 건너뛴다 — 저장은 valueSetter 가 이미 했다.)
   */
  const numberCol = (
    field: "order",
    headerName: string,
    width: number,
    headerTooltip?: string,
  ): ColDef<PoolGridRow> => ({
    colId: field,
    headerName,
    width,
    headerTooltip,
    editable: true,
    cellDataType: "number",
    type: "numericColumn",
    valueGetter: (params) => {
      const raw = params.data?.[field];
      return raw === "" || raw === undefined ? null : Number(raw);
    },
    valueSetter: (params) => {
      if (!params.data) return false;
      const next = params.newValue === null || params.newValue === "" ? "" : String(Math.trunc(Number(params.newValue)));
      if (next === params.data[field]) return false;
      updateDraft(params.data.ticker_type, field, next);
      return true;
    },
  });

  const columnDefs: ColDef<PoolGridRow>[] = [
    // 컬럼이 많아 가로로 스크롤한다 — 어느 풀의 행인지 잃지 않게 ID·이름은 왼쪽에 고정한다.
    { field: "ticker_type", headerName: "ID", width: 112, pinned: "left" },
    { field: "name", headerName: "이름", width: 176, editable: true, pinned: "left" },
    ...(showBacktests ? STRATEGY_COLUMNS : []).map<ColDef<PoolGridRow>>(({ key, label }) => ({
      colId: `backtest_${key}`,
      headerName: label,
      width: 186,
      sortable: false,
      headerTooltip: `${label} 전략의 저장 설정으로 돌린 12개월 백테스트 — CAGR · MDD · 소르티노. 상단 「백테스트」로 갱신합니다.`,
      cellRenderer: (params: ICellRendererParams<PoolGridRow>) => {
        const pool = params.data?.ticker_type;
        if (!pool) return null;
        const result = backtests[pool]?.[key];
        if (backtestRunning === `${pool}:${key}`) {
          return <span style={{ color: "var(--text-muted)" }}>계산 중…</span>;
        }
        if (result?.no_settings) {
          return <span style={{ color: "var(--text-muted)" }}>설정없음</span>;
        }
        if (!result || result.cagr_pct == null) {
          return null;
        }
        return (
          <span title={result.updated_at ? `계산 ${formatKstDateTime(result.updated_at)}` : undefined}>
            {`${fmt(result.cagr_pct, 1)}% / ${fmt(result.mdd_pct, 1)}% / ${fmt(result.sortino, 2)}`}
          </span>
        );
      },
    })),
    selectCol("icon", "아이콘", 72, () => ["🇰🇷", "🇦🇺", "🇺🇸"]),
    numberCol("order", "순서", 64),
    selectCol("country_code", "국가", 76, () => [...COUNTRY_OPTIONS]),
    selectCol("currency", "통화", 76, () => [...CURRENCY_OPTIONS]),
    selectCol("pool_kind", "구분", 80, () => ["stock", "etf"], {
      valueFormatter: (params) => ({ stock: "개별주", etf: "ETF" })[String(params.value)] ?? "미설정",
    }),
    selectCol("TOP_N_HOLD", "보유종목 수", 100, () => data.constraints.top_n_hold_options, {
      valueFormatter: (params) => (params.value ? `${params.value}개` : "미설정"),
    }),
    {
      // 그 풀에 담긴 종목 수 — 설정이 아니라 현황이라 편집할 수 없다.
      colId: "stock_count",
      headerName: "종목수",
      width: 76,
      type: "numericColumn",
      headerTooltip: "이 종목풀에 담긴 종목 수. 설정이 아니라 현황이라 편집할 수 없습니다.",
      valueGetter: (params) => params.data?.__stockCount ?? null,
      valueFormatter: (params) => (params.value == null ? "-" : Number(params.value).toLocaleString("ko-KR")),
    },
    selectCol(
      "SHORT_MA_DAYS",
      "단기",
      76,
      (row) => data.constraints.ma_options_by_country[row.country_code]?.short_ma_options ?? [],
      { valueFormatter: (params) => (params.value ? `${params.value}일` : "미설정") },
    ),
    selectCol(
      "LONG_MA_DAYS",
      "장기",
      76,
      (row) => data.constraints.ma_options_by_country[row.country_code]?.long_ma_options ?? [],
      { valueFormatter: (params) => (params.value ? `${params.value}일` : "미설정") },
    ),
    selectCol("BUY_SLIPPAGE_PCT", "매수", 72, () => slippageOptions, {
      valueFormatter: (params) => (params.value === "" ? "미설정" : `${params.value}%`),
    }),
    selectCol("SELL_SLIPPAGE_PCT", "매도", 72, () => slippageOptions, {
      valueFormatter: (params) => (params.value === "" ? "미설정" : `${params.value}%`),
    }),
    selectCol("STOPLOSS_THRESHOLD_PCT", "손절", 72, () => stoplossOptions, {
      valueFormatter: (params) => (params.value === "" ? "미설정" : `${params.value}%`),
      headerTooltip: "보유 종목의 수익률이 이 값 이하면 손절 알림을 보냅니다(계좌에서 손절🔔을 켠 경우).",
    }),
    {
      field: "benchmarkTicker",
      headerName: "벤치마크",
      width: 290,
      sortable: false,
      cellRenderer: (params: ICellRendererParams<PoolGridRow>) => {
        const row = params.data;
        if (!row) return null;
        const label = row.benchmarkTicker ? `${row.benchmarkName} (${row.benchmarkTicker})` : "미설정";
        return (
          // 이름이 길어도 옆 컬럼을 밀지 않게 — 텍스트는 말줄임, 버튼은 고정.
          <span style={{ display: "flex", alignItems: "center", gap: 6, width: "100%", minWidth: 0 }}>
            <span
              title={label}
              style={{
                flex: 1,
                minWidth: 0,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                color: row.benchmarkTicker ? undefined : "var(--text-muted)",
              }}
            >
              {label}
            </span>
            <button
              type="button"
              className="btn btn-sm btn-outline-secondary"
              style={{ padding: "0 6px", lineHeight: 1.4, flexShrink: 0 }}
              onClick={() => setBenchmarkTargetId(row.ticker_type)}
            >
              변경
            </button>
          </span>
        );
      },
    },
    selectCol("marketRegimeTicker", "ADR 기준", 104, () => marketIndices.map((item) => item.ticker), {
      valueFormatter: (params) =>
        marketIndices.find((item) => item.ticker === params.value)?.name ?? (params.value ? String(params.value) : "미설정"),
    }),
    {
      // 마지막 컬럼이 남는 가로를 채운다 — 오른쪽에 빈 공간이 남지 않게.
      field: "__updatedAt",
      headerName: "마지막 저장",
      flex: 1,
      minWidth: 330,
      valueFormatter: (params) => (params.value ? formatKstDateTime(String(params.value)) : "저장 이력 없음"),
      // 저장 경로(사용자 수동 / 모멘텀 전략·튜닝)를 함께 보여준다 — 누가 바꾼 값인지 알아야 한다.
      cellRenderer: (params: ICellRendererParams<PoolGridRow>) => (
        <LastSavedCell value={params.value} method={params.data?.__saveMethod} />
      ),
    },
  ];

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
          // 국기 3개 중 선택 — 저장값은 국기 이모지 그대로라 다른 화면 표시는 지금과 동일하다.
          // 기존에 목록 밖 아이콘이 저장돼 있으면 그 값도 함께 노출한다(빈 셀렉트 방지).
          <select style={{ ...inputStyle, width: 96 }} value={draft.icon} onChange={(event) => onChange("icon", event.target.value)}>
            {draft.icon && !["🇰🇷", "🇦🇺", "🇺🇸"].includes(draft.icon) ? (
              <option value={draft.icon}>{draft.icon}</option>
            ) : null}
            {draft.icon === "" ? <option value="">없음</option> : null}
            <option value="🇰🇷">🇰🇷 한국</option>
            <option value="🇦🇺">🇦🇺 호주</option>
            <option value="🇺🇸">🇺🇸 미국</option>
          </select>,
          { minWidth: 158, labelWidth: 56 },
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
          <SelectField
            value={draft.country_code}
            options={COUNTRY_OPTIONS}
            width={82}
            onChange={(value) => {
              onChange("country_code", value);
              onChange("benchmarkTicker", "");
              onChange("benchmarkName", "");
            }}
          />,
          { minWidth: 144, labelWidth: 44 },
        )}
        {renderField(
          "통화",
          <SelectField value={draft.currency} options={CURRENCY_OPTIONS} width={88} onChange={(value) => onChange("currency", value)} />,
          { minWidth: 150, labelWidth: 44 },
        )}
        {renderField(
          "구분",
          // 풀 성격 토글 — 개별주(stock)면 섹터·업종 개념이 있고, ETF 면 없다.
          // 구 문서는 미설정일 수 있어 그 상태도 그대로 보여준다(저장하면 선택값으로 확정).
          <select
            style={{ ...inputStyle, width: 96 }}
            value={draft.pool_kind}
            onChange={(event) => onChange("pool_kind", event.target.value)}
          >
            {draft.pool_kind === "" ? <option value="">미설정</option> : null}
            <option value="stock">개별주</option>
            <option value="etf">ETF</option>
          </select>,
          { minWidth: 158, labelWidth: 44 },
        )}
        {renderField(
          "보유종목 수",
          <SelectField
            value={draft.TOP_N_HOLD}
            options={data.constraints.top_n_hold_options}
            width={82}
            onChange={(value) => onChange("TOP_N_HOLD", value)}
          />,
          { minWidth: 190, labelWidth: 92 },
        )}
        {renderField(
          "단기",
          <MaDaysSelect value={Number(draft.SHORT_MA_DAYS) || null} options={data.constraints.ma_options_by_country[draft.country_code]?.short_ma_options} onChange={(days) => onChange("SHORT_MA_DAYS", String(days))} />,
          { minWidth: 160, labelWidth: 44 },
        )}
        {renderField(
          "장기",
          <MaDaysSelect value={Number(draft.LONG_MA_DAYS) || null} options={data.constraints.ma_options_by_country[draft.country_code]?.long_ma_options} onChange={(days) => onChange("LONG_MA_DAYS", String(days))} />,
          { minWidth: 160, labelWidth: 44 },
        )}
      </div>

      <div style={rowStyle}>
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
        {renderField(
          "손절 기준",
          <SelectField value={draft.STOPLOSS_THRESHOLD_PCT} options={stoplossOptions} width={90} onChange={(value) => onChange("STOPLOSS_THRESHOLD_PCT", value)} />,
          { minWidth: 196, labelWidth: 94 },
        )}
        <span style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>% 단위</span>
      </div>

      <div style={{ ...rowStyle, marginBottom: 0 }}>
        <span style={{ ...labelStyle, width: 72 }}>벤치마크</span>
        <BenchmarkField
          ticker={draft.benchmarkTicker}
          name={draft.benchmarkName}
          countryCode={draft.country_code}
          tickerTypes={tickerTypesForCountry(draft.country_code)}
          onChange={onChange}
        />
        {renderField(
          "ADR 기준",
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
                    종목풀 구조와 이평선 설정은 DB(pool_settings)가 단일 소스입니다. 셀을 클릭해 고친 뒤 저장하세요.
                  </div>
                </div>
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <UnsavedChangesBadge show={dirtyCount > 0} message={`저장하지 않은 변경 ${dirtyCount}개`} />
                <button
                  type="button"
                  className="btn btn-sm btn-outline-danger"
                  disabled={selectedIds.length === 0 || deletingId !== null}
                  onClick={() => void handleDeleteSelected()}
                >
                  {deletingId ? "삭제 중…" : `삭제${selectedIds.length > 0 ? ` (${selectedIds.length})` : ""}`}
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-outline-secondary"
                  disabled={backtestRunning !== null || loading}
                  title="모든 종목풀의 모멘텀·신고가·포트폴리오를 12개월 백테스트합니다. 한 건씩 순서대로 돌며 결과가 채워집니다."
                  onClick={() => void runAllBacktests()}
                >
                  {backtestProgress
                    ? `${backtestProgress.done}/${backtestProgress.total} 백테스트 진행중`
                    : "백테스트"}
                </button>
                <button type="button" className="btn btn-sm btn-primary" onClick={() => setIsCreatingNew(!isCreatingNew)}>
                  등록
                </button>
                <button type="button" className="btn btn-sm btn-outline-secondary" disabled={loading} onClick={() => void load()}>
                  새로고침
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-dark"
                  disabled={dirtyCount === 0 || savingId !== null}
                  onClick={() => void handleSaveAll()}
                >
                  {savingId ? "저장 중…" : "저장"}
                </button>
              </div>
            </div>
          </div>

          <div className="card-body appCardBodyTight appTableCardBodyFill">
            <div className="appGridFillWrap">
              <AppAgGrid<PoolGridRow>
                className="settingsAgGrid"
                rowData={gridRows}
                columnDefs={columnDefs}
                loading={loading}
                theme={poolSettingsGridTheme}
                minHeight="100%"
                getRowId={(params) => params.data.ticker_type}
                getRowClass={(params) => (params.data?.__dirty ? "settingsDirtyRow" : "")}
                gridOptions={{
                  suppressMovableColumns: true,
                  // 폭에 맞춰 말줄임하므로, 잘린 값은 마우스를 올려 전체를 본다.
                  defaultColDef: {
                    sortable: true,
                    resizable: true,
                    tooltipValueGetter: (params) => params.valueFormatted ?? params.value,
                  },
                  // AppAgGrid 기본값은 셀 포커스를 막는다(읽기 전용 표 기준). 편집하려면 켜야 한다.
                  suppressCellFocus: false,
                  singleClickEdit: true,
                  stopEditingWhenCellsLoseFocus: true,
                  rowSelection: {
                    mode: "multiRow",
                    checkboxes: true,
                    headerCheckbox: true,
                    enableClickSelection: false,
                  },
                  selectionColumnDef: {
                    width: 40,
                    minWidth: 40,
                    maxWidth: 40,
                    pinned: "left",
                    sortable: false,
                    resizable: false,
                    suppressMovable: true,
                    headerName: "",
                  },
                  onSelectionChanged: (params) => {
                    setSelectedIds(params.api.getSelectedRows().map((row) => row.ticker_type));
                  },
                  onCellValueChanged: (params) => {
                    const key = params.colDef.field as keyof PoolDraft | undefined;
                    if (!key || !params.data || params.newValue === params.oldValue) return;
                    updateDraft(params.data.ticker_type, key, String(params.newValue ?? ""));
                    // 시장 레짐은 티커만 고르고 이름은 목록에서 따라온다(직접 입력 불가).
                    if (key === "marketRegimeTicker") {
                      const selected = marketIndices.find((item) => item.ticker === String(params.newValue ?? ""));
                      updateDraft(params.data.ticker_type, "marketRegimeName", selected?.name ?? "");
                    }
                    // 국가를 바꾸면 이평선 선택지가 통째로 바뀐다 — 목록 밖 값이 남지 않게 비운다.
                    if (key === "country_code") {
                      // 기존 국가의 벤치마크가 새 국가 설정에 남지 않게 다시 선택하도록 비운다.
                      updateDraft(params.data.ticker_type, "benchmarkTicker", "");
                      updateDraft(params.data.ticker_type, "benchmarkName", "");
                      const options = data.constraints.ma_options_by_country[String(params.newValue ?? "")];
                      const short = Number(params.data.SHORT_MA_DAYS);
                      const long = Number(params.data.LONG_MA_DAYS);
                      if (options && !options.short_ma_options.includes(short)) {
                        updateDraft(params.data.ticker_type, "SHORT_MA_DAYS", "");
                      }
                      if (options && !options.long_ma_options.includes(long)) {
                        updateDraft(params.data.ticker_type, "LONG_MA_DAYS", "");
                      }
                    }
                  },
                }}
              />
            </div>
          </div>
        </div>
      </section>

      {/* 벤치마크 — 티커 입력 → 조회로 이름 확정. 셀에 담기지 않아 모달로 뺀다. */}
      <AppModal
        open={benchmarkTargetId !== null}
        title="벤치마크"
        subtitle={benchmarkTargetId ? `${benchmarkTargetId} 종목풀의 벤치마크를 지정합니다. 티커를 비우면 미설정.` : ""}
        onClose={() => setBenchmarkTargetId(null)}
        footer={
          <div style={{ display: "flex", justifyContent: "flex-end", width: "100%" }}>
            <button type="button" className="btn btn-primary" onClick={() => setBenchmarkTargetId(null)}>
              닫기
            </button>
          </div>
        }
      >
        {benchmarkTargetId && drafts[benchmarkTargetId] ? (
          <BenchmarkField
            ticker={drafts[benchmarkTargetId].benchmarkTicker}
            name={drafts[benchmarkTargetId].benchmarkName}
            countryCode={drafts[benchmarkTargetId].country_code}
            tickerTypes={tickerTypesForCountry(drafts[benchmarkTargetId].country_code)}
            onChange={(key, value) => updateDraft(benchmarkTargetId, key, value)}
          />
        ) : null}
      </AppModal>

      <AppModal
        open={isCreatingNew}
        title="신규 종목풀"
        subtitle="ticker_type 은 생성 후 변경할 수 없습니다."
        onClose={() => {
          setNewDraft(withDefaultMaDays(EMPTY_DRAFT, maOptionsByCountry));
          setIsCreatingNew(false);
        }}
        size="xl"
        footer={
          <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, width: "100%" }}>
            <button
              type="button"
              className="btn btn-outline-secondary"
              onClick={() => {
                setNewDraft(withDefaultMaDays(EMPTY_DRAFT, maOptionsByCountry));
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
