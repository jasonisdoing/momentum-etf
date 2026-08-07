"use client";

import type { ColDef } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useState } from "react";

import { formatKstDateTime } from "@/lib/datetime";
import { AppAgGrid } from "../components/AppAgGrid";
import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";

const inputStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "4px 7px",
  fontSize: "var(--fs-sm)",
};

const compactRowStyle: React.CSSProperties = {
  display: "flex",
  gap: 8,
  alignItems: "center",
  padding: "5px 0",
  borderBottom: "1px solid rgba(148,163,184,0.15)",
};

const compactLabelStyle: React.CSSProperties = {
  width: 56,
  flexShrink: 0,
  color: "var(--text-muted)",
  fontWeight: 700,
  fontSize: "var(--fs-sm)",
  whiteSpace: "nowrap",
};

const TUNE_MONTH_OPTIONS = [1, 2, 3, 4, 5, 6, 12, 24, 36, 48, 60] as const;
// 슬리피지 편도(%). 종목풀 설정과 동일한 0.05~0.5 (0.05 단위).
const SLIPPAGE_OPTIONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] as const;
const leverageTuneGridTheme = createAppGridTheme();

type Market = "kor" | "us";
type AssetRef = { ticker: string; name?: string };

type MaConfig = {
  strategy: "ma_cross";
  market: Market;
  index: AssetRef;
  leverage: AssetRef;
  defense: AssetRef;
  ma_days: number;
  peak_drawdown_pct: number;
  slippage: number;
  slack_enabled?: boolean;
  tuning?: {
    months: number;
    ma_min: number;
    ma_max: number;
    ma_step: number;
    peak_min: number;
    peak_max: number;
    peak_step: number;
  };
};

type TuneRow = {
  ma_days: number;
  peak_drawdown_pct: number;
  cumulative_pct: number;
  cagr_pct: number | null;
  mdd_pct: number | null;
  sortino: number | null;
  switches: number;
  days: number;
  leverage_days: number;
};

type TuneBenchmarkRow = {
  label: string;
  ticker: string;
  name: string;
  cumulative_pct: number;
  cagr_pct: number | null;
  mdd_pct: number | null;
  sortino: number | null;
  switches: number;
  days: number;
};

// 튜닝 후보(TuneRow)와 벤치마크(지수/레버리지 보유)를 한 그리드에서 렌더하기 위한 통합 행.
// 벤치마크 행은 label 을 채우고 이동선/고점대비/보유일은 비운다(고정행으로 상단에 표시).
type LeverageTuneGridRow = {
  label?: string;
  isBenchmark?: boolean;
  ma_days?: number;
  peak_drawdown_pct?: number;
  cumulative_pct: number;
  cagr_pct: number | null;
  mdd_pct: number | null;
  sortino: number | null;
  switches: number;
  days: number;
  leverage_days?: number;
};

type MaView = {
  market: Market;
  ma_days: number;
  ma_type: string;
  peak_drawdown_pct: number;
  slippage: number;
  judgment: {
    as_of: string;
    index_close: number;
    ma: number;
    gap_pct: number;
    peak_drawdown_pct: number;
    peak_drawdown_limit_pct: number;
    ma_threshold_close: number;
    peak_threshold_close: number;
    required_index_close: number;
    want_leverage: boolean;
  } | null;
  recommendation: {
    side: "leverage" | "defense";
    target_ticker: string;
    target_name: string;
    as_of: string;
    prev_target: string | null;
    is_changed: boolean;
  } | null;
  state: { date?: string; target?: string; target_name?: string; side?: string; updated_at?: string; holding_days?: number; holding_start_date?: string } | null;
  error?: string;
};

type TuneResult = {
  months: number;
  ma_range: { min: number; max: number; step: number };
  peak_drawdown_range: { min: number; max: number; step: number };
  benchmarks?: TuneBenchmarkRow[];
  rows: TuneRow[];
  error?: string;
};

const MARKET_LABEL: Record<Market, string> = { kor: "🇰🇷 한국", us: "🇺🇸 미국" };

// 지수는 시장별로 고정: 한국=코스피(^KS11), 미국=나스닥100(^NDX). 사용자가 편집하지 않는다.
const FIXED_INDEX: Record<Market, AssetRef> = {
  kor: { ticker: "^KS11", name: "코스피" },
  us: { ticker: "^NDX", name: "나스닥 100" },
};

function blankConfig(market: Market): MaConfig {
  return {
    strategy: "ma_cross",
    market,
    index: FIXED_INDEX[market],
    leverage: { ticker: "", name: "" },
    defense: { ticker: "CASH", name: "현금" },
    ma_days: 120,
    peak_drawdown_pct: 7,
    slippage: 0.1,
  };
}

function fmtPct(v: number | null | undefined): string {
  return v == null ? "-" : `${v > 0 ? "+" : ""}${v.toFixed(2)}%`;
}

function fmtWholePct(v: number | string | null | undefined): string {
  if (v == null || v === "") return "";
  const n = typeof v === "number" ? v : Number.parseFloat(v);
  return Number.isFinite(n) ? `${Math.round(n)}%` : "";
}

function fmtIndexPoint(v: number | null | undefined): string {
  if (v == null) return "-";
  return `${v.toLocaleString("ko-KR", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}pt`;
}

function indexPointLabel(market: Market): string {
  return market === "us" ? "나스닥 100" : "코스피";
}

function buildNumberRange(min: number, max: number, step: number, decimals = 0): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || !Number.isFinite(step) || step <= 0 || max < min) {
    return [];
  }
  const scale = 10 ** decimals;
  const start = Math.round(min * scale);
  const end = Math.round(max * scale);
  const interval = Math.max(1, Math.round(step * scale));
  const values: number[] = [];
  for (let value = start; value <= end; value += interval) {
    values.push(value / scale);
  }
  return values;
}

export function LeverageSettingsClient() {
  const toast = useToast();
  const [market, setMarket] = useState<Market>("kor");
  const [config, setConfig] = useState<MaConfig | null>(null);
  const [configMissing, setConfigMissing] = useState(false);
  const [view, setView] = useState<MaView | null>(null);
  const [loadingConfig, setLoadingConfig] = useState(true);
  const [loadingView, setLoadingView] = useState(false);
  const [saving, setSaving] = useState(false);
  const [mounted, setMounted] = useState(false);

  // 튜닝(온디맨드): 사용자가 기간·이동선 범위를 지정하고 버튼으로 실행.
  const [tuneMonths, setTuneMonths] = useState(36);
  const [tuneMin, setTuneMin] = useState(20);
  const [tuneMax, setTuneMax] = useState(120);
  const [tuneStep, setTuneStep] = useState(10);
  const [peakMin, setPeakMin] = useState(1);
  const [peakMax, setPeakMax] = useState(10);
  const [peakStep, setPeakStep] = useState(1);
  const [tuneResult, setTuneResult] = useState<TuneResult | null>(null);
  const [tuning, setTuning] = useState(false);
  const [tuneError, setTuneError] = useState<string | null>(null);
  const [tuneProgress, setTuneProgress] = useState<{ percent: number; message: string } | null>(null);
  // 슬랙 알람: 우선 UI 토글만(백엔드 연결 없음).
  const [slackEnabled, setSlackEnabled] = useState(false);

  const profile = `ma_cross_${market}`;
  const maOptions = useMemo(() => buildNumberRange(tuneMin, tuneMax, tuneStep), [tuneMin, tuneMax, tuneStep]);
  const peakDrawdownOptions = useMemo(() => buildNumberRange(peakMin, peakMax, peakStep, 1), [peakMin, peakMax, peakStep]);

  const loadConfig = useCallback(async (m: Market) => {
    setLoadingConfig(true);
    setConfigMissing(false);
    try {
      const resp = await fetch(`/api/leverage-config?profile=ma_cross_${m}`, { cache: "no-store" });
      const payload = (await resp.json()) as { config?: Partial<MaConfig>; error?: string };
      if (!resp.ok || payload.error || !payload.config?.index) {
        // 아직 설정이 없는 시장 — 빈 폼으로 새로 작성 (임의 계산 기본값 아님, 사용자 입력 대기)
        setConfig(blankConfig(m));
        setConfigMissing(true);
        return;
      }
      const c = payload.config;
      const tuning = c.tuning;
      setConfig({
        strategy: "ma_cross",
        market: m,
        index: FIXED_INDEX[m],
        leverage: c.leverage ?? { ticker: "", name: "" },
        defense: c.defense ?? { ticker: "CASH", name: "현금" },
        ma_days: c.ma_days ?? 120,
        peak_drawdown_pct: Number(c.peak_drawdown_pct ?? tuning?.peak_min ?? 7),
        slippage: c.slippage ?? 0.1,
        slack_enabled: Boolean(c.slack_enabled),
        tuning: tuning && typeof tuning === "object" ? {
          months: Number(tuning.months),
          ma_min: Number(tuning.ma_min),
          ma_max: Number(tuning.ma_max),
          ma_step: Number(tuning.ma_step),
          peak_min: Number(tuning.peak_min ?? 1),
          peak_max: Number(tuning.peak_max ?? 10),
          peak_step: Number(tuning.peak_step ?? 1),
        } : undefined,
      });
      setTuneMonths(Number(tuning?.months ?? 36));
      setTuneMin(Number(tuning?.ma_min ?? 20));
      setTuneMax(Number(tuning?.ma_max ?? 120));
      setTuneStep(Number(tuning?.ma_step ?? 10));
      setPeakMin(Number(tuning?.peak_min ?? 1));
      setPeakMax(Number(tuning?.peak_max ?? 10));
      setPeakStep(Number(tuning?.peak_step ?? 1));
      setSlackEnabled(Boolean(c.slack_enabled));
    } catch {
      setConfig(blankConfig(m));
      setConfigMissing(true);
    } finally {
      setLoadingConfig(false);
    }
  }, []);

  const loadView = useCallback(async (m: Market) => {
    setLoadingView(true);
    try {
      const resp = await fetch(`/api/leverage-ma?market=${m}`, { cache: "no-store" });
      const payload = (await resp.json()) as MaView;
      if (!resp.ok || payload.error) {
        setView(null);
        return;
      }
      setView(payload);
    } catch {
      setView(null);
    } finally {
      setLoadingView(false);
    }
  }, []);

  const runTune = useCallback(async () => {
    if (tuneMin < 2 || tuneStep < 1 || tuneMax < tuneMin) {
      const msg = "이동선 범위를 확인하세요 (min≥2, step≥1, max≥min).";
      setTuneError(msg);
      toast.error(msg);
      return;
    }
    if (peakMin < 0 || peakStep <= 0 || peakMax < peakMin) {
      const msg = "고점대비 범위를 확인하세요 (min≥0, step>0, max≥min).";
      setTuneError(msg);
      toast.error(msg);
      return;
    }
    setTuning(true);
    setTuneError(null);
    setTuneProgress({ percent: 15, message: "시세 데이터 조회 중" });
    // 단일 요청이라 실제 진행률은 없지만, 반응이 없어 보이지 않게 완만히 채운다(compare 방식).
    const timer = window.setInterval(() => {
      setTuneProgress((prev) => (prev ? { ...prev, percent: Math.min(90, prev.percent + 7), message: "이동선별 백테스트 계산 중" } : prev));
    }, 400);
    try {
      const qs = `market=${market}&months=${tuneMonths}&ma_min=${tuneMin}&ma_max=${tuneMax}&ma_step=${tuneStep}&peak_min=${peakMin}&peak_max=${peakMax}&peak_step=${peakStep}`;
      const resp = await fetch(`/api/leverage-ma/tune?${qs}`, { cache: "no-store" });
      const payload = (await resp.json()) as TuneResult;
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "이동선 튜닝에 실패했습니다.");
      setTuneProgress({ percent: 100, message: "결과 반영 중" });
      setTuneResult(payload);
      if (!payload.rows?.length) setTuneError("결과가 없습니다. 설정을 저장했는지, 기간/범위가 데이터에 맞는지 확인하세요.");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "이동선 튜닝에 실패했습니다.";
      setTuneError(msg);
      toast.error(msg);
    } finally {
      window.clearInterval(timer);
      setTuning(false);
      setTuneProgress(null);
    }
  }, [market, tuneMonths, tuneMin, tuneMax, tuneStep, peakMin, peakMax, peakStep, toast]);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    void loadConfig(market);
    void loadView(market);
    setTuneResult(null); // 시장 전환 시 이전 시장의 튜닝 결과는 폐기
    setTuneError(null);
  }, [market, loadConfig, loadView]);

  const setAsset = (key: "index" | "leverage" | "defense", field: "ticker" | "name", value: string) =>
    setConfig((c) => (c ? { ...c, [key]: { ...c[key], [field]: value } } : c));

  const resolveAsset = useCallback(async (key: "index" | "leverage" | "defense", raw: string) => {
    const ticker = raw.trim();
    if (!ticker) return;
    if (ticker.toUpperCase() === "CASH") {
      setAsset(key, "ticker", "CASH");
      setAsset(key, "name", "현금");
      return;
    }
    try {
      const resp = await fetch(`/api/leverage-config/resolve?ticker=${encodeURIComponent(ticker)}`);
      const data = (await resp.json()) as { name?: string; error?: string };
      if (!resp.ok || data.error || !data.name) {
        toast.error(data.error ?? "존재하지 않는 티커입니다.");
        return;
      }
      setAsset(key, "name", data.name);
      toast.success(`${data.name}(${ticker}) 확인 완료`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "티커 조회 중 오류가 발생했습니다.");
    }
  }, [toast]);

  const setDefenseCash = (isCash: boolean) =>
    setConfig((c) => (c ? { ...c, defense: isCash ? { ticker: "CASH", name: "현금" } : { ticker: "", name: "" } } : c));

  const save = async () => {
    if (!config) return;
    if (!config.index.ticker || !config.index.name) {
      toast.error("지수 티커를 입력하고 확인하세요.");
      return;
    }
    if (!config.leverage.ticker || !config.leverage.name) {
      toast.error("레버리지 티커를 입력하고 확인하세요.");
      return;
    }
    if (!config.defense.ticker || !config.defense.name) {
      toast.error("방어 티커(또는 현금)를 확인하세요.");
      return;
    }
    if (!maOptions.includes(config.ma_days)) {
      toast.error("이동선은 튜닝 설정의 이동선 범위 안에서 선택하세요.");
      return;
    }
    if (!Number.isFinite(config.peak_drawdown_pct) || config.peak_drawdown_pct < 0) {
      toast.error("고점대비 기준을 선택하세요.");
      return;
    }
    if (!peakDrawdownOptions.includes(config.peak_drawdown_pct)) {
      toast.error("고점대비는 튜닝 설정의 고점대비 범위 안에서 선택하세요.");
      return;
    }
    setSaving(true);
    try {
      const configToSave: MaConfig = {
        ...config,
        slack_enabled: slackEnabled,
        tuning: {
          months: tuneMonths,
          ma_min: tuneMin,
          ma_max: tuneMax,
          ma_step: tuneStep,
          peak_min: peakMin,
          peak_max: peakMax,
          peak_step: peakStep,
        },
      };
      const resp = await fetch(`/api/leverage-config?profile=${profile}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: configToSave }),
      });
      const payload = (await resp.json()) as { error?: string };
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "저장에 실패했습니다.");
      setConfig(configToSave);
      toast.success(`${MARKET_LABEL[market]} 전략 설정 저장 완료`);
      setConfigMissing(false);
      await loadView(market);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "저장에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  };

  // 슬랙 토글: 변경 즉시 config 에 저장(장 마감 직후 배치가 이 값을 보고 발송).
  const saveSlackToggle = async (enabled: boolean) => {
    setSlackEnabled(enabled);
    if (!config || configMissing) {
      toast.error("먼저 전략 설정을 저장하세요.");
      setSlackEnabled(!enabled);
      return;
    }
    try {
      const configToSave: MaConfig = { ...config, slack_enabled: enabled };
      const resp = await fetch(`/api/leverage-config?profile=${profile}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: configToSave }),
      });
      const payload = (await resp.json()) as { error?: string };
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "저장에 실패했습니다.");
      setConfig(configToSave);
      toast.success(`${MARKET_LABEL[market]} 슬랙 알람 ${enabled ? "켜짐" : "꺼짐"} 저장`);
    } catch (err) {
      setSlackEnabled(!enabled);
      toast.error(err instanceof Error ? err.message : "슬랙 설정 저장에 실패했습니다.");
    }
  };

  // 슬랙 수동 발송(토글/장 마감 무관, 테스트 표식으로 지금 판정을 보냄).
  const [slackSending, setSlackSending] = useState(false);
  const sendSlackManual = async () => {
    setSlackSending(true);
    try {
      const resp = await fetch(`/api/leverage-ma/slack-test?market=${market}`, { method: "POST" });
      const payload = (await resp.json()) as { sent?: boolean; error?: string };
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "슬랙 발송에 실패했습니다.");
      if (payload.sent) toast.success(`${MARKET_LABEL[market]} 슬랙 수동 발송 완료`);
      else toast.error("슬랙 발송에 실패했습니다 (판정 데이터/채널 확인).");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "슬랙 발송에 실패했습니다.");
    } finally {
      setSlackSending(false);
    }
  };

  const columnDefs = useMemo<ColDef<LeverageTuneGridRow>[]>(() => [
    { headerName: "이동선", field: "ma_days", width: 120, valueGetter: (p) => p.data?.label ?? p.data?.ma_days },
    { headerName: "고점대비", field: "peak_drawdown_pct", width: 100, valueFormatter: (p) => fmtWholePct(p.value) },
    { headerName: "누적수익", field: "cumulative_pct", width: 104, valueFormatter: (p) => fmtPct(p.value) },
    { headerName: "CAGR", field: "cagr_pct", width: 150, sort: "desc", sortIndex: 1, valueFormatter: (p) => fmtPct(p.value) },
    { headerName: "MDD", field: "mdd_pct", width: 82, valueFormatter: (p) => fmtPct(p.value) },
    { headerName: "소르티노", field: "sortino", width: 150, sort: "desc", sortIndex: 0, valueFormatter: (p) => (p.value == null ? "-" : p.value.toFixed(2)) },
    { headerName: "전환수", field: "switches", width: 76 },
    { headerName: "거래일", field: "days", width: 76 },
    { headerName: "보유일", field: "leverage_days", width: 76, valueFormatter: (p) => (p.value == null ? "" : String(p.value)) },
  ], []);

  // 벤치마크(지수/레버리지 보유)를 후보 행과 함께 섞어 현재 정렬 기준에 맞는 위치에 넣는다.
  const gridRows = useMemo<LeverageTuneGridRow[]>(() => {
    const candidates = tuneResult?.rows ?? [];
    const benchmarks: LeverageTuneGridRow[] = (tuneResult?.benchmarks ?? []).map((b) => ({
      label: b.label,
      isBenchmark: true,
      cumulative_pct: b.cumulative_pct,
      cagr_pct: b.cagr_pct,
      mdd_pct: b.mdd_pct,
      sortino: b.sortino,
      switches: b.switches,
      days: b.days,
    }));
    return [...candidates, ...benchmarks];
  }, [tuneResult]);

  const marketSwitcher = useMemo(() => (
    <div style={{ display: "flex", gap: 4, background: "rgba(148,163,184,0.15)", borderRadius: 8, padding: 3 }}>
      {(["kor", "us"] as Market[]).map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => setMarket(m)}
          style={{
            border: "none",
            borderRadius: 6,
            padding: "5px 14px",
            fontWeight: 700,
            fontSize: "var(--fs-base)",
            cursor: "pointer",
            background: market === m ? "var(--surface, #fff)" : "transparent",
            color: market === m ? "var(--text-normal)" : "var(--text-muted)",
            boxShadow: market === m ? "0 1px 2px rgba(0,0,0,0.1)" : "none",
          }}
        >
          {MARKET_LABEL[m]}
        </button>
      ))}
    </div>
  ), [market]);

  if (!mounted) {
    return <PageFrame title="레버리지 설정"><div style={{ color: "var(--text-muted)", padding: 20 }}>불러오는 중…</div></PageFrame>;
  }

  const assetRow = (key: "index" | "leverage" | "defense", label: string, allowCash: boolean) => {
    const a = config?.[key];
    const isCash = a?.ticker?.toUpperCase() === "CASH";
    return (
      <div style={compactRowStyle}>
        <span style={compactLabelStyle}>{label}</span>
        <div style={{ flex: 1, minWidth: 0, display: "flex", gap: 4, alignItems: "center", flexWrap: "nowrap" }}>
          {allowCash && (
            <label style={{ display: "flex", gap: 3, alignItems: "center", fontSize: "var(--fs-sm)", color: "var(--text-muted)", flexShrink: 0 }}>
              <input type="checkbox" checked={isCash} onChange={(e) => setDefenseCash(e.target.checked)} />
              현금
            </label>
          )}
          {!isCash && (
            <>
              <input
                style={{ ...inputStyle, width: 84, flexShrink: 0 }}
                placeholder="티커"
                value={a?.ticker ?? ""}
                onChange={(e) => setAsset(key, "ticker", e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); void resolveAsset(key, e.currentTarget.value); } }}
              />
              <button type="button" className="btn btn-sm btn-outline-secondary" style={{ padding: "3px 6px", fontSize: "var(--fs-sm)", flexShrink: 0 }} onClick={() => void resolveAsset(key, a?.ticker ?? "")}>확인</button>
            </>
          )}
          <input
            style={{ ...inputStyle, flex: "1 1 0", minWidth: 0, backgroundColor: "#f8fafc", color: "var(--text-muted)", cursor: "not-allowed" }}
            value={a?.name ?? ""}
            readOnly
          />
        </div>
      </div>
    );
  };


  const rec = view?.recommendation;
  const state = view?.state;

  return (
    <PageFrame title="레버리지 설정" fullHeight fullWidth>
      <div className="appPageStack appPageStackFill">
        <section className="appSection">
          <div className="card appCard">
            <div className="card-header">
              <div className="appMainHeader">
                <div className="appMainHeaderLeft">
                  {marketSwitcher}
                </div>
                <div className="appMainHeaderRight">
                  <button type="button" className="btn btn-sm btn-dark" disabled={tuning} onClick={() => void runTune()}>
                    {tuning ? "계산 중…" : "튜닝"}
                  </button>
                  <button type="button" className="btn btn-sm btn-primary" disabled={saving || loadingConfig || !config} onClick={() => void save()}>
                    {saving ? "저장 중…" : "저장"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>
        <div style={{ display: "flex", flexDirection: "column", flex: "1 1 auto", gap: 16, minHeight: 0 }}>
          <div style={{ display: "flex", gap: 12, alignItems: "flex-start", flexWrap: "wrap" }}>
            <div style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
            <div className="card appCard" style={{ minWidth: 0 }}>
              <div className="card-body">
                <h2 style={{ fontSize: "var(--fs-lg)", fontWeight: 800, marginBottom: 4 }}>전략 설정</h2>
                <p style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", lineHeight: 1.45, marginBottom: 8 }}>
                  지수 종가가 이동선 위면 <b>레버리지</b>, 아래면 <b>방어(현금/종목)</b>를 보유합니다. {MARKET_LABEL[market]} 시장.
                </p>
                {configMissing && <div className="alert alert-warning py-2" style={{ fontSize: "var(--fs-sm)" }}>이 시장의 설정이 아직 없습니다. 값을 채우고 저장하세요.</div>}
                {loadingConfig || !config ? (
                  <div style={{ color: "var(--text-muted)", padding: 12 }}>불러오는 중…</div>
                ) : (
                  <>
                    <div style={compactRowStyle}>
                      <span style={compactLabelStyle}>지수</span>
                      <input
                        style={{ ...inputStyle, flex: "1 1 0", minWidth: 0, backgroundColor: "#f8fafc", color: "var(--text-muted)", cursor: "not-allowed" }}
                        value={config.index?.name ? `${config.index.name}(${config.index.ticker})` : ""}
                        readOnly
                      />
                      <span style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", whiteSpace: "nowrap" }}>고정</span>
                    </div>
                    {assetRow("leverage", "레버리지", false)}
                    {assetRow("defense", "방어", true)}
                    <div style={compactRowStyle}>
                      <span style={compactLabelStyle}>이동선</span>
                      <select style={{ ...inputStyle, width: 96 }} value={config.ma_days} onChange={(e) => setConfig((c) => c && { ...c, ma_days: Number(e.target.value) })}>
                        {!maOptions.includes(config.ma_days) && <option value={config.ma_days}>{view?.ma_type ?? ""} {config.ma_days}일</option>}
                        {maOptions.map((n) => <option key={n} value={n}>{view?.ma_type ?? ""} {n}일</option>)}
                      </select>
                      <span style={{ ...compactLabelStyle, width: "auto", marginLeft: 14 }}>고점대비</span>
                      <select
                        style={{ ...inputStyle, width: 76 }}
                        value={config.peak_drawdown_pct}
                        onChange={(e) => setConfig((c) => c && { ...c, peak_drawdown_pct: Number(e.target.value) })}
                      >
                        {!peakDrawdownOptions.includes(config.peak_drawdown_pct) && <option value={config.peak_drawdown_pct}>{config.peak_drawdown_pct}%</option>}
                        {peakDrawdownOptions.map((n) => <option key={n} value={n}>{n}%</option>)}
                      </select>
                    </div>
                    <div style={compactRowStyle}>
                      <span style={compactLabelStyle}>슬리피지</span>
                      <select style={{ ...inputStyle, width: 88 }} value={config.slippage} onChange={(e) => setConfig((c) => c && { ...c, slippage: Number(e.target.value) })}>
                        {SLIPPAGE_OPTIONS.map((s) => <option key={s} value={s}>{s.toFixed(2)}%</option>)}
                      </select>
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="card appCard" style={{ minWidth: 0 }}>
              <div className="card-body">
                <h2 style={{ fontSize: "var(--fs-lg)", fontWeight: 800, marginBottom: 4 }}>튜닝 설정</h2>
                <p style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", marginBottom: 12 }}>
                  아래 튜닝 결과와 전략 설정의 선택 가능 범위에 함께 사용됩니다.
                </p>
                <div style={{ display: "flex", gap: "10px 12px", alignItems: "flex-end", flexWrap: "wrap" }}>
                  <label className="appLabeledField" style={{ minWidth: 130, flex: "0 0 auto" }}>
                    <span className="appLabeledFieldLabel">기간(개월)</span>
                    <select className="form-select form-select-sm" value={tuneMonths} onChange={(e) => setTuneMonths(Number(e.target.value))}>
                      {TUNE_MONTH_OPTIONS.map((m) => (
                        <option key={m} value={m}>
                          최근 {m}개월
                        </option>
                      ))}
                    </select>
                  </label>
                  <div style={{ display: "flex", gap: 7, alignItems: "flex-end", flexWrap: "wrap" }}>
                    <span style={{ color: "var(--text-muted)", fontWeight: 700, paddingBottom: 7, fontSize: "var(--fs-sm)" }}>이동선</span>
                    {([["min", tuneMin, setTuneMin], ["max", tuneMax, setTuneMax], ["step", tuneStep, setTuneStep]] as const).map(([lbl, val, setter]) => (
                      <label key={lbl} className="appInlineField" style={{ flex: "0 0 auto" }}>
                        <span className="appInlineFieldLabel">{lbl}</span>
                        <input type="number" min={1} step={1} className="form-control form-control-sm" style={{ width: 66 }} value={val} onChange={(e) => setter(Number(e.target.value))} />
                      </label>
                    ))}
                  </div>
                  <div style={{ display: "flex", gap: 7, alignItems: "flex-end", flexWrap: "wrap" }}>
                    <span style={{ color: "var(--text-muted)", fontWeight: 700, paddingBottom: 7, fontSize: "var(--fs-sm)" }}>고점대비(%)</span>
                    {([["min", peakMin, setPeakMin], ["max", peakMax, setPeakMax], ["step", peakStep, setPeakStep]] as const).map(([lbl, val, setter]) => (
                      <label key={lbl} className="appInlineField" style={{ flex: "0 0 auto" }}>
                        <span className="appInlineFieldLabel">{lbl}</span>
                        <input type="number" min={0} step={0.1} className="form-control form-control-sm" style={{ width: 66 }} value={val} onChange={(e) => setter(Number(e.target.value))} />
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </div>
            </div>

            <div style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
            <div className="card appCard" style={{ minWidth: 0 }}>
              <div className="card-body">
                <h2 style={{ fontSize: "var(--fs-lg)", fontWeight: 800, marginBottom: 4 }}>슬랙 알람</h2>
                <p style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", lineHeight: 1.45, marginBottom: 12 }}>
                  켜두면 배치가 장 마감 직후 {MARKET_LABEL[market]} 추천을 슬랙으로 보냅니다.
                </p>
                <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                  <div className="form-check form-switch" style={{ paddingLeft: "2.6em", marginBottom: 0 }}>
                    <input
                      className="form-check-input"
                      type="checkbox"
                      role="switch"
                      id="leverageSlackToggle"
                      style={{ width: "2.2em", height: "1.2em" }}
                      checked={slackEnabled}
                      onChange={(e) => void saveSlackToggle(e.target.checked)}
                    />
                    <label className="form-check-label" htmlFor="leverageSlackToggle" style={{ fontWeight: 700, marginLeft: 6 }}>
                      슬랙 알람 {slackEnabled ? "켜짐" : "꺼짐"}
                    </label>
                  </div>
                  <button
                    type="button"
                    className="btn btn-sm btn-outline-secondary"
                    disabled={slackSending}
                    onClick={() => void sendSlackManual()}
                  >
                    {slackSending ? "발송 중…" : "지금 발송(테스트)"}
                  </button>
                </div>
              </div>
            </div>

            <div className="card appCard" style={{ minWidth: 0 }}>
              <div className="card-body">
                <h2 style={{ fontSize: "var(--fs-lg)", fontWeight: 800, marginBottom: 12 }}>추천 상태</h2>
                {view?.judgment && rec ? (
                  <>
                    {/* 지금 판정 헤더 */}
                    <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap", marginBottom: 2 }}>
                      <span style={{ fontSize: "var(--fs-lg)", fontWeight: 800 }}>
                        {rec.side === "leverage" ? "🟢 레버리지 보유" : "🔵 방어 보유"}
                      </span>
                      <span style={{ color: "var(--text-muted)", fontWeight: 700 }}>· {rec.target_name}({rec.target_ticker})</span>
                    </div>
                    <div style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", marginBottom: 10 }}>
                      기준일 {view.judgment.as_of} · {view.ma_type} {view.ma_days}일 / 고점대비 한도 {view.judgment.peak_drawdown_limit_pct.toFixed(0)}%
                    </div>
                    {(() => {
                      const j = view.judgment;
                      const requiredMovePct = (j.required_index_close / j.index_close - 1) * 100;
                      const isLeverage = j.want_leverage;
                      return (
                        <div
                          style={{
                            color: "var(--text-primary)",
                            fontSize: "var(--fs-sm)",
                            fontWeight: 800,
                            marginBottom: 10,
                          }}
                        >
                          {isLeverage ? "방어 전환 기준 지수" : "전환 필요 지수"}: {indexPointLabel(view.market)}{" "}
                          <span style={{ color: isLeverage ? "#2563eb" : "#dc2626" }}>
                            {fmtIndexPoint(j.required_index_close)} {isLeverage ? "이하" : "이상"} ({fmtPct(requiredMovePct)})
                          </span>
                        </div>
                      );
                    })()}

                    {/* 근거 표: 이격 · 고점대비 */}
                    {(() => {
                      const j = view.judgment;
                      const gapOk = j.gap_pct >= 0;
                      const pctToTarget = (target: number) => ((target / j.index_close) - 1) * 100;
                      const maRecoveryPct = pctToTarget(j.ma_threshold_close);
                      const peakRecoveryPct = pctToTarget(j.peak_threshold_close);
                      const gapMargin = gapOk
                        ? `${view.ma_days}일 이동평균선 아래로 ${fmtPct(maRecoveryPct)} 하락하기 전까지 레버리지 보유`
                        : `${view.ma_days}일 이동평균선 위로 ${fmtPct(maRecoveryPct)} 회복 필요`;
                      const peakOk = j.peak_drawdown_pct >= -j.peak_drawdown_limit_pct;
                      const peakMargin = peakOk
                        ? `전고점 대비 -${j.peak_drawdown_limit_pct.toFixed(0)}% 아래로 ${fmtPct(peakRecoveryPct)} 하락하기 전까지 레버리지 보유`
                        : `전고점 대비 -${j.peak_drawdown_limit_pct.toFixed(0)}% 이내로 ${fmtPct(peakRecoveryPct)} 회복 필요`;
                      const rows = [
                        { name: `이격 (${view.ma_days}일 이동평균선)`, needValue: maRecoveryPct, need: fmtPct(maRecoveryPct), base: `${view.ma_days}일 이평선`, ok: gapOk, margin: gapMargin },
                        { name: "고점대비", needValue: peakRecoveryPct, need: fmtPct(peakRecoveryPct), base: `≥ -${j.peak_drawdown_limit_pct.toFixed(0)}%`, ok: peakOk, margin: peakMargin },
                      ];
                      const signedColor = (value: number) => (value > 0 ? "#dc2626" : value < 0 ? "#2563eb" : "var(--text-primary)");
                      const th: React.CSSProperties = { textAlign: "left", color: "var(--text-muted)", fontWeight: 600, fontSize: "var(--fs-sm)", padding: "2px 6px" };
                      const td: React.CSSProperties = { padding: "5px 6px", borderTop: "1px solid rgba(148,163,184,0.15)", fontSize: "var(--fs-sm)", whiteSpace: "nowrap" };
                      return (
                        <table style={{ width: "100%", borderCollapse: "collapse" }}>
                          <thead>
                            <tr>
                              <th style={th}>기준</th>
                              <th style={{ ...th, textAlign: "right" }}>필요/여유(%)</th>
                              <th style={{ ...th, textAlign: "right" }}>조건</th>
                              <th style={{ ...th, textAlign: "center" }}>상태</th>
                            </tr>
                          </thead>
                          <tbody>
                            {rows.map((r) => (
                              <tr key={r.name}>
                                <td style={{ ...td, color: "var(--text-muted)", whiteSpace: "normal" }}>{r.margin}</td>
                                <td style={{ ...td, textAlign: "right", fontWeight: 700, color: signedColor(r.needValue) }}>{r.need}</td>
                                <td style={{ ...td, textAlign: "right", color: "var(--text-muted)" }}>{r.base}</td>
                                <td style={{ ...td, textAlign: "center" }}>{r.ok ? "✅" : "❌"}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      );
                    })()}

                    {rec.is_changed ? (
                      <div className="alert alert-warning py-2 mb-0" style={{ marginTop: 10, fontSize: "var(--fs-sm)" }}>
                        ⚠ 전환 예정: {rec.prev_target ?? "-"} → {rec.target_ticker} (다음 종가 확정 시)
                      </div>
                    ) : null}
                  </>
                ) : (
                  <div style={{ color: "var(--text-muted)", padding: 8 }}>{loadingView ? "계산 중…" : "판정 데이터 없음 (설정 저장 후 표시)"}</div>
                )}

                {/* 현재 상태 (실제 보유, 읽기 전용) */}
                <div style={{ marginTop: 12, paddingTop: 8, borderTop: "1px solid rgba(148,163,184,0.25)" }}>
                  <div style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", fontWeight: 700, marginBottom: 4 }}>현재 상태</div>
                  {([
                    ["보유 종목", state?.target_name ? `${state.target_name}(${state.target})` : "-"],
                    ["보유 시작일", state?.holding_start_date ?? "-"],
                    ["보유일", state?.holding_days != null ? `${state.holding_days}거래일째` : "-"],
                    ["갱신 시각", state?.updated_at ? formatKstDateTime(state.updated_at) : "-"],
                  ] as const).map(([label, value]) => (
                    <div key={label} style={{ display: "flex", gap: 12, padding: "4px 0" }}>
                      <span style={{ width: 84, flexShrink: 0, color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>{label}</span>
                      <span style={{ fontWeight: 600, fontSize: "var(--fs-sm)" }}>{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            </div>
          </div>

          <div className="card appCard appTableCardFill" style={{ width: "100%", minHeight: 0 }}>
            <div className="card-body appTableCardBodyFill">
              <h2 style={{ fontSize: "var(--fs-lg)", fontWeight: 800, marginBottom: 4 }}>튜닝</h2>
              <p style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", marginBottom: 12 }}>
                기간과 이동선 범위를 지정해 실행하면, 후보 이동선별 성과를 소르티노·CAGR 내림차순으로 보여줍니다.
                특정 값만 튀지 않고 넓게 완만하면 강건한 규칙입니다.
              </p>
              {tuning ? (
                <div className="compareLoading" style={{ marginBottom: 12 }}>
                  <div className="compareLoadingText">
                    <span>이동선 튜닝 계산 중…</span>
                    <strong>{tuneProgress?.percent ?? 0}%</strong>
                  </div>
                  <div className="compareLoadingBar" aria-hidden="true">
                    <div style={{ width: `${tuneProgress?.percent ?? 0}%` }} />
                  </div>
                  <small>{tuneProgress?.message ?? "계산 중"}</small>
                </div>
              ) : null}
              {tuneError ? <div className="alert alert-warning py-2" style={{ fontSize: "var(--fs-sm)" }}>{tuneError}</div> : null}
              {tuneResult && !tuneError ? (
                <div style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", marginBottom: 6 }}>
                  최근 {tuneResult.months}개월 · 이동선 {tuneResult.ma_range.min}~{tuneResult.ma_range.max} (step {tuneResult.ma_range.step})
                  {" · "}고점대비 {tuneResult.peak_drawdown_range.min}~{tuneResult.peak_drawdown_range.max}% (step {tuneResult.peak_drawdown_range.step}) · 후보 {tuneResult.rows.length}개
                </div>
              ) : null}
              <AppAgGrid<LeverageTuneGridRow>
                rowData={gridRows}
                columnDefs={columnDefs}
                loading={tuning}
                theme={leverageTuneGridTheme}
                minHeight={0}
                height="100%"
                getRowClass={(p) => {
                  if (p.data?.isBenchmark) return "appBenchmarkRow";
                  const currentSmaDays = config?.ma_days ?? view?.ma_days;
                  const currentPeakDrawdownPct = config?.peak_drawdown_pct ?? view?.peak_drawdown_pct;
                  return p.data &&
                    p.data.ma_days === currentSmaDays &&
                    currentPeakDrawdownPct != null &&
                    p.data.peak_drawdown_pct != null &&
                    Math.abs(p.data.peak_drawdown_pct - currentPeakDrawdownPct) < 0.0001
                    ? "appHeldRow"
                    : "";
                }}
                getRowId={(p) => (p.data.isBenchmark ? `bm:${p.data.label}` : `${p.data.ma_days}:${p.data.peak_drawdown_pct}`)}
                gridOptions={{ overlayNoRowsTemplate: "기간·범위를 정하고 '튜닝' 버튼을 누르세요." }}
              />
            </div>
          </div>
        </div>
      </div>
    </PageFrame>
  );
}
