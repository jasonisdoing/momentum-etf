"use client";

import { useCallback, useEffect, useState } from "react";

import { useToast } from "../components/ToastProvider";

export type AssetRef = { ticker: string; name?: string };
export type CutoffRange = { min?: number; max?: number; step?: number };
export type CandidateKey = "offense_candidates" | "defense_candidates";

export type TuningConfig = {
  offense_candidates?: AssetRef[];
  defense_candidates?: AssetRef[];
  buy_cutoff_range?: CutoffRange;
  sell_cutoff_range?: CutoffRange;
};

export type LeverageConfig = {
  backtested_date?: string;
  strategy?: string;
  market?: string;
  months_range?: number;
  start_date?: string;
  signal?: AssetRef;
  offense?: AssetRef;
  defense?: AssetRef;
  slippage?: number;
  drawdown_buy_cutoff?: number;
  drawdown_sell_cutoff?: number;
  benchmarks?: AssetRef[];
  tuning?: TuningConfig;
  [key: string]: unknown;
};

export type LeverageState = {
  date?: string;
  target?: string;
  target_name?: string;
  holding_days?: number;
  updated_at?: string;
};

type LeverageResponse = {
  profile: string;
  config: LeverageConfig;
  state: LeverageState;
  error?: string;
};

export const inputStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "5px 8px",
  fontSize: "0.9rem",
};

/** min~max(끝값 포함)를 step 간격으로 나열 (백엔드 _arange_inclusive 와 동일 의미). */
export function rangeValues(r?: CutoffRange): number[] {
  if (!r || r.min == null || r.max == null || !r.step || r.step <= 0 || r.max < r.min) return [];
  const out: number[] = [];
  for (let v = r.min; v <= r.max + r.step / 2; v += r.step) out.push(Number(v.toFixed(4)));
  return out;
}

/** 설정 로드·저장·티커조회 공통 훅 (설정/튜닝 페이지 공용). */
export function useLeverageConfig() {
  const [config, setConfig] = useState<LeverageConfig | null>(null);
  const [state, setState] = useState<LeverageState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [mounted, setMounted] = useState(false);
  const toast = useToast();

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const resp = await fetch("/api/leverage-config?profile=switch", { cache: "no-store" });
      const payload = (await resp.json()) as LeverageResponse;
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "레버리지 설정을 불러오지 못했습니다.");
      setConfig(payload.config);
      setState(payload.state);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "레버리지 설정을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setMounted(true);
    void load();
  }, [load]);

  const saveConfigDirect = useCallback(async (updatedConfig: LeverageConfig): Promise<boolean> => {
    try {
      setSaving(true);
      setError(null);
      const resp = await fetch("/api/leverage-config?profile=switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: updatedConfig }),
      });
      const payload = (await resp.json()) as LeverageResponse;
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "저장에 실패했습니다.");
      setConfig(payload.config);
      setState(payload.state);
      return true;
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "저장에 실패했습니다.");
      return false;
    } finally {
      setSaving(false);
    }
  }, [toast]);

  const fetchTickerName = useCallback(async (ticker: string): Promise<{ name: string; error?: string }> => {
    const clean = ticker.trim();
    if (!clean) return { name: "", error: "티커를 입력해주세요." };
    try {
      const resp = await fetch(`/api/leverage-config/resolve?ticker=${encodeURIComponent(clean)}`);
      const data = (await resp.json()) as { name?: string; error?: string };
      if (!resp.ok || data.error) return { name: "", error: data.error ?? "티커 조회에 실패했습니다." };
      return { name: data.name ?? "", error: data.name ? undefined : "존재하지 않는 티커입니다." };
    } catch (err) {
      return { name: "", error: err instanceof Error ? err.message : "티커 조회 중 오류가 발생했습니다." };
    }
  }, []);

  return { config, setConfig, state, error, setError, loading, saving, mounted, toast, load, saveConfigDirect, fetchTickerName };
}
