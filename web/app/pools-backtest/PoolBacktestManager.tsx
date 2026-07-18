"use client";

import { useCallback, useEffect, useState } from "react";

import { formatPoolLabel } from "@/lib/pool-label";
import { useToast } from "../components/ToastProvider";

type SignalRow = {
  order?: number;
  label: string;
  avg_return: number;
  avg_return_diff: number;
  avg_return_error: number;
  avg_return_significant: boolean;
  up_rate: number;
  up_rate_diff: number;
  up_rate_significant: boolean;
  samples: number;
};

type LongShort = {
  insufficient: boolean;
  mean: number;
  wins: number;
  losses: number;
  t_value: number | null;
  independent_samples: number;
  required_samples: number;
  significant: boolean;
};

type StrategyStats = { cumulative_pct: number; mdd_pct: number | null; sortino: number | null };
type Performance = {
  top_n_hold: number;
  rounds: number;
  cash_rounds: number;
  partial_rounds: number;
  mean_return: number;
  wins: number;
  losses: number;
  turnover_pct: number;
  round_trip_pct: number;
  cost_per_round_pct: number;
  rule: StrategyStats;
  pool_hold: StrategyStats;
  benchmark: (StrategyStats & { ticker: string; name: string }) | null;
};

type BacktestResult = {
  pool_id: string;
  forward_days: number;
  months: number;
  ma_rule: { short_ma_days: number; long_ma_days: number; slope_days: number };
  ticker_count: number;
  excluded_fixed_count: number;
  row_count: number;
  trading_days: number;
  base_return: number;
  base_rate: number;
  performance: Performance | null;
  effective_samples: number;
  rate_error: number;
  date_from: string;
  date_to: string;
  quantile_count: number;
  disparity: SignalRow[];
  short_disparity: SignalRow[];
  slope: SignalRow[];
  disparity_long_short: LongShort | null;
  short_disparity_long_short: LongShort | null;
  slope_long_short: LongShort | null;
  disparity_ic: LongShort | null;
  short_disparity_ic: LongShort | null;
  slope_ic: LongShort | null;
  error?: string;
};

type PoolSettingField = { value: string | number | null };
type PoolOption = {
  ticker_type: string;
  name: string;
  order: number;
  icon: string;
  settings?: Partial<Record<"TOP_N_HOLD" | "SHORT_MA_DAYS" | "LONG_MA_DAYS" | "SLOPE_DAYS", PoolSettingField>>;
};
type PoolSettingsResponse = {
  pools?: PoolOption[];
  constraints?: { ma_day_options?: number[]; slope_day_options?: number[] };
  error?: string;
};
type BacktestOptions = { forward_day_options?: number[]; month_options?: number[]; max_months?: number; error?: string };

const FORWARD_DAY_OPTIONS = [5, 10, 20, 40, 60];
const DEFAULT_MONTH_OPTIONS = [1, 2, 3, 4, 5, 6, 12, 24, 36, 48, 60];
const DEFAULT_MA_DAY_OPTIONS = [5, 10, 20, 40, 60, 120, 240];
const DEFAULT_SLOPE_DAY_OPTIONS = [1, 2, 3, 5, 10, 20, 40, 60];

/** 종목풀 설정 필드에서 정수를 꺼낸다. 값이 없으면 null. */
function fieldToInt(field: PoolSettingField | undefined): number | null {
  const raw = field?.value;
  if (raw === null || raw === undefined || raw === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

function signedClass(value: number): string {
  if (value === 0) return "";
  return value < 0 ? "metricNegative" : "metricPositive";
}

function formatSigned(value: number, digits = 1): string {
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
}

export function PoolBacktestManager() {
  const toast = useToast();
  const [pools, setPools] = useState<PoolOption[]>([]);
  const [poolId, setPoolId] = useState("");
  const [forwardDays, setForwardDays] = useState(20);
  const [months, setMonths] = useState(12);
  const [monthOptions, setMonthOptions] = useState(DEFAULT_MONTH_OPTIONS);
  // 파라미터 오버라이드(실험용). 종목풀 선택 시 그 설정값으로 채워지고, 사용자가 바꿀 수 있다.
  const [topN, setTopN] = useState<number | null>(null);
  const [shortMa, setShortMa] = useState<number | null>(null);
  const [longMa, setLongMa] = useState<number | null>(null);
  const [slopeDays, setSlopeDays] = useState<number | null>(null);
  const [maDayOptions, setMaDayOptions] = useState(DEFAULT_MA_DAY_OPTIONS);
  const [slopeDayOptions, setSlopeDayOptions] = useState(DEFAULT_SLOPE_DAY_OPTIONS);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [poolResp, optionsResp] = await Promise.all([
          fetch("/api/pool-settings", { cache: "no-store" }),
          fetch("/api/pool-backtest/options", { cache: "no-store" }),
        ]);
        const payload = (await poolResp.json()) as PoolSettingsResponse;
        const options = (await optionsResp.json()) as BacktestOptions;
        if (!poolResp.ok || payload.error) throw new Error(payload.error ?? "종목풀 목록을 불러오지 못했습니다.");
        if (!optionsResp.ok || options.error) throw new Error(options.error ?? "백테스트 옵션을 불러오지 못했습니다.");
        if (alive) {
          const list = payload.pools ?? [];
          const loadedMonthOptions = (options.month_options ?? []).filter((month) => Number.isFinite(month) && month > 0);
          setPools(list);
          if (payload.constraints?.ma_day_options?.length) setMaDayOptions(payload.constraints.ma_day_options);
          if (payload.constraints?.slope_day_options?.length) setSlopeDayOptions(payload.constraints.slope_day_options);
          if (loadedMonthOptions.length > 0) {
            setMonthOptions(loadedMonthOptions);
            setMonths((current) =>
              loadedMonthOptions.includes(current)
                ? current
                : (loadedMonthOptions.includes(12) ? 12 : loadedMonthOptions[0]),
            );
          }
          if (list.length > 0) setPoolId(list[0].ticker_type);
        }
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : "종목풀 목록을 불러오지 못했습니다.");
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // 종목풀을 바꾸면 그 종목풀의 저장된 설정값으로 파라미터를 채운다(사용자가 이후 수정 가능).
  useEffect(() => {
    const pool = pools.find((p) => p.ticker_type === poolId);
    if (!pool?.settings) return;
    setTopN(fieldToInt(pool.settings.TOP_N_HOLD));
    setShortMa(fieldToInt(pool.settings.SHORT_MA_DAYS));
    setLongMa(fieldToInt(pool.settings.LONG_MA_DAYS));
    setSlopeDays(fieldToInt(pool.settings.SLOPE_DAYS));
  }, [poolId, pools]);

  const runBacktest = useCallback(async () => {
    if (!poolId) {
      toast.error("종목풀을 선택해주세요.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ pool_id: poolId, forward_days: String(forwardDays), months: String(months) });
      if (topN != null) params.set("top_n", String(topN));
      if (shortMa != null) params.set("short_ma_days", String(shortMa));
      if (longMa != null) params.set("long_ma_days", String(longMa));
      if (slopeDays != null) params.set("slope_days", String(slopeDays));
      const resp = await fetch(`/api/pool-backtest?${params.toString()}`, { cache: "no-store" });
      const payload = (await resp.json()) as BacktestResult & { detail?: string };
      if (!resp.ok || payload.error) throw new Error(payload.error ?? payload.detail ?? "백테스트에 실패했습니다.");
      setResult(payload);
    } catch (e) {
      setError(e instanceof Error ? e.message : "백테스트에 실패했습니다.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [forwardDays, months, poolId, topN, shortMa, longMa, slopeDays, toast]);

  /** 롱숏·IC 공통 렌더. digits/suffix 로 표기만 달라진다. */
  const renderStat = (
    stat: LongShort | null,
    title: string,
    opts: { digits: number; suffix: string; winLabel: string; hint: string },
  ) => {
    if (!stat) {
      return (
        <div className="poolBtLongShort" style={{ color: "var(--text-muted)" }}>
          {title} — 계산에 데이터가 부족합니다.
        </div>
      );
    }
    const value = (
      <span
        className={stat.insufficient ? undefined : signedClass(stat.mean)}
        style={{ fontWeight: 800, fontSize: "1.05rem", color: stat.insufficient ? "var(--text-muted)" : undefined }}
      >
        {formatSigned(stat.mean, opts.digits)}
        {opts.suffix}
      </span>
    );
    const borderColor = stat.insufficient ? "#b45309" : stat.significant ? "#2f9e44" : "rgba(148,163,184,0.4)";
    return (
      <div
        className="poolBtLongShort"
        style={{ borderColor, background: stat.insufficient ? "rgba(180,83,9,0.06)" : undefined }}
      >
        <strong>{title}</strong> {value}
        <span style={{ color: "var(--text-muted)" }}>
          {" · "}
          {opts.winLabel} {stat.wins}승 {stat.losses}패
          {!stat.insufficient && <>{" · "}t {stat.t_value === null ? "-" : formatSigned(stat.t_value, 2)}</>}
        </span>
        <strong style={{ marginLeft: 8, color: stat.insufficient || !stat.significant ? "#b45309" : "#2f9e44" }}>
          {stat.insufficient
            ? `→ 표본 부족 (최소 ${stat.required_samples}개 필요) — 판단 불가`
            : stat.significant
              ? "→ 유의 (신호 있음)"
              : "→ 유의하지 않음 (우연과 구분 불가)"}
        </strong>
        <div style={{ color: "var(--text-muted)", fontSize: "0.82rem", marginTop: 2 }}>
          {stat.insufficient ? "이 표본으로는 t 검정이 무의미해 유의 판정을 내지 않습니다. 기간을 늘리거나 전망일수를 줄이세요." : opts.hint}
        </div>
      </div>
    );
  };

  const renderTable = (
    title: string,
    note: string,
    rows: SignalRow[],
    ls: LongShort | null,
    ic: LongShort | null,
    spreadLabel: string,
  ) => (
    <div className="card appCard">
      <div className="card-body">
        <h3 style={{ fontSize: "0.98rem", fontWeight: 800, marginBottom: 2 }}>{title}</h3>
        <p style={{ color: "var(--text-muted)", fontSize: "0.82rem", margin: "0 0 8px" }}>{note}</p>
        {renderStat(ls, `롱숏(${spreadLabel})`, {
          digits: 2,
          suffix: "%p",
          winLabel: "독립구간",
          hint: "같은 날 최상·최하 등급을 빼서 시장 요인을 상쇄한 값입니다.",
        })}
        {renderStat(ic, "IC(날짜별 상관)", {
          digits: 3,
          suffix: "",
          winLabel: "독립구간",
          hint: "그날 신호 순위와 향후 수익 순위의 상관. 아래 등급 평균의 매끄러운 단조성은 10개의 증거가 아니라 같은 날을 10칸으로 나눈 것이라, 미약한 경향도 단조로 보입니다. IC 는 날짜마다 관계가 실제로 성립했는지를 봅니다.",
        })}
        <div style={{ overflowX: "auto" }}>
          <table className="poolBtTable">
            <thead>
              <tr>
                <th style={{ textAlign: "left", width: "1%", whiteSpace: "nowrap" }}>구간</th>
                <th>평균수익</th>
                <th>기저 대비</th>
                <th>상승확률</th>
                <th>기저 대비</th>
                <th>표본</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.label}>
                  <td style={{ textAlign: "left", width: "1%", whiteSpace: "nowrap" }}>{row.label}</td>
                  <td style={{ fontWeight: 800 }} className={signedClass(row.avg_return)}>
                    {formatSigned(row.avg_return, 2)}%
                  </td>
                  <td className={signedClass(row.avg_return_diff)}>
                    {formatSigned(row.avg_return_diff, 2)}%p
                    <span style={{ color: "var(--text-muted)", fontWeight: 400 }}> (±{row.avg_return_error.toFixed(2)})</span>
                  </td>
                  <td style={{ color: "var(--text-muted)" }}>{row.up_rate.toFixed(1)}%</td>
                  <td style={{ color: "var(--text-muted)" }}>{formatSigned(row.up_rate_diff)}%p</td>
                  <td style={{ color: "var(--text-muted)" }}>{row.samples.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft" style={{ flexWrap: "wrap", gap: "12px 16px" }}>
                <label className="appLabeledField" style={{ minWidth: 280, flex: "0 0 auto" }}>
                  <span className="appLabeledFieldLabel">종목풀</span>
                  <select className="form-select form-select-sm" value={poolId} onChange={(e) => setPoolId(e.target.value)}>
                    {pools.length === 0 ? <option value="">불러오는 중…</option> : null}
                    {pools.map((p) => (
                      <option key={p.ticker_type} value={p.ticker_type}>
                        {formatPoolLabel(p)}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField" style={{ minWidth: 84, flex: "0 0 auto" }}>
                  <span className="appLabeledFieldLabel">리밸런싱 주기</span>
                  <select
                    className="form-select form-select-sm"
                    value={forwardDays}
                    onChange={(e) => setForwardDays(Number(e.target.value))}
                  >
                    {FORWARD_DAY_OPTIONS.map((d) => (
                      <option key={d} value={d}>
                        {d}일
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField" style={{ minWidth: 130, flex: "0 0 auto" }}>
                  <span className="appLabeledFieldLabel">기간(개월)</span>
                  <select className="form-select form-select-sm" value={months} onChange={(e) => setMonths(Number(e.target.value))}>
                    {monthOptions.map((m) => (
                      <option key={m} value={m}>
                        최근 {m}개월
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField" style={{ minWidth: 96, flex: "0 0 auto" }}>
                  <span className="appLabeledFieldLabel">보유 종목수</span>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    className="form-control form-control-sm"
                    value={topN ?? ""}
                    onChange={(e) => setTopN(e.target.value === "" ? null : Number(e.target.value))}
                  />
                </label>
                <label className="appLabeledField" style={{ minWidth: 104, flex: "0 0 auto" }}>
                  <span className="appLabeledFieldLabel">단기 이평선</span>
                  <select
                    className="form-select form-select-sm"
                    value={shortMa ?? ""}
                    onChange={(e) => setShortMa(e.target.value === "" ? null : Number(e.target.value))}
                  >
                    {maDayOptions.map((d) => (
                      <option key={d} value={d}>
                        {d}일
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField" style={{ minWidth: 104, flex: "0 0 auto" }}>
                  <span className="appLabeledFieldLabel">장기 이평선</span>
                  <select
                    className="form-select form-select-sm"
                    value={longMa ?? ""}
                    onChange={(e) => setLongMa(e.target.value === "" ? null : Number(e.target.value))}
                  >
                    {maDayOptions.map((d) => (
                      <option key={d} value={d}>
                        {d}일
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField" style={{ minWidth: 104, flex: "0 0 auto" }}>
                  <span className="appLabeledFieldLabel">기울기 일수</span>
                  <select
                    className="form-select form-select-sm"
                    value={slopeDays ?? ""}
                    onChange={(e) => setSlopeDays(e.target.value === "" ? null : Number(e.target.value))}
                  >
                    {slopeDayOptions.map((d) => (
                      <option key={d} value={d}>
                        {d}일
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "flex-end", flexShrink: 0 }}>
                <button type="button" className="btn btn-sm btn-primary" disabled={loading || !poolId} onClick={() => void runBacktest()}>
                  {loading ? "백테스트 중…" : "백테스트"}
                </button>
              </div>
            </div>
          </div>

          <div className="card-body appCardBodyTight appTableCardBodyFill" style={{ overflowY: "auto", padding: "16px 20px" }}>
            <p style={{ fontSize: "0.88rem", color: "var(--text-muted)", lineHeight: 1.6, margin: "0 0 16px" }}>
              등급은 <strong>같은 날짜 안에서 종목끼리 상대비교</strong>해 나눕니다(시장 타이밍 효과 제거 — 순위 화면과 같은 관점). <strong>1등급 = 신호 상위</strong>. 고정 보유 종목 제외.
              <br />
              주 지표는 <strong>평균수익</strong>입니다 — 모멘텀은 승률이 아니라 손익비로 벌기 때문에 상승확률로는 안 잡힙니다.{" "}
              <strong>기저 대비 차이가 오차(±)를 넘어야만 의미</strong>가 있습니다.
              <br />
              표보다 위의 <strong>롱숏·IC</strong>를 믿으세요 — 등급 평균의 매끄러운 단조성은 같은 날을 10칸으로 나눈 것이라 <strong>미약한 경향도 단조로 보입니다</strong>.
              롱숏·IC는 시장 요인을 상쇄하고 <strong>날짜마다 관계가 성립했는지</strong>를 봅니다. <strong>t &gt; 2</strong>여야 우연이 아닙니다.
            </p>

            {error ? <div className="alert alert-danger mb-0">{error}</div> : null}
            {!loading && !result && !error ? (
              <div style={{ color: "var(--text-muted)", padding: "16px 0" }}>종목풀·리밸런싱 주기·기간을 고르고 백테스트를 눌러주세요.</div>
            ) : null}

      {result ? (
        <>
          {/* 현재 설정 그대로 돌렸을 때의 기간 실적 — 순위 화면의 추천(✅) 규칙과 동일 */}
          {result.performance ? (
            <div className="card appCard">
              <div className="card-body">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 8 }}>
                  <h3 style={{ fontSize: "0.98rem", fontWeight: 800, margin: "0 0 2px" }}>
                    최근 {result.months}개월 실적 — 현재 설정 그대로
                  </h3>
                  <span style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--text-muted)" }}>
                    분석 기간 {result.date_from} ~ {result.date_to}
                  </span>
                </div>
                <p style={{ color: "var(--text-muted)", fontSize: "0.82rem", margin: "0 0 10px" }}>
                  순위 화면의 추천(✅)과 같은 규칙: 이격 상위 {result.performance.top_n_hold}종목, 단기이격이 음수면 제외.
                  {result.forward_days}일마다 리밸런싱({result.performance.rounds}회
                  {result.performance.cash_rounds > 0 ? `, 전부 현금 ${result.performance.cash_rounds}회` : ""}
                  {result.performance.partial_rounds > 0 ? `, 일부 현금 ${result.performance.partial_rounds}회` : ""}).
                  조건 맞는 종목이 {result.performance.top_n_hold}개 미만이면 부족분은 현금으로 둡니다.
                </p>
                {(() => {
                  const p = result.performance!;
                  const bench = p.benchmark;
                  const cell = (v: number) => (
                    <span className={signedClass(v)} style={{ fontWeight: 800 }}>
                      {formatSigned(v, 1)}%
                    </span>
                  );
                  const mdd = (v: number | null) =>
                    v === null ? <span style={{ color: "var(--text-muted)" }}>-</span> : <span>{v.toFixed(1)}%</span>;
                  const sortino = (v: number | null) =>
                    v === null ? <span style={{ color: "var(--text-muted)" }}>-</span> : <span style={{ fontWeight: 700 }}>{v.toFixed(2)}</span>;
                  const th: React.CSSProperties = { textAlign: "right", padding: "6px 10px", fontWeight: 700 };
                  const td: React.CSSProperties = { textAlign: "right", padding: "6px 10px" };
                  const labelCell: React.CSSProperties = { textAlign: "left", padding: "6px 10px", color: "var(--text-muted)" };
                  return (
                    <div style={{ overflowX: "auto" }}>
                      <table className="poolBtTable" style={{ minWidth: 460 }}>
                        <thead>
                          <tr>
                            <th style={labelCell}>지표</th>
                            <th style={th}>종목풀 규칙</th>
                            <th style={th}>종목풀 보유</th>
                            <th style={th}>{bench ? `벤치마크 (${bench.name})` : "벤치마크"}</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td style={labelCell}>누적수익 (슬리피지 차감)</td>
                            <td style={td}>{cell(p.rule.cumulative_pct)}</td>
                            <td style={td}>{cell(p.pool_hold.cumulative_pct)}</td>
                            <td style={td}>
                              {bench ? cell(bench.cumulative_pct) : <span style={{ color: "var(--text-muted)" }}>미설정</span>}
                            </td>
                          </tr>
                          <tr>
                            <td style={labelCell}>최대낙폭 (MDD)</td>
                            <td style={td}>{mdd(p.rule.mdd_pct)}</td>
                            <td style={td}>{mdd(p.pool_hold.mdd_pct)}</td>
                            <td style={td}>{bench ? mdd(bench.mdd_pct) : <span style={{ color: "var(--text-muted)" }}>-</span>}</td>
                          </tr>
                          <tr>
                            <td style={labelCell}>소르티노 (연율)</td>
                            <td style={td}>{sortino(p.rule.sortino)}</td>
                            <td style={td}>{sortino(p.pool_hold.sortino)}</td>
                            <td style={td}>{bench ? sortino(bench.sortino) : <span style={{ color: "var(--text-muted)" }}>-</span>}</td>
                          </tr>
                        </tbody>
                      </table>
                      <p style={{ fontSize: "0.85rem", margin: "8px 2px 0", color: "var(--text-muted)" }}>
                        <strong>규칙 − 벤치마크(기여한 몫):</strong>{" "}
                        {bench ? (
                          <span className={signedClass(p.rule.cumulative_pct - bench.cumulative_pct)} style={{ fontWeight: 800 }}>
                            {formatSigned(p.rule.cumulative_pct - bench.cumulative_pct, 1)}%p
                          </span>
                        ) : (
                          "벤치마크 미설정"
                        )}
                        {" · "}회차당 {formatSigned(p.mean_return, 2)}% · {p.wins}승 {p.losses}패 · 회전율{" "}
                        {p.turnover_pct.toFixed(0)}% · 회차비용 −{p.cost_per_round_pct.toFixed(2)}%
                      </p>
                    </div>
                  );
                })()}
                <p style={{ fontSize: "0.83rem", color: "var(--text-muted)", lineHeight: 1.6, margin: "10px 0 0" }}>
                  <strong>기대수익이 아니라 지나간 {result.months}개월의 실적입니다</strong> — 리밸런싱{" "}
                  {result.performance.rounds}회가 표본의 전부라, 큰 상승장이 통째로 들어오면 숫자가 커집니다. 다음 기간에
                  이만큼을 기대하면 안 됩니다. <strong>벤치마크와의 차이(규칙이 기여한 몫)</strong>를 보세요 — 이 값이 작으면
                  수익의 대부분은 규칙이 아니라 시장이 만든 것입니다.
                </p>
              </div>
            </div>
          ) : null}


          <div className="poolBtSplit">
            {renderTable(
              `이격 ${result.quantile_count}등급`,
              `종가가 장기 이평선(${result.ma_rule.long_ma_days}일) 대비 얼마나 벌어졌나 · 1등급 = 이격 상위`,
              result.disparity,
              result.disparity_long_short,
              result.disparity_ic,
              `1등급 − ${result.quantile_count}등급`,
            )}
            {renderTable(
              `단기이격 ${result.quantile_count}등급`,
              `종가가 단기 이평선(${result.ma_rule.short_ma_days}일) 대비 얼마나 벌어졌나 · 1등급 = 단기이격 상위`,
              result.short_disparity,
              result.short_disparity_long_short,
              result.short_disparity_ic,
              `1등급 − ${result.quantile_count}등급`,
            )}
            {renderTable(
              `기울기 ${result.quantile_count}등급`,
              `단기 이평선(${result.ma_rule.short_ma_days}일)의 ${result.ma_rule.slope_days}일 전 대비 변화율 · 1등급 = 기울기 상위`,
              result.slope,
              result.slope_long_short,
              result.slope_ic,
              `1등급 − ${result.quantile_count}등급`,
            )}
          </div>
        </>
      ) : null}
          </div>
        </div>
      </section>

      <style jsx global>{`
        .poolBtTable {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.9rem;
          white-space: nowrap;
        }
        .poolBtTable th,
        .poolBtTable td {
          padding: 5px 10px;
          text-align: right;
          border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        }
        .poolBtTable th {
          color: var(--text-muted);
          font-weight: 600;
        }
        .poolBtLongShort {
          border: 1px solid rgba(148, 163, 184, 0.4);
          border-radius: 6px;
          padding: 6px 10px;
          margin-bottom: 10px;
          font-size: 0.88rem;
          background: rgba(148, 163, 184, 0.06);
          line-height: 1.5;
        }
        /* 이격 · 단기이격 · 기울기 나란히. 좁아지면 2열 → 1열로 접힌다. */
        .poolBtSplit {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 12px;
          align-items: start;
        }
        @media (max-width: 1500px) {
          .poolBtSplit {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
        }
        @media (max-width: 1100px) {
          .poolBtSplit {
            grid-template-columns: minmax(0, 1fr);
          }
        }
      `}</style>
    </div>
  );
}
