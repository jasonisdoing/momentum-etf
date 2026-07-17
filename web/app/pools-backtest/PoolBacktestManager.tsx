"use client";

import { useCallback, useEffect, useState } from "react";

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
  win_rate: number;
  t_value: number | null;
  independent_samples: number;
  required_samples: number;
  significant: boolean;
};

type BacktestResult = {
  pool_id: string;
  forward_days: number;
  months: number;
  ma_rule: { short_ma_days: number; main_ma_days: number; slope_days: number };
  ticker_count: number;
  excluded_fixed_count: number;
  row_count: number;
  trading_days: number;
  base_return: number;
  base_rate: number;
  effective_samples: number;
  rate_error: number;
  date_from: string;
  date_to: string;
  quantile_count: number;
  disparity: SignalRow[];
  slope: SignalRow[];
  disparity_long_short: LongShort | null;
  slope_long_short: LongShort | null;
  disparity_ic: LongShort | null;
  slope_ic: LongShort | null;
  error?: string;
};

type PoolOption = { ticker_type: string; name: string };
type BacktestOptions = { forward_day_options?: number[]; month_options?: number[]; max_months?: number; error?: string };

const FORWARD_DAY_OPTIONS = [5, 10, 20, 40, 60];
const DEFAULT_MONTH_OPTIONS = [1, 2, 3, 4, 5, 6, 12, 24, 36, 48, 60];

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
  const [months, setMonths] = useState(36);
  const [monthOptions, setMonthOptions] = useState(DEFAULT_MONTH_OPTIONS);
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
        const payload = (await poolResp.json()) as { pools?: PoolOption[]; error?: string };
        const options = (await optionsResp.json()) as BacktestOptions;
        if (!poolResp.ok || payload.error) throw new Error(payload.error ?? "종목풀 목록을 불러오지 못했습니다.");
        if (!optionsResp.ok || options.error) throw new Error(options.error ?? "백테스트 옵션을 불러오지 못했습니다.");
        if (alive) {
          const list = payload.pools ?? [];
          const loadedMonthOptions = (options.month_options ?? []).filter((month) => Number.isFinite(month) && month > 0);
          setPools(list);
          if (loadedMonthOptions.length > 0) {
            setMonthOptions(loadedMonthOptions);
            setMonths((current) =>
              loadedMonthOptions.includes(current)
                ? current
                : (loadedMonthOptions.includes(36) ? 36 : loadedMonthOptions[loadedMonthOptions.length - 1]),
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

  const runBacktest = useCallback(async () => {
    if (!poolId) {
      toast.error("종목풀을 선택해주세요.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(
        `/api/pool-backtest?pool_id=${encodeURIComponent(poolId)}&forward_days=${forwardDays}&months=${months}`,
        { cache: "no-store" },
      );
      const payload = (await resp.json()) as BacktestResult & { detail?: string };
      if (!resp.ok || payload.error) throw new Error(payload.error ?? payload.detail ?? "백테스트에 실패했습니다.");
      setResult(payload);
    } catch (e) {
      setError(e instanceof Error ? e.message : "백테스트에 실패했습니다.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [forwardDays, months, poolId, toast]);

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
          {opts.winLabel} {stat.win_rate.toFixed(0)}%
          {!stat.insufficient && <>{" · "}t {stat.t_value === null ? "-" : formatSigned(stat.t_value, 2)}</>}
          {" · "}독립표본 {stat.independent_samples}개
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
          winLabel: "이긴 날",
          hint: "같은 날 최상·최하 등급을 빼서 시장 요인을 상쇄한 값입니다.",
        })}
        {renderStat(ic, "IC(날짜별 상관)", {
          digits: 3,
          suffix: "",
          winLabel: "양수인 날",
          hint: "그날 신호 순위와 향후 수익 순위의 상관. 아래 등급 평균의 매끄러운 단조성은 10개의 증거가 아니라 같은 날을 10칸으로 나눈 것이라, 미약한 경향도 단조로 보입니다. IC 는 날짜마다 관계가 실제로 성립했는지를 봅니다.",
        })}
        <div style={{ overflowX: "auto" }}>
          <table className="poolBtTable">
            <thead>
              <tr>
                <th style={{ textAlign: "left" }}>구간</th>
                <th>평균수익</th>
                <th>기저 대비</th>
                <th>유의?</th>
                <th>상승확률</th>
                <th>기저 대비</th>
                <th>표본</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.label}>
                  <td style={{ textAlign: "left" }}>{row.label}</td>
                  <td style={{ fontWeight: 800 }} className={signedClass(row.avg_return)}>
                    {formatSigned(row.avg_return, 2)}%
                  </td>
                  <td className={signedClass(row.avg_return_diff)}>
                    {formatSigned(row.avg_return_diff, 2)}%p
                    <span style={{ color: "var(--text-muted)", fontWeight: 400 }}> (±{row.avg_return_error.toFixed(2)})</span>
                  </td>
                  <td style={{ fontWeight: 700, color: row.avg_return_significant ? "#2f9e44" : "var(--text-muted)" }}>
                    {row.avg_return_significant ? "O" : "–"}
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
    <div className="appPageStack">
      {/* 상단 옵션 + 실행 */}
      <div className="card appCard">
        <div className="card-body">
          <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "flex-end" }}>
            <label className="appLabeledField" style={{ minWidth: 180 }}>
              <span className="appLabeledFieldLabel">종목풀</span>
              <select className="form-select form-select-sm" value={poolId} onChange={(e) => setPoolId(e.target.value)}>
                {pools.length === 0 ? <option value="">불러오는 중…</option> : null}
                {pools.map((p) => (
                  <option key={p.ticker_type} value={p.ticker_type}>
                    {p.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="appLabeledField" style={{ minWidth: 130 }}>
              <span className="appLabeledFieldLabel">전망일수</span>
              <select
                className="form-select form-select-sm"
                value={forwardDays}
                onChange={(e) => setForwardDays(Number(e.target.value))}
              >
                {FORWARD_DAY_OPTIONS.map((d) => (
                  <option key={d} value={d}>
                    {d}일 후
                  </option>
                ))}
              </select>
            </label>
            <label className="appLabeledField" style={{ minWidth: 130 }}>
              <span className="appLabeledFieldLabel">기간(개월)</span>
              <select className="form-select form-select-sm" value={months} onChange={(e) => setMonths(Number(e.target.value))}>
                {monthOptions.map((m) => (
                  <option key={m} value={m}>
                    최근 {m}개월
                  </option>
                ))}
              </select>
            </label>
            <button type="button" className="btn btn-sm btn-primary" disabled={loading || !poolId} onClick={() => void runBacktest()}>
              {loading ? "백테스트 중…" : "백테스트"}
            </button>
          </div>
          <p style={{ fontSize: "0.88rem", color: "var(--text-muted)", lineHeight: 1.6, margin: "12px 0 0" }}>
            등급은 <strong>같은 날짜 안에서 종목끼리 상대비교</strong>해 나눕니다(시장 타이밍 효과 제거 — 순위 화면과 같은 관점). <strong>1등급 = 신호 상위</strong>. 고정 보유 종목 제외.
            <br />
            주 지표는 <strong>평균수익</strong>입니다 — 모멘텀은 승률이 아니라 손익비로 벌기 때문에 상승확률로는 안 잡힙니다.{" "}
            <strong>기저 대비 차이가 오차(±)를 넘어야만 의미</strong>가 있습니다.
            <br />
            표보다 위의 <strong>롱숏·IC</strong>를 믿으세요 — 등급 평균의 매끄러운 단조성은 같은 날을 10칸으로 나눈 것이라 <strong>미약한 경향도 단조로 보입니다</strong>.
            롱숏·IC는 시장 요인을 상쇄하고 <strong>날짜마다 관계가 성립했는지</strong>를 봅니다. <strong>t &gt; 2</strong>여야 우연이 아닙니다.
          </p>
        </div>
      </div>

      {error ? <div className="alert alert-danger mb-0">{error}</div> : null}
      {!loading && !result && !error ? (
        <div style={{ color: "var(--text-muted)", padding: 16 }}>종목풀·전망일수·기간을 고르고 백테스트를 눌러주세요.</div>
      ) : null}

      {result ? (
        <>
          {/* 요약: 기저율/오차범위를 가장 먼저 크게 */}
          <div className="card appCard">
            <div className="card-body">
              <div style={{ display: "flex", flexWrap: "wrap", gap: 24, alignItems: "baseline" }}>
                <div>
                  <div style={{ color: "var(--text-muted)", fontSize: "0.82rem" }}>
                    기저 평균수익 (아무 종목이나 {result.forward_days}일 보유)
                  </div>
                  <div style={{ fontSize: "1.6rem", fontWeight: 800 }} className={signedClass(result.base_return)}>
                    {formatSigned(result.base_return, 2)}%
                  </div>
                </div>
                <div>
                  <div style={{ color: "var(--text-muted)", fontSize: "0.82rem" }}>기저 상승확률</div>
                  <div style={{ fontSize: "1.6rem", fontWeight: 800, color: "var(--text-muted)" }}>
                    {result.base_rate.toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div style={{ color: "var(--text-muted)", fontSize: "0.82rem" }}>유효 독립구간</div>
                  <div style={{ fontSize: "1.6rem", fontWeight: 800, color: "#b45309" }}>{result.effective_samples}개</div>
                </div>
                <div>
                  <div style={{ color: "var(--text-muted)", fontSize: "0.82rem" }}>실제 분석 기간</div>
                  <div style={{ fontSize: "1.05rem", fontWeight: 800 }}>
                    {result.date_from} ~ {result.date_to}
                  </div>
                </div>
              </div>
              <p style={{ fontSize: "0.83rem", color: "var(--text-muted)", lineHeight: 1.6, margin: "10px 0 0" }}>
                종목 {result.ticker_count}개(고정 {result.excluded_fixed_count}개 제외) · 거래일 {result.trading_days}일 · 행{" "}
                {result.row_count.toLocaleString()}개 · 단기 {result.ma_rule.short_ma_days}일 / 메인 {result.ma_rule.main_ma_days}일 / 기울기{" "}
                {result.ma_rule.slope_days}일
                <br />
                <strong>행 {result.row_count.toLocaleString()}개가 곧 표본이 아닙니다</strong> — {result.forward_days}일 수익률이 매일 겹쳐
                실제 독립구간은 <strong>{result.effective_samples}개</strong>뿐입니다. 종목 간 동조까지 감안하면 더 적습니다.
                <br />
                {result.forward_days}일 뒤 수익을 알아야 하므로 <strong>최근 {result.forward_days}거래일은 분석에서 빠집니다</strong> — 끝일자가 오늘이
                아닌 이유입니다. 최근 구간을 보려면 전망일수를 짧게 잡으세요.
              </p>
            </div>
          </div>

          <div className="poolBtSplit">
            {renderTable(
              `이격 ${result.quantile_count}등급`,
              `종가가 메인 이평선(${result.ma_rule.main_ma_days}일) 대비 얼마나 벌어졌나 · 1등급 = 이격 상위`,
              result.disparity,
              result.disparity_long_short,
              result.disparity_ic,
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
        /* 이격(좌) · 기울기(우) 나란히. 좁아지면 1열로 접힌다. */
        .poolBtSplit {
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
          gap: 12px;
          align-items: start;
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
