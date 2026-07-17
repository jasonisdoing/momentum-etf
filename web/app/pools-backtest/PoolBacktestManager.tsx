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
  disparity: SignalRow[];
  slope: SignalRow[];
  arrangement: SignalRow[];
  error?: string;
};

type PoolOption = { ticker_type: string; name: string };

const FORWARD_DAY_OPTIONS = [5, 10, 20, 40, 60];
const MONTH_OPTIONS = [6, 12, 24, 36, 60, 120];

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
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const resp = await fetch("/api/pool-settings", { cache: "no-store" });
        const payload = (await resp.json()) as { pools?: PoolOption[]; error?: string };
        if (!resp.ok || payload.error) throw new Error(payload.error ?? "종목풀 목록을 불러오지 못했습니다.");
        if (alive) {
          const list = payload.pools ?? [];
          setPools(list);
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

  const renderTable = (title: string, note: string, rows: SignalRow[]) => (
    <div className="card appCard">
      <div className="card-body">
        <h3 style={{ fontSize: "0.98rem", fontWeight: 800, marginBottom: 2 }}>{title}</h3>
        <p style={{ color: "var(--text-muted)", fontSize: "0.82rem", margin: "0 0 8px" }}>{note}</p>
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
                {MONTH_OPTIONS.map((m) => (
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
            분위는 <strong>같은 날짜 안에서 종목끼리 상대비교</strong>해 나눕니다(시장 타이밍 효과 제거 — 순위 화면과 같은 관점). 고정 보유 종목 제외.
            <br />
            주 지표는 <strong>평균수익</strong>입니다 — 모멘텀은 승률이 아니라 손익비로 벌기 때문에 상승확률로는 안 잡힙니다.{" "}
            <strong>기저 대비 차이가 오차(±)를 넘어야만 의미</strong>가 있습니다.
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
              </div>
              <p style={{ fontSize: "0.83rem", color: "var(--text-muted)", lineHeight: 1.6, margin: "10px 0 0" }}>
                종목 {result.ticker_count}개(고정 {result.excluded_fixed_count}개 제외) · 거래일 {result.trading_days}일 · 행{" "}
                {result.row_count.toLocaleString()}개 · 단기 {result.ma_rule.short_ma_days}일 / 메인 {result.ma_rule.main_ma_days}일 / 기울기{" "}
                {result.ma_rule.slope_days}일
                <br />
                <strong>행 {result.row_count.toLocaleString()}개가 곧 표본이 아닙니다</strong> — {result.forward_days}일 수익률이 매일 겹쳐
                실제 독립구간은 <strong>{result.effective_samples}개</strong>뿐입니다. 종목 간 동조까지 감안하면 더 적습니다.
              </p>
            </div>
          </div>

          {renderTable("이격 5분위", `종가가 메인 이평선(${result.ma_rule.main_ma_days}일) 대비 얼마나 벌어졌나`, result.disparity)}
          {renderTable(
            "기울기 5분위",
            `단기 이평선(${result.ma_rule.short_ma_days}일)의 ${result.ma_rule.slope_days}일 전 대비 변화율`,
            result.slope,
          )}
          {renderTable("배열", "단기 이평선이 메인 이평선 위(정배열)인지", result.arrangement)}
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
      `}</style>
    </div>
  );
}
