"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { CartesianGrid, LabelList, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";

import { formatPoolLabel } from "@/lib/pool-label";
import { readRememberedTickerType, writeRememberedTickerType } from "../components/account-selection";
import { MONTH_OPTIONS, MonthsSelect } from "../components/MonthsSelect";
import { useTickerDetailModal } from "../components/TickerDetailModalProvider";

/** 종목 하나 = 점 하나. 가로는 MDD(낙폭), 세로는 CAGR(연 수익률). */
type RiskReturnPoint = {
  /** 시스템 표준 티커(호주는 `ASX:` 접두사 포함) — 상세 열기에 그대로 쓴다. */
  ticker: string;
  name: string;
  cagr_pct: number;
  mdd_pct: number;
  /** CAGR ÷ |MDD| — 낙폭이 없으면 null. */
  calmar: number | null;
  /** 지금 계좌에 보유 중인 종목 — 다른 화면과 같은 녹색. */
  is_held: boolean;
};

type RiskReturnResult = {
  pool_id: string;
  /** 요청한 기간(개월). */
  months: number;
  points: RiskReturnPoint[];
  /** 데이터가 모자라 점을 못 찍은 종목 수. */
  excluded: number;
  error?: string;
};

type PoolOption = { ticker_type: string; name: string; order: number; icon: string };

const COLOR_NORMAL = "#206bc4";
const COLOR_HELD = "#2fb344";
const TICK_STYLE = { fontSize: 12, fill: "#4a5568", fontWeight: 500 };
/** 점이 이 개수 이하일 때만 티커를 점 옆에 적는다 — 더 많으면 겹쳐서 읽을 수 없다(마우스를 올리면 나온다). */
const LABEL_MAX_POINTS = 60;
const DEFAULT_MONTHS = 60;
/** 칼마 비율 기준선 — 같은 칼마 값은 원점을 지나는 직선 위에 놓인다. 선보다 위쪽일수록 효율이 좋다. */
const CALMAR_GUIDES = [0.5, 1, 2];

function RiskReturnTooltip({ active, payload }: { active?: boolean; payload?: { payload: RiskReturnPoint }[] }) {
  const point = active ? payload?.[0]?.payload : null;
  if (!point) return null;
  return (
    <div className="card" style={{ padding: "8px 12px", fontSize: "var(--fs-sm)", minWidth: 180 }}>
      <div style={{ fontWeight: 700 }}>
        {point.name} <span style={{ color: "var(--text-muted)" }}>{point.ticker}</span>
      </div>
      <div>
        CAGR{" "}
        <strong className={point.cagr_pct < 0 ? "metricNegative" : "metricPositive"}>{point.cagr_pct.toFixed(2)}%</strong>
      </div>
      <div>
        MDD <strong>{point.mdd_pct.toFixed(2)}%</strong>
      </div>
      <div>
        칼마 <strong>{point.calmar == null ? "-" : point.calmar.toFixed(2)}</strong>
      </div>
    </div>
  );
}

export function PoolRiskReturnManager() {
  const openTicker = useTickerDetailModal();
  const [pools, setPools] = useState<PoolOption[]>([]);
  const [poolId, setPoolId] = useState("");
  const [months, setMonths] = useState(DEFAULT_MONTHS);
  const [monthOptions, setMonthOptions] = useState(MONTH_OPTIONS);
  const [result, setResult] = useState<RiskReturnResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const response = await fetch("/api/pool-settings", { cache: "no-store" });
        const payload = (await response.json()) as { pools?: PoolOption[]; error?: string };
        if (!response.ok || payload.error) throw new Error(payload.error ?? "종목풀 목록을 불러오지 못했습니다.");
        if (!alive) return;
        const list = payload.pools ?? [];
        setPools(list);
        if (list.length > 0) {
          const remembered = readRememberedTickerType("pools-risk-return");
          setPoolId(remembered && list.some((pool) => pool.ticker_type === remembered) ? remembered : list[0].ticker_type);
        }
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : "종목풀 목록을 불러오지 못했습니다.");
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // 기간 선택지 — 백테스트와 같은 공용 목록(가격 캐시가 못 채우는 구간은 서버가 뺀다).
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const response = await fetch("/api/pool-backtest/options", { cache: "no-store" });
        const payload = (await response.json()) as { month_options?: number[]; error?: string };
        if (!response.ok || payload.error) throw new Error(payload.error ?? "기간 선택지를 불러오지 못했습니다.");
        const loaded = (payload.month_options ?? []).filter((month) => Number.isFinite(month) && month > 0);
        if (!alive || loaded.length === 0) return;
        setMonthOptions(loaded);
        // 가격 캐시가 기본 기간을 못 채우면 가장 긴 선택지로 둔다.
        setMonths((current) => (loaded.includes(current) ? current : Math.max(...loaded)));
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : "기간 선택지를 불러오지 못했습니다.");
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  /** 마지막 요청 번호 — 풀·기간을 빠르게 바꿨을 때 늦게 도착한 이전 응답을 버린다. */
  const requestSequenceRef = useRef(0);
  useEffect(() => {
    if (!poolId) return;
    const sequence = ++requestSequenceRef.current;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const params = new URLSearchParams({ pool_id: poolId, months: String(months) });
        const response = await fetch(`/api/pool-risk-return?${params.toString()}`, { cache: "no-store" });
        const payload = (await response.json()) as RiskReturnResult & { detail?: string };
        if (sequence !== requestSequenceRef.current) return;
        if (!response.ok || payload.error) throw new Error(payload.error ?? payload.detail ?? "위험·수익을 불러오지 못했습니다.");
        setResult(payload);
      } catch (e) {
        if (sequence !== requestSequenceRef.current) return;
        setResult(null);
        setError(e instanceof Error ? e.message : "위험·수익을 불러오지 못했습니다.");
      } finally {
        if (sequence === requestSequenceRef.current) setLoading(false);
      }
    })();
  }, [poolId, months]);

  const points = result?.points ?? [];
  const { xDomain, yDomain } = useMemo(() => {
    const mdds = points.map((point) => point.mdd_pct);
    const cagrs = points.map((point) => point.cagr_pct);
    const xMin = mdds.length ? Math.floor(Math.min(...mdds) / 5) * 5 : -50;
    const yMin = cagrs.length ? Math.floor(Math.min(0, ...cagrs) / 10) * 10 : -10;
    const yMax = cagrs.length ? Math.ceil(Math.max(0, ...cagrs) / 10) * 10 : 10;
    // MDD 는 음수라 축을 뒤집는다 — 0(낙폭 없음)이 왼쪽, 클수록 오른쪽. 왼쪽 위가 좋은 자리다.
    return { xDomain: [xMin, 0] as [number, number], yDomain: [yMin, yMax] as [number, number] };
  }, [points]);

  const showLabels = points.length <= LABEL_MAX_POINTS;
  const normalPoints = points.filter((point) => !point.is_held);
  const heldPoints = points.filter((point) => point.is_held);

  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft" style={{ flexWrap: "wrap", gap: "12px 16px" }}>
                <label className="appLabeledField" style={{ minWidth: 280, flex: "0 0 auto" }}>
                  <span className="appLabeledFieldLabel">종목풀</span>
                  <select
                    className="form-select form-select-sm"
                    value={poolId}
                    onChange={(event) => {
                      setPoolId(event.target.value);
                      writeRememberedTickerType("pools-risk-return", event.target.value);
                    }}
                  >
                    {pools.length === 0 ? <option value="">불러오는 중…</option> : null}
                    {pools.map((pool) => (
                      <option key={pool.ticker_type} value={pool.ticker_type}>
                        {formatPoolLabel(pool)}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="appLabeledField" style={{ minWidth: 130, flex: "0 0 auto" }}>
                  <span className="appLabeledFieldLabel">기간</span>
                  <MonthsSelect value={months} options={monthOptions} onChange={setMonths} />
                </label>
                <span
                  style={{ alignSelf: "flex-end", paddingBottom: 6, color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}
                >
                  출시 {months}개월 이하는 제외하였습니다
                </span>
              </div>
              <div style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
                {loading
                  ? "불러오는 중…"
                  : `점 ${points.length}개${result?.excluded ? ` (데이터 부족 ${result.excluded}개 제외)` : ""} · 클릭하면 종목 상세`}
              </div>
            </div>
          </div>
          <div className="card-body" style={{ flex: 1, minHeight: 480, display: "flex", flexDirection: "column", gap: 8 }}>
            {error ? <div className="alert alert-danger mb-0">{error}</div> : null}
            <div style={{ flex: 1, minHeight: 420, minWidth: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 16, right: 32, bottom: 28, left: 8 }}>
                  <CartesianGrid stroke="#e6e8ec" strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    dataKey="mdd_pct"
                    name="MDD"
                    domain={xDomain}
                    reversed
                    tick={TICK_STYLE}
                    tickFormatter={(value: number) => `${value}%`}
                    label={{ value: "하락 변동성 (MDD) →  클수록 위험", position: "insideBottom", offset: -16, fill: "#4a5568", fontSize: 12 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="cagr_pct"
                    name="CAGR"
                    domain={yDomain}
                    tick={TICK_STYLE}
                    tickFormatter={(value: number) => `${value}%`}
                    label={{ value: "연 수익률 (CAGR)", angle: -90, position: "insideLeft", fill: "#4a5568", fontSize: 12 }}
                  />
                  <ReferenceLine y={0} stroke="#94a3b8" />
                  {CALMAR_GUIDES.map((calmar) => {
                    // y = 칼마 × |x| — 세로 범위 안에서 끝나도록 선 끝을 잘라 그린다.
                    const reach = Math.min(Math.abs(xDomain[0]), yDomain[1] / calmar);
                    if (!(reach > 0)) return null;
                    return (
                      <ReferenceLine
                        key={calmar}
                        segment={[
                          { x: 0, y: 0 },
                          { x: -reach, y: calmar * reach },
                        ]}
                        stroke="#94a3b8"
                        strokeDasharray="4 4"
                        label={{ value: `칼마 ${calmar}`, position: "insideTopRight", fill: "#64748b", fontSize: 12 }}
                      />
                    );
                  })}
                  <Tooltip content={<RiskReturnTooltip />} cursor={{ strokeDasharray: "3 3" }} />
                  {[
                    { data: normalPoints, color: COLOR_NORMAL },
                    { data: heldPoints, color: COLOR_HELD },
                  ].map(({ data, color }) => (
                    <Scatter
                      key={color}
                      data={data}
                      fill={color}
                      isAnimationActive={false}
                      cursor="pointer"
                      onClick={(point: unknown) => {
                        const ticker = (point as { ticker?: string })?.ticker;
                        if (ticker) openTicker(ticker);
                      }}
                      shape={(props: { cx?: number; cy?: number; payload?: RiskReturnPoint }) => (
                        <circle
                          cx={props.cx}
                          cy={props.cy}
                          r={6}
                          fill={color}
                          fillOpacity={0.85}
                          stroke="#ffffff"
                          strokeWidth={1}
                        />
                      )}
                    >
                      {showLabels ? (
                        <LabelList dataKey="ticker" position="right" style={{ fontSize: 11, fill: "#4a5568" }} />
                      ) : null}
                    </Scatter>
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <div style={{ display: "flex", gap: 16, fontSize: "var(--fs-sm)", color: "var(--text-muted)" }}>
              <span>
                <span style={{ color: COLOR_NORMAL }}>●</span> 종목
              </span>
              <span>
                <span style={{ color: COLOR_HELD }}>●</span> 보유 중
              </span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
