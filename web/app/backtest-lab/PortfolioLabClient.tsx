"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ColorType, LineSeries, createChart } from "lightweight-charts";
import type { Time } from "lightweight-charts";

import { AppModal } from "../components/AppModal";
import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";

type LabTicker = { ticker: string; name?: string };

type LabSummary = { total_return_pct: number; cagr_pct: number; mdd_pct: number; sortino: number };

type LabPosition = LabTicker & {
  buy_date: string;
  late_entry: boolean;
  shares: number;
  buy_price: number;
  last_price: number;
  return_pct: number;
  mdd_pct: number;
  mdd_start: string;
  mdd_end: string;
  sortino: number;
  value: number;
};

type LabResult = {
  months: number;
  rebalance?: string;
  buy_date: string;
  end_date: string;
  has_late_entry?: boolean;
  initial_capital: number;
  final_value: number;
  summary: LabSummary;
  benchmark: LabTicker & { summary: LabSummary };
  positions: LabPosition[];
  chart: { dates: string[]; portfolio_pct: number[]; benchmark_pct: number[] };
  error?: string;
};

type SavedPortfolio = { name: string; tickers: LabTicker[]; months: number; benchmark?: LabTicker; rebalance?: string; updated_at?: string | null };

const DEFAULT_BENCHMARK: LabTicker = { ticker: "069500", name: "KODEX 200" };

const REBALANCE_OPTIONS: { value: string; label: string }[] = [
  { value: "none", label: "리밸런싱 없음 (보유)" },
  { value: "weekly", label: "매주 (금요일)" },
  { value: "monthly", label: "매월 (말일)" },
  { value: "quarterly", label: "분기 (분기말)" },
  { value: "yearly", label: "매년 (연말)" },
];

const rebalanceLabel = (v?: string) => REBALANCE_OPTIONS.find((o) => o.value === (v ?? "none"))?.label ?? "리밸런싱 없음 (보유)";

const inputStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "5px 8px",
  fontSize: "0.9rem",
};

function formatKrw(value: number): string {
  return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
}

function signedClass(value: number): string {
  if (value > 0) return "#d63939";
  if (value < 0) return "#206bc4";
  return "#475569";
}

/** 포트폴리오 vs 벤치마크 누적수익률(%) 라인 차트. */
function LabChart({ result }: { result: LabResult }) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const chart = createChart(container, {
      height: 280,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#64748b" },
      grid: { vertLines: { color: "rgba(148,163,184,0.12)" }, horzLines: { color: "rgba(148,163,184,0.12)" } },
      rightPriceScale: { borderVisible: false },
      // 마지막 데이터가 우측 끝에 붙지 않도록 여백을 준다 (값 라벨과의 겹침 완화)
      timeScale: { borderVisible: false, rightOffset: 6 },
      autoSize: true,
    });

    const toLine = (values: number[]) =>
      result.chart.dates.map((d, i) => ({ time: d as Time, value: values[i] }));

    // 시리즈명(title) 라벨은 차트 끝을 가리므로 끄고, 아래 HTML 범례로 대체한다.
    chart
      .addSeries(LineSeries, { color: "#2563eb", lineWidth: 2, lastValueVisible: false, priceLineVisible: false })
      .setData(toLine(result.chart.portfolio_pct));
    chart
      .addSeries(LineSeries, { color: "#94a3b8", lineWidth: 1, lastValueVisible: false, priceLineVisible: false })
      .setData(toLine(result.chart.benchmark_pct));
    chart.timeScale().fitContent();

    return () => chart.remove();
  }, [result]);

  return (
    <div>
      <div style={{ display: "flex", gap: 16, marginBottom: 6, fontSize: "0.82rem", fontWeight: 600 }}>
        <span style={{ display: "flex", alignItems: "center", gap: 5, color: "#2563eb" }}>
          <span style={{ width: 14, height: 3, background: "#2563eb", borderRadius: 2 }} /> 포트폴리오
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 5, color: "#94a3b8" }}>
          <span style={{ width: 14, height: 3, background: "#94a3b8", borderRadius: 2 }} /> {result.benchmark.name}
        </span>
      </div>
      <div ref={containerRef} style={{ width: "100%", height: 280 }} />
    </div>
  );
}

export function PortfolioLabClient() {
  const toast = useToast();
  const [tickers, setTickers] = useState<LabTicker[]>([{ ticker: "" }]);
  const [months, setMonths] = useState(12);
  const [rebalance, setRebalance] = useState("none");
  const [benchmark, setBenchmark] = useState<LabTicker>(DEFAULT_BENCHMARK);
  const [benchInput, setBenchInput] = useState(DEFAULT_BENCHMARK.ticker);
  const [name, setName] = useState("");
  const [saved, setSaved] = useState<SavedPortfolio[]>([]);
  const [result, setResult] = useState<LabResult | null>(null);
  const [running, setRunning] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showLoadModal, setShowLoadModal] = useState(false);

  const loadSaved = useCallback(async () => {
    try {
      const resp = await fetch("/api/backtest-lab/saved", { cache: "no-store" });
      const data = (await resp.json()) as { portfolios?: SavedPortfolio[]; error?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? "저장 목록을 불러오지 못했습니다.");
      setSaved(data.portfolios ?? []);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "저장 목록을 불러오지 못했습니다.");
    }
  }, [toast]);

  useEffect(() => {
    void loadSaved();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const resolveTicker = async (index: number) => {
    const raw = (tickers[index]?.ticker ?? "").trim();
    if (!raw) {
      toast.error("티커를 입력해주세요.");
      return;
    }
    const dup = tickers.some((t, i) => i !== index && t.ticker.trim().toUpperCase() === raw.toUpperCase() && t.name);
    if (dup) {
      toast.error("이미 등록된 종목입니다.");
      return;
    }
    try {
      const resp = await fetch(`/api/backtest-lab/resolve?ticker=${encodeURIComponent(raw)}`);
      const data = (await resp.json()) as { ticker?: string; name?: string; error?: string; detail?: string };
      if (!resp.ok || data.error || !data.name) throw new Error(data.error ?? data.detail ?? "존재하지 않는 티커입니다.");
      setTickers((list) => list.map((t, i) => (i === index ? { ticker: data.ticker ?? raw, name: data.name } : t)));
      toast.success(`${data.name}(${data.ticker}) 확인 완료`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "티커 조회에 실패했습니다.");
    }
  };

  const resolveBenchmark = async () => {
    const raw = benchInput.trim();
    if (!raw) {
      toast.error("벤치마크 티커를 입력해주세요.");
      return;
    }
    try {
      const resp = await fetch(`/api/backtest-lab/resolve?ticker=${encodeURIComponent(raw)}`);
      const data = (await resp.json()) as { ticker?: string; name?: string; error?: string; detail?: string };
      if (!resp.ok || data.error || !data.name) throw new Error(data.error ?? data.detail ?? "존재하지 않는 티커입니다.");
      setBenchmark({ ticker: data.ticker ?? raw, name: data.name });
      setBenchInput(data.ticker ?? raw);
      toast.success(`[벤치마크] ${data.name}(${data.ticker}) 확인 완료`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "벤치마크 조회에 실패했습니다.");
    }
  };

  const benchConfirmed = benchInput.trim().toUpperCase() === benchmark.ticker.toUpperCase() && !!benchmark.name;

  const validTickers = tickers.filter((t) => t.ticker.trim() && t.name);

  const run = async () => {
    if (validTickers.length === 0) {
      toast.error("확인된 종목이 1개 이상 필요합니다.");
      return;
    }
    if (!benchConfirmed) {
      toast.error("벤치마크 티커를 확인해주세요.");
      return;
    }
    try {
      setRunning(true);
      setResult(null);
      const resp = await fetch("/api/backtest-lab/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers: validTickers, months, benchmark, rebalance }),
      });
      const data = (await resp.json()) as LabResult & { detail?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? data.detail ?? "실행에 실패했습니다.");
      setResult(data);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "실행에 실패했습니다.");
    } finally {
      setRunning(false);
    }
  };

  const save = async () => {
    if (!name.trim()) {
      toast.error("포트폴리오 이름을 입력해주세요.");
      return;
    }
    if (validTickers.length === 0) {
      toast.error("확인된 종목이 1개 이상 필요합니다.");
      return;
    }
    try {
      setSaving(true);
      const resp = await fetch("/api/backtest-lab/saved", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim(), tickers: validTickers, months, benchmark, rebalance }),
      });
      const data = (await resp.json()) as { portfolios?: SavedPortfolio[]; error?: string; detail?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? data.detail ?? "저장에 실패했습니다.");
      setSaved(data.portfolios ?? []);
      toast.success(`[포트폴리오] ${name.trim()} 저장 완료`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "저장에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  };

  const loadPortfolio = (p: SavedPortfolio) => {
    setName(p.name);
    setMonths(p.months);
    setRebalance(p.rebalance ?? "none");
    setTickers(p.tickers.length > 0 ? p.tickers : [{ ticker: "" }]);
    const bench = p.benchmark ?? DEFAULT_BENCHMARK;
    setBenchmark(bench);
    setBenchInput(bench.ticker);
    setResult(null);
    setShowLoadModal(false);
    toast.success(`[포트폴리오] ${p.name} 불러오기 완료`);
  };

  const deletePortfolio = async (p: SavedPortfolio) => {
    try {
      const resp = await fetch("/api/backtest-lab/saved", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: p.name }),
      });
      const data = (await resp.json()) as { portfolios?: SavedPortfolio[]; error?: string; detail?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? data.detail ?? "삭제에 실패했습니다.");
      setSaved(data.portfolios ?? []);
      toast.success(`[포트폴리오] ${p.name} 삭제 완료`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "삭제에 실패했습니다.");
    }
  };

  const summaryChip = (label: string, value: string, color?: string) => (
    <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 96 }}>
      <span style={{ color: "#94a3b8", fontSize: "0.78rem", fontWeight: 600 }}>{label}</span>
      <span style={{ fontWeight: 800, fontSize: "1.02rem", color: color ?? "#182433" }}>{value}</span>
    </div>
  );

  return (
    <PageFrame title="🧪 백테스트 실험">
      <div className="appPageStack" style={{ maxWidth: 1400 }}>
        {/* 상단: 포트폴리오 구성 | 결과 (가로, 높이 맞춤) */}
        <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "stretch" }}>
          <div style={{ flex: "1 1 460px", minWidth: 0, display: "flex" }}>
            <div className="card appCard" style={{ width: "100%" }}>
              <div className="card-body">
                <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>포트폴리오 구성 (한국 전용)</h2>
                <p style={{ color: "#94a3b8", fontSize: "0.85rem", marginBottom: 12 }}>
                  N개월 전 첫 거래일에 균등비중 매수 후 그대로 보유했다고 가정합니다. (수정주가 기준, 슬리피지 0.5%)
                </p>

                {tickers.map((t, i) => {
                  const confirmed = !!t.name;
                  return (
                    <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", padding: "4px 0", flexWrap: "wrap" }}>
                      <input
                        style={{ ...inputStyle, width: 110, backgroundColor: confirmed ? "#f8fafc" : undefined, color: confirmed ? "#64748b" : undefined }}
                        placeholder="티커"
                        value={t.ticker}
                        readOnly={confirmed}
                        onChange={(e) => setTickers((list) => list.map((x, idx) => (idx === i ? { ticker: e.target.value } : x)))}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            void resolveTicker(i);
                          }
                        }}
                      />
                      {!confirmed && (
                        <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => void resolveTicker(i)}>
                          확인
                        </button>
                      )}
                      <input
                        style={{ ...inputStyle, flex: 1, minWidth: 150, backgroundColor: "#f8fafc", color: "#64748b" }}
                        placeholder="이름 (티커 입력 후 확인)"
                        value={t.name ?? ""}
                        readOnly
                      />
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-danger"
                        onClick={() => setTickers((list) => (list.length > 1 ? list.filter((_, idx) => idx !== i) : [{ ticker: "" }]))}
                      >
                        삭제
                      </button>
                    </div>
                  );
                })}
                <button
                  type="button"
                  className="btn btn-sm btn-outline-secondary"
                  style={{ marginTop: 6 }}
                  onClick={() => setTickers((list) => [...list, { ticker: "" }])}
                >
                  + 종목 추가
                </button>

                <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 14, flexWrap: "wrap" }}>
                  <span style={{ color: "#64748b", fontWeight: 600, fontSize: "0.85rem", flexShrink: 0 }}>벤치마크</span>
                  <input
                    style={{ ...inputStyle, width: 110 }}
                    placeholder="티커"
                    value={benchInput}
                    onChange={(e) => setBenchInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        void resolveBenchmark();
                      }
                    }}
                  />
                  <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => void resolveBenchmark()}>
                    확인
                  </button>
                  <span style={{ fontSize: "0.85rem", color: benchConfirmed ? "#16a34a" : "#dc2626", fontWeight: 600 }}>
                    {benchConfirmed ? `${benchmark.name}` : "미확인"}
                  </span>
                </div>

                <div style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 12, flexWrap: "wrap" }}>
                  <span style={{ color: "#64748b", fontWeight: 600, fontSize: "0.85rem" }}>기간(개월)</span>
                  <select className="form-select form-select-sm" style={{ width: 80 }} value={months} onChange={(e) => setMonths(Number(e.target.value))}>
                    {[6, 12, 24].map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                  <span style={{ color: "#64748b", fontWeight: 600, fontSize: "0.85rem" }}>리밸런싱</span>
                  <select className="form-select form-select-sm" style={{ width: 170 }} value={rebalance} onChange={(e) => setRebalance(e.target.value)}>
                    {REBALANCE_OPTIONS.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                  <button type="button" className="btn btn-dark" disabled={running || validTickers.length === 0 || !benchConfirmed} onClick={() => void run()}>
                    {running ? "실행 중…" : `실행 (${validTickers.length}종목)`}
                  </button>
                </div>

                <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 14, flexWrap: "wrap", borderTop: "1px solid rgba(148,163,184,0.2)", paddingTop: 12 }}>
                  <input style={{ ...inputStyle, width: 180 }} placeholder="포트폴리오 이름" value={name} onChange={(e) => setName(e.target.value)} />
                  <button type="button" className="btn btn-outline-dark btn-sm" disabled={saving} onClick={() => void save()}>
                    {saving ? "저장 중…" : "저장"}
                  </button>
                  <button type="button" className="btn btn-outline-secondary btn-sm" onClick={() => setShowLoadModal(true)}>
                    불러오기 ({saved.length})
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* 상단 오른쪽: 결과 (요약 + 차트) */}
          <div style={{ flex: "1 1 460px", minWidth: 0, display: "flex" }}>
            {result ? (
              <div className="card appCard" style={{ width: "100%" }}>
                <div className="card-body">
                  <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>
                    결과 — {result.buy_date} ~ {result.end_date} ({result.months}개월)
                  </h2>
                  <p style={{ color: "#94a3b8", fontSize: "0.82rem", marginBottom: 12 }}>
                    초기 {formatKrw(result.initial_capital)} → 최종 {formatKrw(result.final_value)} · 리밸런싱: {rebalanceLabel(result.rebalance)}
                  </p>
                  <div style={{ display: "flex", gap: 20, flexWrap: "wrap", marginBottom: 8 }}>
                    {summaryChip("총수익률", `${result.summary.total_return_pct.toFixed(2)}%`, signedClass(result.summary.total_return_pct))}
                    {summaryChip("CAGR", `${result.summary.cagr_pct.toFixed(2)}%`, signedClass(result.summary.cagr_pct))}
                    {summaryChip("MDD", `${result.summary.mdd_pct.toFixed(2)}%`, "#d63939")}
                    {summaryChip("Sortino", result.summary.sortino.toFixed(2))}
                  </div>
                  <div style={{ color: "#94a3b8", fontSize: "0.8rem", marginBottom: 10 }}>
                    벤치마크 {result.benchmark.name}: 총 {result.benchmark.summary.total_return_pct.toFixed(2)}% · MDD{" "}
                    {result.benchmark.summary.mdd_pct.toFixed(2)}% · Sortino {result.benchmark.summary.sortino.toFixed(2)}
                  </div>
                  <LabChart result={result} />
                </div>
              </div>
            ) : (
              <div className="card appCard" style={{ width: "100%" }}>
                <div className="card-body" style={{ color: "#94a3b8", padding: 24 }}>
                  종목을 확인한 뒤 실행을 누르면 결과가 여기에 표시됩니다.
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 하단: 종목별 성과 (전체 폭) */}
        {result ? (
          <>
                <div className="card appCard">
                  <div className="card-body">
                    <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 8 }}>종목별 성과</h2>
                    {result.has_late_entry ? (
                      <p style={{ color: "#b45309", background: "rgba(245,158,11,0.08)", fontSize: "0.8rem", padding: "6px 10px", borderRadius: 6, marginBottom: 10 }}>
                        ⚠️ 실험 시작 이후 상장된 종목은 배정 예산을 현금으로 대기시켰다가 상장일 종가에 편입합니다 (상장 전 구간은 합성하지 않음).
                      </p>
                    ) : null}
                    <div style={{ overflowX: "auto" }}>
                      <table className="table table-sm" style={{ fontSize: "0.85rem" }}>
                        <thead>
                          <tr>
                            <th>종목</th>
                            <th>매수일</th>
                            <th style={{ textAlign: "right" }}>매수가</th>
                            <th style={{ textAlign: "right" }}>현재가</th>
                            <th style={{ textAlign: "right" }}>수익률</th>
                            <th style={{ textAlign: "right" }}>MDD</th>
                            <th style={{ textAlign: "right" }}>Sortino</th>
                            <th style={{ textAlign: "right" }}>평가금액</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.positions.map((p) => (
                            <tr key={p.ticker}>
                              <td>
                                {p.name} <span style={{ color: "#94a3b8" }}>({p.ticker})</span>
                              </td>
                              <td style={{ whiteSpace: "nowrap", color: p.late_entry ? "#b45309" : "#64748b" }}>
                                {p.buy_date}
                                {p.late_entry ? " ↩" : ""}
                              </td>
                              <td style={{ textAlign: "right" }}>{new Intl.NumberFormat("ko-KR").format(p.buy_price)}</td>
                              <td style={{ textAlign: "right" }}>{new Intl.NumberFormat("ko-KR").format(p.last_price)}</td>
                              <td style={{ textAlign: "right", fontWeight: 700, color: signedClass(p.return_pct) }}>
                                {p.return_pct.toFixed(2)}%
                              </td>
                              <td style={{ textAlign: "right", color: "#d63939", whiteSpace: "nowrap" }}>
                                {p.mdd_pct.toFixed(2)}%
                                <span style={{ color: "#94a3b8", fontSize: "0.72rem", marginLeft: 4 }}>
                                  ({p.mdd_start.replaceAll("-", "/")}~{p.mdd_end.replaceAll("-", "/")})
                                </span>
                              </td>
                              <td style={{ textAlign: "right" }}>{p.sortino.toFixed(2)}</td>
                              <td style={{ textAlign: "right" }}>{formatKrw(p.value)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
          </>
        ) : null}
      </div>

      <AppModal open={showLoadModal} title="저장된 포트폴리오" size="xl" onClose={() => setShowLoadModal(false)}>
        {saved.length === 0 ? (
          <div style={{ color: "#94a3b8", fontSize: "0.9rem", padding: 8 }}>저장된 포트폴리오가 없습니다.</div>
        ) : (
          saved.map((p) => (
            <div key={p.name} style={{ display: "flex", gap: 8, alignItems: "center", padding: "8px 0", borderBottom: "1px solid rgba(148,163,184,0.15)" }}>
              <span style={{ fontWeight: 700, flexShrink: 0, minWidth: 120 }}>{p.name}</span>
              <span style={{ color: "#94a3b8", fontSize: "0.82rem", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {p.months}개월 · {rebalanceLabel(p.rebalance)} · vs {(p.benchmark ?? DEFAULT_BENCHMARK).name} · {p.tickers.map((t) => t.name ?? t.ticker).join(", ")}
              </span>
              <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => loadPortfolio(p)}>
                불러오기
              </button>
              <button type="button" className="btn btn-sm btn-outline-danger" onClick={() => void deletePortfolio(p)}>
                삭제
              </button>
            </div>
          ))
        )}
      </AppModal>
    </PageFrame>
  );
}
