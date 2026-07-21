"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CandlestickSeries, ColorType, LineSeries, LineStyle, createChart, createSeriesMarkers } from "lightweight-charts";
import type {
  CandlestickData,
  IChartApi,
  IPriceLine,
  ISeriesApi,
  ISeriesMarkersPluginApi,
  LineData,
  SeriesMarker,
  Time,
  UTCTimestamp,
} from "lightweight-charts";

import { PageFrame } from "../components/PageFrame";

// 레버리지 단타 — 위: 나스닥100 선물(캔들만, 눈으로 참고), 아래: KODEX 레버리지(신호·화살표 표시).
// 둘 다 1분봉을 KST 09:00~15:00 세션으로 잘라 시간축을 동기화해 겹쳐 본다.
// 전략: EMA 돌파 + 슈퍼트렌드 조합(가격>EMA & 슈퍼트렌드 상승 → 매수, 하나라도 깨지면 매도).
// 기록·백테스트는 이 세션 데이터로 계산한다. 선물 차트는 눈으로 참고만(매매 로직 미사용).
type Candle = { t: number; o: number; h: number; l: number; c: number };
type Trade = { type: "buy" | "sell"; time: number; price: number; pnlPct?: number };
type Overlay = { color: string; data: LineData[]; width?: number };

const LEV_CODE = "A122630";
const LEV_NAME = "KODEX 레버리지";
const FUT_CODE = "RFU.NQc1";
const FUT_NAME = "나스닥100 선물";
const LEV_INTERVAL = "min:1";
const FUT_INTERVAL = "min:1";
const SESSION_START_MIN = 9 * 60; // KST 09:00
const SESSION_END_MIN = 15 * 60 + 30; // KST 15:30(이하)
const KST_OFFSET_SEC = 9 * 3600;
const POLL_MS = 2000;
const CANDLE_COUNT = 450;
const UP_COLOR = "#d63939";
const DOWN_COLOR = "#206bc4";
const EMA_COLOR = "rgba(230,145,56,0.9)"; // EMA선(주황)
const ST_COLOR = "rgba(120,130,150,0.75)"; // 슈퍼트렌드선(회색)
// 선물 참고용 이동평균선(SMA) 20·60·120·240.
const FUT_MAS = [
  { period: 20, color: "rgba(230,145,56,0.95)" },
  { period: 60, color: "rgba(46,164,79,0.95)" },
  { period: 120, color: "rgba(32,107,196,0.95)" },
  { period: 240, color: "rgba(150,90,205,0.95)" },
];

function fmtKrw(value: number): string {
  return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
}
function fmtPoints(value: number): string {
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 2 }).format(value);
}

const WEEKDAYS = ["일", "월", "화", "수", "목", "금", "토"];
function fmtDateTime(msTimestamp: number): string {
  const d = new Date(msTimestamp);
  const date = `${d.getMonth() + 1}월 ${d.getDate()}일(${WEEKDAYS[d.getDay()]})`;
  const time = d.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" });
  return `${date} ${time}`;
}
function toTime(msTimestamp: number): UTCTimestamp {
  return (Math.floor(msTimestamp / 1000) + KST_OFFSET_SEC) as UTCTimestamp;
}
function intervalLabel(interval: string): string {
  const m = interval.match(/^min:(\d+)$/);
  return m ? `${m[1]}분봉` : interval;
}
// KST 09:00~15:30 세션 안의 봉인지.
function inSession(msTimestamp: number): boolean {
  const kstMin = Math.floor((msTimestamp / 1000 + KST_OFFSET_SEC) / 60) % 1440;
  return kstMin >= SESSION_START_MIN && kstMin <= SESSION_END_MIN;
}

// 지수이동평균(EMA). 첫 값은 종가로 시드.
function computeEma(values: number[], period: number): number[] {
  const k = 2 / (Math.max(1, period) + 1);
  const out: number[] = [];
  let prev = 0;
  for (let i = 0; i < values.length; i += 1) {
    prev = i === 0 ? values[0] : values[i] * k + prev * (1 - k);
    out.push(prev);
  }
  return out;
}
// 단순이동평균(SMA). 기간 미만 구간은 가용분 평균(부분).
function computeSma(values: number[], period: number): number[] {
  const p = Math.max(1, period);
  const out: number[] = [];
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i];
    if (i >= p) sum -= values[i - p];
    out.push(sum / Math.min(i + 1, p));
  }
  return out;
}

// 지수가중이동평균(alpha 지정) — ATR 의 Wilder 스무딩용.
function ewmAlpha(values: number[], alpha: number): number[] {
  const out: number[] = [];
  let prev = 0;
  for (let i = 0; i < values.length; i += 1) {
    prev = i === 0 ? values[0] : values[i] * alpha + prev * (1 - alpha);
    out.push(prev);
  }
  return out;
}

// 슈퍼트렌드(백엔드 _calculate_supertrend 와 동일 로직: ATR ewm + final band).
function computeSupertrend(candles: Candle[], period: number, mult: number): { supertrend: number[]; direction: number[] } {
  const n = candles.length;
  const high = candles.map((c) => c.h);
  const low = candles.map((c) => c.l);
  const close = candles.map((c) => c.c);
  const tr: number[] = [];
  for (let i = 0; i < n; i += 1) {
    const t1 = high[i] - low[i];
    const t2 = i > 0 ? Math.abs(high[i] - close[i - 1]) : t1;
    const t3 = i > 0 ? Math.abs(low[i] - close[i - 1]) : t1;
    tr.push(Math.max(t1, t2, t3));
  }
  const atr = ewmAlpha(tr, 1 / Math.max(1, period));
  const basicUpper = candles.map((c, i) => (c.h + c.l) / 2 + mult * atr[i]);
  const basicLower = candles.map((c, i) => (c.h + c.l) / 2 - mult * atr[i]);
  const finalUpper = new Array<number>(n).fill(0);
  const finalLower = new Array<number>(n).fill(0);
  const supertrend = new Array<number>(n).fill(0);
  const direction = new Array<number>(n).fill(1);
  for (let i = 1; i < n; i += 1) {
    const pu = finalUpper[i - 1];
    const pl = finalLower[i - 1];
    const pc = close[i - 1];
    finalUpper[i] = basicUpper[i] < pu || pc > pu ? basicUpper[i] : pu;
    finalLower[i] = basicLower[i] > pl || pc < pl ? basicLower[i] : pl;
    if (direction[i - 1] === 1) {
      if (close[i] < finalLower[i]) {
        direction[i] = -1;
        supertrend[i] = finalUpper[i];
      } else {
        direction[i] = 1;
        supertrend[i] = finalLower[i];
      }
    } else if (close[i] > finalUpper[i]) {
      direction[i] = 1;
      supertrend[i] = finalLower[i];
    } else {
      direction[i] = -1;
      supertrend[i] = finalUpper[i];
    }
  }
  if (n > 0) {
    finalUpper[0] = basicUpper[0];
    finalLower[0] = basicLower[0];
    supertrend[0] = basicLower[0];
  }
  return { supertrend, direction };
}

type StratMode = "both" | "ema" | "st"; // EMA+ST 조합 / EMA 단독 / 슈퍼트렌드 단독
const MODE_LABELS: Record<StratMode, string> = { both: "EMA + 슈퍼트렌드", ema: "EMA 단독", st: "슈퍼트렌드 단독" };

// 강세(bull) 판정: both=가격>EMA & ST상승, ema=가격>EMA, st=ST상승. 강세 진입=매수 이벤트.
// 매도는 강세가 아닌 봉에서 실행하므로(computeStrategyTrades) 강세 상태 배열을 함께 반환한다.
function buildSignals(mode: StratMode, candles: Candle[], emaPeriod: number, stPeriod: number, stMult: number): { buy: boolean[]; bull: boolean[] } {
  const n = candles.length;
  const closes = candles.map((c) => c.c);
  let bull: boolean[];
  if (mode === "ema") {
    const ema = computeEma(closes, emaPeriod);
    bull = closes.map((c, i) => c > ema[i]);
  } else if (mode === "st") {
    const { direction } = computeSupertrend(candles, stPeriod, stMult);
    bull = direction.map((d) => d === 1);
  } else {
    const ema = computeEma(closes, emaPeriod);
    const { direction } = computeSupertrend(candles, stPeriod, stMult);
    bull = closes.map((c, i) => c > ema[i] && direction[i] === 1);
  }
  const buy = new Array<boolean>(n).fill(false);
  for (let i = 1; i < n; i += 1) {
    buy[i] = bull[i] && !bull[i - 1];
  }
  return { buy, bull };
}

// 매매 시뮬레이션: 강세 진입 봉에서 매수, 강세가 깨지는 봉에서 매도.
function computeStrategyTrades(candles: Candle[], buy: boolean[], bull: boolean[]): Trade[] {
  const n = Math.min(candles.length, buy.length, bull.length);
  const trades: Trade[] = [];
  let entryPrice: number | null = null;
  for (let i = 0; i < n; i += 1) {
    if (entryPrice !== null && !bull[i]) {
      const price = candles[i].c;
      trades.push({ type: "sell", time: candles[i].t, price, pnlPct: (price / entryPrice - 1) * 100 });
      entryPrice = null;
    }
    if (entryPrice === null && buy[i]) {
      entryPrice = candles[i].c;
      trades.push({ type: "buy", time: candles[i].t, price: candles[i].c });
    }
  }
  return trades;
}

type Metrics = { trades: number; winRate: number; cum: number; avg: number; mdd: number };
function backtestMetrics(trades: Trade[]): Metrics {
  const sells = trades.filter((t) => t.type === "sell" && t.pnlPct != null);
  const n = sells.length;
  const wins = sells.filter((t) => (t.pnlPct ?? 0) > 0).length;
  let equity = 1;
  let peak = 1;
  let mdd = 0;
  for (const s of sells) {
    equity *= 1 + (s.pnlPct ?? 0) / 100;
    peak = Math.max(peak, equity);
    mdd = Math.max(mdd, (peak - equity) / peak);
  }
  const avg = n > 0 ? sells.reduce((acc, t) => acc + (t.pnlPct ?? 0), 0) / n : 0;
  return { trades: n, winRate: n > 0 ? (wins / n) * 100 : 0, cum: (equity - 1) * 100, avg, mdd: mdd * 100 };
}

function runCombo(levC: Candle[], mode: StratMode, emaPeriod: number, stPeriod: number, stMult: number): Trade[] {
  const { buy, bull } = buildSignals(mode, levC, emaPeriod, stPeriod, stMult);
  return computeStrategyTrades(levC, buy, bull);
}

// 스윕: EMA 기간 × 슈퍼트렌드 기간 × 배수. 모드별로 관련 파라미터만 조합한다.
const SWEEP_EMA = [5, 10, 20, 30];
const SWEEP_ST = [7, 10, 14];
const SWEEP_MULT = [1, 1.5, 2, 2.5, 3];
const SWEEP_TOP = 40; // 표에는 누적수익 상위 N개만 표시.

type SweepRow = { label: string; mode: StratMode; ema: number; st: number; mult: number; m: Metrics };
type ScalpSettings = { mode: StratMode; ema_period: number; st_period: number; st_mult: number };
type BacktestResult = { rangeLabel: string; levBars: number; current: Metrics; sweep: SweepRow[] };

// 단일 차트(캔들 + 오버레이 선 + 화살표). 선물은 overlays·markers 를 비워 캔들만 표시한다.
type ScalpChartProps = {
  name: string;
  code: string;
  feed: "kr-stock" | "us-futures";
  interval: string;
  note?: string;
  formatPrice: (value: number) => string;
  overlays: Overlay[]; // 그릴 선(최대 4개)
  legend?: { label: string; color: string }[]; // 선 범례(선택)
  priceLines?: { price: number; color: string; title?: string }[]; // 수평 가격선(현재가·트리거)
  markers: Trade[];
  hint?: { text: string; color: string }; // 가격 옆 다음 트리거 안내(레버리지만)
  onData?: (candles: Candle[]) => void;
  onChart?: (chart: IChartApi | null) => void;
};

const MAX_OVERLAYS = 4;

function ScalpChart({ name, code, feed, interval, note, formatPrice, overlays, legend, priceLines, markers, hint, onData, onChart }: ScalpChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const lineSeriesRef = useRef<ISeriesApi<"Line">[]>([]);
  const priceLinesRef = useRef<IPriceLine[]>([]);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const firstLoadRef = useRef(true);
  const barCountRef = useRef(0); // 직전 봉 개수(실시간 추적 여부 판단용)
  const lastSigRef = useRef(""); // 직전 데이터 시그니처(변화 없으면 재그리기 생략 → 마감 후 깜빡임 방지)

  const [candles, setCandles] = useState<Candle[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<string>("");
  const [summary, setSummary] = useState<{ close: number; changePct: number } | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const chart = createChart(container, {
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#5b6675" },
      grid: { vertLines: { color: "rgba(148,163,184,0.12)" }, horzLines: { color: "rgba(148,163,184,0.12)" } },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, timeVisible: true, secondsVisible: false, rightOffset: 4 },
      autoSize: true,
    });
    candleSeriesRef.current = chart.addSeries(CandlestickSeries, {
      upColor: UP_COLOR,
      downColor: DOWN_COLOR,
      borderVisible: false,
      wickUpColor: UP_COLOR,
      wickDownColor: DOWN_COLOR,
      priceLineVisible: false, // 기본 현재가선 끄고 커스텀 가격선(priceLines)을 쓴다.
    });
    lineSeriesRef.current = Array.from({ length: MAX_OVERLAYS }, () =>
      chart.addSeries(LineSeries, { color: "transparent", lineWidth: 2, lastValueVisible: false, priceLineVisible: false }),
    );
    markersRef.current = createSeriesMarkers(candleSeriesRef.current, []);
    chartRef.current = chart;
    onChart?.(chart);
    return () => {
      onChart?.(null);
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      lineSeriesRef.current = [];
      markersRef.current = null;
    };
    // onChart 는 부모에서 안정적으로 전달(변경 시 차트 재생성 방지 위해 deps 제외).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let alive = true;
    const fetchCandles = async () => {
      try {
        const resp = await fetch(
          `/api/leverage-candles?code=${encodeURIComponent(code)}&interval=${encodeURIComponent(interval)}&count=${CANDLE_COUNT}&feed=${feed}`,
          { cache: "no-store" },
        );
        const data = (await resp.json()) as { candles?: Candle[]; error?: string };
        if (!resp.ok || data.error) throw new Error(data.error ?? "캔들을 불러오지 못했습니다.");
        if (!alive || !candleSeriesRef.current) return;
        // KST 09:00~15:30 세션만 남겨 두 차트의 시간축을 맞춘다.
        const next = (data.candles ?? []).filter((c) => inSession(c.t));
        // 데이터가 직전과 동일하면(장 마감 후 등) 재그리기·스크롤을 건너뛴다(깜빡임 방지).
        const last = next[next.length - 1];
        const sig = last ? `${next.length}:${last.t}:${last.c}` : "";
        if (!firstLoadRef.current && sig === lastSigRef.current) return;
        lastSigRef.current = sig;
        const ts = chartRef.current?.timeScale();
        // 데이터 교체 전에 "지금 오른쪽 끝(실시간)을 보고 있었는지" 판단(과거로 스크롤했으면 추적 안 함).
        const vr = ts?.getVisibleLogicalRange();
        const following = !vr || vr.to >= barCountRef.current - 1;
        candleSeriesRef.current.setData(
          next.map<CandlestickData>((c) => ({ time: toTime(c.t), open: c.o, high: c.h, low: c.l, close: c.c })),
        );
        setCandles(next);
        barCountRef.current = next.length;
        if (firstLoadRef.current && next.length > 0) {
          // 최초 표시폭은 최근 약 1시간(60개 1분봉). 이후 사용자가 자유롭게 확대/축소.
          ts?.setVisibleLogicalRange({ from: Math.max(0, next.length - 60), to: next.length + 4 });
          firstLoadRef.current = false;
        } else if (following) {
          // 오른쪽 끝을 보고 있었으면 새 봉을 따라 실시간으로 이동(줌 유지).
          ts?.scrollToRealTime();
        }
        if (next.length > 0) {
          const dayOpen = next[0].o;
          const cur = next[next.length - 1].c;
          setSummary({ close: cur, changePct: dayOpen > 0 ? (cur / dayOpen - 1) * 100 : 0 });
        }
        setError(null);
        setUpdatedAt(new Date().toLocaleTimeString("ko-KR"));
      } catch (err) {
        if (alive) setError(err instanceof Error ? err.message : "캔들을 불러오지 못했습니다.");
      }
    };
    void fetchCandles();
    const id = setInterval(() => void fetchCandles(), POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [code, feed, interval]);

  useEffect(() => {
    onData?.(candles);
  }, [candles, onData]);

  // 오버레이 선 갱신(레버리지: EMA·슈퍼트렌드 / 선물: 20·60·120·240 이동평균).
  useEffect(() => {
    lineSeriesRef.current.forEach((ref, i) => {
      const ov = overlays[i];
      ref.applyOptions({ color: ov ? ov.color : "transparent", lineWidth: (ov?.width ?? 2) as 1 | 2 | 3 | 4 });
      ref.setData(ov ? ov.data : []);
    });
  }, [overlays]);

  // 수평 가격선(현재가·매수/매도 트리거) 갱신. 매번 지우고 다시 만든다(1~2개라 가벼움).
  useEffect(() => {
    const series = candleSeriesRef.current;
    if (!series) return;
    priceLinesRef.current.forEach((pl) => series.removePriceLine(pl));
    priceLinesRef.current = (priceLines ?? []).map((l) =>
      series.createPriceLine({
        price: l.price,
        color: l.color,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: false, // 우측 축 라벨은 끄고 점선만 표시(차트 가림 방지).
      }),
    );
  }, [priceLines]);

  useEffect(() => {
    const data: SeriesMarker<Time>[] = markers.map((t) =>
      t.type === "buy"
        ? { time: toTime(t.time), position: "belowBar", color: UP_COLOR, shape: "arrowUp", text: "매수", size: 2 }
        : { time: toTime(t.time), position: "aboveBar", color: DOWN_COLOR, shape: "arrowDown", text: "매도", size: 2 },
    );
    markersRef.current?.setMarkers(data);
  }, [markers]);

  const changeColor = summary && summary.changePct > 0 ? UP_COLOR : summary && summary.changePct < 0 ? DOWN_COLOR : "#475569";

  return (
    <div className="card appCard scalpChartCard">
      <div className="card-body">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
            <h2 style={{ fontSize: "1.0rem", fontWeight: 800, margin: 0 }}>
              {name} <span style={{ color: "var(--text-muted)", fontWeight: 500 }}>({code})</span> · {intervalLabel(interval)}
              {note ? <span style={{ color: "var(--text-muted)", fontWeight: 600, fontSize: "0.82rem" }}> · {note}</span> : null}
            </h2>
            {summary ? (
              <span style={{ fontWeight: 800, color: changeColor }}>
                {formatPrice(summary.close)} ({summary.changePct > 0 ? "+" : ""}
                {summary.changePct.toFixed(2)}%)
              </span>
            ) : null}
          </div>
          <span className="text-muted small">
            {error ? <span style={{ color: UP_COLOR }}>{error}</span> : `실시간(2초) · ${updatedAt || "-"}`}
          </span>
        </div>
        {/* 트리거 안내: 가격 옆이 아니라 종목명 아래(차트 좌상단)에 표시 */}
        {hint ? <div style={{ marginTop: -4, marginBottom: 6, fontWeight: 800, fontSize: "0.88rem", color: hint.color }}>{hint.text}</div> : null}
        {legend && legend.length > 0 ? (
          <div style={{ marginTop: -2, marginBottom: 6, display: "flex", gap: 10, flexWrap: "wrap", fontSize: "0.8rem", fontWeight: 700 }}>
            {legend.map((l) => (
              <span key={l.label} style={{ color: l.color }}>
                ━ {l.label}
              </span>
            ))}
          </div>
        ) : null}
        {/* 높이는 인라인으로 지정한다(styled-jsx 스코프가 자식 컴포넌트에 닿지 않음). */}
        <div ref={containerRef} style={{ width: "100%", flex: "1 1 auto", minHeight: 0 }} />
      </div>
    </div>
  );
}

export function LeverageScalpClient() {
  const [mode, setMode] = useState<StratMode>("both");
  const [emaPeriod, setEmaPeriod] = useState("10");
  const [stPeriod, setStPeriod] = useState("10");
  const [stMult, setStMult] = useState("2");
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [saveError, setSaveError] = useState<string | null>(null);
  const [levCandles, setLevCandles] = useState<Candle[]>([]);
  const handleLevData = useCallback((c: Candle[]) => setLevCandles(c), []);
  const [futCandles, setFutCandles] = useState<Candle[]>([]);
  const handleFutData = useCallback((c: Candle[]) => setFutCandles(c), []);
  const [btOpen, setBtOpen] = useState(false);

  // 선물 이동평균선(SMA 20/60/120/240) 오버레이 + 범례.
  const futOverlays = useMemo<Overlay[]>(() => {
    if (futCandles.length === 0) return [];
    const closes = futCandles.map((c) => c.c);
    return FUT_MAS.map((m) => ({
      color: m.color,
      width: 1,
      data: computeSma(closes, m.period).map<LineData>((v, i) => ({ time: toTime(futCandles[i].t), value: v })),
    }));
  }, [futCandles]);
  const futLegend = FUT_MAS.map((m) => ({ label: String(m.period), color: m.color }));

  // 두 차트 시간축 동기화(한쪽 스크롤/줌 → 다른 쪽도 같은 시간범위).
  const futChartRef = useRef<IChartApi | null>(null);
  const levChartRef = useRef<IChartApi | null>(null);
  const syncingRef = useRef(false);
  const syncWiredRef = useRef(false);
  const wireSync = useCallback(() => {
    const a = futChartRef.current;
    const b = levChartRef.current;
    if (!a || !b || syncWiredRef.current) return;
    syncWiredRef.current = true;
    const link = (src: IChartApi, dst: IChartApi) => {
      src.timeScale().subscribeVisibleTimeRangeChange((range) => {
        if (!range || syncingRef.current) return;
        syncingRef.current = true;
        try {
          dst.timeScale().setVisibleRange(range);
        } catch {
          /* 범위가 데이터 밖이면 무시 */
        }
        syncingRef.current = false;
      });
    };
    link(a, b);
    link(b, a);
  }, []);
  const handleFutChart = useCallback(
    (chart: IChartApi | null) => {
      futChartRef.current = chart;
      if (chart) wireSync();
      else syncWiredRef.current = false;
    },
    [wireSync],
  );
  const handleLevChart = useCallback(
    (chart: IChartApi | null) => {
      levChartRef.current = chart;
      if (chart) wireSync();
      else syncWiredRef.current = false;
    },
    [wireSync],
  );

  // 마운트 시 DB 설정 로드(없으면 프런트 기본값).
  useEffect(() => {
    let alive = true;
    void (async () => {
      try {
        const resp = await fetch("/api/leverage-scalp-settings", { cache: "no-store" });
        const data = (await resp.json()) as { settings?: ScalpSettings | null; error?: string };
        if (!resp.ok || data.error) throw new Error(data.error ?? "설정을 불러오지 못했습니다.");
        if (!alive || !data.settings) return;
        setMode(data.settings.mode);
        setEmaPeriod(String(data.settings.ema_period));
        setStPeriod(String(data.settings.st_period));
        setStMult(String(data.settings.st_mult));
      } catch {
        // 로드 실패 → 기본값 표시.
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const emaP = Math.max(1, Math.trunc(Number(emaPeriod) || 1));
  const stP = Math.max(1, Math.trunc(Number(stPeriod) || 1));
  const stM = Math.max(0.1, Number(stMult) || 0.1);

  // 결합 신호. 형성 중 마지막 봉 제외.
  const trades = useMemo(() => {
    const lev = levCandles.slice(0, Math.max(0, levCandles.length - 1));
    if (lev.length === 0) return [];
    return runCombo(lev, mode, emaP, stP, stM);
  }, [levCandles, mode, emaP, stP, stM]);

  // 레버리지 지표(EMA·슈퍼트렌드) — 오버레이선·트리거 안내에 공용.
  const levInd = useMemo(() => {
    if (levCandles.length === 0) return null;
    const ema = computeEma(levCandles.map((c) => c.c), emaP);
    const { supertrend, direction } = computeSupertrend(levCandles, stP, stM);
    return { ema, supertrend, direction };
  }, [levCandles, emaP, stP, stM]);

  const levOverlays = useMemo<Overlay[]>(() => {
    if (!levInd) return [];
    const emaLine: Overlay = { color: EMA_COLOR, data: levCandles.map<LineData>((c, i) => ({ time: toTime(c.t), value: levInd.ema[i] })) };
    const stLine: Overlay = { color: ST_COLOR, data: levCandles.map<LineData>((c, i) => ({ time: toTime(c.t), value: levInd.supertrend[i] })) };
    if (mode === "ema") return [emaLine];
    if (mode === "st") return [stLine];
    return [emaLine, stLine];
  }, [levCandles, levInd, mode]);

  const realizedPct = trades.reduce((acc, t) => acc + (t.type === "sell" ? t.pnlPct ?? 0 : 0), 0);
  const holding = trades.length > 0 && trades[trades.length - 1].type === "buy";
  const lastPrice = levCandles.length > 0 ? levCandles[levCandles.length - 1].c : null;
  const openEntry = holding ? trades[trades.length - 1].price : null;
  const unrealizedPct = openEntry != null && lastPrice != null ? (lastPrice / openEntry - 1) * 100 : null;
  const effPnls = [
    ...trades.filter((t) => t.type === "sell" && t.pnlPct != null).map((t) => t.pnlPct as number),
    ...(unrealizedPct != null ? [unrealizedPct] : []),
  ];
  const totalPct = realizedPct + (unrealizedPct ?? 0);
  const winCount = effPnls.filter((p) => p > 0).length;
  const winRate = effPnls.length > 0 ? (winCount / effPnls.length) * 100 : 0;
  const avgPct = effPnls.length > 0 ? effPnls.reduce((a, p) => a + p, 0) / effPnls.length : 0;

  // 레버리지 가격 옆 안내: 미보유면 매수까지 상승%, 보유 중이면 매도까지 하락%.
  // 트리거 레벨 — 매수: 가격이 EMA 위 & 슈퍼트렌드 상승이어야 하므로 넘어야 할 레벨 = 하락추세면 max(EMA, ST밴드), 상승추세면 EMA.
  //           매도: 가격이 EMA 아래 또는 슈퍼트렌드 하락 전환 시 → 먼저 닿는 레벨 = max(EMA, ST밴드).
  const priceHint = useMemo<{ text: string; color: string; pct: number; kind: "buy" | "sell" } | undefined>(() => {
    if (!levInd || levCandles.length < 2) return undefined;
    const i = levCandles.length - 2; // 마지막 확정봉 지표
    const E = levInd.ema[i];
    const S = levInd.supertrend[i];
    const D = levInd.direction[i];
    const P = levCandles[levCandles.length - 1].c; // 현재가(형성 봉 종가)
    if (!(P > 0)) return undefined;
    // 트리거 레벨 — ema: EMA / st: ST밴드 / both: 매수는 하락추세면 max(EMA,ST) 상승추세면 EMA, 매도는 max(EMA,ST).
    if (holding) {
      const level = mode === "ema" ? E : mode === "st" ? S : Math.max(E, S);
      const pct = (level / P - 1) * 100;
      return { text: `${pct.toFixed(2)}% 하락하면 매도`, color: DOWN_COLOR, pct, kind: "sell" };
    }
    const level = mode === "ema" ? E : mode === "st" ? S : D === 1 ? E : Math.max(E, S);
    const pct = (level / P - 1) * 100;
    return { text: `${pct > 0 ? "+" : ""}${pct.toFixed(2)}% 상승하면 매수`, color: UP_COLOR, pct, kind: "buy" };
  }, [levInd, levCandles, holding, mode]);

  // 레버리지 수평 가격선: 현재가(검은 점선) + 대기 시 매수선(빨강) / 보유 시 매도선(파랑).
  const levPriceLines = useMemo<{ price: number; color: string; title?: string }[]>(() => {
    const lines: { price: number; color: string; title?: string }[] = [];
    const P = levCandles.length > 0 ? levCandles[levCandles.length - 1].c : null;
    if (P != null) lines.push({ price: P, color: "#111111", title: "현재가" });
    if (levInd && levCandles.length >= 2) {
      const i = levCandles.length - 2;
      const E = levInd.ema[i];
      const S = levInd.supertrend[i];
      const D = levInd.direction[i];
      if (holding) {
        lines.push({ price: mode === "ema" ? E : mode === "st" ? S : Math.max(E, S), color: DOWN_COLOR, title: "매도" });
      } else {
        lines.push({ price: mode === "ema" ? E : mode === "st" ? S : D === 1 ? E : Math.max(E, S), color: UP_COLOR, title: "매수" });
      }
    }
    return lines;
  }, [levCandles, levInd, holding, mode]);

  // 트리거까지 거리(|pct|)가 줄면 "가까워짐", 늘면 "멀어짐". 직전 갱신과 비교(종류 바뀌면 리셋).
  const prevHintRef = useRef<{ pct: number; kind: "buy" | "sell" } | null>(null);
  const [hintTrend, setHintTrend] = useState<"closer" | "farther" | null>(null);
  useEffect(() => {
    if (!priceHint) {
      prevHintRef.current = null;
      setHintTrend(null);
      return;
    }
    const prev = prevHintRef.current;
    if (prev && prev.kind === priceHint.kind) {
      const d = Math.abs(priceHint.pct) - Math.abs(prev.pct);
      setHintTrend(d < -0.001 ? "closer" : d > 0.001 ? "farther" : null);
    } else {
      setHintTrend(null);
    }
    prevHintRef.current = { pct: priceHint.pct, kind: priceHint.kind };
  }, [priceHint]);

  const hintDisplay = priceHint
    ? { text: `${priceHint.text}${hintTrend === "closer" ? " ▼가까워짐" : hintTrend === "farther" ? " ▲멀어짐" : ""}`, color: priceHint.color }
    : undefined;

  const backtest: BacktestResult | null = useMemo(() => {
    const lev = levCandles.slice(0, Math.max(0, levCandles.length - 1));
    if (lev.length === 0) return null;
    const current = backtestMetrics(runCombo(lev, mode, emaP, stP, stM));
    const sweep: SweepRow[] = [];
    const push = (m: StratMode, label: string, e: number, s: number, mlt: number, buy: boolean[], bull: boolean[]) => {
      sweep.push({ label, mode: m, ema: e, st: s, mult: mlt, m: backtestMetrics(computeStrategyTrades(lev, buy, bull)) });
    };
    // both: EMA × ST × 배수
    for (const e of SWEEP_EMA) {
      for (const s of SWEEP_ST) {
        for (const mlt of SWEEP_MULT) {
          const { buy, bull } = buildSignals("both", lev, e, s, mlt);
          push("both", `EMA${e}+ST${s}/${mlt}`, e, s, mlt, buy, bull);
        }
      }
    }
    // ema 단독: EMA만
    for (const e of SWEEP_EMA) {
      const { buy, bull } = buildSignals("ema", lev, e, 0, 0);
      push("ema", `EMA${e}만`, e, stP, stM, buy, bull);
    }
    // st 단독: ST × 배수
    for (const s of SWEEP_ST) {
      for (const mlt of SWEEP_MULT) {
        const { buy, bull } = buildSignals("st", lev, 0, s, mlt);
        push("st", `ST${s}/${mlt}만`, emaP, s, mlt, buy, bull);
      }
    }
    sweep.sort((x, y) => y.m.cum - x.m.cum);
    const rangeLabel = `${fmtDateTime(lev[0].t)} ~ ${fmtDateTime(lev[lev.length - 1].t)}`;
    return { rangeLabel, levBars: lev.length, current, sweep: sweep.slice(0, SWEEP_TOP) };
  }, [levCandles, mode, emaP, stP, stM]);

  // 명시적 값으로 DB 저장(setState 는 비동기라 현재 state 를 읽지 않고 인자로 받은 값을 저장).
  const saveSettings = async (settings: { mode: StratMode; ema_period: number; st_period: number; st_mult: number }) => {
    setSaveState("saving");
    setSaveError(null);
    try {
      const resp = await fetch("/api/leverage-scalp-settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ settings }),
      });
      const data = (await resp.json()) as { settings?: ScalpSettings; error?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? "저장에 실패했습니다.");
      setSaveState("saved");
      setTimeout(() => setSaveState("idle"), 2000);
    } catch (err) {
      setSaveState("error");
      setSaveError(err instanceof Error ? err.message : "저장에 실패했습니다.");
    }
  };

  const handleSaveSettings = () => saveSettings({ mode, ema_period: emaP, st_period: stP, st_mult: stM });

  // 스윕 행을 입력값에 반영(라이브 미리보기). applyAndSave 는 반영 + DB 저장까지.
  const applyRow = (row: SweepRow) => {
    setMode(row.mode);
    if (row.mode !== "st") setEmaPeriod(String(row.ema));
    if (row.mode !== "ema") {
      setStPeriod(String(row.st));
      setStMult(String(row.mult));
    }
  };
  const applyAndSaveRow = (row: SweepRow) => {
    applyRow(row);
    void saveSettings({ mode: row.mode, ema_period: row.ema, st_period: row.st, st_mult: row.mult });
  };

  return (
    <PageFrame title="레버리지 단타" fullHeight>
      <div className="leverageScalpLayout">
        <div className="scalpCharts">
          <ScalpChart
            name={FUT_NAME}
            code={FUT_CODE}
            feed="us-futures"
            interval={FUT_INTERVAL}
            note="참고용 · 이동평균"
            formatPrice={fmtPoints}
            overlays={futOverlays}
            legend={futLegend}
            markers={[]}
            onData={handleFutData}
            onChart={handleFutChart}
          />
          <ScalpChart
            name={LEV_NAME}
            code={LEV_CODE}
            feed="kr-stock"
            interval={LEV_INTERVAL}
            formatPrice={fmtKrw}
            overlays={levOverlays}
            priceLines={levPriceLines}
            markers={trades}
            hint={hintDisplay}
            onData={handleLevData}
            onChart={handleLevChart}
          />
        </div>

        <div className="card appCard">
          <div className="card-body">
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 12 }}>
              <h2 style={{ fontSize: "1.05rem", fontWeight: 800, margin: 0 }}>EMA 돌파 + 슈퍼트렌드</h2>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                {saveState === "saved" ? <span style={{ color: UP_COLOR, fontSize: "0.82rem", fontWeight: 700 }}>저장됨</span> : null}
                {saveState === "error" ? <span style={{ color: DOWN_COLOR, fontSize: "0.82rem", fontWeight: 700 }}>{saveError}</span> : null}
                <button type="button" className="btn btn-sm btn-primary" onClick={handleSaveSettings} disabled={saveState === "saving"}>
                  {saveState === "saving" ? "저장 중…" : "저장"}
                </button>
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <label style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                <span style={{ color: "var(--text-muted)", fontWeight: 700 }}>지표</span>
                <select className="form-select form-select-sm" style={{ width: 160 }} value={mode} onChange={(e) => setMode(e.target.value as StratMode)}>
                  {(Object.keys(MODE_LABELS) as StratMode[]).map((m) => (
                    <option key={m} value={m}>
                      {MODE_LABELS[m]}
                    </option>
                  ))}
                </select>
              </label>
              {mode !== "st" ? (
                <label style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                  <span style={{ color: "var(--text-muted)", fontWeight: 700 }}>EMA 기간</span>
                  <input type="number" min={1} step={1} className="form-control form-control-sm" style={{ width: 120 }} value={emaPeriod} onChange={(e) => setEmaPeriod(e.target.value)} />
                </label>
              ) : null}
              {mode !== "ema" ? (
                <>
                  <label style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                    <span style={{ color: "var(--text-muted)", fontWeight: 700 }}>슈퍼트렌드 기간</span>
                    <input type="number" min={1} step={1} className="form-control form-control-sm" style={{ width: 120 }} value={stPeriod} onChange={(e) => setStPeriod(e.target.value)} />
                  </label>
                  <label style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                    <span style={{ color: "var(--text-muted)", fontWeight: 700 }}>슈퍼트렌드 배수</span>
                    <input type="number" min={0.1} step={0.1} className="form-control form-control-sm" style={{ width: 120 }} value={stMult} onChange={(e) => setStMult(e.target.value)} />
                  </label>
                </>
              ) : null}
              <p style={{ margin: 0, color: "var(--text-muted)", fontSize: "0.82rem", lineHeight: 1.6 }}>
                <span style={{ color: EMA_COLOR, fontWeight: 800 }}>━</span> EMA · <span style={{ color: ST_COLOR, fontWeight: 800 }}>━</span> 슈퍼트렌드
                <br />
                {mode === "both"
                  ? "매수: 가격 > EMA 그리고 슈퍼트렌드 상승. 매도: 둘 중 하나라도 깨질 때."
                  : mode === "ema"
                    ? "매수: 가격이 EMA 상향 돌파. 매도: EMA 하향 이탈."
                    : "매수: 슈퍼트렌드 상승 전환. 매도: 슈퍼트렌드 하락 전환."}
              </p>
            </div>

            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, margin: "16px 0 6px" }}>
              <button
                type="button"
                onClick={() => setBtOpen((v) => !v)}
                style={{ display: "flex", alignItems: "center", gap: 6, background: "none", border: "none", padding: 0, cursor: "pointer" }}
              >
                <span style={{ color: "var(--text-muted)", fontWeight: 800 }}>{btOpen ? "▾" : "▸"}</span>
                <h2 style={{ fontSize: "1.05rem", fontWeight: 800, margin: 0 }}>백테스트 · 파라미터 스윕</h2>
              </button>
              {backtest && !btOpen ? (
                <span style={{ fontSize: "0.82rem", fontWeight: 700, color: backtest.current.cum >= 0 ? UP_COLOR : DOWN_COLOR }}>
                  현재 {backtest.current.cum >= 0 ? "+" : ""}
                  {backtest.current.cum.toFixed(2)}%
                </span>
              ) : null}
            </div>
            {btOpen ? (
              backtest ? (
                <div style={{ marginBottom: 8 }}>
                  <div style={{ color: "var(--text-muted)", fontSize: "0.78rem", marginBottom: 6 }}>
                    구간 {backtest.rangeLabel} · {backtest.levBars}봉 · 비용 미반영(총수익). 스윕: EMA+ST / EMA단독 / ST단독 × 파라미터, 누적 상위 {SWEEP_TOP}개.
                  </div>
                  <div style={{ fontSize: "0.85rem", fontWeight: 700, marginBottom: 8 }}>
                    현재 · {backtest.current.trades}건 · 승률 {backtest.current.winRate.toFixed(0)}% · 누적{" "}
                    <span style={{ color: backtest.current.cum >= 0 ? UP_COLOR : DOWN_COLOR }}>
                      {backtest.current.cum >= 0 ? "+" : ""}
                      {backtest.current.cum.toFixed(2)}%
                    </span>{" "}
                    · MDD {backtest.current.mdd.toFixed(2)}%
                  </div>
                  <div style={{ maxHeight: 200, overflowY: "auto" }}>
                    <table style={{ width: "100%", fontSize: "0.8rem", borderCollapse: "collapse" }}>
                      <thead>
                        <tr style={{ color: "var(--text-muted)", textAlign: "right" }}>
                          <th style={{ textAlign: "left", padding: "2px 4px" }}>조합</th>
                          <th style={{ padding: "2px 4px" }}>건수</th>
                          <th style={{ padding: "2px 4px" }}>승률</th>
                          <th style={{ padding: "2px 4px" }}>누적</th>
                          <th style={{ padding: "2px 4px" }}>MDD</th>
                          <th style={{ padding: "2px 4px" }}></th>
                        </tr>
                      </thead>
                      <tbody>
                        {backtest.sweep.map((row) => {
                          const paramsMatch =
                            row.mode === "ema"
                              ? row.ema === emaP
                              : row.mode === "st"
                                ? row.st === stP && row.mult === stM
                                : row.ema === emaP && row.st === stP && row.mult === stM;
                          const isCurrent = row.mode === mode && paramsMatch;
                          return (
                            <tr
                              key={row.label}
                              onClick={() => applyRow(row)}
                              style={{ cursor: "pointer", textAlign: "right", background: isCurrent ? "rgba(32,107,196,0.08)" : undefined, fontWeight: isCurrent ? 800 : 500 }}
                            >
                              <td style={{ textAlign: "left", padding: "2px 4px" }}>{row.label}</td>
                              <td style={{ padding: "2px 4px" }}>{row.m.trades}</td>
                              <td style={{ padding: "2px 4px" }}>{row.m.winRate.toFixed(0)}%</td>
                              <td style={{ padding: "2px 4px", color: row.m.cum >= 0 ? UP_COLOR : DOWN_COLOR, fontWeight: 700 }}>
                                {row.m.cum >= 0 ? "+" : ""}
                                {row.m.cum.toFixed(2)}%
                              </td>
                              <td style={{ padding: "2px 4px" }}>{row.m.mdd.toFixed(1)}%</td>
                              <td style={{ padding: "2px 4px", textAlign: "center" }}>
                                <button
                                  type="button"
                                  className="btn btn-sm btn-outline-primary"
                                  style={{ padding: "0 8px", fontSize: "0.75rem", lineHeight: 1.6 }}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    applyAndSaveRow(row);
                                  }}
                                >
                                  적용
                                </button>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <div style={{ color: "var(--text-muted)", fontSize: "0.75rem", marginTop: 4 }}>행 클릭 = 미리보기(라이브 반영), <b>적용</b> = 반영 + DB 저장.</div>
                </div>
              ) : (
                <div style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginBottom: 8 }}>데이터 로딩 중…</div>
              )
            ) : null}

            <h2 style={{ fontSize: "1.05rem", fontWeight: 800, margin: "16px 0 6px" }}>
              기록{" "}
              <span style={{ fontSize: "0.85rem", fontWeight: 700, color: totalPct > 0 ? UP_COLOR : totalPct < 0 ? DOWN_COLOR : "#475569" }}>
                (실현 {realizedPct > 0 ? "+" : ""}
                {realizedPct.toFixed(2)}%
                {unrealizedPct != null ? (
                  <>
                    {" · 평가 "}
                    {unrealizedPct > 0 ? "+" : ""}
                    {unrealizedPct.toFixed(2)}% · 보유 중
                  </>
                ) : null}
                )
              </span>
            </h2>
            <div style={{ display: "flex", gap: 16, marginBottom: 8, fontSize: "0.85rem", fontWeight: 700 }}>
              <span style={{ color: "var(--text-muted)" }}>
                승률 <span style={{ color: "#475569" }}>{winRate.toFixed(1)}%</span>
                <span style={{ color: "var(--text-muted)", fontWeight: 500 }}> ({winCount}/{effPnls.length})</span>
              </span>
              <span style={{ color: "var(--text-muted)" }}>
                평균 수익률{" "}
                <span style={{ color: avgPct > 0 ? UP_COLOR : avgPct < 0 ? DOWN_COLOR : "#475569" }}>
                  {avgPct > 0 ? "+" : ""}
                  {avgPct.toFixed(2)}%
                </span>
              </span>
            </div>
            <div className="recordScroll">
              {trades.length === 0 ? (
                <div style={{ color: "var(--text-muted)", fontSize: "0.88rem" }}>아직 신호가 없습니다.</div>
              ) : (
                <ul style={{ margin: 0, padding: 0, listStyle: "none", fontSize: "0.88rem", lineHeight: 1.7 }}>
                  {[...trades].reverse().map((t, idx) => {
                    const win = t.type === "sell" && t.pnlPct != null && t.pnlPct > 0;
                    const resultColor = t.pnlPct != null && t.pnlPct >= 0 ? UP_COLOR : DOWN_COLOR;
                    return (
                      <li key={`${t.time}-${t.type}-${idx}`} style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
                        <span style={{ color: "var(--text-muted)", minWidth: 150 }}>{fmtDateTime(t.time)}</span>
                        <span style={{ fontWeight: 800, color: t.type === "buy" ? UP_COLOR : DOWN_COLOR, minWidth: 34 }}>
                          {t.type === "buy" ? "매수" : "매도"}
                        </span>
                        <span style={{ fontWeight: 700 }}>{fmtKrw(t.price)}</span>
                        {t.type === "sell" && t.pnlPct != null ? (
                          <span style={{ fontWeight: 800, color: resultColor }}>
                            {win ? "성공!" : "실패"} ({t.pnlPct > 0 ? "+" : ""}
                            {t.pnlPct.toFixed(2)}%)
                          </span>
                        ) : null}
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          </div>
        </div>
      </div>
      <style jsx>{`
        .leverageScalpLayout {
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
          grid-template-rows: minmax(0, 1fr);
          gap: 12px;
          flex: 1 1 auto;
          min-height: 0;
          align-items: stretch;
        }
        .scalpCharts {
          display: flex;
          flex-direction: column;
          gap: 12px;
          min-height: 0;
        }
        .scalpCharts > :global(.scalpChartCard) {
          display: flex;
          flex-direction: column;
          flex: 1 1 0;
          min-height: 0;
        }
        .leverageScalpLayout > :global(.card),
        .scalpCharts > :global(.card) {
          display: flex;
          flex-direction: column;
          min-height: 0;
        }
        .leverageScalpLayout :global(.card-body) {
          display: flex;
          flex-direction: column;
          flex: 1 1 auto;
          min-height: 0;
        }
        .recordScroll {
          flex: 1 1 auto;
          min-height: 0;
          overflow-y: auto;
        }
        @media (max-width: 900px) {
          .leverageScalpLayout {
            grid-template-columns: minmax(0, 1fr);
            grid-template-rows: minmax(0, 1fr) minmax(0, 1fr);
          }
        }
      `}</style>
    </PageFrame>
  );
}
