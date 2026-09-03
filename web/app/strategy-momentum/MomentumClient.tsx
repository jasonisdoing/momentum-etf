"use client";

import type { ColDef } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { IconCheck } from "@tabler/icons-react";

import { AppAgGrid } from "../components/AppAgGrid";
import { excessColumn } from "../components/BacktestPeriodTables";
import { MonthsSelect } from "../components/MonthsSelect";
import { AppLoadingProgress, startProgressRamp, type LoadingProgress } from "../components/AppLoadingProgress";
import { useRealtimeQuotes } from "../components/useRealtimeQuotes";
import { StrategyNotes } from "../components/StrategyNotes";
import { StrategyTuning } from "../components/StrategyTuning";
import { BacktestSummary } from "../components/BacktestSummary";
import { BacktestTradeStats } from "../components/BacktestTradeStats";
import { type HoldingChartData } from "../components/HoldingChart";
import { StrategyHoldingCharts } from "../components/StrategyHoldingCharts";
import { NavTabs } from "../components/NavTabs";
import { PageFrame } from "../components/PageFrame";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";
import { formatDateWithWeekday } from "@/lib/datetime";
import { readRememberedTickerType, writeRememberedTickerType } from "../components/account-selection";
import { formatPoolLabel, type PoolLabelSource } from "@/lib/pool-label";
import { formatMarketCapWon } from "@/lib/market-cap-format";
import {
  type BacktestDayRow,
  type BacktestMonthRow,
  type BacktestYearRow,
  toCalendarMonthRows,
  toYearRows,
} from "@/lib/backtest-periods";
import {
  industryColumn,
  STOCK_NAME_COLUMN_MIN_WIDTH,
  adrColumn as sharedAdrColumn,
  formatSignedPct,
  maExitGapColumn,
  renderHighDrawdownCell,
  signColor,
  marketCapRankColumn,
  stockMemoColumn,
  tradeValueMultColumn,
} from "@/lib/grid-cells";
import { isTrendBroken, renderStockNameCell } from "@/lib/name-highlight";
import { updateStockMemo } from "@/lib/stocks-store";
import { poolHasIndustry, poolHasMarketCap } from "@/lib/pool-industry";
import { MaDaysSelect } from "../components/MaDaysSelect";
import { UnsavedChangesBadge } from "../components/UnsavedChangesBadge";
import { formatPrice } from "../../lib/price-format";

const gridTheme = createAppGridTheme();

/** 접이식 전략 설명 — 운용 현황·백테스트 섹션 상단(기본 접힘). */
const CURRENT_NOTES = [
  {
    title: "선정",
    body:
      "주 마지막 거래일 종가 기준, 보유 자격(장기 이격 > 0, 단기 이격 ≥ 0)을 갖춘 종목을 " +
      "장기 이평선 이격(%) 순으로 줄 세워 상위 종목을 고릅니다. " +
      "종목 수는 종목풀 설정값을 사용합니다.",
  },
  {
    title: "체결",
    body:
      "다음 주 첫 거래일 시가에 교체합니다. 유지 종목도 슬리브 자산의 1/N로 되돌립니다(동일가중) — " +
      "오른 종목은 초과분을 팔고 내린 종목은 미달분을 삽니다.",
  },
  {
    title: "ADR 게이트",
    body:
      "ADR 하한을 설정하면, 판정일의 시장 ADR(20일 상승/하락 종목수 비율 — 시장은 종목풀 설정의 시장 레짐 지수)이 " +
      "하한 미만일 때 다음 주 첫 거래일 시가에 전량 매도하고 그 주는 신규 진입도 하지 않습니다. " +
      "주중에도 매일 보아, 하한 아래로 내려간 날 남은 보유를 다음 거래일 시가에 전부 팝니다. " +
      "ADR이 하한 위로 회복하면 다음 교체부터 정상 선정을 재개합니다. 없음이면 게이트를 쓰지 않습니다.",
  },
  {
    title: "주중 이탈 (상시)",
    body:
      "보유 자격(장기 이격 > 0, 단기 이격 ≥ 0)을 잃으면 다음 거래일 시가에 전량 매도합니다. " +
      "ADR 하한을 함께 쓰면 주중에 시장 ADR이 하한 아래로 내려간 날에도 남은 보유를 전부 팝니다 " +
      "— 주간 게이트가 다음 교체일에 할 일을 그날로 앞당깁니다. " +
      "판 슬롯은 다음 교체까지 현금입니다(전략 고정 — 끄는 설정은 없습니다).",
  },
  {
    title: "표시",
    body: "판정 결과는 5분 캐시이고, 현재가·일간(%)만 60초마다 갱신됩니다.",
  },
];


// 풀별로 따로 저장되는 설정 — 풀 셀렉트를 바꾸면 그 풀의 저장분으로 폼이 전환된다.
// 슬리피지는 종목풀 설정을, 백테스트 기간은 실행할 때 고른 값을 쓴다 — 여기 없다.
type PoolSettings = {
  /** 종목풀 설정의 보유종목 수 — 이 화면에서는 표시·계산에만 쓴다. */
  top_n: number;
  short_ma_days: number;
  long_ma_days: number;
  /** ADR 하한 — 판정일의 시장 ADR(풀의 시장 레짐 지수 시장)이 미만이면 그 주 전량 현금. null = 없음. */
  adr_floor?: number | null;
};

type Settings = PoolSettings & { pool: string };

type PickRow = {
  rank: number | null;
  // 다음 주 예상 순위 — 현재 가격 기준으로 같은 선정 규칙을 돌린 순위 (자격 미달은 null).
  expected_rank: number | null;
  is_reserve: boolean;
  // 현재 표(선정+후보) 밖인데 다음 주 편입이 예상되는 종목 — 하단 별도 행.
  is_expected_only: boolean;
  /** 지금 실제로 들고 있는가 — 교체가 확정됐어도 체결 전이면 거짓. */
  is_held?: boolean;
  // 주중 매도 — 보유 자격(장기>0 & 단기≥0) 상실로 매도됨(체결 완료) / 다음 시가 매도 예정.
  is_exited?: boolean;
  is_exit_pending?: boolean;
  exit_date?: string | null;
  /** 주중 매도 사유 — "주중 이탈"(자격 상실) · "ADR 게이트"(시장 하한 미달). */
  exit_reason?: string | null;
  streak_weeks: number | null;
  /** 연속 편입이 시작된 교체일 — 차트 탭의 Buy 마커 위치. 선정분만 값이 있다. */
  entry_date?: string | null;
  /** 편입 후 수익률(%) — 연속 편입 시작 교체일 시가 대비. 보유 중인 종목만 값이 있다. */
  entry_return_pct?: number | null;
  next_week_expected: boolean;
  /** 주중 이탈 **예상** — 현재(장중) 가격 기준, 오늘 종가로 확정 전. */
  is_exit_forecast?: boolean;
  exit_forecast_reason?: string | null;
  ticker: string;
  name: string;
  // 종목의 소속 마켓(KOSPI/KOSDAQ) — 한국 통합 풀 구분 표시용, 없으면 빈 값.
  market: string;
  industry: string;
  currency: string;
  price: number | null;
  monthly_returns: Record<string, number | null>;
  daily_change_pct: number | null;
  high_drawdown_pct: number | null;
  market_cap: number | null;
  market_cap_rank: number | null;
  /** 20일 평균 대비 거래대금 배수 — 순위·신고가 화면과 같은 소스(배치 확정 + 실시간). */
  value_mult?: number | null;
  value_mult_live?: number | null;
  /** 종목 메모 — 계좌가 아니라 종목에 붙는다(자산 관리·순위 화면과 같은 값). */
  memo?: string;
  signal_short_pct: number | null;
  signal_long_pct: number | null;
  current_short_pct: number | null;
  current_long_pct: number | null;
};

type PicksResult = {
  as_of: string;
  portfolio_week: string;
  rebalance_date: string;
  signal_date: string;
  universe_count: number;
  candidate_count: number;
  // 풀의 국가·통화 — 마켓·시가총액 컬럼 표시와 티커 표기(ASX: 등)를 정한다.
  country: string;
  currency: string;
  monthly_return_labels: string[];
  /**
   * ADR 게이트 상태 — 하한 미설정이면 null. blocked 면 이번 주는 전량 현금.
   * `intraweek` 는 마지막 확정 종가 기준의 주중 게이트 — 주간 판정을 통과한 뒤 ADR 이
   * 무너지면 여기가 blocked 가 되고 다음 거래일 시가에 전량 매도한다.
   */
  adr_gate?: {
    market: string | null;
    floor: number;
    value: number | null;
    blocked: boolean;
    intraweek: { date: string; value: number | null; blocked: boolean };
  } | null;
  rows: PickRow[];
};

// 월간 행은 집계만 담는다 — 매매 내역(편입·편출·교체율·보유 수)은 주간 행이 담당한다.
// 타입·집계는 자산 헬퍼와 공용(@/lib/backtest-periods).

// 주간 행 — 달력 주 단위. 기준일은 그 주 마지막 거래일, 편입·편출은 그 주 체결분.
type BacktestWeekRow = {
  week_end: string;
  /** 판정일(기준일)의 시장 ADR — 다음 주 게이트를 결정한 값. */
  adr?: number | null;
  strategy_pct: number | null;
  benchmark_pct: number | null;
  holdings_count: number;
  holdings_start?: number;
  /** 주중 이탈 매도 — "종목명(코드) · 사유" 형식. */
  exited?: string[];
  turnover_pct: number | null;
  added: string[];
  removed: string[];
  // 다음 교체일에 실행될 예정 행 — 아직 수익률 없음.
  is_pending?: boolean;
};

// 체결 행 — 편입~편출 한 쌍. exit_date 가 없으면 아직 보유 중이다.
type BacktestTradeRow = {
  ticker: string;
  name: string;
  entry_date: string;
  entry_price: number;
  exit_date: string | null;
  exit_price: number | null;
  return_pct: number | null;
  days: number;
  reason: string;
};

type BacktestResult = {
  start_date: string;
  end_date: string;
  months: number;
  strategy_total_pct: number;
  benchmark_total_pct: number;
  strategy_mdd_pct: number | null;
  benchmark_mdd_pct: number | null;
  strategy_sortino: number | null;
  benchmark_sortino: number | null;
  strategy_cagr_pct: number | null;
  benchmark_cagr_pct: number | null;
  benchmark_name: string;
  benchmark_ticker: string;
  weekly: BacktestWeekRow[];
  daily: BacktestDayRow[];
  trades: BacktestTradeRow[];
  /** 체결 통계 — 백엔드 공용 계산(utils/trade_stats.py). 청산분만 센다. */
  trade_count: number;
  win_rate_pct: number | null;
  avg_win_pct: number | null;
  avg_loss_pct: number | null;
  reason_counts?: Record<string, number>;
};

// 풀 옵션 — 공용 라벨 소스에 국가·통화·풀 성격(stock/etf)이 붙는다.
type PoolOption = PoolLabelSource & {
  country_code?: string;
  currency?: string;
  pool_kind?: string | null;
};

type View = {
  /** 선택지 밖 저장값을 첫 선택지로 보정한 내역 — 있으면 화면이 저장을 요구한다. */
  coerced?: string[];
  settings: Settings;
  // 풀별 저장 설정 맵 — 셀렉트 전환 시 즉시 그 풀의 값으로 폼을 채운다.
  settings_by_pool?: Record<string, PoolSettings>;
  pool_options?: PoolOption[];
  // 이평선 — 종목풀 설정(`pool_settings`)에 저장된다. 순위 화면·보유종목 알림과 같은 값이다.
  ma_rule?: {
    short_ma_days: number;
    long_ma_days: number;
    short_ma_options: number[];
    long_ma_options: number[];
  };
  // 기간 선택지는 서버가 가격 캐시 범위로 계산해 내려준다 (종목풀 백테스트와 동일).
  month_options?: number[];
  /** 튜닝 전용 기간 — ADR 축이 항상 있으므로 ADR 이력이 덮는 범위만 온다. */
  tuning_month_options?: number[];
  // 셀렉트 선택지 — 백엔드 상수가 단일 소스(프론트에 복사본을 두지 않는다).
  constraints?: {
    adr_floor_options?: (number | null)[];
  };
  picks: PicksResult | null;
};

// 운용 현황 안쪽 탭 — 신고가 화면과 같은 구성. 차트는 선정 종목 수만큼 그리므로 열 때만 그린다.
const CURRENT_TABS = [
  { key: "list", label: "종목" },
  { key: "chart", label: "차트" },
] as const;
type CurrentTab = (typeof CURRENT_TABS)[number]["key"];

// 백테스트 표 보기 단위 — /compare 의 연간·월간·일간 구분에 주간을 더한 것.
const VIEW_MODES = [
  { key: "yearly", label: "연간" },
  { key: "monthly", label: "월간" },
  { key: "weekly", label: "주간" },
  { key: "daily", label: "일간" },
  { key: "trades", label: "체결" },
] as const;
type ViewMode = (typeof VIEW_MODES)[number]["key"];

const hintStyle: React.CSSProperties = { color: "var(--text-muted)", fontSize: "var(--fs-sm)" };
const numberInputStyle: React.CSSProperties = { width: 88, textAlign: "right" };

function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return value.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

// 부호 %·색 헬퍼는 공용 모듈(@/lib/grid-cells)을 쓴다.
const formatSigned = formatSignedPct;

export function MomentumClient() {
  const toast = useToast();
  /** 종목 메모 저장 — 계좌가 아니라 종목에 붙는다(순위·자산 관리 화면과 같은 값·같은 API). */
  const saveMemo = useCallback(
    async (ticker: string, memo: string) => {
      try {
        await updateStockMemo(ticker, memo);
        toast.success("메모 저장 완료");
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "메모 저장에 실패했습니다.");
      }
    },
    [toast],
  );
  const [view, setView] = useState<View | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [picking, setPicking] = useState(false);
  const [backtesting, setBacktesting] = useState(false);
  const [backtest, setBacktest] = useState<BacktestResult | null>(null);
  const [pickProgress, setPickProgress] = useState<LoadingProgress | null>(null);
  const [pickFailed, setPickFailed] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [backtestProgress, setBacktestProgress] = useState<LoadingProgress | null>(null);
  // 기본 보기는 주간 — 매매 내역(편입·편출)이 실제 체결 주에 붙는 표라 주간 점검의 기준이다.
  const [viewMode, setViewMode] = useState<ViewMode>("weekly");
  const autoPickedRef = useRef(false);

  // 설정 입력 초안 (문자열로 보관해 입력 중 상태를 그대로 둔다)
  const [draft, setDraft] = useState<Record<string, string>>({});
  // 초안 초기값은 자리만 잡는다 — 설정을 받은 뒤 applyView 가 항상 덮어쓰며,
  // 설정을 못 받으면 폼 자체를 그리지 않으므로 이 값이 화면에 보이는 경우는 없다.
  const [draftPool, setDraftPool] = useState<string>("");
  // 이평선 초안 — 종목풀 설정에 풀별로 저장된다(순위 화면·보유종목 알림과 같은 값).
  const [draftMaRule, setDraftMaRule] = useState<{ short: number; long: number } | null>(null);
  // ADR 하한 — "" 은 없음(기본). 시장은 풀 설정의 시장 레짐 지수를 따른다.
  const [draftAdrFloor, setDraftAdrFloor] = useState<string>("");

  // 풀별 설정을 폼 초안에 채운다 — 풀 셀렉트 전환과 응답 반영이 같은 경로를 쓴다.
  const fillDrafts = useCallback((values: PoolSettings) => {
    setDraft({});
    setDraftMaRule({ short: values.short_ma_days, long: values.long_ma_days });
    setDraftAdrFloor(values.adr_floor == null ? "" : String(values.adr_floor));
  }, []);

  const applyView = useCallback(
    (data: View) => {
      setView(data);
      setDraftPool(data.settings.pool);
      fillDrafts(data.settings);
    },
    [fillDrafts],
  );

  const isNewPoolDraft = view != null && draftPool !== "" && view.settings_by_pool?.[draftPool] == null;

  const load = useCallback(async (): Promise<string | null> => {
    setLoading(true);
    try {
      // 마지막으로 고른 풀은 브라우저에 기억한다(다른 화면들과 같은 공용 키).
      const remembered = readRememberedTickerType();
      const query = remembered ? `?pool=${encodeURIComponent(remembered)}` : "";
      const resp = await fetch(`/api/strategy-momentum${query}`, { cache: "no-store" });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "설정을 불러오지 못했습니다.");
      setLoadError(null);
      const data = payload as View;
      applyView(data);
      if (data.coerced?.length) {
        toast.warning(`저장값이 선택지에 없어 보정했습니다: ${data.coerced.join(", ")} — 확인 후 저장하세요.`);
      }
      // 기억된 풀에 저장분이 없으면(첫 설정) 그 풀을 초안으로 띄운다 — 폼 값은 저장된
      // 첫 풀의 설정을 시작점으로 쓰고, 선정은 저장 전이라 실행하지 않는다(null 반환).
      const requested = (payload as { requested_pool?: string }).requested_pool;
      if (requested && data.settings_by_pool?.[requested] == null) {
        setDraftPool(requested);
        return null;
      }
      return data.settings.pool;
    } catch (error) {
      // 설정을 못 받으면 값을 지어내지 않는다 — 폼을 그리지 않고 실패만 알린다.
      // (기본값을 그렸다가 그대로 저장되면 저장돼 있던 설정이 덮어써진다.)
      const message = error instanceof Error ? error.message : "설정을 불러오지 못했습니다.";
      setLoadError(message);
      toast.error(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, [applyView, toast]);

  const runPicks = useCallback(async (pool: string) => {
    setPicking(true);
    setPickFailed(false);
    setPickProgress({ percent: 10, message: "월 확정 포트폴리오 계산 중" });
    const stopRamp = startProgressRamp(setPickProgress);
    try {
      // 지금 화면이 고른 풀로 계산한다 — 안 넘기면 서버가 기본 풀로 돌린다.
      const resp = await fetch(`/api/strategy-momentum/picks?pool=${encodeURIComponent(pool)}`, {
        method: "POST",
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "선정에 실패했습니다.");
      setPickProgress({ percent: 100, message: "선정 결과 반영 중" });
      setView((prev) => (prev ? { ...prev, picks: payload as PicksResult } : prev));
    } catch (error) {
      setPickFailed(true);
      toast.error(error instanceof Error ? error.message : "선정에 실패했습니다.");
    } finally {
      stopRamp();
      setPicking(false);
      setPickProgress(null);
    }
  }, [toast]);

  // 진입 시 저장된 설정으로 선정을 한 번 자동 실행한다 (가격 캐시 기반이라 수 초).
  useEffect(() => {
    void (async () => {
      const appliedPool = await load();
      if (appliedPool && !autoPickedRef.current) {
        autoPickedRef.current = true;
        await runPicks(appliedPool);
      }
    })();
  }, [load, runPicks]);

  // 저장 공용 경로 — 폼 저장 버튼과 풀 전환(저장된 풀의 설정 자동 적용)이 함께 쓴다.
  const persistSettings = useCallback(
    async (settings: Settings, successMessage: string) => {
      setSaving(true);
      try {
        const resp = await fetch("/api/strategy-momentum", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ settings }),
        });
        const payload = await resp.json();
        if (!resp.ok) throw new Error(payload?.error ?? "설정을 저장하지 못했습니다.");
        const saved = payload as View;
        // 이 폼의 설정(종목풀·이평선·ADR 하한)은 전부 선정·매도 예정에
        // 영향을 준다 — 저장하면 무조건 다시 계산한다. (예전의 "선정 무관 설정" 예외 목록은
        // 슬리피지가 이 폼에 있던 시절의 유물이라 제거했다.)
        applyView({ ...saved, picks: null });
        // 백테스트는 어느 설정이 바뀌든 결과가 달라지므로 비운다.
        setBacktest(null);
        toast.success(`${successMessage} 선정을 다시 계산합니다.`);
        await runPicks(saved.settings.pool);
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "설정을 저장하지 못했습니다.");
      } finally {
        setSaving(false);
      }
    },
    [applyView, runPicks, toast, view],
  );

  // 업종 UI 노출 — 종목풀 설정의 풀 성격(pool_kind) 토글이 1순위:
  // 개별주(stock)면 표시, ETF 면 숨김. 미설정(구 문서)이면 선정 행에 값이 있는지로
  // 추정한다(pools-rank 와 같은 패턴). 업종을 모르는 종목엔 상한이 원래 미적용이라
  // 숨겨도 동작은 그대로다.
  const selectedPoolOption = view?.pool_options?.find((option) => option.ticker_type === view?.settings.pool);
  const poolKind = selectedPoolOption?.pool_kind ?? "";
  // 업종 컬럼·업종 상한 노출 — 판정은 전 화면 공용(`@/lib/pool-industry`).
  const hasIndustryData = poolHasIndustry(selectedPoolOption);
  const hasMarketCap = poolHasMarketCap(selectedPoolOption);
  const saveSettings = useCallback(async () => {
    if (draftMaRule == null) {
      toast.error("설정 값이 올바르지 않습니다.");
      return;
    }
    await persistSettings(
      {
        pool: draftPool,
        // 종목 수(공통 config)·업종 상한(폐기)은 더 이상 보내지 않는다.
        top_n: view?.settings.top_n ?? 0,
        // 이평선은 전략 전용 값 — 설정의 일부로 풀별 저장한다(종목풀 설정과 무관).
        short_ma_days: draftMaRule.short,
        long_ma_days: draftMaRule.long,
        adr_floor: draftAdrFloor === "" ? null : Number(draftAdrFloor),
      },
      "설정을 저장했습니다.",
    );
  }, [draftAdrFloor, draftMaRule, draftPool, persistSettings, toast, view?.settings.top_n]);

  // 풀 셀렉트 변경 — 그 풀의 저장 설정이 있으면 **즉시 전환·저장·재선정**한다
  // (전환은 초안이 아니라 컨텍스트 스위치다). 저장분이 없는 풀(첫 설정)만 초안으로
  // 남기고, 현재 폼 값을 시작점으로 보여준 뒤 저장을 요구한다.
  const handlePoolChange = useCallback(
    (pool: string) => {
      setDraftPool(pool);
      writeRememberedTickerType(pool);
      // 풀이 바뀌면 이전 풀의 백테스트 결과는 의미가 없다 — 저장 이력이 없어 아래 분기를 타지
      // 않는 풀(그 풀 첫 진입)도 마찬가지라 분기 밖에서 비운다.
      // 튜닝 결과는 StrategyTuning 이 key={draftPool} 로 재마운트되며 함께 비워진다.
      setBacktest(null);
      const saved = view?.settings_by_pool?.[pool];
      if (saved) {
        fillDrafts(saved);
        void persistSettings({ pool, ...saved }, "풀을 전환했습니다.");
      }
    },
    [fillDrafts, persistSettings, view],
  );

  const [backtestMonths, setBacktestMonths] = useState<number>(12);

  const runBacktest = useCallback(async () => {
    const months = backtestMonths;
    setBacktesting(true);
    setBacktestProgress({ percent: 10, message: "월별 리밸런싱 시뮬레이션 중" });
    const stopRamp = startProgressRamp(setBacktestProgress);
    try {
      const resp = await fetch(`/api/strategy-momentum/backtest?pool=${encodeURIComponent(draftPool)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // 한 번 실행으로 연간·월간·주간·일간을 모두 만든다 — 탭 전환 시 재실행이 없도록.
        body: JSON.stringify({ months, include_daily: true }),
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "백테스트에 실패했습니다.");
      setBacktestProgress({ percent: 100, message: "결과 반영 중" });
      setBacktest(payload as BacktestResult);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "백테스트에 실패했습니다.");
    } finally {
      stopRamp();
      setBacktesting(false);
      setBacktestProgress(null);
    }
  }, [backtestMonths, draftPool, toast]);

  // 표시용 현재가·일간(%)만 60초마다 갱신한다 — 선정 판정은 종가 기준이라 5분 캐시로 둔다.
  const quotes = useRealtimeQuotes(view?.picks?.country ?? "", (view?.picks?.rows ?? []).map((row) => row.ticker));
  const pickRows = useMemo(() => {
    const rows = view?.picks?.rows ?? [];
    if (Object.keys(quotes).length === 0) return rows;
    return rows.map((row) => {
      const quote = quotes[row.ticker];
      return quote ? { ...row, price: quote.price, daily_change_pct: quote.change_pct } : row;
    });
  }, [view?.picks?.rows, quotes]);

  // ── 차트 탭 (신고가 화면과 같은 구성 — 공용 HoldingChart) ──
  const [currentTab, setCurrentTab] = useState<CurrentTab>("list");
  const [charts, setCharts] = useState<HoldingChartData[] | null>(null);
  const [chartsLoading, setChartsLoading] = useState(false);
  const [chartsError, setChartsError] = useState<string | null>(null);
  // 차트 기간(개월) — 백엔드 config.HOLDING_CHART_MONTHS 가 단일 소스. 응답에서 받아 문구에 쓴다.
  const [chartMonths, setChartMonths] = useState<number | null>(null);
  // 차트를 그릴 대상 — 선정분(1~N)만. 차순위·예상 전용·이미 매도된 종목은 뺀다.
  const chartRows = useMemo(
    () => pickRows.filter((row) => !row.is_reserve && !row.is_expected_only && !row.is_exited),
    [pickRows],
  );
  // 풀·구성이 바뀌면 이전 차트는 버린다.
  // 사용자가 고른 풀(draftPool)을 앞에 둔다 — `view` 는 저장 응답이 와야 바뀌므로,
  // 그것만 보면 풀을 바꾼 뒤 응답이 오기까지 이전 풀의 차트가 그대로 남는다.
  const chartKey = useMemo(
    () => `${draftPool}|${view?.settings.pool ?? ""}|${chartRows.map((row) => row.ticker).join(",")}`,
    [draftPool, view?.settings.pool, chartRows],
  );
  useEffect(() => {
    setCharts(null);
    setChartsError(null);
  }, [chartKey]);
  // 차트 탭을 열 때만 받는다 — 선정 종목 수만큼 일봉을 실어 오므로 목록 탭에서는 낭비다.
  useEffect(() => {
    if (currentTab !== "chart" || !view?.picks || charts || chartsLoading || chartsError) return;
    // 풀을 막 바꾼 직후에는 `view` 가 아직 이전 풀 것이다 — 그 목록으로 차트를 받으면
    // 다른 풀의 종목이 뜬다. 저장 응답이 와서 두 값이 맞을 때까지 기다린다.
    if (saving || picking || view.settings.pool !== draftPool) return;
    if (chartRows.length === 0) {
      setCharts([]);
      return;
    }
    setChartsLoading(true);
    void (async () => {
      try {
        const response = await fetch("/api/strategy-momentum/charts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pool: view.settings.pool, tickers: chartRows.map((row) => row.ticker) }),
        });
        const payload = (await response.json()) as { charts?: HoldingChartData[]; months?: number; error?: string };
        if (!response.ok) throw new Error(payload.error ?? "차트를 불러오지 못했습니다.");
        setCharts(payload.charts ?? []);
        setChartMonths(payload.months ?? null);
      } catch (chartError) {
        const message = chartError instanceof Error ? chartError.message : "차트를 불러오지 못했습니다.";
        setChartsError(message);
        toast.error(message);
      } finally {
        setChartsLoading(false);
      }
    })();
  }, [currentTab, view, charts, chartsLoading, chartsError, chartRows, draftPool, saving, picking, toast]);

  // 저장하지 않은 입력이 있으면 실행 결과가 화면 값과 어긋난다 — 저장을 먼저 요구한다.
  const isDirty = useMemo(() => {
    if (!view) return false;
    const saved = view.settings;
    // 서버가 선택지 밖 저장값을 보정해 보낸 경우도 '저장 필요'로 본다 (저장하면 coerced 가 비워진다).
    if (view.coerced?.length) return true;
    return (
      draftPool !== saved.pool ||
      (draftAdrFloor === "" ? null : Number(draftAdrFloor)) !== (saved.adr_floor ?? null) ||
      (draftMaRule != null &&
        view.ma_rule != null &&
        (draftMaRule.short !== view.ma_rule.short_ma_days ||
          draftMaRule.long !== view.ma_rule.long_ma_days))
    );
  }, [draftAdrFloor, draftMaRule, draftPool, view]);

  const monthlyLabels = view?.picks?.monthly_return_labels ?? [];
  // 선정 결과 풀의 국가 — 마켓·시가총액 컬럼 표시와 티커 표기(ASX:)를 정한다.
  const picksCountry = view?.picks?.country ?? "";
  const pickColumns = useMemo<ColDef<PickRow>[]>(() => {
    return [
      {
        headerName: "순위",
        field: "rank",
        headerTooltip:
          "선정 1~N 은 판정일 종가 이격 순(확정된 편입 순서). 그 아래(차순위·예상)는 현재 이격 순 — " +
          "다음 교체에 무엇이 올라오는지 보기 위함. 매도예정 = 자격 상실로 다음 시가 매도, 매도 = 주중에 이미 매도됨",
        width: 68,
        type: "numericColumn",
        cellDataType: "text",
        cellRenderer: (p: { value?: number | null; data?: PickRow }) => {
          if (p.data?.is_exited) return <span style={{ color: "var(--text-muted)" }}>매도</span>;
          if (p.data?.is_exit_pending) return <span style={{ color: "#d62828", fontWeight: 700 }}>매도예정</span>;
          return <span>{p.value == null ? "-" : String(p.value)}</span>;
        },
      },
      {
        headerName: "연속",
        field: "streak_weeks",
        headerTooltip:
          "이번 포트폴리오까지 몇 주 연속 편입됐는지 (신규 = 이번 주 첫 편입, 최대 12주 추적). " +
          "화살표 — →유지/→신규(초록)는 다음 주 예상, 확정은 교체일 직전 판정일 종가. " +
          "빨강은 빠지는 종목이다: →교체예정은 다음 주 순위 교체, →매도예정은 주중 이탈 확정 후 다음 시가 매도, " +
          "→이탈(굵게)은 주중에 이미 매도된 것. →이탈(예상)은 장중 가격 기준 예보로 " +
          "오늘 종가로 확정되면 다음 거래일 시가에 판다.",
        width: 108,
        cellDataType: "text",
        cellRenderer: (p: { value?: number | null; data?: PickRow }) => {
          const streak =
            p.value == null ? "-" : p.value <= 1 ? "신규" : p.value >= 12 ? "12+" : `${p.value}주`;
          const isNewPick = p.value != null && p.value <= 1 && !p.data?.is_reserve;
          const held = Boolean(p.data && !p.data.is_reserve && !p.data.is_expected_only);
          // 빠지는 종목은 빨강으로 쓰고 원인을 문구로 가른다 — 순위 경쟁이면 교체예정,
          // 주중 이탈이 확정됐지만 미체결이면 매도예정, 이미 매도됐으면 사유를 굵게 표시한다.
          let next: React.ReactNode = null;
          if (p.data?.is_exited) {
            next = <span style={{ color: "var(--up-color, #d64545)", fontWeight: 700 }}> →이탈</span>;
          } else if (p.data?.is_exit_pending) {
            next = <span style={{ color: "var(--up-color, #d64545)" }}> →매도예정</span>;
          } else if (p.data?.is_exit_forecast) {
            // 주중 이탈 **예보** — 다음주 순위 예상보다 우선한다(오늘 종가 확정 시 내일 시가 매도).
            next = (
              <span style={{ color: "var(--up-color, #d64545)", opacity: 0.75 }} title="오늘 종가로 확정 시 다음 거래일 시가 매도">
                {" "}→이탈(예상)
              </span>
            );
          } else if (p.data?.next_week_expected) {
            next = <span style={{ color: "#2f9e44", fontWeight: 700 }}> →{held ? "유지" : "신규"}</span>;
          } else if (held) {
            next = <span style={{ color: "var(--up-color, #d64545)" }}> →교체예정</span>;
          }
          return (
            <span>
              <span style={{ color: isNewPick ? "var(--up-color, #d64545)" : "inherit" }}>{streak}</span>
              {next}
            </span>
          );
        },
      },
      {
        headerName: "수익률",
        field: "entry_return_pct",
        headerTooltip: "편입 후 수익률 — 연속 편입이 시작된 교체일 시가 대비 현재가. 보유 중인 종목만 표시.",
        width: 92,
        type: "numericColumn",
        valueFormatter: (p) => (p.value == null ? "-" : formatSignedPct(p.value as number)),
        cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 600 }),
      },
      maExitGapColumn<PickRow>({
        field: "current_short_pct",
        maDays: view?.settings.short_ma_days,
      }),
      {
        headerName: "고점",
        field: "high_drawdown_pct",
        headerTooltip: "캐시 전 기간 최고가 대비 마지막 종가(%) — pools-rank 고점과 같은 규칙, 0 = 신고점",
        width: 80,
        type: "rightAligned",
        // `/pools-rank` 고점 컬럼과 같은 공용 렌더러 — 0 이면 ⭐신고점.
        cellRenderer: (p: { value?: number | null }) => renderHighDrawdownCell(p.value, 1),
      },
      // 시총은 개별주에만 있는 값이라 업종과 판정이 다르다(`@/lib/pool-industry`).
      marketCapRankColumn<PickRow>("market_cap_rank", !hasMarketCap),
      {
        headerName: "티커",
        field: "ticker",
        // 한국 6자리 코드 + 상세 링크 아이콘이 잘리지 않는 폭.
        width: 96,
        // 호주 풀은 미국 동일 심볼과 구분되게 ASX: 접두를 붙인다 (pools-rank 와 동일).
        cellRenderer: (p: { value: string | null | undefined }) => {
          const raw = String(p.value ?? "-");
          const display = picksCountry === "au" && raw !== "-" && !raw.startsWith("ASX:") ? `ASX:${raw}` : raw;
          return <TickerDetailLink ticker={display} displayTicker={display} />;
        },
      },
      {
        headerName: "종목명",
        field: "name",
        // 이 표에서 유일한 flex 컬럼 — 남는 폭을 종목명이 가져간다.
        // 상한(maxWidth)을 두지 않아 넓은 화면에서 계속 늘어난다.
        flex: 1,
        minWidth: 220,
        // 종목명 셀 표준(`@/lib/name-highlight`) — 말줄임·레버리지 강조·추세 이탈 배지가 전 화면 공통이다.
        // 이탈 판정은 `현재-단기/장기` 기준이다. 선정 당시가 아니라 지금 상태를 보여준다.
        cellRenderer: (p: { value?: string | null; data?: PickRow }) =>
          renderStockNameCell(p.value, {
            trendBroken: isTrendBroken(p.data?.current_short_pct, p.data?.current_long_pct),
          }),
      },
      // 종목 메모 — 순위·자산 관리 화면과 같은 값(종목에 붙는다). 셀을 벗어나면 저장.
      stockMemoColumn<PickRow>({
        field: "memo",
        onSave: (row, memo) => void saveMemo(row.ticker, memo),
      }),
      // 업종 데이터가 아예 없는 풀(ETF 모음 등)에서는 빈 컬럼을 숨긴다.
      industryColumn<PickRow>({ hide: !hasIndustryData }),
      {
        headerName: "일간(%)",
        field: "daily_change_pct",
        headerTooltip: "실시간 일간 등락률 (실시간 실패 시 캐시 종가 기준)",
        width: 88,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value, 2),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
      {
        headerName: "현재가",
        field: "price",
        headerTooltip: "실시간 현재가 (실시간 실패 시 가격 캐시 최신 종가, 통화는 종목풀 기준)",
        width: 100,
        type: "numericColumn",
        valueFormatter: (p) => formatPrice(p.value, p.data?.currency),
      },
      // 표준 배치(일간(%) → 현재가 → 거래대금) — 순위·신고가 화면과 같은 공용 컬럼.
      tradeValueMultColumn<PickRow>(),
      // 시가총액 — 배치 B 가 메타 캐시에 적어 둔 값(개별주 풀만). 신고가 화면과 같은 소스·같은 표기.
      ...(hasMarketCap
        ? [
          {
            headerName: "시가총액",
            field: "market_cap",
            headerTooltip: "배치가 하루 한 번 적어 둔 시가총액 (순위·신고가 화면과 같은 값)",
            width: 120,
            type: "numericColumn",
            valueFormatter: (p) => formatMarketCapWon(p.value),
          } as ColDef<PickRow>,
        ]
        : []),
      // 월별 수익률 — pools-rank 월별과 같은 계산(전월 말 종가 대비, 이번 달은 마지막
      // 종가까지)의 최근 6개월. 라벨은 서버가 내려주고 헤더는 (%) 없이 표시한다.
      ...monthlyLabels.map(
        (label): ColDef<PickRow> => ({
          headerName: label.replace("(%)", ""),
          colId: label,
          headerTooltip: "전월 말 종가 대비 수익률 (이번 달은 실시간 현재가까지) — pools-rank 월별과 같은 계산",
          width: 92,
          type: "numericColumn",
          valueGetter: (p) => p.data?.monthly_returns?.[label] ?? null,
          valueFormatter: (p) => formatSigned(p.value, 1),
          cellStyle: (p) => ({ color: signColor(p.value) }),
        }),
      ),
    ];
    // 월별 라벨·국가·업종 유무가 선정 응답에 실려 온다 — 바뀌면(월 전환·풀 전환) 컬럼도 다시 만든다.
  }, [hasIndustryData, monthlyLabels, picksCountry, saveMemo, view?.settings.short_ma_days]);

  // 월간 표 — 연간과 같은 집계형(월/전략/벤치). 매매 내역은 주간 표가 담당한다.
  // 표 헤더의 지수 이름 — 신고가·합성 화면과 같은 표기(값에 % 가 붙으므로 헤더엔 붙이지 않는다).
  const benchmarkLabel = backtest?.benchmark_name ?? "벤치마크";

  // ADR 컬럼 — 값이 하나도 없으면(레짐 시장 없는 풀) 컬럼을 숨긴다.
  const hasAdr = useMemo(() => (backtest?.daily ?? []).some((row) => row.adr != null), [backtest]);
  // 월간·연간의 최저 ADR — 일간 값에서 직접 구한다(집계 공용 함수는 수익률만 다룬다).
  const adrMinByPrefix = useMemo(() => {
    const min = new Map<string, number>();
    for (const row of backtest?.daily ?? []) {
      if (row.adr == null) continue;
      for (const prefix of [row.date.slice(0, 7), row.date.slice(0, 4)]) {
        const known = min.get(prefix);
        if (known == null || row.adr < known) min.set(prefix, row.adr);
      }
    }
    return min;
  }, [backtest]);
  // 주간의 최저 ADR — 월간·연간은 날짜 접두사로 묶이지만 주는 접두사가 없어 **구간**으로 나눈다
  // (직전 기준일 다음날 ~ 이번 기준일). 표는 최신순이라 여기서 오름차순으로 세워 훑는다.
  const adrMinByWeekEnd = useMemo(() => {
    const min = new Map<string, number>();
    const weekEnds = (backtest?.weekly ?? []).map((row) => row.week_end).sort();
    if (weekEnds.length === 0) return min;
    const days = [...(backtest?.daily ?? [])].sort((a, b) => a.date.localeCompare(b.date));
    let index = 0;
    for (const row of days) {
      if (row.adr == null) continue;
      while (index < weekEnds.length && row.date > weekEnds[index]) index += 1;
      if (index >= weekEnds.length) break;
      const known = min.get(weekEnds[index]);
      if (known == null || row.adr < known) min.set(weekEnds[index], row.adr);
    }
    return min;
  }, [backtest]);

  /** ADR 컬럼 — 정의는 공용(@/lib/grid-cells), 여기서는 숨김 여부만 바인딩한다. */
  const adrColumn = useCallback(
    <T,>(headerName: string, getter: (row: T) => number | null | undefined, headerTooltip: string): ColDef<T> =>
      sharedAdrColumn<T>({ headerName, headerTooltip, hide: !hasAdr, getter }),
    [hasAdr],
  );

  const backtestColumns = useMemo<ColDef<BacktestMonthRow>[]>(() => {
    if (!backtest) return [];
    const columns: ColDef<BacktestMonthRow>[] = [
      {
        headerName: "월",
        field: "month",
        width: 148,
        cellStyle: () => ({ fontWeight: 700 }),
      },
      {
        headerName: "전략",
        field: "strategy_pct",
        flex: 1,
        minWidth: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value),
        cellStyle: (p) => ({ color: signColor(p.value), fontWeight: 700 }),
      },
      {
        headerName: benchmarkLabel,
        headerTooltip: `벤치마크 ${backtest.benchmark_name}(${backtest.benchmark_ticker})`,
        field: "benchmark_pct",
        flex: 1,
        minWidth: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value),
        cellStyle: (p) => ({ color: signColor(p.value) }),
      },
    ];
    columns.push(excessColumn<BacktestMonthRow>());
    columns.push(adrColumn<BacktestMonthRow>("최저 ADR", (row) => adrMinByPrefix.get(row.month) ?? null, "그 달의 시장 ADR 최저값"));
    return columns;
  }, [backtest, adrColumn, adrMinByPrefix]);

  const dailyColumns = useMemo<ColDef<BacktestDayRow>[]>(() => {
    if (!backtest) return [];
    const pctColumn = (headerName: string, field: keyof BacktestDayRow, headerTooltip?: string): ColDef<BacktestDayRow> => ({
      headerName,
      field,
      headerTooltip,
      flex: 1,
      minWidth: 110,
      type: "numericColumn",
      valueFormatter: (p) => formatSigned(p.value),
      cellStyle: (p) => ({ color: signColor(p.value), fontWeight: field === "strategy_pct" ? 700 : 400 }),
    });
    const columns: ColDef<BacktestDayRow>[] = [
      { headerName: "날짜", field: "date", width: 148, cellStyle: () => ({ fontWeight: 700 }) },
      pctColumn("전략", "strategy_pct", "보유 종목 동일가중 일간 변동률 (교체일에는 리밸런싱 비용 반영)"),
      pctColumn(benchmarkLabel, "benchmark_pct", `${backtest.benchmark_name}(${backtest.benchmark_ticker})`),
    ];
    columns.push(excessColumn<BacktestDayRow>());
    columns.push(adrColumn<BacktestDayRow>("ADR", (row) => row.adr, "그날의 시장 ADR(20일 등락비율)"));
    return columns;
  }, [backtest, adrColumn]);

  const tradeColumns = useMemo<ColDef<BacktestTradeRow>[]>(
    () => [
      { headerName: "티커", field: "ticker", width: 96 },
      {
        headerName: "종목명",
        field: "name",
        flex: 1,
        minWidth: STOCK_NAME_COLUMN_MIN_WIDTH,
        cellRenderer: (p: { value?: string | null }) => renderStockNameCell(p.value),
      },
      { headerName: "편입일", field: "entry_date", width: 116 },
      {
        headerName: "매수가",
        field: "entry_price",
        width: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatPrice(p.value as number, view?.picks?.currency),
      },
      {
        headerName: "청산일",
        field: "exit_date",
        width: 116,
        valueFormatter: (p) => (p.value ? String(p.value) : "-"),
      },
      {
        headerName: "청산가",
        field: "exit_price",
        headerTooltip: "보유중 행은 마지막 종가",
        width: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatPrice(p.value as number, view?.picks?.currency),
      },
      {
        headerName: "수익률(%)",
        field: "return_pct",
        width: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatSigned(p.value),
        cellStyle: (p) => ({ color: signColor(p.value), fontWeight: 700 }),
      },
      { headerName: "보유일", field: "days", width: 84, type: "numericColumn" },
      { headerName: "사유", field: "reason", width: 110 },
    ],
    [view?.picks?.currency],
  );

  // 월간·연간은 **달력 기준**으로 다시 만든다(신고가·합성 화면과 같은 경계).
  const monthRows = useMemo<BacktestMonthRow[]>(
    () => (backtest ? toCalendarMonthRows(backtest.daily) : []),
    [backtest],
  );
  const yearRows = useMemo<BacktestYearRow[]>(() => toYearRows(monthRows), [monthRows]);


  // 주간 표 — 매매 일지. 주 수익률 + 그 주에 체결된 편입·편출과 주말 보유 수·교체율.
  const weeklyColumns = useMemo<ColDef<BacktestWeekRow>[]>(() => {
    if (!backtest) return [];
    const pctColumn = (headerName: string, field: keyof BacktestWeekRow, headerTooltip?: string): ColDef<BacktestWeekRow> => ({
      headerName,
      field,
      headerTooltip,
      width: 120,
      type: "numericColumn",
      valueFormatter: (p) => formatSigned(p.value as number | null),
      cellStyle: (p) => ({ color: signColor(p.value as number | null), fontWeight: field === "strategy_pct" ? 700 : 400 }),
    });
    const columns: ColDef<BacktestWeekRow>[] = [
      {
        headerName: "기준일",
        field: "week_end",
        headerTooltip: "그 주 마지막 거래일 — 수익률은 그 주 성과, 편입·편출은 그 주에 체결된 매매",
        width: 148,
        valueFormatter: (p) => formatDateWithWeekday(String(p.value ?? "")),
        cellStyle: () => ({ fontWeight: 700 }),
      },
      pctColumn("전략", "strategy_pct", "그 주 보유 포트폴리오의 수익률 (교체 비용 반영)"),
      pctColumn(benchmarkLabel, "benchmark_pct", `${backtest.benchmark_name}(${backtest.benchmark_ticker})`),
      adrColumn<BacktestWeekRow>("판정일 ADR", (row) => row.adr, "기준일(판정일) 종가 기준 시장 ADR — 다음 주 게이트를 결정한 값"),
      adrColumn<BacktestWeekRow>(
        "최저 ADR",
        (row) => adrMinByWeekEnd.get(row.week_end) ?? null,
        "그 주의 시장 ADR 최저값 — 주중 게이트가 걸렸는지 본다",
      ),
    ];
    columns.push(
      {
        headerName: "종목 수",
        field: "holdings_count",
        headerTooltip: "교체 직후 → 주말 종목 수. 화살표가 있으면 주중 이탈로 줄어든 것(이탈 컬럼 참고).",
        width: 84,
        type: "numericColumn",
        cellDataType: "text",
        valueFormatter: (p) => {
          const start = p.data?.holdings_start;
          const end = p.value as number | null;
          if (end == null) return "-";
          return start != null && start !== end ? `${start} → ${end}` : String(end);
        },
      },
      {
        headerName: "교체율(%)",
        field: "turnover_pct",
        headerTooltip: "이 교체에서 편입된 슬롯 비중 (편입 수 ÷ 종목 수 설정)",
        width: 84,
        type: "numericColumn",
        valueFormatter: (p) => (p.value == null ? "-" : formatNumber(p.value)),
      },
      {
        headerName: "편입",
        field: "added",
        flex: 1,
        minWidth: 200,
        wrapText: true,
        autoHeight: true,
        cellClass: "momentumWrapCell",
        valueFormatter: (p) => (p.value?.length ? p.value.join(", ") : "-"),
        cellStyle: () => ({ color: "var(--up-color, #d64545)" }),
      },
      {
        headerName: "편출",
        field: "removed",
        flex: 1,
        minWidth: 200,
        wrapText: true,
        autoHeight: true,
        cellClass: "momentumWrapCell",
        valueFormatter: (p) => (p.value?.length ? p.value.join(", ") : "-"),
        cellStyle: () => ({ color: "var(--down-color, #2f6fd0)" }),
      },
      {
        headerName: "이탈",
        field: "exited",
        headerTooltip: "주중에 자격 상실로 매도된 종목 (교체 편출과 별개, 판 슬롯은 다음 교체까지 현금)",
        flex: 1,
        minWidth: 200,
        wrapText: true,
        autoHeight: true,
        cellClass: "momentumWrapCell",
        valueFormatter: (p) => (p.value?.length ? p.value.join(", ") : "-"),
        cellStyle: () => ({ color: "var(--down-color, #2f6fd0)", opacity: 0.8 }),
      },
    );
    columns.push(excessColumn<BacktestWeekRow>());
    return columns;
  }, [backtest, adrColumn, adrMinByWeekEnd]);

  const yearColumns = useMemo<ColDef<BacktestYearRow>[]>(() => {
    if (!backtest) return [];
    // 부분 기간은 /compare 와 같은 규칙으로 값 뒤에 `*` 를 붙인다.
    const pctColumn = (
      headerName: string,
      field: "strategy_pct" | "benchmark_pct",
      partialField: "strategy_partial" | "benchmark_partial",
      headerTooltip?: string,
    ): ColDef<BacktestYearRow> => ({
      headerName,
      field,
      headerTooltip,
      flex: 1,
      minWidth: 110,
      type: "numericColumn",
      valueFormatter: (p) =>
        p.value == null ? "-" : `${formatSigned(p.value)}${p.data?.[partialField] ? "*" : ""}`,
      cellStyle: (p) => ({ color: signColor(p.value), fontWeight: field === "strategy_pct" ? 700 : 400 }),
      tooltipValueGetter: (p) =>
        p.data?.[partialField] ? "12개월이 다 차지 않은 해 — 있는 달만 합성한 부분 기간" : undefined,
    });

    const columns: ColDef<BacktestYearRow>[] = [
      { headerName: "연도", field: "year", width: 148, cellStyle: () => ({ fontWeight: 700 }) },
      pctColumn("전략", "strategy_pct", "strategy_partial"),
      pctColumn(
        benchmarkLabel,
        "benchmark_pct",
        "benchmark_partial",
        `${backtest.benchmark_name}(${backtest.benchmark_ticker})`,
      ),
    ];
    columns.push(excessColumn<BacktestYearRow>());
    columns.push(adrColumn<BacktestYearRow>("최저 ADR", (row) => adrMinByPrefix.get(row.year) ?? null, "그 해의 시장 ADR 최저값"));
    return columns;
  }, [backtest, adrColumn, adrMinByPrefix]);

  if (loading && !view) {
    return (
      <PageFrame title="모멘텀 전략" fullWidth>
        <div style={{ ...hintStyle, padding: 20 }}>불러오는 중…</div>
      </PageFrame>
    );
  }
  if (!view) {
    // 설정을 못 받은 상태 — 값을 지어내 폼을 그리지 않고 실패와 재시도만 제공한다.
    return (
      <PageFrame title="모멘텀 전략" fullWidth>
        <div className="card appCard">
          <div className="card-body" style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ fontWeight: 700 }}>설정을 불러오지 못했습니다.</span>
            <span style={hintStyle}>{loadError ?? "원인을 알 수 없습니다."}</span>
            <button type="button" className="btn btn-sm btn-primary" onClick={() => void load()} disabled={loading}>
              {loading ? "다시 시도 중…" : "다시 시도"}
            </button>
          </div>
        </div>
      </PageFrame>
    );
  }

  // 보유 수 — 판정일 선정 − 주중 매도(체결 완료).
  const selectedCount = view.picks?.rows.filter((row) => !row.is_reserve && !row.is_exited).length ?? 0;
  const reserveCount = view.picks?.rows.filter((row) => row.is_reserve).length ?? 0;

  return (
    <PageFrame title="모멘텀 전략" fullWidth>
      <div className="appPageStack">
        {/* ① 변수 설정 */}
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">종목풀</span>
                  <select
                    className="form-select form-select-sm"
                    value={draftPool}
                    onChange={(e) => handlePoolChange(e.target.value)}
                  >
                    {/* 설정을 못 받으면 폼 자체를 그리지 않으므로(로드 실패 화면) 폴백 목록이 필요 없다. */}
                    {(view.pool_options ?? []).map((pool) => (
                      <option key={pool.ticker_type} value={pool.ticker_type}>
                        {formatPoolLabel(pool)}
                      </option>
                    ))}
                  </select>
                </label>
                {draftMaRule != null && view.ma_rule != null ? (
                  <>
                    <label className="appLabeledField">
                      <span className="appLabeledFieldLabel">이평선</span>
                      <span className="appMaRuleRow">
                        <MaDaysSelect
                          title="단기 이평선"
                          value={draftMaRule.short}
                          options={view.ma_rule.short_ma_options}
                          onChange={(days) => setDraftMaRule((r) => r && { ...r, short: days })}
                        />
                        <MaDaysSelect
                          title="장기 이평선"
                          value={draftMaRule.long}
                          options={view.ma_rule.long_ma_options}
                          onChange={(days) => setDraftMaRule((r) => r && { ...r, long: days })}
                        />
                      </span>
                    </label>
                    <label className="appLabeledField">
                      <span className="appLabeledFieldLabel">ADR 하한</span>
                      <select
                        className="form-select form-select-sm"
                        style={{ width: 88 }}
                        value={draftAdrFloor}
                        onChange={(e) => setDraftAdrFloor(e.target.value)}
                        title="판정일의 시장 ADR(20일 등락비율)이 이 값 미만이면 그 주는 전량 현금. 시장은 종목풀 설정의 시장 레짐 지수를 따른다."
                      >
                        {(view.constraints?.adr_floor_options ?? []).map((value) => (
                          <option key={String(value)} value={value == null ? "" : String(value)}>
                            {value == null ? "없음" : String(value)}
                          </option>
                        ))}
                      </select>
                    </label>
                  </>
                ) : null}
              </div>
              <div className="appMainHeaderRight">
                {isNewPoolDraft ? (
                  <span style={{ ...hintStyle, color: "var(--up-color, #d64545)", fontWeight: 700 }}>
                    이 풀은 첫 설정 — 저장해야 선정·백테스트가 실행됩니다
                  </span>
                ) : (
                  <UnsavedChangesBadge show={isDirty} />
                )}
                <button
                  type="button"
                  className="btn btn-success btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                  onClick={() => void saveSettings()}
                  disabled={saving || !isDirty}
                >
                  <IconCheck size={16} />
                  <span>{saving ? "저장 중…" : "저장"}</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* ② 운용 현황 */}
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <span style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>운용 현황</span>
                {view.picks ? (
                  <span style={{ ...hintStyle, fontSize: "var(--fs-sm)" }}>
                    <b style={{ color: "inherit" }}>{formatDateWithWeekday(view.picks.portfolio_week)} 포트폴리오</b> ·
                    체결 {view.picks.rebalance_date} (판정 {view.picks.signal_date}) · {view.picks.universe_count} →{" "}
                    {view.picks.candidate_count} → {selectedCount}
                    {reserveCount > 0 ? ` (+${reserveCount})` : ""}
                    {view.picks.adr_gate ? (
                      view.picks.adr_gate.blocked ? (
                        <b style={{ color: "#d9480f" }}>
                          {" "}· ADR 게이트 발동 — {view.picks.adr_gate.market} {view.picks.adr_gate.value ?? "-"} &lt;{" "}
                          {view.picks.adr_gate.floor}, 이번 주 전량 현금
                        </b>
                      ) : view.picks.adr_gate.intraweek.blocked ? (
                        // 주간 판정은 통과했는데 그 뒤 ADR 이 무너진 상태 — 판정일 값만 보여 주면
                        // 왜 전량 매도 예정인지 화면에서 알 수가 없다. 최신 값을 함께 세운다.
                        <b
                          style={{ color: "#d9480f" }}
                          title={`판정일 ${view.picks.signal_date} 에는 하한을 넘어 선정이 실행됐지만, 그 뒤 ${view.picks.adr_gate.intraweek.date} 에 하한 아래로 내려갔습니다.`}
                        >
                          {" "}· ADR {view.picks.adr_gate.market} 판정일 {view.picks.adr_gate.value ?? "-"} → 주중 게이트 발동{" "}
                          {view.picks.adr_gate.intraweek.value ?? "-"} &lt; {view.picks.adr_gate.floor}, 다음 거래일 시가 전량 매도
                        </b>
                      ) : (
                        // 두 값의 시점이 다르다 — 앞은 이번 주를 정한 판정일, 뒤는 마지막 거래일.
                        // 라벨 없이 숫자만 두면 무엇을 보는 값인지 알 수가 없다.
                        <span
                          title={`판정일 ${view.picks.signal_date} 의 ADR 이 하한 이상이라 이번 주 선정이 실행됐습니다. 최신은 마지막 거래일 ${view.picks.adr_gate.intraweek.date} 값으로, 주중 게이트가 이 값을 봅니다.`}
                        >
                          {" "}· ADR {view.picks.adr_gate.market} 판정일 {view.picks.adr_gate.value ?? "-"}
                          {view.picks.adr_gate.intraweek.value != null
                            ? ` · 최신 ${view.picks.adr_gate.intraweek.value}`
                            : ""}
                          {" "}(하한 {view.picks.adr_gate.floor})
                        </span>
                      )
                    ) : null}
                  </span>
                ) : (
                  <span style={{ ...hintStyle, fontSize: "var(--fs-sm)" }}>
                    {pickFailed
                      ? "선정 결과를 불러오지 못했습니다. 설정을 저장하거나 새로고침하세요."
                      : "계산 중…"}
                  </span>
                )}
              </div>
            </div>
            <StrategyNotes items={CURRENT_NOTES} />
            <NavTabs
              items={CURRENT_TABS}
              value={currentTab}
              onChange={setCurrentTab}
              label="운용 현황 보기"
              style={{ marginBottom: 12 }}
            />
            {picking ? <AppLoadingProgress title="선정 계산 중..." progress={pickProgress} /> : null}
            {view.picks && !picking && currentTab === "list" ? (
              // autoHeight — 그리드가 행 수만큼만 높이를 차지해 하단 낭비가 없다.
              <AppAgGrid<PickRow>
                rowData={pickRows}
                columnDefs={pickColumns}
                theme={gridTheme}
                minHeight={0}
                height="auto"
                gridOptions={{ domLayout: "autoHeight" }}
                getRowClass={(p) => {
                  // 추세 이탈은 종목명 뒤 ❗ 와 같은 조건으로 행을 연한 회색으로 눌러 둔다.
                  const classes: string[] = [];
                  if (p.data?.is_reserve) classes.push("momentumReserveRow");
                  // 주중 매도 예정 — 판정만 끝나고 체결 전 (백테스트 예정 행과 같은 스타일).
                  if (p.data?.is_exit_pending) classes.push("momentumPendingRow");
                  // 주중 매도 완료 — 더는 보유가 아니다.
                  if (p.data?.is_exited) classes.push("appTrendBrokenRow");
                  if (isTrendBroken(p.data?.current_short_pct, p.data?.current_long_pct)) {
                    classes.push("appTrendBrokenRow");
                  }
                  return classes.join(" ");
                }}
                getRowId={(p) => p.data.ticker}
              />
            ) : null}
            {view.picks && !picking && currentTab === "chart" ? (
              <StrategyHoldingCharts
                charts={charts}
                loading={chartsLoading || saving || picking}
                error={chartsError}
                emptyMessage="선정된 종목이 없습니다."
                hint="장기선 위 & 단기선 위(자격)를 잃으면 편출되고, 편입이 시작된 교체일에 Buy 화살표가 표시됩니다."
                months={chartMonths}
                chartProps={(item) => {
                  const row = chartRows.find((candidate) => candidate.ticker === item.ticker);
                  return {
                    strategyLabel: "모멘텀",
                    entryDate: row?.entry_date,
                    returnPct: row?.entry_return_pct,
                    // 아직 안 산 종목은 0주 — streak_weeks 는 '이번 주 선정 1주차'라 1 이 들어온다.
                    days: row?.is_held ? row.streak_weeks : 0,
                    daysUnit: "주",
                  };
                }}
              />
            ) : null}
          </div>
        </div>

        {/* ③ 백테스트 */}
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <span style={{ fontWeight: 700, fontSize: "var(--fs-base)" }}>백테스트</span>
              </div>
              <div className="appMainHeaderRight">
                {isDirty ? <span style={hintStyle}>설정을 저장해야 실행할 수 있습니다</span> : null}
                <MonthsSelect
                  value={backtestMonths}
                  options={view.month_options ?? [12]}
                  disabled={backtesting}
                  onChange={setBacktestMonths}
                />
                <button
                  type="button"
                  className="btn btn-sm btn-dark"
                  onClick={() => void runBacktest()}
                  disabled={backtesting || isDirty}
                >
                  {backtesting ? "실행 중…" : "실행"}
                </button>
              </div>
            </div>
            {backtesting ? <AppLoadingProgress title="백테스트 실행 중..." progress={backtestProgress} /> : null}
            {backtest && !backtesting ? (
              <>
                <BacktestSummary
                  startDate={backtest.start_date}
                  endDate={backtest.end_date}
                  strategy={{
                    label: "전략",
                    totalPct: backtest.strategy_total_pct,
                    cagrPct: backtest.strategy_cagr_pct,
                    mddPct: backtest.strategy_mdd_pct,
                    sortino: backtest.strategy_sortino,
                  }}
                  benchmark={{
                    label: `${backtest.benchmark_name}(${backtest.benchmark_ticker})`,
                    totalPct: backtest.benchmark_total_pct,
                    cagrPct: backtest.benchmark_cagr_pct,
                    mddPct: backtest.benchmark_mdd_pct,
                    sortino: backtest.benchmark_sortino,
                  }}
                />
                <BacktestTradeStats stats={backtest} style={{ marginBottom: 12 }} />
                <NavTabs
                  items={VIEW_MODES}
                  value={viewMode}
                  onChange={setViewMode}
                  label="백테스트 보기 단위"
                  style={{ marginBottom: 10 }}
                />
                {viewMode === "trades" ? (
                  <AppAgGrid<BacktestTradeRow>
                    rowData={backtest.trades}
                    columnDefs={tradeColumns}
                    theme={gridTheme}
                    minHeight={0}
                    height="auto"
                    gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                    getRowClass={(p) => (p.data?.exit_date ? "" : "momentumPendingRow")}
                  />
                ) : viewMode === "weekly" ? (
                  <AppAgGrid<BacktestWeekRow>
                    rowData={backtest.weekly}
                    columnDefs={weeklyColumns}
                    theme={gridTheme}
                    minHeight={0}
                    height="auto"
                    gridOptions={{ domLayout: "autoHeight" }}
                    getRowClass={(p) => (p.data?.is_pending ? "momentumPendingRow" : "")}
                    getRowId={(p) => p.data.week_end}
                  />
                ) : viewMode === "daily" ? (
                  // 월간·연간과 같이 autoHeight — 카드 안에서 스크롤하지 않고 브라우저 스크롤로 본다.
                  <AppAgGrid<BacktestDayRow>
                    rowData={backtest.daily}
                    columnDefs={dailyColumns}
                    theme={gridTheme}
                    minHeight={0}
                    height="auto"
                    gridOptions={{ domLayout: "autoHeight" }}
                    getRowId={(p) => p.data.date}
                  />
                ) : viewMode === "monthly" ? (
                  <AppAgGrid<BacktestMonthRow>
                    rowData={monthRows}
                    columnDefs={backtestColumns}
                    theme={gridTheme}
                    minHeight={0}
                    height="auto"
                    gridOptions={{ domLayout: "autoHeight" }}
                    getRowId={(p) => p.data.month}
                  />
                ) : (
                  <>
                    <AppAgGrid<BacktestYearRow>
                      rowData={yearRows}
                      columnDefs={yearColumns}
                      theme={gridTheme}
                      minHeight={0}
                      height="auto"
                      gridOptions={{ domLayout: "autoHeight" }}
                      getRowId={(p) => p.data.year}
                    />
                    {yearRows.some(
                      (row) => row.strategy_partial || row.benchmark_partial,
                    ) ? (
                      <span style={{ ...hintStyle, fontSize: "var(--fs-sm)" }}>
                        * 12개월이 다 차지 않은 해 — 있는 달만 합성한 부분 기간입니다.
                      </span>
                    ) : null}
                  </>
                )}
              </>
            ) : !backtesting ? (
              <span style={{ ...hintStyle, fontSize: "var(--fs-sm)" }}>
                실행을 누르면 월별 성과가 표시됩니다. 기간은 위 변수 설정에서 바꿉니다.
              </span>
            ) : null}
          </div>
        </div>

        {/* 튜닝 — 백테스트 아래. 백테스트와 같이 **저장된** 설정을 기준으로 아래 범위의 조합을
            전부 돌린다. 이평 축은 상단 셀렉트와 같은 선택지(서버 상수)를 쓴다. */}
        <StrategyTuning
          // 풀이 바뀌면 재마운트 — 이전 풀의 튜닝 결과가 남지 않게 한다(백테스트와 같은 시점).
          key={draftPool}
          monthOptions={view.tuning_month_options ?? view.month_options ?? [backtestMonths]}
          defaultMonths={backtestMonths}
          // 튜닝도 백테스트와 같이 **저장된 설정** 기준이라 실행 조건을 같게 둔다.
          disabled={backtesting || isDirty}
          disabledHint={isDirty ? "설정을 저장해야 실행할 수 있습니다" : undefined}
          fixedLabel={`저장된 설정 기준 (종목풀 ${draftPool} · 종목 수 ${view.settings.top_n} 공통 고정 · 교체 규칙 자격 유지 · 주중 이탈 사용)`}
          current={{
            short_ma_days: view.settings.short_ma_days,
            long_ma_days: view.settings.long_ma_days,
            adr_floor: view.settings.adr_floor ?? null,
          }}
          axes={[
            // 축 값 = 상단 셀렉트 선택지(서버 상수) — 여기서 따로 정하지 않는다.
            // 종목 수(공통 고정)·업종 상한(폐기)·교체 규칙·주중 이탈(전략 고정)은 축이 아니다
            // — 튜닝은 시장의 이평 반응만 잰다.
            { key: "short_ma_days", label: "단기 이평", values: (view.ma_rule?.short_ma_options ?? []).map((n) => ({ value: n, label: `${n}일` })) },
            { key: "long_ma_days", label: "장기 이평", values: (view.ma_rule?.long_ma_options ?? []).map((n) => ({ value: n, label: `${n}일` })) },
            {
              key: "adr_floor",
              label: "ADR 하한",
              values: (view.constraints?.adr_floor_options ?? []).map((n) => (n == null ? { value: null, label: "없음" } : { value: n, label: String(n) })),
            },
          ]}
          onApply={async (params) => {
            // 조합을 상단 폼에 넣고 그대로 저장한다 (저장 응답이 폼·선정을 갱신한다).
            const next = {
              ...view.settings,
              pool: draftPool,
              short_ma_days: Number(params.short_ma_days),
              long_ma_days: Number(params.long_ma_days),
              adr_floor: params.adr_floor == null ? null : Number(params.adr_floor),
            };
            fillDrafts(next);
            await persistSettings(next, "튜닝 조합을 적용해 저장했습니다.");
          }}
          // 응답은 SSE 스트림(진행 이벤트 + 결과 이벤트) — 파싱은 StrategyTuning 이 한다.
          run={async (months, ranges, signal) =>
            fetch(`/api/strategy-momentum/tuning?pool=${encodeURIComponent(draftPool)}`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ months, ranges }),
              signal,
            })
          }
          cancelRun={() => fetch("/api/strategy-momentum/tuning", { method: "DELETE" })}
        />
      </div>
    </PageFrame>
  );
}
