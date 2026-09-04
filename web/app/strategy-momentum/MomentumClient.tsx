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
  VIEW_MODES,
  toPeriodRows,
  weekKeyOf,
  type PeriodRow,
  type ViewMode,
} from "@/lib/backtest-periods";
import {
  industryColumn,
  STOCK_NAME_COLUMN_WIDTH,
  adrColumn as sharedAdrColumn,
  formatSignedPct,
  maExitGapColumn,
  highDrawdownColumn,
  signColor,
  marketCapRankColumn,
  slotStatusColumn,
  slotTradeColumns,
  stockMemoColumn,
  tradeValueMultColumn,
  STATUS_COLUMN_MIN_WIDTH,
  STATUS_COLUMN_WIDTH,
  type SlotPlan,
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
    title: "주중 매도 (ADR 게이트만)",
    body:
      "주중 매도는 시장 ADR이 하한 아래로 내려간 날의 전량 매도(다음 거래일 시가) 하나뿐입니다. " +
      "보유 종목이 주중에 이평선을 이탈해도 ADR이 하한 이상이면 주말 판정까지 계속 보유합니다 " +
      "— 개별 이탈은 주말 교체(자격 유지 재선정)가 처리합니다. " +
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

/** 보유 표 한 행 — 보유·매도 예정·진입 예정·이탈·빈 슬롯을 한 표에 담는다(신고가와 같다). */
type PlanRow = {
  ticker: string;
  name: string;
  industry: string;
  market_cap_rank?: number | null;
  change_pct: number | null;
  /** 현재 시세 — 이탈 행도 지금 값이다(청산가는 exit_price). */
  price: number | null;
  /** 청산가 — 마지막 세션에 이탈한 행에만 있다. */
  exit_price: number | null;
  /** 청산일 — 이탈 행에만 있다. 상태 문구에 붙인다. */
  exit_date?: string | null;
  entry_date: string | null;
  entry_price: number | null;
  return_pct: number | null;
  plan: SlotPlan;
  value_mult?: number | null;
  value_mult_live?: number | null;
  days: number | null;
  is_new: boolean;
  exit_reason: string | null;
  /** 종목에 붙는 메모 — 순위·신고가·자산 관리 화면과 같은 값. */
  memo?: string;
  /** 단기 이평선까지 남은 여유(%) — 0 이하면 이탈. */
  short_gap_pct?: number | null;
  /** 장기 이평선까지 남은 여유(%) — 0 이하면 이탈. */
  long_gap_pct?: number | null;
  high_drawdown_pct?: number | null;
  /** 실계좌 보유 여부 — 전략 보유와 뜻이 다르다. */
  account_held?: boolean;
};

/** 진입 후보 한 행 — 자리가 나면 담을 순서대로. 상태는 「보유 중」과 「후보」 둘뿐이다. */
type CandidateRow = {
  ticker: string;
  name: string;
  industry: string;
  market_cap_rank?: number | null;
  change_pct: number | null;
  price: number | null;
  value_mult?: number | null;
  value_mult_live?: number | null;
  memo?: string;
  /** 이 순위표에서의 자리 — 우선순위(장기 이격률) 순. */
  rank: number;
  short_gap_pct: number | null;
  long_gap_pct: number | null;
  high_drawdown_pct: number | null;
  market_cap: number | null;
  account_held?: boolean;
};

/** 체결 한 건 — 진입~청산 한 쌍. */
type Trade = {
  ticker: string;
  name: string;
  industry: string;
  entry_date: string;
  entry_price: number;
  exit_date: string;
  exit_price: number;
  return_pct: number;
  days: number;
  reason: string;
  price?: number | null;
  change_pct?: number | null;
  market_cap_rank?: number | null;
  value_mult?: number | null;
  value_mult_live?: number | null;
  memo?: string;
  account_held?: boolean;
  /** 표시용 — 운용 현황 표가 보유 행과 같은 칸을 채운다. */
  short_gap_pct?: number | null;
  long_gap_pct?: number | null;
  high_drawdown_pct?: number | null;
};

/** 엔진이 주는 보유 한 행 — 표는 여기에 `plan` 을 붙여 한 축으로 다룬다. */
type Holding = Omit<PlanRow, "plan"> & { status: "hold" | "sell" };

type Positions = {
  as_of: string;
  /** 동시 보유 상한 — 빈 슬롯 행 수를 세는 데 쓴다. */
  top_n: number;
  /** 표시용 시세 갱신에 쓰는 국가 코드(시세 소스가 국가별로 다르다). */
  country: string;
  /** 가격 표기 통화 — 화면이 원·달러 표기를 이 값으로 정한다. */
  currency: string;
  /** 진입 예정·매도 예정이 실제로 체결되는 날. */
  next_session: string | null;
  /** 장중인가 — 참이면 오늘 종가 확정 전이라 판정이 잠정이다. */
  live: boolean;
  /** 가격 캐시가 마지막으로 갱신된 시각(KST). */
  cache_refreshed_at?: string | null;
  holdings: Holding[];
  /** 다음 시가에 살 종목 — 자리·자격·우선순위를 모두 적용한 결과. */
  planned_entries: CandidateRow[];
  /** 마지막 세션에 청산된 종목. */
  exited_today: Trade[];
  /** 진입 후보 — 우선순위 순 top_n 개. */
  candidates: CandidateRow[];
  /** ADR 게이트 — 하한 미설정이면 null. blocked 면 오늘은 신규 진입이 없다. */
  adr_gate?: {
    market: string | null;
    floor: number;
    value: number | null;
    blocked: boolean;
  } | null;
};

// 체결 행 — 진입~청산 한 쌍.
type BacktestTradeRow = {
  ticker: string;
  name: string;
  industry: string;
  entry_date: string;
  entry_price: number;
  exit_date: string;
  exit_price: number;
  return_pct: number;
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
  /** 일별 누적(%) 곡선 — 연간·월간·주간·일간 표를 전부 여기서 잘라 낸다. */
  daily: { date: string; strategy_pct: number; benchmark_pct: number; adr?: number | null }[];
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
  positions: Positions | null;
};

// 운용 현황 안쪽 탭 — 신고가 화면과 같은 구성. 차트는 선정 종목 수만큼 그리므로 열 때만 그린다.
const CURRENT_TABS = [
  { key: "list", label: "종목" },
  { key: "chart", label: "차트" },
] as const;
type CurrentTab = (typeof CURRENT_TABS)[number]["key"];


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
      const resp = await fetch(`/api/strategy-momentum/positions?pool=${encodeURIComponent(pool)}`, {
        method: "POST",
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "선정에 실패했습니다.");
      setPickProgress({ percent: 100, message: "선정 결과 반영 중" });
      setView((prev) => (prev ? { ...prev, positions: payload as Positions } : prev));
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
        applyView({ ...saved, positions: null });
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
        body: JSON.stringify({ months }),
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

  // 표시용 현재가·일간(%)만 60초마다 갱신한다 — 진입·청산 판정은 종가 기준이라 다시 계산하지 않는다.
  const positions = view?.positions ?? null;
  const quoteTickers = useMemo(
    () =>
      [
        ...(positions?.holdings ?? []).map((row) => row.ticker),
        ...(positions?.planned_entries ?? []).map((row) => row.ticker),
        ...(positions?.candidates ?? []).map((row) => row.ticker),
      ].filter((ticker, index, all) => all.indexOf(ticker) === index),
    [positions],
  );
  const quotes = useRealtimeQuotes(positions?.country ?? "", quoteTickers);
  /** 시세가 들어온 종목만 현재가·등락률을 덮어쓴다(나머지 값은 판정 결과 그대로). */
  const withQuote = useCallback(
    <T extends { ticker: string; price: number | null; change_pct: number | null }>(row: T): T => {
      const quote = quotes[row.ticker];
      return quote ? { ...row, price: quote.price, change_pct: quote.change_pct } : row;
    },
    [quotes],
  );

  // 보유 + 다음 시가 매수 + 빈 슬롯 + 이탈을 한 표로 — 이 표만 보고 주문을 낼 수 있게.
  const planRows = useMemo<PlanRow[]>(() => {
    if (!positions) return [];
    // 엔진은 `status`(hold/sell) 로 준다 — 표는 진입 예정·빈 슬롯까지 한 축(`plan`)으로 본다.
    const held: PlanRow[] = positions.holdings.map((row) => ({ ...row, plan: row.status }));
    const buys: PlanRow[] = positions.planned_entries.map((row) => ({
      ticker: row.ticker,
      name: row.name,
      industry: row.industry,
      market_cap_rank: row.market_cap_rank ?? null,
      change_pct: row.change_pct,
      price: row.price,
      value_mult: row.value_mult ?? null,
      value_mult_live: row.value_mult_live ?? null,
      exit_price: null,
      entry_date: null,
      entry_price: null,
      return_pct: null,
      // 아직 안 샀다 — days 를 null 로 두면 보유일 칸과 차트 배지가 통째로 비어 진입 전인지 알 수 없다.
      plan: "buy",
      days: 0,
      is_new: false,
      exit_reason: null,
      memo: row.memo,
      short_gap_pct: row.short_gap_pct,
      long_gap_pct: row.long_gap_pct,
      high_drawdown_pct: row.high_drawdown_pct,
      account_held: row.account_held,
    }));
    const exited: PlanRow[] = positions.exited_today.map((trade) => ({
      ticker: trade.ticker,
      name: trade.name,
      industry: trade.industry,
      market_cap_rank: trade.market_cap_rank ?? null,
      change_pct: trade.change_pct ?? null,
      price: trade.price ?? null,
      exit_price: trade.exit_price,
      exit_date: trade.exit_date,
      value_mult: trade.value_mult ?? null,
      value_mult_live: trade.value_mult_live ?? null,
      entry_date: trade.entry_date,
      entry_price: trade.entry_price,
      return_pct: trade.return_pct,
      plan: "exited",
      days: trade.days,
      is_new: false,
      exit_reason: trade.reason,
      memo: trade.memo,
      account_held: trade.account_held,
      // 이탈 행도 표의 모든 칸이 채워져야 한다 — 판 뒤의 상태를 같은 기준으로 본다.
      short_gap_pct: trade.short_gap_pct,
      long_gap_pct: trade.long_gap_pct,
      high_drawdown_pct: trade.high_drawdown_pct,
    }));
    // 빈 슬롯 — 상한에서 '다음 시가 이후에 실제로 차 있을 자리' 를 뺀 만큼. 매도 예정은 곧
    // 비고, 진입 예정은 곧 찬다. 자리가 남았다는 것은 자격을 갖춘 후보가 없었다는 뜻이라,
    // 표에 자리를 그려 두면 "왜 안 샀지" 를 표만 보고 알 수 있다.
    const filled = held.filter((row) => row.plan !== "sell").length + buys.length;
    const empty: PlanRow[] = Array.from({ length: Math.max(positions.top_n - filled, 0) }, (_, index) => ({
      ticker: `__EMPTY_${index}__`,
      name: "(빈 슬롯)",
      industry: "",
      market_cap_rank: null,
      change_pct: null,
      price: null,
      exit_price: null,
      value_mult: null,
      value_mult_live: null,
      entry_date: null,
      entry_price: null,
      return_pct: null,
      plan: "empty",
      days: null,
      is_new: false,
      exit_reason: null,
    }));
    // 계속 들고 갈 것 → 팔 것 → 살 것 → 빈 자리 → 이미 끝난 것 순.
    // 오늘 계좌에서 할 일의 순서다: 그대로 두고, 팔고, 사고, 남은 자리를 확인한다.
    // 같은 묶음 안에서는 **오래 들고 있는 것이 위** — 편입일이 이른 순이다.
    const order = { hold: 0, sell: 1, buy: 2, empty: 3, exited: 4 } as const;
    return [...held, ...buys, ...empty, ...exited].sort(
      (a, b) => order[a.plan] - order[b.plan] || (a.entry_date ?? "").localeCompare(b.entry_date ?? ""),
    );
  }, [positions]);

  const candidateRows = useMemo(() => (positions?.candidates ?? []).map(withQuote), [positions, withQuote]);
  const heldCount = planRows.filter((row) => row.plan === "hold" || row.plan === "sell").length;

  // ── 차트 탭 (신고가 화면과 같은 구성 — 공용 HoldingChart) ──
  const [currentTab, setCurrentTab] = useState<CurrentTab>("list");
  const [charts, setCharts] = useState<HoldingChartData[] | null>(null);
  const [chartsLoading, setChartsLoading] = useState(false);
  const [chartsError, setChartsError] = useState<string | null>(null);
  // 차트 기간(개월) — 백엔드 config.HOLDING_CHART_MONTHS 가 단일 소스. 응답에서 받아 문구에 쓴다.
  const [chartMonths, setChartMonths] = useState<number | null>(null);
  // 차트를 그릴 대상 — 이미 나간 종목과 빈 슬롯은 뺀다. 표와 같은 순서로 그린다.
  const chartRows = useMemo(
    () => planRows.filter((row) => row.plan !== "exited" && row.plan !== "empty"),
    [planRows],
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
    if (currentTab !== "chart" || !view || !positions || charts || chartsLoading || chartsError) return;
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

  const fillDay = positions?.next_session ?? "다음 거래일";
  const country = positions?.country ?? "";

  /** 티커 셀 — 호주는 `ASX:` 접두사를 붙여 다른 화면과 같게 보여준다. */
  const renderTicker = useCallback(
    (value: string | null | undefined) => {
      const raw = String(value ?? "").trim();
      if (!raw) return <span>-</span>;
      return <TickerDetailLink ticker={country === "au" && !raw.startsWith("ASX:") ? `ASX:${raw}` : raw} />;
    },
    [country],
  );

  // 보유 표 — 신고가와 같은 구성(공용 빌더). 모멘텀 고유는 이평선 이격 둘이다.
  const holdingColumns = useMemo<ColDef<PlanRow>[]>(
    () => [
      slotStatusColumn<PlanRow>({ live: Boolean(positions?.live), fillDay: positions?.next_session }),
      marketCapRankColumn<PlanRow>("market_cap_rank", !hasMarketCap),
      highDrawdownColumn<PlanRow>("high_drawdown_pct"),
      {
        field: "ticker",
        headerName: "티커",
        width: 108,
        cellRenderer: (p: { value?: string | null }) => renderTicker(p.value),
      },
      {
        field: "name",
        headerName: "종목명",
        // 고정 폭 — 보유 표와 후보 표의 앞쪽 칸(상태~거래대금)을 맞춘다.
        width: STOCK_NAME_COLUMN_WIDTH,
        cellRenderer: (p: { value?: string | null }) => renderStockNameCell(p.value),
      },
      stockMemoColumn<PlanRow>({
        field: "memo",
        editable: (row) => row?.plan !== "empty",
        onSave: (row, memo) => void saveMemo(row.ticker, memo),
      }),
      industryColumn<PlanRow>({ hide: !hasIndustryData }),
      {
        field: "change_pct",
        headerName: "일간(%)",
        width: 96,
        type: "numericColumn",
        valueFormatter: (p) => (p.value == null ? "-" : formatSignedPct(p.value as number, 2)),
        cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 600 }),
      },
      {
        field: "price",
        headerName: "현재가",
        width: 110,
        type: "numericColumn",
        headerTooltip: "이탈한 종목도 지금 시세다 — 판 뒤의 흐름을 청산가와 견줘 볼 수 있다.",
        valueFormatter: (p) => (p.value == null ? "-" : formatPrice(p.value as number, positions?.currency)),
      },
      tradeValueMultColumn<PlanRow>(),
      ...slotTradeColumns<PlanRow>({ fillDay }),
      // 이탈까지 남은 여유 — 둘 중 하나라도 0 이하가 되면 다음 거래일 시가에 판다.
      maExitGapColumn<PlanRow>({ field: "short_gap_pct", maDays: view?.settings.short_ma_days }),
      maExitGapColumn<PlanRow>({ field: "long_gap_pct", maDays: view?.settings.long_ma_days }),
    ],
    [
      fillDay,
      hasIndustryData,
      hasMarketCap,
      positions?.live,
      renderTicker,
      saveMemo,
      view?.settings.long_ma_days,
      view?.settings.short_ma_days,
    ],
  );

  // 진입 후보 표 — 자리가 나면 담을 순서. 상태는 「보유 중」과 「후보」 둘뿐이다.
  const candidateColumns = useMemo<ColDef<CandidateRow>[]>(
    () => [
      // 첫 컬럼은 보유 표의 '상태' 와 **같은 폭**이다 — 두 표가 위아래로 붙어 있어
      // 거래대금까지 칸이 어긋나면 읽기 어렵다. 이 표에 담긴 것은 전부 후보라 상태를
      // 따로 쓸 것이 없으므로 그 자리에 순위를 넣는다.
      {
        field: "rank",
        headerName: "순위",
        width: STATUS_COLUMN_WIDTH,
        minWidth: STATUS_COLUMN_MIN_WIDTH,
        headerTooltip: "장기 이격률이 큰 순 — 자리가 나면 이 순서로 담는다.",
        cellStyle: { display: "flex", alignItems: "center", justifyContent: "center" },
        valueFormatter: (p) => (p.value == null ? "-" : String(p.value)),
      },
      marketCapRankColumn<CandidateRow>("market_cap_rank", !hasMarketCap),
      highDrawdownColumn<CandidateRow>("high_drawdown_pct"),
      {
        field: "ticker",
        headerName: "티커",
        width: 108,
        cellRenderer: (p: { value?: string | null }) => renderTicker(p.value),
      },
      {
        field: "name",
        headerName: "종목명",
        // 고정 폭 — 보유 표와 후보 표의 앞쪽 칸(상태~거래대금)을 맞춘다.
        width: STOCK_NAME_COLUMN_WIDTH,
        cellRenderer: (p: { value?: string | null }) => renderStockNameCell(p.value),
      },
      stockMemoColumn<CandidateRow>({
        field: "memo",
        onSave: (row, memo) => void saveMemo(row.ticker, memo),
      }),
      industryColumn<CandidateRow>({ hide: !hasIndustryData }),
      {
        field: "change_pct",
        headerName: "일간(%)",
        width: 96,
        type: "numericColumn",
        valueFormatter: (p) => (p.value == null ? "-" : formatSignedPct(p.value as number, 2)),
        cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 600 }),
      },
      {
        field: "price",
        headerName: "현재가",
        width: 110,
        type: "numericColumn",
        valueFormatter: (p) => (p.value == null ? "-" : formatPrice(p.value as number, positions?.currency)),
      },
      tradeValueMultColumn<CandidateRow>(),
      maExitGapColumn<CandidateRow>({ field: "short_gap_pct", maDays: view?.settings.short_ma_days }),
      maExitGapColumn<CandidateRow>({ field: "long_gap_pct", maDays: view?.settings.long_ma_days }),
      {
        field: "market_cap",
        headerName: "시가총액",
        width: 128,
        hide: !hasMarketCap,
        type: "numericColumn",
        valueFormatter: (p) => (p.value == null ? "-" : formatMarketCapWon(p.value as number)),
      },
    ],
    [
      hasIndustryData,
      hasMarketCap,
      renderTicker,
      saveMemo,
      view?.settings.long_ma_days,
      view?.settings.short_ma_days,
    ],
  );

  // 표 헤더의 지수 이름 — 신고가·합성 화면과 같은 표기(값에 % 가 붙으므로 헤더엔 붙이지 않는다).
  const benchmarkLabel = backtest?.benchmark_name ?? "벤치마크";
  // ADR 컬럼 — 값이 하나도 없으면(레짐 시장 없는 풀) 컬럼을 숨긴다.
  const hasAdr = useMemo(() => (backtest?.daily ?? []).some((row) => row.adr != null), [backtest]);

  // 기간 표(연간·월간·주간·일간) — 신고가와 **같은 공용 계산**(`toPeriodRows`)을 쓴다.
  // 일별 누적 곡선 하나에서 잘라 내므로 기간이 달라도 합계가 어긋나지 않는다.
  const periodColumns = useMemo<ColDef<PeriodRow>[]>(
    () => [
      {
        field: "period",
        headerName:
          viewMode === "yearly" ? "연도" : viewMode === "monthly" ? "월" : viewMode === "weekly" ? "주" : "일자",
        width: 148,
        valueFormatter: (p) =>
          viewMode === "weekly" || viewMode === "daily"
            ? formatDateWithWeekday(String(p.value ?? ""))
            : String(p.value ?? ""),
      },
      {
        field: "strategy_pct",
        headerName: "전략",
        flex: 1,
        minWidth: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatSignedPct(p.value as number, 2),
        cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 700 }),
      },
      {
        field: "benchmark_pct",
        headerName: benchmarkLabel,
        flex: 1,
        minWidth: 110,
        type: "numericColumn",
        valueFormatter: (p) => formatSignedPct(p.value as number, 2),
        cellStyle: (p) => ({ color: signColor(p.value as number) }),
      },
      excessColumn<PeriodRow>(),
      // ADR — 일간=당일, 주간=판정일(주 마지막 거래일), 월간·연간=기간 최저. 신고가 표와 같다.
      viewMode === "daily"
        ? sharedAdrColumn<PeriodRow>({
            headerName: "ADR",
            headerTooltip: "그날의 시장 ADR(20일 등락비율)",
            hide: !hasAdr,
            getter: (row) => row.adr,
          })
        : viewMode === "weekly"
          ? sharedAdrColumn<PeriodRow>({
              headerName: "판정일 ADR",
              headerTooltip: "그 주 마지막 거래일의 시장 ADR — 다음 거래일 진입 게이트를 결정한 값",
              hide: !hasAdr,
              getter: (row) => row.adr,
            })
          : sharedAdrColumn<PeriodRow>({
              headerName: "최저 ADR",
              headerTooltip: `${viewMode === "yearly" ? "그 해" : "그 달"}의 시장 ADR 최저값`,
              hide: !hasAdr,
              getter: (row) => row.adr_min,
            }),
    ],
    [viewMode, benchmarkLabel, hasAdr],
  );

  const periodRows = useMemo<PeriodRow[]>(() => {
    if (!backtest) return [];
    if (viewMode === "weekly") return toPeriodRows(backtest.daily, weekKeyOf, true);
    const keyLength = viewMode === "yearly" ? 4 : viewMode === "monthly" ? 7 : viewMode === "daily" ? 10 : 0;
    return keyLength ? toPeriodRows(backtest.daily, (date) => date.slice(0, keyLength)) : [];
  }, [backtest, viewMode]);

  const tradeColumns = useMemo<ColDef<BacktestTradeRow>[]>(() => {
    const price = (value: unknown) => formatPrice(value as number, positions?.currency);
    return [
      { headerName: "티커", field: "ticker", width: 96 },
      {
        headerName: "종목명",
        field: "name",
        // 고정 폭 — 보유 표와 후보 표의 앞쪽 칸(상태~거래대금)을 맞춘다.
        width: STOCK_NAME_COLUMN_WIDTH,
        cellRenderer: (p: { value?: string | null }) => renderStockNameCell(p.value),
      },
      industryColumn<BacktestTradeRow>({ hide: !hasIndustryData }),
      { headerName: "편입일", field: "entry_date", width: 116 },
      { headerName: "매수가", field: "entry_price", width: 110, type: "numericColumn", valueFormatter: (p) => price(p.value) },
      { headerName: "청산일", field: "exit_date", width: 116 },
      { headerName: "청산가", field: "exit_price", width: 110, type: "numericColumn", valueFormatter: (p) => price(p.value) },
      {
        headerName: "수익률(%)",
        field: "return_pct",
        width: 116,
        type: "numericColumn",
        valueFormatter: (p) => (p.value == null ? "-" : formatSignedPct(p.value as number, 2)),
        cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 700 }),
      },
      { headerName: "보유일", field: "days", width: 84, type: "numericColumn" },
      { headerName: "사유", field: "reason", width: 110 },
    ];
  }, [hasIndustryData, positions?.currency]);

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
                {positions ? (
                  <span style={{ ...hintStyle, fontSize: "var(--fs-sm)" }}>
                    <b style={{ color: "inherit" }}>
                      보유 {heldCount} / {positions.top_n}
                    </b>{" "}
                    · 기준 {formatDateWithWeekday(positions.as_of)}
                    {planRows.some((row) => row.plan === "sell")
                      ? ` · ${fillDay} 매도 ${planRows.filter((row) => row.plan === "sell").length}`
                      : ""}
                    {positions.planned_entries.length ? ` · ${fillDay} 매수 ${positions.planned_entries.length}` : ""}
                    {positions.exited_today.length ? ` · 이탈 ${positions.exited_today.length}` : ""}
                    {positions.adr_gate ? (
                      positions.adr_gate.blocked ? (
                        <b style={{ color: "#d9480f" }}>
                          {" "}· ADR 게이트 발동 — {positions.adr_gate.market} {positions.adr_gate.value ?? "-"} &lt;{" "}
                          {positions.adr_gate.floor}, 오늘 신규 진입 없음
                        </b>
                      ) : (
                        <span title="하한 이상이면 빈 슬롯을 후보로 채운다. 미만이면 신규 진입만 멈추고 보유는 그대로 둔다.">
                          {" "}· ADR {positions.adr_gate.market} {positions.adr_gate.value ?? "-"} (하한{" "}
                          {positions.adr_gate.floor})
                        </span>
                      )
                    ) : null}
                  </span>
                ) : (
                  <span style={{ ...hintStyle, fontSize: "var(--fs-sm)" }}>
                    {pickFailed
                      ? "운용 현황을 불러오지 못했습니다. 설정을 저장하거나 새로고침하세요."
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
            {picking ? <AppLoadingProgress title="운용 현황 계산 중..." progress={pickProgress} /> : null}
            {positions && !picking && currentTab === "list" ? (
              <>
                <div style={{ ...hintStyle, fontWeight: 700, margin: "4px 0 6px" }}>보유 종목 ({heldCount}개)</div>
                {/* autoHeight — 그리드가 행 수만큼만 높이를 차지해 하단 낭비가 없다. */}
                <AppAgGrid<PlanRow>
                  rowData={planRows.map(withQuote)}
                  columnDefs={holdingColumns}
                  theme={gridTheme}
                  minHeight={0}
                  height="auto"
                  // 빈 슬롯은 값을 비우고, 실계좌 보유는 행 배경 녹색 — 시장 화면과 같은 표준.
                  getRowClass={(p) =>
                    p.data?.plan === "empty" ? "appEmptySlotRow" : p.data?.account_held ? "appHeldRow" : ""
                  }
                  gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                />
                <div style={{ ...hintStyle, fontWeight: 700, margin: "16px 0 6px" }}>
                  진입 후보 ({candidateRows.length}개)
                </div>
                <AppAgGrid<CandidateRow>
                  rowData={candidateRows}
                  columnDefs={candidateColumns}
                  theme={gridTheme}
                  minHeight={0}
                  height="auto"
                  // 실계좌 보유 종목은 행 배경 녹색 — 시장 화면과 같은 표준.
                  getRowClass={(p) => (p.data?.account_held ? "appHeldRow" : "")}
                  gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                />
              </>
            ) : null}
            {positions && !picking && currentTab === "chart" ? (
              <StrategyHoldingCharts
                charts={charts}
                loading={chartsLoading || saving || picking}
                error={chartsError}
                emptyMessage="보유 중인 종목이 없습니다."
                hint="장기선 위 & 단기선 위(자격)를 잃으면 다음 거래일 시가에 팔고, 진입일에 Buy 화살표가 표시됩니다."
                months={chartMonths}
                chartProps={(item) => {
                  const row = chartRows.find((candidate) => candidate.ticker === item.ticker);
                  return {
                    strategyLabel: "모멘텀",
                    entryDate: row?.entry_date ?? undefined,
                    returnPct: row?.return_pct,
                    days: row?.days ?? 0,
                    daysUnit: "일",
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
                    label: backtest.benchmark_name,
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
                  />
                ) : (
                  // 연간·월간·주간·일간은 같은 일별 곡선에서 잘라 낸다 — 기간이 달라도 합계가 맞는다.
                  <AppAgGrid<PeriodRow>
                    rowData={periodRows}
                    columnDefs={periodColumns}
                    theme={gridTheme}
                    minHeight={0}
                    height="auto"
                    gridOptions={{ domLayout: "autoHeight", suppressMovableColumns: true }}
                    getRowId={(p) => p.data.period}
                  />
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
          fixedLabel={`저장된 설정 기준 (종목풀 ${draftPool} · 종목 수 ${view.settings.top_n} 공통 고정 · 교체 규칙 자격 유지 · 주중 매도는 ADR 게이트만)`}
          current={{
            short_ma_days: view.settings.short_ma_days,
            long_ma_days: view.settings.long_ma_days,
            adr_floor: view.settings.adr_floor ?? null,
          }}
          axes={[
            // 축 값 = 상단 셀렉트 선택지(서버 상수) — 여기서 따로 정하지 않는다.
            // 종목 수(공통 고정)·업종 상한(폐기)·교체 규칙·주중 매도(전략 고정)는 축이 아니다
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
