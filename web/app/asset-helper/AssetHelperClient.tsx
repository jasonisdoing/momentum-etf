"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ColDef, GridOptions } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";
import { GridToolbarButton } from "../components/GridToolbarButton";
import { PageFrame } from "../components/PageFrame";
import { StableInlineInput } from "../components/StableInlineInput";
import { useAddingTickerRow } from "../components/useAddingTickerRow";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { AssetHelperBacktestResult, type LabResult } from "../components/AssetHelperBacktestResult";
import { BUCKET_THEME } from "@/lib/bucket-theme";
import { renderNameWithLeverageHighlight } from "@/lib/name-highlight";
import { reorderHoldings } from "@/lib/holdings-store";
import { fetchAlertBadges, normalizeBadgeTicker, type AlertBadges } from "@/lib/alert-badges";

function getBucketName(bucketId: number | undefined): string {
  return bucketId ? BUCKET_THEME[String(bucketId)]?.name ?? "-" : "-";
}
function getBucketCellClass(bucketId: number | undefined): string {
  return bucketId ? `rankBucketCell rankBucketCell${bucketId}` : "rankBucketCell";
}
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";
import {
  readRememberedMomentumEtfAccountId,
  writeRememberedMomentumEtfAccountId,
} from "../components/account-selection";

// 자산 헬퍼: 실제 계좌와 별개로, 계좌별 테스트 포트폴리오(수동 종목+비중)를 관리·백테스트한다.
// 저장/계산/백테스트는 기존 fixed 모드 엔진을 그대로 재활용한다.

// 종목 비중 합의 나머지는 현금으로 보유한다(백엔드 fixed 엔진이 __CASH__ = 1 - 합 으로 처리).
// 현금 행은 편집 가능하며, 종목합 + 현금 = 100% 여야 유효하다(백엔드는 종목만 저장, 현금=100-종목합 파생).
const CASH_TICKER = "__CASH__";

// 저장된 종목들의 고정비중 합의 나머지를 현금(%)으로 계산한다(로드 시 현금 초기화용).
function cashFromTickers(rows: Array<{ ticker: string; name?: string; fixed_weight_pct?: number | null }>): number {
  const sum = rows.reduce((acc, item) => acc + (item.ticker.trim() && item.name ? Number(item.fixed_weight_pct) || 0 : 0), 0);
  return Math.max(0, 100 - sum);
}

type MarketRegimeIndex = { ticker: string; name: string };
type AccountOption = { account_id: string; name: string; icon?: string; order?: number; market_regime_index?: MarketRegimeIndex | null };

// /market-trend 화면과 동일한 시장 레짐 데이터(계좌 연결 지수의 추세만 사용).
type MarketRegimeKey = "accel_up" | "accel_down";
type MarketTrendItem = {
  ticker: string;
  name: string;
  current_regime: MarketRegimeKey | null;
  current_regime_days: number | null;
  days_since_last_up: number | null;
  pct_from_high: number | null;
};

// 계좌별 최근 거래일 기간 수익률(입출금 제거). 1주/2주/3주/1달 = 5/10/15/21 거래일.
// series: [손익, 총자산] 최신순 — 임의 N거래일(예: 시장 레짐 지속일수) 수익률 계산용.
type AccountReturns = {
  d5: number | null;
  d10: number | null;
  d15: number | null;
  d21: number | null;
  d63: number | null;
  d126: number | null;
  d252: number | null;
  series?: [number, number][];
};
type AccountReturnKey = "d5" | "d10" | "d15" | "d21" | "d63" | "d126" | "d252";
const ACCOUNT_RETURN_PERIODS: { key: AccountReturnKey; label: string }[] = [
  { key: "d5", label: "1주" },
  { key: "d10", label: "2주" },
  { key: "d15", label: "3주" },
  { key: "d21", label: "1달" },
  { key: "d63", label: "3달" },
  { key: "d126", label: "6달" },
  { key: "d252", label: "1년" },
];

// 시장 상황 불렛 — /market-trend 의 「기간(거래일)」 컬럼과 동일한 문구·색상 규칙.
function formatMarketRegimeBullet(item: MarketTrendItem): { text: string; color: string } | null {
  const key = item.current_regime;
  if (!key) return null;
  if (key === "accel_up") {
    const d = item.current_regime_days;
    return { text: d != null ? `⬆️ 상승 ${d}일째` : "⬆️ 상승", color: "#d62828" };
  }
  const sinceUp = item.days_since_last_up;
  return {
    text: sinceUp != null ? `⬇️ 하락 ${sinceUp}일째` : "⬇️ 1년 내 상승 없음",
    color: "#1971c2",
  };
}

// 계좌 셀렉터 표기 — /pools-backtest 종목풀 셀렉터(formatPoolLabel)와 같은 조합 형식.
// `1. 💰 연금저축 계좌(pension_account)`. order/icon 이 없으면 그 부분만 빠진다.
function formatAccountLabel(acc: AccountOption): string {
  const name = String(acc.name ?? "").trim() || acc.account_id;
  const prefix = [
    acc.order === null || acc.order === undefined ? null : `${acc.order}.`,
    String(acc.icon ?? "").trim() || null,
  ]
    .filter(Boolean)
    .join(" ");
  const body = `${name}(${acc.account_id})`;
  return prefix ? `${prefix} ${body}` : body;
}

type HelperTicker = {
  ticker: string;
  name?: string;
  ticker_type?: string;
  country_code?: string;
  currency?: string;
  bucket?: number;
  fixed_weight_pct?: number | null;
  current_weight_pct?: number | null;
  // 미저장(임시) 행 — 확인만 된 상태. 저장 버튼에서 보유목록에 커밋한다(/assets 와 동일).
  pending?: boolean;
};

// 종목 목록의 소스는 자산 관리(보유 종목)다. 자산 헬퍼는 이 목록을 같은 순서로 보여주고 비중만 붙인다.
type HoldingRow = {
  ticker: string;
  name: string;
  bucket_id?: number;
  sort_order?: number;
  ticker_type?: string;
  country_code?: string;
  currency?: string;
  weight_pct?: number | null;
};

// 보유 티커의 시장 접두어(ASX:/KR: 등)를 제거한다. 가격 캐시·비중 계산은 접두어 없는 티커를 쓴다.
function stripMarketPrefix(ticker: string): string {
  return String(ticker ?? "").replace(/^[A-Za-z]+:/, "").trim().toUpperCase();
}

// 표시용 티커 — 호주(au)는 rankings/자산 관리와 동일하게 ASX: 접두어를 붙여 보여준다(내부 티커는 접두어 없음).
function displayTickerFor(ticker: string, countryCode?: string): string {
  return String(countryCode ?? "").toLowerCase() === "au" && !ticker.startsWith("ASX:") ? `ASX:${ticker}` : ticker;
}

// 보유 종목(순서 유지) + 저장된 비중 맵을 합쳐 자산 헬퍼용 종목 목록을 만든다.
function mergeHoldingsWithWeights(holdings: HoldingRow[], weightTickers: HelperTicker[] | undefined): HelperTicker[] {
  const weightMap: Record<string, number | null> = {};
  for (const t of weightTickers ?? []) {
    const tk = stripMarketPrefix(String(t.ticker ?? ""));
    if (tk) weightMap[tk] = t.fixed_weight_pct ?? null;
  }
  return holdings
    .filter((r) => {
      const tk = stripMarketPrefix(r.ticker);
      return tk && tk !== "IS"; // IS(International Shares)는 수동 고정자산이라 제외
    })
    .slice()
    .sort((a, b) => (a.sort_order ?? 0) - (b.sort_order ?? 0))
    .map((r) => {
      const tk = stripMarketPrefix(r.ticker);
      return {
        ticker: tk,
        name: r.name,
        ticker_type: r.ticker_type,
        country_code: r.country_code,
        currency: r.currency,
        bucket: r.bucket_id,
        fixed_weight_pct: weightMap[tk] ?? null,
        current_weight_pct: r.weight_pct ?? null,
      };
    });
}

type WeightRow = {
  ticker: string;
  current_price?: number | null;
  daily_change_pct?: number | null;
  return_1m_pct?: number | null;
  return_3m_pct?: number | null;
  return_6m_pct?: number | null;
  return_12m_pct?: number | null;
  mdd_pct?: number | null;
  sortino?: number | null;
};

type HelperSettings = Record<string, unknown> & { ACCOUNT_ID?: string; STOCK_MAX_WEIGHT?: number };

type GridRow = HelperTicker & { row_index: number; is_adding: boolean } & Partial<WeightRow>;

// 백테스트 기간(개월) 선택 옵션 — 백엔드 ALLOWED_BACKTEST_MONTHS 와 동일해야 한다(자유 입력 시 허용값 밖 에러 방지).
const BACKTEST_MONTH_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 36];

const REBALANCE_OPTIONS = [
  { value: "none", label: "리밸런싱 없음 (보유)" },
  { value: "weekly", label: "매주 (금요일)" },
  { value: "monthly", label: "매월 (말일)" },
  { value: "quarterly", label: "분기 (분기말)" },
  { value: "yearly", label: "매년 (연말)" },
];

function signColor(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "#475569";
  if (value > 0) return "#d63939";
  if (value < 0) return "#206bc4";
  return "#475569";
}

// 순위 화면과 동일한 색상 셀(양수 빨강/음수 파랑).
function renderPctCell(params: { value?: number | null }) {
  return <span style={{ color: signColor(params.value) }}>{fmtPct(params.value)}</span>;
}

const gridTheme = createAppGridTheme();

// 확정된 종목만 담는다(빈 행 없음). 신규 입력 행은 useAddingTickerRow 가 별도로 관리한다.
function buildRows(items: HelperTicker[] | undefined): HelperTicker[] {
  return (items ?? [])
    .filter((item) => (item?.ticker ?? "").trim() && (item?.name ?? "").trim())
    .map((item) => ({ ...item, ticker: item.ticker ?? "" }));
}

// 지표(금일/수익률/MDD/소르티노/현재가)를 계산해 티커→지표 맵으로 반환한다. 실패해도 빈 맵(그리드는 표시).
async function fetchMetrics(validList: HelperTicker[], settingsArg: HelperSettings): Promise<Record<string, WeightRow>> {
  try {
    const resp = await fetch("/api/asset-helper-settings/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tickers: validList,
        settings: settingsArg,
        weight_mode: "fixed",
        backtest_settings: { months: 12, rebalance: "none", initial_amount_manwon: 10000 },
      }),
    });
    const data = (await resp.json()) as { rows?: (WeightRow & { bucket?: unknown })[]; error?: string };
    if (!resp.ok || data.error) return {};
    const map: Record<string, WeightRow> = {};
    for (const row of data.rows ?? []) {
      map[row.ticker] = {
        ticker: row.ticker,
        current_price: row.current_price,
        daily_change_pct: row.daily_change_pct,
        return_1m_pct: row.return_1m_pct,
        return_3m_pct: row.return_3m_pct,
        return_6m_pct: row.return_6m_pct,
        return_12m_pct: row.return_12m_pct,
        mdd_pct: row.mdd_pct,
        sortino: row.sortino,
      };
    }
    return map;
  } catch {
    return {};
  }
}

function fmtPct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "-";
  return `${value.toFixed(2)}%`;
}

function fmtNum(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
}

// 현재가 — 행별 통화(currency)로 표기(자산 관리 formatPrice 와 동일 규칙).
// 혼합 환종 계좌에서 country_code 는 비어 올 수 있어 currency 를 우선 쓰고, 없으면 country_code 로 추정한다.
function fmtPrice(value: number | null | undefined, currency?: string, countryCode?: string): string {
  if (value == null || Number.isNaN(value)) return "-";
  let cur = String(currency ?? "").toUpperCase();
  if (!cur) {
    const cc = String(countryCode ?? "").toLowerCase();
    cur = cc === "us" ? "USD" : cc === "au" ? "AUD" : "KRW";
  }
  if (cur === "AUD") return `A$${new Intl.NumberFormat("en-AU", { minimumFractionDigits: 2, maximumFractionDigits: 4 }).format(value)}`;
  if (cur === "USD") return `$${new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 4 }).format(value)}`;
  return `${new Intl.NumberFormat("ko-KR").format(Math.round(value))}원`;
}

// /assets 메모와 동일한 저장시간 표기.
function formatNoteUpdatedAt(value: string | null): string {
  if (!value) return "아직 저장된 메모가 없습니다.";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

export function AssetHelperClient() {
  const toast = useToast();
  const [accounts, setAccounts] = useState<AccountOption[]>([]);
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null);
  // 드래그 순서 저장(onRowDragEnd)은 gridOptions(deps [])의 클로저에서 실행돼 최신 계좌가 필요하다.
  const selectedAccountRef = useRef<string | null>(null);
  useEffect(() => {
    selectedAccountRef.current = selectedAccount;
  }, [selectedAccount]);
  const [marketTrendItems, setMarketTrendItems] = useState<MarketTrendItem[]>([]);
  // 알람 배지(이동선 이탈·손절 아이콘) — /alarms 설정·판정 그대로. 보조 정보라 실패 시 빈 맵.
  const [alertBadges, setAlertBadges] = useState<AlertBadges>({});
  const [accountReturns, setAccountReturns] = useState<Record<string, AccountReturns>>({});
  const [memo, setMemo] = useState("");
  const [savedMemo, setSavedMemo] = useState("");
  const [noteSaving, setNoteSaving] = useState(false);
  const [noteUpdatedAt, setNoteUpdatedAt] = useState<string | null>(null);
  const [tickers, setTickers] = useState<HelperTicker[]>(() => buildRows(undefined));
  const [cashWeight, setCashWeight] = useState(100); // 현금 비중(%) — 편집 가능. 로드 시 100-종목합으로 초기화.
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]); // 삭제 선택(내부 티커=접두어 없음)
  const [settings, setSettings] = useState<HelperSettings | null>(null);
  const [metricByTicker, setMetricByTicker] = useState<Record<string, WeightRow>>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [btAmount, setBtAmount] = useState("10000");
  const [btMonths, setBtMonths] = useState("12");
  const [btRebalance, setBtRebalance] = useState("monthly");
  // 기간(개월) 옵션 — pools-backtest 와 동일하게 가격 캐시 데이터 길이 기준으로 서버에서 받는다.
  const [monthOptions, setMonthOptions] = useState<number[]>(BACKTEST_MONTH_OPTIONS);
  const [btResult, setBtResult] = useState<LabResult | null>(null);
  const [btRunning, setBtRunning] = useState(false);

  // 기간(개월) 옵션 로드 — pools-backtest 와 같은 엔드포인트/데이터 기준 재사용.
  useEffect(() => {
    let alive = true;
    void (async () => {
      try {
        const resp = await fetch("/api/pool-backtest/options", { cache: "no-store" });
        const data = (await resp.json()) as { month_options?: number[] };
        if (!resp.ok || !alive) return;
        const loaded = (data.month_options ?? []).filter((m) => Number.isFinite(m) && m > 0);
        if (loaded.length > 0) setMonthOptions(loaded);
      } catch {
        /* 실패 시 기본 옵션 유지 */
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const validTickers = useMemo(
    () => tickers.filter((item) => item.ticker.trim() && item.name),
    [tickers],
  );

  const load = useCallback(
    async (accountId?: string, opts?: { silent?: boolean }) => {
      // silent: 재로드 중 로딩 오버레이를 띄우지 않아 그리드가 깜빡이지 않는다(종목 추가/삭제 후 사용).
      try {
        if (!opts?.silent) setLoading(true);
        // 전체 계좌·이름은 account-settings 가 단일 소스다.
        const acctSettingsResp = await fetch("/api/account-settings", { cache: "no-store" });
        const acctSettingsData = (await acctSettingsResp.json()) as {
          accounts?: { account_id: string; name?: string; icon?: string; order?: number; market_regime_index?: MarketRegimeIndex | null }[];
          error?: string;
        };
        if (!acctSettingsResp.ok || acctSettingsData.error) {
          throw new Error(acctSettingsData.error ?? "계좌 목록을 불러오지 못했습니다.");
        }
        const accountList: AccountOption[] = (acctSettingsData.accounts ?? [])
          .map((acc) => ({
            account_id: acc.account_id,
            name: acc.name ?? acc.account_id,
            icon: acc.icon,
            order: acc.order,
            market_regime_index: acc.market_regime_index ?? null,
          }))
          .sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
        setAccounts(accountList);

        const allIds = accountList.map((a) => a.account_id);
        const remembered = readRememberedMomentumEtfAccountId();
        const target = accountId ?? (remembered && allIds.includes(remembered) ? remembered : allIds[0]);
        if (!target) throw new Error("등록된 계좌가 없습니다.");
        writeRememberedMomentumEtfAccountId(target);

        // 종목 목록은 자산 관리(보유 종목)가 소스. 비중은 자산 헬퍼 설정, 메모는 /api/note.
        const [settingsResp, noteResp, holdingsResp] = await Promise.all([
          fetch(`/api/asset-helper-settings?account_id=${encodeURIComponent(target)}`, { cache: "no-store" }),
          fetch(`/api/note?account=${encodeURIComponent(target)}`, { cache: "no-store" }),
          fetch(`/api/assets?account=${encodeURIComponent(target)}`, { cache: "no-store" }),
        ]);
        const data = (await settingsResp.json()) as {
          tickers?: HelperTicker[];
          settings?: HelperSettings;
          error?: string;
        };
        if (!settingsResp.ok || data.error) throw new Error(data.error ?? "포트폴리오를 불러오지 못했습니다.");
        const holdingsData = (await holdingsResp.json()) as { rows?: HoldingRow[]; error?: string };
        if (!holdingsResp.ok || holdingsData.error) throw new Error(holdingsData.error ?? "보유 종목을 불러오지 못했습니다.");
        const noteData = (await noteResp.json()) as { content?: string; updated_at?: string };
        const noteContent = noteResp.ok ? String(noteData.content ?? "") : "";
        setSelectedAccount(target);
        setMemo(noteContent);
        setSavedMemo(noteContent);
        setNoteUpdatedAt(noteResp.ok ? noteData.updated_at ?? null : null);
        const loadedTickers = mergeHoldingsWithWeights(holdingsData.rows ?? [], data.tickers);
        setTickers(loadedTickers);
        setCashWeight(cashFromTickers(loadedTickers)); // 저장된 종목합의 나머지를 현금으로 초기화
        setSettings(data.settings ?? null);
        // 지표까지 로드가 끝난 뒤 그리드를 한 번에 표시한다(종목 먼저 뜨고 값이 나중에 채워지는 깜빡임 방지).
        const validLoaded = loadedTickers.filter((t) => t.ticker.trim() && t.name);
        if (validLoaded.length >= 3 && data.settings?.ACCOUNT_ID) {
          setMetricByTicker(await fetchMetrics(validLoaded, data.settings));
        } else {
          setMetricByTicker({});
        }
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "불러오지 못했습니다.");
      } finally {
        if (!opts?.silent) setLoading(false);
      }
    },
    [toast],
  );

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 시장 상황 — 계좌 연결 지수의 추세만 보여준다(보조 정보라 실패해도 화면은 유지).
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const resp = await fetch("/api/market-trend", { cache: "no-store" });
        const payload = (await resp.json()) as { items?: MarketTrendItem[] };
        if (alive && resp.ok) setMarketTrendItems(payload.items ?? []);
      } catch {
        /* 시장 추세는 보조 정보 — 실패 시 무시 */
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // 종목명 알람 배지 — 선택 계좌의 이동선 이탈·손절 트리거를 티커→아이콘 맵으로 로드.
  useEffect(() => {
    if (!selectedAccount) {
      setAlertBadges({});
      return;
    }
    let alive = true;
    void fetchAlertBadges(selectedAccount).then((badges) => {
      if (alive) setAlertBadges(badges);
    });
    return () => {
      alive = false;
    };
  }, [selectedAccount]);

  // 계좌별 기간 수익률(스냅샷 기반, 보조 정보라 실패해도 화면 유지).
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const resp = await fetch("/api/account-returns", { cache: "no-store" });
        const payload = (await resp.json()) as { returns?: Record<string, AccountReturns> };
        if (alive && resp.ok) setAccountReturns(payload.returns ?? {});
      } catch {
        /* 보조 정보 — 실패 시 무시 */
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const selectedReturns = selectedAccount ? accountReturns[selectedAccount] : undefined;

  // 선택 계좌에 연결된 시장 레짐 지수 + 그 추세 불렛.
  const marketRegime = useMemo(() => {
    const index = accounts.find((a) => a.account_id === selectedAccount)?.market_regime_index;
    if (!index?.ticker) return null;
    const item = marketTrendItems.find((it) => it.ticker === index.ticker);
    const bullet = item ? formatMarketRegimeBullet(item) : null;
    // 레짐 지속일수(불렛의 "N일째"와 동일 기준): 상승=current_regime_days, 하락=days_since_last_up.
    const regimeDays = item
      ? item.current_regime === "accel_up"
        ? item.current_regime_days
        : item.days_since_last_up
      : null;
    return { indexName: index.name || index.ticker, bullet, pctFromHigh: item?.pct_from_high ?? null, regimeDays };
  }, [accounts, selectedAccount, marketTrendItems]);

  // 시장 지수가 현재 레짐인 기간(regimeDays 거래일) 동안의 계좌 수익률.
  const regimeAccountReturn = useMemo(() => {
    const n = marketRegime?.regimeDays;
    const series = selectedReturns?.series;
    if (n == null || !series || series.length <= n) return null;
    const [profitThen, totalThen] = series[n];
    if (totalThen <= 0) return null;
    return ((series[0][0] - profitThen) / totalThen) * 100;
  }, [marketRegime, selectedReturns]);

  // 종목 추가 — /assets 와 동일한 addingRow 흐름을 공용 훅(useAddingTickerRow)으로 재사용한다.
  // 추가 → 상단 신규 행에 티커 입력 → 확인(조회) → 목록에 추가. 저장은 그리드 저장 버튼으로.
  const addResolve = useCallback(
    async (raw: string): Promise<HelperTicker & { name: string }> => {
      if (tickers.some((item) => item.ticker.trim().toUpperCase() === raw && item.name)) {
        throw new Error("이미 등록된 종목입니다.");
      }
      const accountParam = selectedAccount ? `&account_id=${encodeURIComponent(selectedAccount)}` : "";
      const resp = await fetch(`/api/ticker-resolve?ticker=${encodeURIComponent(raw)}${accountParam}`);
      const data = (await resp.json()) as HelperTicker & { name?: string; error?: string; detail?: string };
      if (!resp.ok || data.error || !data.name) {
        throw new Error(data.error ?? data.detail ?? "존재하지 않는 티커입니다.");
      }
      return {
        ticker: data.ticker ?? raw,
        name: data.name,
        ticker_type: data.ticker_type,
        country_code: data.country_code,
        bucket: data.bucket,
      };
    },
    [tickers, selectedAccount],
  );
  // /assets 와 동일: 확인 시 즉시 저장하지 않고 상단에 '미저장(pending)' 행으로 둔다.
  // 실제 보유목록 등록·비중 저장은 '저장' 버튼에서 한 번에 처리하고, 저장 전 새로고침하면 사라진다.
  const addOnValidated = useCallback(
    (resolved: HelperTicker) => {
      const key = resolved.ticker.trim().toUpperCase();
      // 종목풀에 없는 티커는 resolve 가 이름 대신 티커를 돌려준다 — 그 경우 티커를 이름으로 쓴다(저장 후 실제 이름 확정).
      const nm = resolved.name && resolved.name.trim() ? resolved.name.trim() : resolved.ticker;
      setTickers((cur) =>
        cur.some((t) => t.ticker.trim().toUpperCase() === key)
          ? cur
          : [{ ...resolved, name: nm, pending: true, fixed_weight_pct: null }, ...cur],
      );
      toast.success(`조회 성공: ${nm}`); // /assets 와 동일한 확인 성공 안내
    },
    [toast],
  );
  const addOnError = useCallback((message: string) => toast.error(message), [toast]);
  const add = useAddingTickerRow({
    resolve: addResolve,
    onValidated: addOnValidated,
    onError: addOnError,
    normalize: (raw) => raw.trim().toUpperCase(),
    resetOnValidated: true,
  });

  // 선택 종목을 보유 목록에서 삭제한다(공통 목록이라 자산 관리 보유에서도 사라짐 — 수량·비중 함께 삭제).
  const handleDelete = async () => {
    if (!selectedTickers.length || !selectedAccount) return;
    if (!window.confirm(`${selectedTickers.length}개 종목을 보유 목록에서 삭제할까요? (수량·비중 함께 삭제)`)) return;
    try {
      setDeleting(true);
      for (const bare of selectedTickers) {
        const item = tickers.find((t) => t.ticker === bare);
        if (item?.pending) continue; // 미저장 행은 보유목록에 없으니 로컬 제거만(아래).
        const delTicker = displayTickerFor(bare, item?.country_code); // 보유 저장 형식(au는 ASX: 접두어)
        const resp = await fetch(
          `/api/assets?account=${encodeURIComponent(selectedAccount)}&ticker=${encodeURIComponent(delTicker)}`,
          { method: "DELETE" },
        );
        const data = (await resp.json()) as { error?: string };
        if (!resp.ok || data.error) throw new Error(data.error ?? "삭제에 실패했습니다.");
      }
      // 티커만 제거한다 — 남은 종목의 비중과 현금 비중은 건드리지 않는다.
      // (재로드하면 현금을 100−종목합으로 다시 계산해 제거분이 현금으로 채워지므로 재로드도 하지 않는다.)
      setTickers((cur) => cur.filter((t) => !selectedTickers.includes(t.ticker)));
      setSelectedTickers([]);
      toast.success("삭제 완료");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "삭제에 실패했습니다.");
    } finally {
      setDeleting(false);
    }
  };


  // 메모는 /assets 와 같은 계좌 메모(/api/note) — 양쪽에서 편집·저장하면 서로 공유된다.
  const handleSaveNote = async () => {
    if (!selectedAccount) {
      toast.error("계좌를 선택해주세요.");
      return;
    }
    try {
      setNoteSaving(true);
      const resp = await fetch("/api/note", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account_id: selectedAccount, content: memo }),
      });
      const data = (await resp.json()) as { updated_at?: string; error?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? "메모 저장에 실패했습니다.");
      setSavedMemo(memo);
      setNoteUpdatedAt(data.updated_at ?? null);
      toast.success("메모 저장 완료");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "메모 저장에 실패했습니다.");
    } finally {
      setNoteSaving(false);
    }
  };

  // 종목 + 현금 = 100% 여야 저장·백테스트 가능. 아니면 명시적 에러(현금으로 자동 보정하지 않음).
  const ensureWeightSum100 = useCallback((): boolean => {
    const sum = validTickers.reduce((acc, item) => acc + (Number(item.fixed_weight_pct) || 0), 0);
    if (Math.abs(sum + cashWeight - 100) >= 0.05) {
      toast.error(`종목 ${sum.toFixed(1)}% + 현금 ${cashWeight.toFixed(1)}% = ${(sum + cashWeight).toFixed(1)}%. 합계를 100%로 맞춰주세요.`);
      return false;
    }
    return true;
  }, [validTickers, cashWeight, toast]);

  const saveSettings = async () => {
    if (!selectedAccount || !settings) {
      toast.error("계좌를 선택해주세요.");
      return;
    }
    const unresolved = tickers.filter((item) => item.ticker.trim() && !item.name);
    if (unresolved.length > 0) {
      toast.error(`확인되지 않은 티커가 ${unresolved.length}개 있습니다. "확인"으로 조회해주세요.`);
      return;
    }
    if (validTickers.length === 0) {
      toast.error("확인된 종목이 1개 이상 필요합니다.");
      return;
    }
    if (!ensureWeightSum100()) return;
    try {
      setSaving(true);
      // 1) 미저장(pending) 종목을 보유목록에 커밋(수량 0) — /assets 처럼 '저장' 시점에 등록한다.
      const pendings = tickers.filter((t) => t.pending && t.ticker.trim());
      for (const p of pendings) {
        const addResp = await fetch("/api/assets", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ account_id: selectedAccount, ticker: p.ticker, quantity: 0, average_buy_price: 0, target_ratio: 0 }),
        });
        const addData = (await addResp.json()) as { error?: string };
        if (!addResp.ok || addData.error) throw new Error(addData.error ?? `${p.ticker} 추가에 실패했습니다.`);
      }
      // 2) 비중 저장(종목은 보유목록이 소스, 여기선 비중만).
      const resp = await fetch("/api/asset-helper-settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          account_id: selectedAccount,
          tickers: validTickers,
          weight_mode: "fixed",
          settings,
        }),
      });
      const data = (await resp.json()) as { tickers?: HelperTicker[]; error?: string; detail?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? data.detail ?? "저장에 실패했습니다.");
      await load(selectedAccount ?? undefined); // 보유 목록 순서 기준으로 다시 병합(pending 반영 + 정렬)
      toast.success("저장 완료");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "저장에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  };

  const runBacktest = async () => {
    if (!settings || !settings.ACCOUNT_ID) {
      toast.error("계좌를 선택해주세요.");
      return;
    }
    if (validTickers.length < 3) {
      toast.error("백테스트에는 확인된 종목이 3개 이상 필요합니다.");
      return;
    }
    // 종목 + 현금 = 100% 여야 한다(백엔드는 종목만 저장, 현금=100-종목합 파생).
    if (!ensureWeightSum100()) return;
    if (!Number.isInteger(Number(btAmount)) || Number(btAmount) <= 0) {
      toast.error("최초 금액(만원)은 1 이상의 정수여야 합니다.");
      return;
    }
    try {
      setBtRunning(true);
      setBtResult(null);
      const resp = await fetch("/api/asset-helper-settings/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: validTickers,
          settings,
          weight_mode: "fixed",
          backtest_settings: { months: Number(btMonths), rebalance: btRebalance, initial_amount_manwon: Number(btAmount) },
        }),
      });
      const data = (await resp.json()) as LabResult;
      if (!resp.ok || data.error) throw new Error(data.error ?? data.detail ?? "백테스트에 실패했습니다.");
      setBtResult(data);
      toast.success("백테스트 완료");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "백테스트에 실패했습니다.");
    } finally {
      setBtRunning(false);
    }
  };

  const weightSum = validTickers.reduce((acc, item) => acc + (Number(item.fixed_weight_pct) || 0), 0);
  const totalWeight = weightSum + cashWeight; // 종목합 + 현금
  const weightOk = Math.abs(totalWeight - 100) < 0.05; // 종목 + 현금 = 100% 여야 유효

  const gridRows = useMemo<GridRow[]>(() => {
    // 맨 위 고정 현금 행(읽기전용, row_index=-1 로 종목·추가행과 구분).
    const cashRow: GridRow = { ticker: CASH_TICKER, name: "현금", row_index: -1, is_adding: false, fixed_weight_pct: cashWeight };
    const committed = tickers.map((item, idx) => ({
      ...item,
      ...(metricByTicker[item.ticker.trim().toUpperCase()] ?? {}),
      row_index: idx + 1,
      is_adding: false,
    }));
    const addingRows: GridRow[] = add.addingRow
      ? [{ ticker: add.addingRow.ticker, name: add.addingRow.name, row_index: 0, is_adding: true }]
      : [];
    return [cashRow, ...addingRows, ...committed];
  }, [tickers, metricByTicker, add.addingRow, cashWeight]);

  const columnDefs = useMemo<ColDef<GridRow>[]>(
    () => [
      {
        colId: "drag",
        headerName: "",
        width: 36,
        maxWidth: 36,
        pinned: "left",
        sortable: false,
        resizable: false,
        suppressMovable: true,
        // 현금 고정행·추가행은 드래그 불가. 확정된 종목만 순서 변경.
        rowDrag: (params) => Boolean(params.data && !params.data.is_adding && params.data.ticker !== CASH_TICKER),
        cellClass: "assetsDragCell",
        valueGetter: () => "",
      },
      {
        colId: "bucket",
        headerName: "버킷",
        minWidth: 108,
        width: 108,
        pinned: "left",
        valueGetter: (params) => getBucketName(params.data?.bucket),
        cellClass: (params) => getBucketCellClass(params.data?.bucket),
        cellRenderer: (params: { data?: GridRow }) => <span>{getBucketName(params.data?.bucket)}</span>,
      },
      {
        field: "ticker",
        headerName: "티커",
        minWidth: 110,
        width: 110,
        pinned: "left",
        cellRenderer: (params: { data?: GridRow; value?: string }) => {
          const row = params.data;
          if (!row) return "-";
          if (row.ticker === CASH_TICKER) return <span>-</span>;
          if (row.is_adding) {
            return (
              <StableInlineInput
                className="form-control form-control-sm assetsInlineInput assetsInlineInputTicker"
                placeholder="티커"
                initialValue={row.ticker}
                disabled={add.addingRow?.isValidating}
                submitOnBlur={false}
                onChange={(value) => add.setTicker(value)}
                onSave={(value) => void add.validate(value)}
              />
            );
          }
          {
            const dt = displayTickerFor(row.ticker, row.country_code);
            return <TickerDetailLink ticker={dt} displayTicker={dt} />;
          }
        },
      },
      {
        field: "name",
        headerName: "종목명",
        minWidth: 220,
        flex: 1,
        cellRenderer: (params: { data?: GridRow; value?: string }) => {
          const row = params.data;
          if (!row) return null;
          if (row.is_adding) {
            return (
              <div className="assetsNameLookup">
                <span className="assetsNameLookupStatus">
                  {add.addingRow?.isValidating ? "조회 중…" : "티커를 입력한 뒤 확인하세요."}
                </span>
                <button
                  type="button"
                  className="btn btn-outline-primary btn-sm assetsInlineButton d-inline-flex align-items-center gap-1"
                  disabled={add.addingRow?.isValidating}
                  onClick={() => void add.validate()}
                >
                  확인
                </button>
              </div>
            );
          }
          const badge = alertBadges[normalizeBadgeTicker(row.ticker)] ?? "";
          return (
            <span className="rankNameCellText" title={params.value ?? ""}>
              {renderNameWithLeverageHighlight(String(params.value ?? ""))}
              {badge ? <span> {badge}</span> : null}
            </span>
          );
        },
      },
      { field: "daily_change_pct", headerName: "일간", minWidth: 88, width: 88, type: "rightAligned", cellRenderer: renderPctCell },
      {
        field: "current_price",
        headerName: "현재가",
        minWidth: 96,
        width: 96,
        type: "rightAligned",
        valueFormatter: (p) => fmtPrice(p.value as number | null, p.data?.currency, p.data?.country_code),
      },
      { field: "return_1m_pct", headerName: "1달", minWidth: 84, width: 84, type: "rightAligned", cellRenderer: renderPctCell },
      { field: "return_3m_pct", headerName: "3달", minWidth: 84, width: 84, type: "rightAligned", cellRenderer: renderPctCell },
      { field: "return_6m_pct", headerName: "6달", minWidth: 84, width: 84, type: "rightAligned", cellRenderer: renderPctCell },
      { field: "return_12m_pct", headerName: "1년", minWidth: 84, width: 84, type: "rightAligned", cellRenderer: renderPctCell },
      { field: "mdd_pct", headerName: "MDD", minWidth: 84, width: 84, type: "rightAligned", cellRenderer: renderPctCell },
      { field: "sortino", headerName: "Sortino", minWidth: 90, width: 90, type: "rightAligned", valueFormatter: (p) => fmtNum(p.value as number | null) },
      {
        field: "current_weight_pct",
        headerName: "현재 비중",
        minWidth: 92,
        width: 92,
        type: "rightAligned",
        valueFormatter: (p) => (p.value == null || p.value === "" ? "-" : `${Number(p.value).toFixed(1)}`),
      },
      {
        field: "fixed_weight_pct",
        headerName: "비중",
        minWidth: 92,
        width: 92,
        type: "rightAligned",
        editable: (params) => Boolean(params.data && !params.data.is_adding),
        valueFormatter: (p) => (p.value == null || p.value === "" ? "-" : `${Number(p.value).toFixed(1)}`),
        cellClass: "assetsEditableCell",
      },
    ],
    // add(useAddingTickerRow) 는 매 렌더 새로 생성되므로 의존성에 포함해
    // 낡은 클로저가 빈 상태를 읽어 "티커를 입력해주세요"가 뜨는 것을 막는다.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [add, alertBadges],
  );

  const gridOptions = useMemo<GridOptions<GridRow>>(
    () => ({
      domLayout: "autoHeight",
      suppressMovableColumns: true,
      stopEditingWhenCellsLoseFocus: true,
      animateRows: true,
      // /assets 와 동일한 임시(미저장) 행 강조 — 같은 CSS 클래스(assetsAddingRow)를 재사용한다.
      rowClassRules: {
        assetsAddingRow: (params: { data?: GridRow }) => Boolean(params.data?.is_adding || params.data?.pending),
      },
      // 드래그로 종목 순서 변경(/assets 와 동일). 순서의 단일 소스는 holdings 라
      // 드래그 즉시 holdings sort_order 를 저장한다(안 하면 재조회 시 holdings 순서로 되돌아감).
      rowDragManaged: true,
      onRowDragEnd: (params) => {
        const orderedTickers: string[] = [];
        params.api.forEachNode((node) => {
          const row = node.data;
          if (row && !row.is_adding && row.ticker && row.ticker !== CASH_TICKER) {
            orderedTickers.push(row.ticker.trim().toUpperCase());
          }
        });
        if (!orderedTickers.length) return;
        setTickers((current) => {
          const byTicker = new Map(current.map((item) => [item.ticker.trim().toUpperCase(), item]));
          const seen = new Set<string>();
          const reordered: HelperTicker[] = [];
          for (const key of orderedTickers) {
            const item = byTicker.get(key);
            if (item && !seen.has(key)) {
              seen.add(key);
              reordered.push(item);
            }
          }
          // 순서 목록에 없는 항목(방어)은 기존 순서로 뒤에 붙인다.
          const rest = current.filter((item) => !seen.has(item.ticker.trim().toUpperCase()));
          return [...reordered, ...rest];
        });
        const accountId = selectedAccountRef.current;
        if (accountId) {
          void reorderHoldings(accountId, orderedTickers).catch((err) =>
            toast.error(err instanceof Error ? err.message : "순서 저장에 실패했습니다."),
          );
        }
      },
      rowSelection: {
        mode: "multiRow",
        checkboxes: (params) => Boolean(params.data && !params.data.is_adding && params.data.ticker !== CASH_TICKER),
        headerCheckbox: true,
        hideDisabledCheckboxes: true,
        enableClickSelection: false,
      },
      selectionColumnDef: { width: 52, minWidth: 52, maxWidth: 52, pinned: "left", sortable: false, resizable: false, headerName: "", cellClass: "assetsSelectCell" },
      onSelectionChanged: (params) => {
        setSelectedTickers(
          params.api
            .getSelectedRows()
            .filter((row) => row.ticker && row.ticker !== CASH_TICKER && !row.is_adding)
            .map((row) => row.ticker),
        );
      },
      onCellValueChanged: (params) => {
        if (params.colDef.field !== "fixed_weight_pct") return;
        if (params.data.ticker === CASH_TICKER) {
          const raw = params.newValue === "" || params.newValue == null ? 0 : Number(params.newValue);
          setCashWeight(Number.isFinite(raw) ? Math.max(0, raw) : 0);
          return;
        }
        const newWeight = params.newValue === "" || params.newValue == null ? null : Number(params.newValue);
        const index = params.data.row_index - 1;
        setTickers((current) =>
          current.map((item, i) => (i === index ? { ...item, fixed_weight_pct: newWeight } : item)),
        );
      },
    }),
    [toast],
  );

  return (
    <PageFrame title="자산 헬퍼">
      <div className="appPageStack">
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">계좌</span>
                  <select
                    className="form-select form-select-sm"
                    value={selectedAccount ?? ""}
                    disabled={loading || accounts.length === 0}
                    onChange={(event) => void load(event.target.value)}
                  >
                    {accounts.length === 0 ? (
                      <option value="">계좌 불러오는 중...</option>
                    ) : (
                      accounts.map((a) => (
                        <option key={a.account_id} value={a.account_id}>
                          {formatAccountLabel(a)}
                        </option>
                      ))
                    )}
                  </select>
                </label>
              </div>
            </div>
          </div>
        </div>

        <div className="assetHelperMemoLayout">
          <div className="card appCard">
            <div className="card-body">
              <h2 style={{ fontSize: "1.0rem", fontWeight: 800, marginBottom: 6 }}>시장 상황</h2>
              <div style={{ minHeight: "16rem" }}>
                <ul style={{ margin: 0, paddingLeft: "1.1rem", fontSize: "0.9rem", lineHeight: 1.6 }}>
                  {marketRegime ? (
                    <li>
                      <span style={{ fontWeight: 700 }}>{marketRegime.indexName}</span>{" "}
                      {marketRegime.bullet ? (
                        <span style={{ color: marketRegime.bullet.color, fontWeight: 700 }}>
                          {marketRegime.bullet.text}
                          {marketRegime.pctFromHigh != null ? `(${marketRegime.pctFromHigh.toFixed(2)}%)` : ""}
                        </span>
                      ) : (
                        <span style={{ color: "var(--text-muted)" }}>추세 데이터 없음</span>
                      )}
                    </li>
                  ) : null}
                  {marketRegime?.regimeDays != null && regimeAccountReturn != null ? (
                    <li>
                      같은 기간({marketRegime.regimeDays}거래일) 동안 계좌{" "}
                      <span style={{ color: signColor(regimeAccountReturn), fontWeight: 700 }}>
                        {regimeAccountReturn > 0 ? "⬆️ +" : "⬇️ "}
                        {regimeAccountReturn.toFixed(1)}%
                      </span>
                    </li>
                  ) : null}
                  <li>
                    <span style={{ fontWeight: 700, marginRight: 6 }}>계좌 수익률</span>
                    {selectedReturns ? (
                      ACCOUNT_RETURN_PERIODS.map((p, i) => {
                        const v = selectedReturns[p.key];
                        return (
                          <span key={p.key}>
                            {i > 0 ? <span style={{ color: "var(--text-muted)" }}> · </span> : null}
                            {p.label}{" "}
                            <span style={{ color: signColor(v), fontWeight: 700 }}>
                              {v == null ? "-" : `${v > 0 ? "+" : ""}${v.toFixed(1)}%`}
                            </span>
                          </span>
                        );
                      })
                    ) : (
                      <span style={{ color: "var(--text-muted)" }}>이력 부족</span>
                    )}
                  </li>
                </ul>
              </div>
            </div>
          </div>
          <div className="card appCard">
            <div className="card-body">
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 6 }}>
                <span className="text-muted small">메모 저장: {formatNoteUpdatedAt(noteUpdatedAt)}</span>
                <GridToolbarButton
                  variant="save"
                  disabled={noteSaving || memo === savedMemo}
                  onClick={() => void handleSaveNote()}
                >
                  {noteSaving ? "저장 중..." : "메모 저장"}
                </GridToolbarButton>
              </div>
              <textarea
                className="form-control"
                style={{ fontSize: "0.9rem", minHeight: "16rem" }}
                rows={12}
                placeholder="이 계좌에 대한 투자 전략이나 주의사항을 메모하세요. AI가 요약할 때 함께 참고합니다."
                value={memo}
                onChange={(e) => setMemo(e.target.value)}
              />
            </div>
          </div>
          <style jsx>{`
            .assetHelperMemoLayout {
              display: grid;
              grid-template-columns: minmax(0, 2fr) minmax(0, 3fr);
              gap: 12px;
              align-items: stretch;
            }
            @media (max-width: 900px) {
              .assetHelperMemoLayout {
                grid-template-columns: minmax(0, 1fr);
              }
            }
          `}</style>
        </div>

        <div className="card appCard">
          <div className="card-body">
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 8 }}>
              <div>
                <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>종목</h2>
                <p style={{ margin: 0, fontSize: "0.9rem", fontWeight: 700, color: weightOk ? "#16a34a" : "#dc2626" }}>
                  종목 {weightSum.toFixed(1)}% + 현금 {cashWeight.toFixed(1)}% = {totalWeight.toFixed(1)}%{weightOk ? " ✓" : " (100% 필요)"}
                </p>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <GridToolbarButton variant="add" onClick={() => add.start()} disabled={Boolean(add.addingRow)} />
                <GridToolbarButton variant="save" disabled={saving} onClick={() => void saveSettings()}>
                  {saving ? "저장 중..." : "저장"}
                </GridToolbarButton>
                <GridToolbarButton variant="delete" onClick={() => void handleDelete()} disabled={!selectedTickers.length || deleting}>
                  {deleting ? "삭제 중..." : undefined}
                </GridToolbarButton>
              </div>
            </div>
            <AppAgGrid<GridRow>
              className="rankAgGrid assetsAgGrid"
              rowData={gridRows}
              columnDefs={columnDefs}
              loading={loading || deleting}
              minHeight={Math.max(180, Math.min(gridRows.length, 12) * 34 + 36)}
              gridOptions={gridOptions}
              theme={gridTheme}
              getRowId={(params) =>
                params.data.is_adding ? "__adding__" : String(params.data.ticker || `row_${params.data.row_index}`)
              }
            />
          </div>
        </div>

        <div className="card appCard">
          <div className="card-body">
            <div style={{ display: "flex", alignItems: "flex-end", gap: 12, flexWrap: "wrap", marginBottom: 8 }}>
              <label className="appLabeledField">
                <span className="appLabeledFieldLabel">최초 금액(만원)</span>
                <input type="number" min={1} className="form-control form-control-sm" value={btAmount} onChange={(e) => setBtAmount(e.target.value)} />
              </label>
              <label className="appLabeledField">
                <span className="appLabeledFieldLabel">기간(개월)</span>
                <select className="form-select form-select-sm" value={btMonths} onChange={(e) => setBtMonths(e.target.value)}>
                  {monthOptions.map((m) => (
                    <option key={m} value={m}>
                      최근 {m}개월
                    </option>
                  ))}
                </select>
              </label>
              <label className="appLabeledField">
                <span className="appLabeledFieldLabel">리밸런싱</span>
                <select className="form-select form-select-sm" value={btRebalance} onChange={(e) => setBtRebalance(e.target.value)}>
                  {REBALANCE_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </label>
              <button type="button" className="btn btn-sm btn-outline-dark" disabled={btRunning} onClick={() => void runBacktest()}>
                {btRunning ? "백테스트 중..." : "백테스트"}
              </button>
            </div>

            {btResult ? (
              <AssetHelperBacktestResult result={btResult} />
            ) : (
              <div style={{ color: "var(--text-muted)", fontSize: "0.9rem", padding: "6px 0" }}>백테스트 버튼을 눌러 결과를 확인하세요.</div>
            )}
          </div>
        </div>
      </div>
    </PageFrame>
  );
}
