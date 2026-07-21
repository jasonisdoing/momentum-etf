"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ColDef, GridOptions } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";
import { GridToolbarButton } from "../components/GridToolbarButton";
import { PageFrame } from "../components/PageFrame";
import { StableInlineInput } from "../components/StableInlineInput";
import { useAddingTickerRow } from "../components/useAddingTickerRow";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { TopPickBacktestResult, type LabResult } from "../components/TopPickBacktestResult";
import { BUCKET_THEME } from "@/lib/bucket-theme";
import { renderNameWithLeverageHighlight } from "@/lib/name-highlight";

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
// 저장/계산/백테스트는 기존 top_pick_settings(fixed 모드) 엔진을 그대로 재활용한다.

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
  bucket?: number;
  fixed_weight_pct?: number | null;
};

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

// 원본 top-pick 과 동일한 색상 셀(양수 빨강/음수 파랑).
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

function fmtPct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "-";
  return `${value.toFixed(2)}%`;
}

function fmtNum(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
}

// 현재가 — 통화 단위가 섞여 있어(원/달러/호주달러) 숫자만 천단위 구분해 표기한다.
function fmtPrice(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 2 }).format(value);
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
  const [marketTrendItems, setMarketTrendItems] = useState<MarketTrendItem[]>([]);
  const [accountReturns, setAccountReturns] = useState<Record<string, AccountReturns>>({});
  const [memo, setMemo] = useState("");
  const [savedMemo, setSavedMemo] = useState("");
  const [noteSaving, setNoteSaving] = useState(false);
  const [noteUpdatedAt, setNoteUpdatedAt] = useState<string | null>(null);
  const [tickers, setTickers] = useState<HelperTicker[]>(() => buildRows(undefined));
  const [settings, setSettings] = useState<HelperSettings | null>(null);
  const [metricByTicker, setMetricByTicker] = useState<Record<string, WeightRow>>({});
  const [selectedRowIndexes, setSelectedRowIndexes] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [running, setRunning] = useState(false);
  const [btAmount, setBtAmount] = useState("10000");
  const [btMonths, setBtMonths] = useState("12");
  const [btRebalance, setBtRebalance] = useState("monthly");
  const [btResult, setBtResult] = useState<LabResult | null>(null);
  const [btRunning, setBtRunning] = useState(false);


  const validTickers = useMemo(
    () => tickers.filter((item) => item.ticker.trim() && item.name),
    [tickers],
  );

  const load = useCallback(
    async (accountId?: string) => {
      try {
        setLoading(true);
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

        // 상단 메모는 /assets 와 동일한 계좌 메모(/api/note).
        const [settingsResp, noteResp] = await Promise.all([
          fetch(`/api/top-pick-settings?account_id=${encodeURIComponent(target)}`, { cache: "no-store" }),
          fetch(`/api/note?account=${encodeURIComponent(target)}`, { cache: "no-store" }),
        ]);
        const data = (await settingsResp.json()) as {
          tickers?: HelperTicker[];
          settings?: HelperSettings;
          error?: string;
        };
        if (!settingsResp.ok || data.error) throw new Error(data.error ?? "포트폴리오를 불러오지 못했습니다.");
        const noteData = (await noteResp.json()) as { content?: string; updated_at?: string };
        const noteContent = noteResp.ok ? String(noteData.content ?? "") : "";
        setSelectedAccount(target);
        setMemo(noteContent);
        setSavedMemo(noteContent);
        setNoteUpdatedAt(noteResp.ok ? noteData.updated_at ?? null : null);
        setTickers(buildRows(data.tickers));
        setSettings(data.settings ?? null);
        setMetricByTicker({});
        setSelectedRowIndexes([]);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "불러오지 못했습니다.");
      } finally {
        setLoading(false);
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

  // 지표(금일/수익률/MDD/소르티노)는 계좌 로드 시 자동으로 계산한다(버튼 없이).
  const autoRunAccountRef = useRef<string | null>(null);
  useEffect(() => {
    if (
      !loading &&
      selectedAccount &&
      settings?.ACCOUNT_ID &&
      validTickers.length >= 3 &&
      autoRunAccountRef.current !== selectedAccount
    ) {
      autoRunAccountRef.current = selectedAccount;
      void runMetrics();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading, selectedAccount, settings?.ACCOUNT_ID, validTickers.length]);

  // 종목 추가 — /assets 와 동일한 addingRow 흐름을 공용 훅(useAddingTickerRow)으로 재사용한다.
  // 추가 → 상단 신규 행에 티커 입력 → 확인(조회) → 목록에 추가. 저장은 그리드 저장 버튼으로.
  const addResolve = useCallback(
    async (raw: string): Promise<HelperTicker & { name: string }> => {
      if (tickers.some((item) => item.ticker.trim().toUpperCase() === raw && item.name)) {
        throw new Error("이미 등록된 종목입니다.");
      }
      const resp = await fetch(`/api/ticker-resolve?ticker=${encodeURIComponent(raw)}`);
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
    [tickers],
  );
  const addOnValidated = useCallback((resolved: HelperTicker) => {
    setTickers((prev) => [resolved, ...prev]);
  }, []);
  const addOnError = useCallback((message: string) => toast.error(message), [toast]);
  const add = useAddingTickerRow({
    resolve: addResolve,
    onValidated: addOnValidated,
    onError: addOnError,
    normalize: (raw) => raw.trim().toUpperCase(),
    resetOnValidated: true,
  });

  const handleDelete = useCallback(() => {
    if (!selectedRowIndexes.length) return;
    const toRemove = new Set(selectedRowIndexes);
    setTickers((current) => current.filter((_, idx) => !toRemove.has(idx + 1)));
    setSelectedRowIndexes([]);
  }, [selectedRowIndexes]);

  // 비중계산 — 지표(금일/수익률/MDD/소르티노)를 채운다. fixed 모드로 top_pick 엔진 재활용.
  const runMetrics = useCallback(async () => {
    if (!settings || !settings.ACCOUNT_ID) return;
    if (validTickers.length < 3) {
      toast.error("지표 계산에는 확인된 종목이 3개 이상 필요합니다.");
      return;
    }
    try {
      setRunning(true);
      const resp = await fetch("/api/top-pick-settings/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: validTickers,
          settings,
          weight_mode: "fixed",
          backtest_settings: { months: 12, rebalance: "none", initial_amount_manwon: 10000 },
        }),
      });
      const data = (await resp.json()) as { rows?: (WeightRow & { bucket?: unknown })[]; error?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? "지표 계산에 실패했습니다.");
      // 지표 필드만 병합한다(run 응답의 bucket 은 문자열이라 티커의 숫자 bucket 을 덮어쓰지 않도록 제외).
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
      setMetricByTicker(map);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "지표 계산에 실패했습니다.");
    } finally {
      setRunning(false);
    }
  }, [settings, toast, validTickers]);

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
    try {
      setSaving(true);
      const resp = await fetch("/api/top-pick-settings", {
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
      if (data.tickers) setTickers(buildRows(data.tickers));
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
    // 비중 합계가 100%가 아니면 백테스트를 막는다.
    const sum = validTickers.reduce((acc, item) => acc + (Number(item.fixed_weight_pct) || 0), 0);
    if (Math.abs(sum - 100) >= 0.05) {
      toast.error(`비중 합계가 ${sum.toFixed(1)}%입니다. 100%로 맞춰야 백테스트할 수 있습니다.`);
      return;
    }
    if (!Number.isInteger(Number(btAmount)) || Number(btAmount) <= 0) {
      toast.error("최초 금액(만원)은 1 이상의 정수여야 합니다.");
      return;
    }
    try {
      setBtRunning(true);
      setBtResult(null);
      const resp = await fetch("/api/top-pick-settings/backtest", {
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

  const gridRows = useMemo<GridRow[]>(() => {
    const committed = tickers.map((item, idx) => ({
      ...item,
      ...(metricByTicker[item.ticker.trim().toUpperCase()] ?? {}),
      row_index: idx + 1,
      is_adding: false,
    }));
    if (!add.addingRow) return committed;
    const addingRow: GridRow = {
      ticker: add.addingRow.ticker,
      name: add.addingRow.name,
      row_index: 0,
      is_adding: true,
    };
    return [addingRow, ...committed];
  }, [tickers, metricByTicker, add.addingRow]);

  const weightSum = validTickers.reduce((acc, item) => acc + (Number(item.fixed_weight_pct) || 0), 0);
  const weightOk = Math.abs(weightSum - 100) < 0.05;

  const columnDefs = useMemo<ColDef<GridRow>[]>(
    () => [
      {
        colId: "drag",
        headerName: "",
        width: 42,
        maxWidth: 42,
        pinned: "left",
        sortable: false,
        resizable: false,
        suppressMovable: true,
        rowDrag: (params) => Boolean(params.data && !params.data.is_adding),
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
          if (row.is_adding) {
            return (
              <StableInlineInput
                className="form-control form-control-sm assetsInlineInput assetsInlineInputTicker"
                placeholder="티커"
                initialValue={row.ticker}
                disabled={add.addingRow?.isValidating}
                onChange={(value) => add.setTicker(value)}
                onSave={(value) => void add.validate(value)}
              />
            );
          }
          return <TickerDetailLink ticker={row.ticker} displayTicker={row.ticker} />;
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
                  disabled={!row.ticker.trim() || add.addingRow?.isValidating}
                  onClick={() => void add.validate()}
                >
                  확인
                </button>
              </div>
            );
          }
          return (
            <span className="rankNameCellText" title={params.value ?? ""}>
              {renderNameWithLeverageHighlight(String(params.value ?? ""))}
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
        valueFormatter: (p) => fmtPrice(p.value as number | null),
      },
      { field: "return_1m_pct", headerName: "1달", minWidth: 84, width: 84, type: "rightAligned", cellRenderer: renderPctCell },
      { field: "return_3m_pct", headerName: "3달", minWidth: 84, width: 84, type: "rightAligned", cellRenderer: renderPctCell },
      { field: "return_6m_pct", headerName: "6달", minWidth: 84, width: 84, type: "rightAligned", cellRenderer: renderPctCell },
      { field: "return_12m_pct", headerName: "1년", minWidth: 84, width: 84, type: "rightAligned", cellRenderer: renderPctCell },
      { field: "mdd_pct", headerName: "MDD", minWidth: 84, width: 84, type: "rightAligned", cellRenderer: renderPctCell },
      { field: "sortino", headerName: "Sortino", minWidth: 90, width: 90, type: "rightAligned", valueFormatter: (p) => fmtNum(p.value as number | null) },
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
    [add],
  );

  const gridOptions = useMemo<GridOptions<GridRow>>(
    () => ({
      domLayout: "autoHeight",
      suppressMovableColumns: true,
      stopEditingWhenCellsLoseFocus: true,
      rowDragManaged: true,
      animateRows: true,
      rowSelection: {
        mode: "multiRow",
        checkboxes: (params) => Boolean(params.data && !params.data.is_adding),
        headerCheckbox: true,
        hideDisabledCheckboxes: true,
        enableClickSelection: false,
      },
      selectionColumnDef: {
        width: 52,
        minWidth: 52,
        maxWidth: 52,
        pinned: "left",
        sortable: false,
        resizable: false,
        headerName: "",
        cellClass: "assetsSelectCell",
      },
      onSelectionChanged: (params) => {
        setSelectedRowIndexes(
          params.api.getSelectedRows().map((row) => row.row_index).filter((idx): idx is number => Number.isFinite(idx)),
        );
      },
      onRowDragEnd: (params) => {
        const orderedIndexes: number[] = [];
        params.api.forEachNode((node) => {
          if (node.data) orderedIndexes.push(node.data.row_index);
        });
        setTickers((current) => orderedIndexes.map((idx) => current[idx - 1]).filter(Boolean));
      },
      onCellValueChanged: (params) => {
        if (params.colDef.field === "fixed_weight_pct") {
          const newWeight = params.newValue === "" || params.newValue == null ? null : Number(params.newValue);
          const index = params.data.row_index - 1;
          setTickers((current) =>
            current.map((item, i) => (i === index ? { ...item, fixed_weight_pct: newWeight } : item)),
          );
        }
      },
    }),
    [],
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
              <div style={{ minHeight: "13rem" }}>
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
                style={{ fontSize: "0.9rem", minHeight: "13rem" }}
                rows={10}
                placeholder="이 계좌에 대한 투자 전략이나 주의사항을 메모하세요. AI가 요약할 때 함께 참고합니다."
                value={memo}
                onChange={(e) => setMemo(e.target.value)}
              />
            </div>
          </div>
          <style jsx>{`
            .assetHelperMemoLayout {
              display: grid;
              grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
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
                  합계 {weightSum.toFixed(1)}%{weightOk ? " ✓" : " (100% 필요)"}
                </p>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <GridToolbarButton variant="add" onClick={() => add.start()} disabled={Boolean(add.addingRow)} />
                <GridToolbarButton variant="save" disabled={saving} onClick={() => void saveSettings()}>
                  {saving ? "저장 중..." : "저장"}
                </GridToolbarButton>
                <GridToolbarButton variant="delete" onClick={handleDelete} disabled={!selectedRowIndexes.length} />
              </div>
            </div>
            <AppAgGrid<GridRow>
              className="rankAgGrid assetsAgGrid"
              rowData={gridRows}
              columnDefs={columnDefs}
              loading={loading}
              minHeight={Math.max(260, Math.min(gridRows.length, 12) * 44 + 46)}
              gridOptions={gridOptions}
              theme={gridTheme}
              getRowId={(params) => String(params.data.row_index)}
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
                <input type="number" min={1} className="form-control form-control-sm" value={btMonths} onChange={(e) => setBtMonths(e.target.value)} />
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
              <TopPickBacktestResult result={btResult} />
            ) : (
              <div style={{ color: "var(--text-muted)", fontSize: "0.9rem", padding: "6px 0" }}>백테스트 버튼을 눌러 결과를 확인하세요.</div>
            )}
          </div>
        </div>
      </div>
    </PageFrame>
  );
}
