"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { ColDef, GridOptions } from "ag-grid-community";

import {
  readRememberedMomentumEtfAccountId,
  writeRememberedMomentumEtfAccountId,
} from "../components/account-selection";
import { AppAgGrid } from "../components/AppAgGrid";
import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";
import { TickerDetailLink } from "../components/TickerDetailLink";

type TopPickRow = {
  ticker: string;
  name: string;
  ticker_type?: string;
  country_code?: string;
  trend_pct?: number | null;
  trend_score?: number | null;
  sortino_score?: number | null;
  sortino?: number | null;
  score?: number | null;
  target_weight_pct: number | null;
  current_price?: number | null;
  target_amount_krw?: number | null;
  target_value_krw?: number | null;
  target_quantity?: number | null;
  current_quantity?: number | null;
  current_amount_krw?: number | null;
  change_quantity?: number | null;
  change_amount_krw?: number | null;
  change_weight_pct?: number | null;
  unallocated_amount_krw?: number | null;
  return_pct?: number | null;
  pnl_krw?: number | null;
  current_weight_pct?: number | null;
  bucket?: string | null;
  daily_change_pct?: number | null;
};

type TopPickTradeSummary = {
  account_id?: string;
  account_name?: string;
  currency?: string;
  account_amount_krw?: number;
  target_asset_amount_krw?: number;
  remaining_cash_krw?: number;
};

type TopPickPayload = {
  as_of_date?: string;
  rows?: TopPickRow[];
  missing_tickers?: string[];
  settings?: {
    MA_TYPE: string;
    MA_MONTHS: number;
    MIN_WEIGHT: number;
    MAX_WEIGHT: number;
    CASH_MAX_WEIGHT: number;
    ACCOUNT_ID: string;
    START_AMOUNT_MANWON?: number | null;
    START_DATE?: string | null;
  };
  trade_summary?: TopPickTradeSummary;
  error?: string;
};

function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return value.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatWeightPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${value.toFixed(1)}%`;
}

function formatAccountMoney(value: number | null | undefined, currency: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  if (currency === "KRW") return `${Math.round(value).toLocaleString("ko-KR")}원`;
  const amount = value.toLocaleString("ko-KR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (currency === "AUD") return `A$${amount}`;
  if (currency === "USD") return `$${amount}`;
  return `${amount} ${currency}`;
}

function formatQuantity(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${Math.floor(value).toLocaleString("ko-KR")}주`;
}

function formatChangeQuantity(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  const quantity = Math.trunc(value);
  if (quantity > 0) return `+${quantity.toLocaleString("ko-KR")}주`;
  return `${quantity.toLocaleString("ko-KR")}주`;
}

function signedQuantityColor(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) {
    return "#475569";
  }
  return value > 0 ? "#d63939" : "#206bc4";
}

function formatTopPickTicker(row: TopPickRow | undefined): string {
  const ticker = String(row?.ticker ?? "").trim().toUpperCase();
  if (!ticker || ticker === "__CASH__") return ticker;
  const isAustralian = row?.ticker_type === "aus" || row?.country_code === "au";
  if (!isAustralian || ticker.startsWith("ASX:")) return ticker;
  return ticker.endsWith(".AX") ? `ASX:${ticker.slice(0, -3)}` : `ASX:${ticker}`;
}

const gridTheme = createAppGridTheme();

export function TopPickClient() {
  const toast = useToast();
  const [payload, setPayload] = useState<TopPickPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [topPickAccounts, setTopPickAccounts] = useState<string[]>([]);
  const [accountNames, setAccountNames] = useState<Record<string, string>>({});
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null);

  const load = useCallback(async (accountId?: string) => {
    try {
      setLoading(true);
      // 계좌 목록 먼저 → account_id 미지정이면 기억된 계좌(목록에 있으면) → 없으면 첫 계좌
      const [tpAccountsResp, accountsResp] = await Promise.all([
        fetch("/api/top-pick-settings/accounts", { cache: "no-store" }),
        fetch("/api/holdings-components/accounts", { cache: "no-store" }),
      ]);
      const tpData = (await tpAccountsResp.json()) as { accounts?: string[]; error?: string };
      const accData = (await accountsResp.json()) as { account_id: string; name: string }[] | { error?: string };
      const tpList = Array.isArray(tpData.accounts) ? tpData.accounts : [];
      setTopPickAccounts(tpList);
      if (Array.isArray(accData)) {
        setAccountNames(Object.fromEntries(accData.map((a) => [a.account_id, a.name])));
      }
      const remembered = readRememberedMomentumEtfAccountId();
      const target = accountId ?? (remembered && tpList.includes(remembered) ? remembered : tpList[0]);
      if (!target) {
        throw new Error("등록된 탑픽 계좌가 없습니다.");
      }
      writeRememberedMomentumEtfAccountId(target);
      setSelectedAccount(target);

      const resp = await fetch(`/api/top-pick?account_id=${encodeURIComponent(target)}`, { cache: "no-store" });
      const data = (await resp.json()) as TopPickPayload;
      if (!resp.ok || data.error) {
        throw new Error(data.error ?? "탑픽 비중을 불러오지 못했습니다.");
      }
      setPayload(data);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "탑픽 비중을 불러오지 못했습니다.");
      setPayload(null);
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    void load();
  }, [load]);

  const rows = payload?.rows ?? [];
  const settings = payload?.settings;
  const missing = payload?.missing_tickers ?? [];
  const tradeSummary = payload?.trade_summary ?? {};
  const accountCurrency = String(tradeSummary.currency ?? "KRW").toUpperCase();
  const cashWeight = rows.find((row) => row.ticker === "__CASH__")?.target_weight_pct ?? null;
  const etfCount = rows.filter((row) => row.ticker !== "__CASH__").length;
 
  const totalDailyReturn = useMemo(() => {
    if (!rows || rows.length === 0) return 0;
    let sumWeight = 0;
    let weightedReturnSum = 0;
    for (const row of rows) {
      const weight = row.current_weight_pct ?? 0;
      const dailyChange = row.daily_change_pct ?? 0;
      weightedReturnSum += dailyChange * weight;
      sumWeight += weight;
    }
    return sumWeight > 0 ? weightedReturnSum / sumWeight : 0;
  }, [rows]);
 
  // 누적 수익률 = (현재 계좌 총평가액 ÷ 시작금액 − 1) × 100. 시작금액/현재액이 없으면 null.
  const cumulativeReturn = useMemo(() => {
    const storedStartAmount = settings?.START_AMOUNT_MANWON ?? 0;
    const startAmount = accountCurrency === "KRW" ? storedStartAmount * 10000 : storedStartAmount;
    const currentAmount = tradeSummary.account_amount_krw ?? 0;
    if (startAmount <= 0 || currentAmount <= 0) return null;
    return (currentAmount / startAmount - 1) * 100.0;
  }, [accountCurrency, settings?.START_AMOUNT_MANWON, tradeSummary.account_amount_krw]);

  const titleRight = useMemo(
    () => (
      <div className="appHeaderMetrics rankToolbarMeta">
        <div className="appHeaderMetric">
          <span>종목:</span>
          <span className="appHeaderMetricValue">{etfCount}개</span>
        </div>
        <div className="appHeaderMetric">
          <span>현금:</span>
          <span className="appHeaderMetricValue">{formatWeightPct(cashWeight)} / {formatAccountMoney(tradeSummary.remaining_cash_krw, accountCurrency)}</span>
        </div>
        <div className="appHeaderMetric">
          <span>계좌:</span>
          <span className="appHeaderMetricValue">{tradeSummary.account_name ?? settings?.ACCOUNT_ID ?? "-"}</span>
        </div>
        <div className="appHeaderMetric">
          <span>총자산:</span>
          <span className="appHeaderMetricValue">{formatAccountMoney(tradeSummary.account_amount_krw, accountCurrency)}</span>
        </div>
        <div className="appHeaderMetric">
          <span>기준일:</span>
          <span className="appHeaderMetricValue">{payload?.as_of_date ?? "-"}</span>
        </div>
      </div>
    ),
    [accountCurrency, cashWeight, etfCount, payload?.as_of_date, settings?.ACCOUNT_ID, tradeSummary.account_amount_krw, tradeSummary.account_name, tradeSummary.remaining_cash_krw],
  );

  const columns = useMemo<ColDef<TopPickRow>[]>(
    () => [
      {
        field: "bucket",
        headerName: "버킷",
        width: 100,
        cellClass: (params) => {
          const match = /^(\d+)/.exec(String(params.value || "").trim());
          const num = match ? match[1] : "";
          if (params.data?.ticker === "__CASH__" || params.value?.includes("현금")) {
            return "appBucketCell";
          }
          return num ? `appBucketCell appBucketCell${num}` : "appBucketCell";
        },
        cellRenderer: (params: { value: string | null | undefined; data?: TopPickRow }) => {
          if (!params.value || params.data?.ticker === "__CASH__" || params.value.includes("현금")) return "-";
          return params.value;
        },
      },
      {
        field: "ticker",
        headerName: "티커",
        width: 112,
        cellRenderer: (params: { data?: TopPickRow }) => {
          const ticker = formatTopPickTicker(params.data);
          return <TickerDetailLink ticker={ticker} displayTicker={ticker} />;
        },
      },
      {
        field: "name",
        headerName: "종목명",
        minWidth: 240,
        flex: 1,
        cellClass: "topPickNameCell",
        cellRenderer: (params: { value: string | null | undefined }) => {
          const name = params.value || "-";
          return <span className="topPickNameCellText" title={name}>{name}</span>;
        },
      },
      {
        field: "daily_change_pct",
        headerName: "일간(%)",
        width: 88,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) => {
          if (params.value == null || Number.isNaN(params.value)) return "-";
          const val = params.value;
          const formatted = `${val > 0 ? "+" : ""}${val.toFixed(2)}%`;
          const color = val === 0 ? "var(--text-strong)" : val > 0 ? "#d63939" : "#206bc4";
          return <span style={{ color }}>{formatted}</span>;
        },
      },
      {
        field: "current_price",
        headerName: "현재가",
        width: 98,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatAccountMoney(params.value, accountCurrency),
      },
      {
        field: "return_pct",
        headerName: "수익률",
        width: 88,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) => {
          if (params.value == null || Number.isNaN(params.value)) return "-";
          const val = params.value;
          const formatted = `${val > 0 ? "+" : ""}${val.toFixed(2)}%`;
          const color = val === 0 ? "var(--text-strong)" : val > 0 ? "#d63939" : "#206bc4";
          return <span style={{ color }}>{formatted}</span>;
        },
      },
      {
        field: "pnl_krw",
        headerName: "평가손익",
        width: 114,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) => {
          if (params.value == null || Number.isNaN(params.value)) return "-";
          const val = params.value;
          const sign = val > 0 ? "+" : val < 0 ? "-" : "";
          const formatted = `${sign}${formatAccountMoney(Math.abs(val), accountCurrency)}`;
          const color = val === 0 ? "var(--text-strong)" : val > 0 ? "#d63939" : "#206bc4";
          return <span style={{ color }}>{formatted}</span>;
        },
      },
      {
        field: "current_weight_pct",
        headerName: "현재비중",
        width: 88,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) => {
          if (params.value == null || Number.isNaN(params.value)) return "-";
          return `${params.value.toFixed(1)}%`;
        },
      },
      {
        field: "current_quantity",
        headerName: "현재수량",
        width: 84,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatQuantity(params.value),
      },
      {
        field: "current_amount_krw",
        headerName: "현재금액",
        width: 114,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatAccountMoney(params.value, accountCurrency),
      },
      {
        field: "target_weight_pct",
        headerName: "목표비중",
        width: 84,
        type: "rightAligned",
        cellStyle: { fontWeight: 800 },
        cellRenderer: (params: { value: number | null | undefined }) => formatWeightPct(params.value),
      },
      {
        field: "target_quantity",
        headerName: "목표수량",
        width: 84,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) => formatQuantity(params.value),
      },
      {
        field: "target_value_krw",
        headerName: "목표금액",
        width: 114,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatAccountMoney(params.value, accountCurrency),
      },
      {
        field: "change_weight_pct",
        headerName: "변동비중",
        width: 84,
        type: "rightAligned",
        cellStyle: { fontWeight: 700 },
        cellRenderer: (params: { value: number | null | undefined }) => {
          if (params.value == null || Number.isNaN(params.value)) return "-";
          const val1d = Number(params.value.toFixed(1));
          const formatted = `${val1d > 0 ? "+" : ""}${val1d.toFixed(1)}%`;
          const color = val1d === 0 ? "#475569" : val1d > 0 ? "#d63939" : "#206bc4";
          return <span style={{ color }}>{formatted}</span>;
        },
      },
      {
        field: "change_quantity",
        headerName: "변동수량",
        width: 84,
        type: "rightAligned",
        cellStyle: { fontWeight: 800 },
        cellRenderer: (params: { value: number | null | undefined }) => (
          <span style={{ color: signedQuantityColor(params.value) }}>{formatChangeQuantity(params.value)}</span>
        ),
      },
      {
        field: "change_amount_krw",
        headerName: "변동금액",
        width: 114,
        type: "rightAligned",
        cellStyle: { fontWeight: 800 },
        cellRenderer: (params: { value: number | null | undefined }) => {
          if (params.value == null || Number.isNaN(params.value)) return "-";
          const sign = params.value > 0 ? "+" : params.value < 0 ? "-" : "";
          return (
            <span style={{ color: signedQuantityColor(params.value) }}>
              {sign}{formatAccountMoney(Math.abs(params.value), accountCurrency)}
            </span>
          );
        },
      },
    ],
    [accountCurrency],
  );

  const gridOptions = useMemo<GridOptions<TopPickRow>>(
    () => ({
      suppressMovableColumns: true,
      overlayNoRowsTemplate: '<span style="color:#667382;">표시할 탑픽 비중이 없습니다.</span>',
    }),
    [],
  );

  return (
    <PageFrame title="탑픽 비중" fullHeight fullWidth titleRight={titleRight}>
      <div className="appPageStack appPageStackFill">
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>목표 비중</h2>
                <div style={{ color: "var(--text-muted)", fontSize: "0.86rem" }}>
                  기준일 {payload?.as_of_date ?? "-"} · {settings ? `${settings.MA_TYPE} ${settings.MA_MONTHS}개월 (추세 100%)` : "설정 없음"}
                  {settings ? ` · 적용계좌 ${tradeSummary.account_name ?? settings.ACCOUNT_ID} · 최소 ${settings.MIN_WEIGHT}% · 최대 ${settings.MAX_WEIGHT}% · 현금 최대 ${settings.CASH_MAX_WEIGHT}%` : ""}
                </div>
              </div>
              <div className="appMainHeaderRight" style={{ gap: 8, alignItems: "center" }}>
                <select
                  className="form-select form-select-sm"
                  style={{ minWidth: 160, width: "auto" }}
                  value={selectedAccount ?? ""}
                  disabled={loading || topPickAccounts.length === 0}
                  onChange={(event) => void load(event.target.value)}
                >
                  {topPickAccounts.length === 0 ? (
                    <option value="">계좌 불러오는 중...</option>
                  ) : (
                    [...topPickAccounts]
                      .sort((a, b) => {
                        const num = (id: string) => {
                          const m = (accountNames[id] ?? id).match(/^\s*(\d+)/);
                          return m ? parseInt(m[1], 10) : Number.MAX_SAFE_INTEGER;
                        };
                        return num(a) - num(b);
                      })
                      .map((accId) => (
                        <option key={accId} value={accId}>
                          {accountNames[accId] ?? accId}
                        </option>
                      ))
                  )}
                </select>
                <button type="button" className="btn btn-sm btn-outline-secondary" disabled={loading} onClick={() => void load(selectedAccount ?? undefined)}>
                  새로고침
                </button>
              </div>
            </div>
            {settings ? (
              <div style={{ fontSize: "0.86rem", marginTop: 8, display: "flex", gap: 16, alignItems: "center" }}>
                <div>
                  포트폴리오 일간수익률{" "}
                  <span style={{ color: totalDailyReturn === 0 ? "var(--text-strong)" : totalDailyReturn > 0 ? "#d63939" : "#206bc4", fontWeight: 700 }}>
                    {totalDailyReturn > 0 ? "+" : ""}{totalDailyReturn.toFixed(2)}%
                  </span>
                </div>
                <div style={{ color: "var(--text-muted)" }}>|</div>
                <div title={cumulativeReturn === null ? "설정 화면에서 시작금액·시작일자를 입력해야 누적수익률이 계산됩니다" : `${settings.START_DATE} · 시작금액 ${accountCurrency === "KRW" ? `${settings.START_AMOUNT_MANWON?.toLocaleString()}만원` : formatAccountMoney(settings.START_AMOUNT_MANWON, accountCurrency)} 기준`}>
                  누적 수익률{cumulativeReturn !== null && settings.START_DATE ? `(${settings.START_DATE}~)` : ""}{" "}
                  {cumulativeReturn === null ? (
                    <span style={{ color: "#d63939", fontWeight: 700 }}>⚠ 시작금액 미설정</span>
                  ) : (
                    <span style={{ color: cumulativeReturn === 0 ? "var(--text-strong)" : cumulativeReturn > 0 ? "#d63939" : "#206bc4", fontWeight: 700 }}>
                      {cumulativeReturn > 0 ? "+" : ""}{cumulativeReturn.toFixed(2)}%
                    </span>
                  )}
                </div>
              </div>
            ) : null}
            {missing.length > 0 ? (
              <div className="alert alert-warning" style={{ marginTop: 12, marginBottom: 0 }}>
                가격 캐시 누락: {missing.join(", ")}
              </div>
            ) : null}
          </div>
        </div>

        <div className="card appCard appTableCardFill">
          <div className="card-body p-2 appTableCardBodyFill">
            <div className="appGridFillWrap">
              <AppAgGrid<TopPickRow>
                rowData={rows}
                columnDefs={columns}
                loading={loading}
                minHeight="100%"
                className="topPickWeightGrid assetsAgGrid"
                theme={gridTheme}
                getRowId={(params) => params.data.ticker}
                gridOptions={gridOptions}
              />
            </div>
          </div>
        </div>
      </div>
    </PageFrame>
  );
}
