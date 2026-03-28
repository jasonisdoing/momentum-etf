"use client";

import { useEffect, useMemo, useState } from "react";
import { type GridColDef, type GridRenderCellParams } from "@mui/x-data-grid";

import {
  readRememberedMomentumEtfAccountId,
  writeRememberedMomentumEtfAccountId,
} from "../components/account-selection";
import { AppDataGrid } from "../components/AppDataGrid";

type RankAccount = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
  country_code: string;
};

type RankRow = {
  보유여부: string;
  버킷: string;
  bucket: number;
  티커: string;
  종목명: string;
  상장일: string;
  추세: number | null;
  지속: number | null;
  보유: string;
  현재가: number | null;
  "괴리율": number | null;
  "일간(%)": number | null;
  "1주(%)": number | null;
  "2주(%)": number | null;
  "1달(%)": number | null;
  "3달(%)": number | null;
  "6달(%)": number | null;
  "12달(%)": number | null;
  고점: number | null;
  RSI: number | null;
};

type RankResponse = {
  accounts?: RankAccount[];
  account_id?: string;
  ma_type?: string;
  ma_months?: number;
  ma_type_options?: string[];
  ma_months_max?: number;
  rows?: RankRow[];
  cache_blocked?: boolean;
  latest_trading_day?: string | null;
  cache_updated_at?: string | null;
  ranking_computed_at?: string | null;
  realtime_fetched_at?: string | null;
  missing_tickers?: string[];
  stale_tickers?: string[];
  error?: string;
};

type RankGridRow = RankRow & {
  id: string;
};

type RankToolbarCache = {
  accounts: RankAccount[];
  account_id: string;
  ma_type: string;
  ma_months: number;
  ma_type_options: string[];
  ma_months_max: number;
};

let rankToolbarCache: RankToolbarCache | null = null;

function formatNumber(value: number | null, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function formatPercent(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${value.toFixed(2)}%`;
}

function getSignedClass(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) {
    return "";
  }
  return value > 0 ? "metricPositive" : "metricNegative";
}

function getBucketCellClass(bucketLabel: string): string {
  const match = /^(\d+)/.exec(String(bucketLabel || "").trim());
  if (!match) {
    return "rankBucketCell";
  }
  return `rankBucketCell rankBucketCell${match[1]}`;
}

function formatMetaTime(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("ko-KR", {
    dateStyle: "short",
    timeStyle: "short",
  }).format(date);
}

function renderSignedPercentCell(params: GridRenderCellParams<RankGridRow, number | null>) {
  return <span className={getSignedClass(params.value ?? null)}>{formatPercent(params.value ?? null)}</span>;
}

export function RankManager() {
  const [accounts, setAccounts] = useState<RankAccount[]>(rankToolbarCache?.accounts ?? []);
  const [selectedAccountId, setSelectedAccountId] = useState(
    rankToolbarCache?.account_id ?? readRememberedMomentumEtfAccountId() ?? "",
  );
  const [maType, setMaType] = useState(rankToolbarCache?.ma_type ?? "");
  const [maMonths, setMaMonths] = useState(rankToolbarCache?.ma_months ?? 1);
  const [maTypeOptions, setMaTypeOptions] = useState<string[]>(rankToolbarCache?.ma_type_options ?? []);
  const [maMonthsMax, setMaMonthsMax] = useState(rankToolbarCache?.ma_months_max ?? 12);
  const [rows, setRows] = useState<RankRow[]>([]);
  const [cacheBlocked, setCacheBlocked] = useState(false);
  const [latestTradingDay, setLatestTradingDay] = useState<string | null>(null);
  const [cacheUpdatedAt, setCacheUpdatedAt] = useState<string | null>(null);
  const [rankingComputedAt, setRankingComputedAt] = useState<string | null>(null);
  const [realtimeFetchedAt, setRealtimeFetchedAt] = useState<string | null>(null);
  const [missingTickers, setMissingTickers] = useState<string[]>([]);
  const [staleTickers, setStaleTickers] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load(next?: { account_id?: string; ma_type?: string; ma_months?: number }) {
    setLoading(true);
    setError(null);

    try {
      const search = new URLSearchParams();
      if (next?.account_id) {
        search.set("account_id", next.account_id);
      }
      if (next?.ma_type) {
        search.set("ma_type", next.ma_type);
      }
      if (next?.ma_months) {
        search.set("ma_months", String(next.ma_months));
      }

      const query = search.size > 0 ? `?${search.toString()}` : "";
      const response = await fetch(`/api/rank${query}`, { cache: "no-store" });
      const payload = (await response.json()) as RankResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "순위 데이터를 불러오지 못했습니다.");
      }

      setAccounts(payload.accounts ?? []);
      const nextAccountId = payload.account_id ?? "";
      setSelectedAccountId(nextAccountId);
      writeRememberedMomentumEtfAccountId(nextAccountId);
      setMaType(payload.ma_type ?? "");
      setMaMonths(payload.ma_months ?? 1);
      setMaTypeOptions(payload.ma_type_options ?? []);
      setMaMonthsMax(payload.ma_months_max ?? 12);
      rankToolbarCache = {
        accounts: payload.accounts ?? [],
        account_id: nextAccountId,
        ma_type: payload.ma_type ?? "",
        ma_months: payload.ma_months ?? 1,
        ma_type_options: payload.ma_type_options ?? [],
        ma_months_max: payload.ma_months_max ?? 12,
      };
      setRows(payload.rows ?? []);
      setCacheBlocked(Boolean(payload.cache_blocked));
      setLatestTradingDay(payload.latest_trading_day ?? null);
      setCacheUpdatedAt(payload.cache_updated_at ?? null);
      setRankingComputedAt(payload.ranking_computed_at ?? null);
      setRealtimeFetchedAt(payload.realtime_fetched_at ?? null);
      setMissingTickers(payload.missing_tickers ?? []);
      setStaleTickers(payload.stale_tickers ?? []);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "순위 데이터를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load({ account_id: readRememberedMomentumEtfAccountId() ?? undefined });
  }, []);

  const selectedAccount = useMemo(
    () => accounts.find((account) => account.account_id === selectedAccountId) ?? null,
    [accounts, selectedAccountId],
  );

  const gridRows = useMemo<RankGridRow[]>(
    () =>
      rows.map((row, index) => ({
        ...row,
        id: `${row.티커}-${row.보유여부 || "none"}-${index}`,
      })),
    [rows],
  );

  const columns = useMemo<GridColDef<RankGridRow>[]>(
    () => [
      { field: "보유여부", headerName: "보유여부", minWidth: 88, width: 88 },
      {
        field: "버킷",
        headerName: "버킷",
        minWidth: 108,
        width: 108,
        sortable: false,
        cellClassName: (params) => getBucketCellClass(String(params.value ?? "")),
        renderCell: (params) => <span>{String(params.value ?? "-")}</span>,
      },
      {
        field: "티커",
        headerName: "티커",
        minWidth: 92,
        width: 92,
        renderCell: (params) => <span className="appCodeText">{String(params.value ?? "-")}</span>,
      },
      { field: "종목명", headerName: "종목명", minWidth: 220, flex: 1 },
      {
        field: "현재가",
        headerName: "현재가",
        minWidth: 110,
        width: 110,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatNumber(params.value ?? null, selectedAccount?.country_code === "au" ? 2 : 0),
      },
      {
        field: "일간(%)",
        headerName: "일간(%)",
        minWidth: 96,
        width: 96,
        align: "right",
        headerAlign: "right",
        renderCell: renderSignedPercentCell,
      },
      {
        field: "추세",
        headerName: "추세",
        minWidth: 72,
        width: 72,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatNumber(params.value ?? null, 1),
      },
      {
        field: "고점",
        headerName: "고점",
        minWidth: 92,
        width: 92,
        align: "right",
        headerAlign: "right",
        renderCell: renderSignedPercentCell,
      },
      {
        field: "1주(%)",
        headerName: "1주(%)",
        minWidth: 88,
        width: 88,
        align: "right",
        headerAlign: "right",
        renderCell: renderSignedPercentCell,
      },
      {
        field: "2주(%)",
        headerName: "2주(%)",
        minWidth: 88,
        width: 88,
        align: "right",
        headerAlign: "right",
        renderCell: renderSignedPercentCell,
      },
      {
        field: "1달(%)",
        headerName: "1달(%)",
        minWidth: 88,
        width: 88,
        align: "right",
        headerAlign: "right",
        renderCell: renderSignedPercentCell,
      },
      {
        field: "3달(%)",
        headerName: "3달(%)",
        minWidth: 88,
        width: 88,
        align: "right",
        headerAlign: "right",
        renderCell: renderSignedPercentCell,
      },
      {
        field: "6달(%)",
        headerName: "6달(%)",
        minWidth: 88,
        width: 88,
        align: "right",
        headerAlign: "right",
        renderCell: renderSignedPercentCell,
      },
      {
        field: "12달(%)",
        headerName: "12달(%)",
        minWidth: 94,
        width: 94,
        align: "right",
        headerAlign: "right",
        renderCell: renderSignedPercentCell,
      },
      {
        field: "RSI",
        headerName: "RSI",
        minWidth: 74,
        width: 74,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatNumber(params.value ?? null, 1),
      },
      {
        field: "지속",
        headerName: "지속",
        minWidth: 66,
        width: 66,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatNumber(params.value ?? null, 0),
      },
    ],
    [selectedAccount?.country_code],
  );

  function handleAccountChange(accountId: string) {
    writeRememberedMomentumEtfAccountId(accountId);
    void load({ account_id: accountId, ma_type: maType, ma_months: maMonths });
  }

  function handleMaTypeChange(nextMaType: string) {
    void load({ account_id: selectedAccountId, ma_type: nextMaType, ma_months: maMonths });
  }

  function handleMaMonthsChange(nextMaMonths: number) {
    void load({ account_id: selectedAccountId, ma_type: maType, ma_months: nextMaMonths });
  }

  const blockedMessage = useMemo(() => {
    if (!cacheBlocked) {
      return null;
    }

    const parts: string[] = [];
    if (latestTradingDay) {
      parts.push(`최신 거래일 ${latestTradingDay}`);
    }
    if (cacheUpdatedAt) {
      parts.push(`캐시 기준 ${formatMetaTime(cacheUpdatedAt)}`);
    }
    if (missingTickers.length > 0) {
      parts.push(`누락 ${missingTickers.join(", ")}`);
    }
    if (staleTickers.length > 0) {
      parts.push(`오래된 캐시 ${staleTickers.join(", ")}`);
    }
    return parts.join(" | ");
  }, [cacheBlocked, cacheUpdatedAt, latestTradingDay, missingTickers, staleTickers]);

  return (
    <div className="appPageStack">
      {error ? (
        <div className="appBannerStack">
          <div className="bannerError">{error}</div>
        </div>
      ) : null}
      {blockedMessage ? (
        <div className="appBannerStack">
          <div className="bannerWarning">{blockedMessage}</div>
        </div>
      ) : null}

      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBody">
            <div className="accountToolbar">
              <div className="accountToolbarLeft">
                <div className="accountSelect">
                  <select
                    className="form-select"
                    value={selectedAccountId}
                    onChange={(event) => handleAccountChange(event.target.value)}
                    disabled={accounts.length === 0}
                  >
                    {accounts.length === 0 ? (
                      <option value="">계좌 불러오는 중...</option>
                    ) : (
                      accounts.map((account) => (
                        <option key={account.account_id} value={account.account_id}>
                          {account.name}
                        </option>
                      ))
                    )}
                  </select>
                </div>
                <div className="accountToolbarOptions">
                  <select
                    className="form-select appSelect"
                    value={maType}
                    onChange={(event) => handleMaTypeChange(event.target.value)}
                    disabled={maTypeOptions.length === 0}
                  >
                    {maTypeOptions.length === 0 ? (
                      <option value="">기준 불러오는 중...</option>
                    ) : (
                      maTypeOptions.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))
                    )}
                  </select>
                  <select
                    className="form-select appSelect"
                    value={String(maMonths)}
                    onChange={(event) => handleMaMonthsChange(Number(event.target.value))}
                    disabled={maTypeOptions.length === 0}
                  >
                    {Array.from({ length: maMonthsMax }, (_, index) => index + 1).map((month) => (
                      <option key={month} value={month}>
                        {month}개월
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="accountToolbarRight">
                <div className="accountToolbarMeta rankToolbarMeta">
                <span>{selectedAccount ? `${selectedAccount.icon} ${selectedAccount.name}` : ""}</span>
                {rankingComputedAt ? <span>계산 {formatMetaTime(rankingComputedAt)}</span> : null}
                {realtimeFetchedAt ? <span>실시간 {formatMetaTime(realtimeFetchedAt)}</span> : null}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="appSection appSectionFill">
        <div className="card appCard">
          <div className="card-body appCardBodyTight">
            <div className="rankGridWrap">
              <AppDataGrid
                className="rankDataGrid"
                rows={gridRows}
                columns={columns}
                loading={loading}
                getRowClassName={(params) => (params.row.보유 ? "rankHeldRow" : "")}
                minHeight="70vh"
              />
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
