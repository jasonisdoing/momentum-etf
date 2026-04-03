"use client";

import { useRouter } from "next/navigation";
import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { type GridColDef, type GridRenderCellParams } from "@mui/x-data-grid";

import {
  readRememberedTickerType,
  writeRememberedTickerType,
} from "../components/account-selection";
import { AppDataGrid } from "../components/AppDataGrid";
import { useToast } from "../components/ToastProvider";

type RankTickerType = {
  ticker_type: string;
  order: number;
  name: string;
  icon: string;
  country_code: string;
};

type RankRow = {
  [key: string]: string | number | null;
  순번: string;
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
  "3주(%)": number | null;
  "4주(%)": number | null;
  "1달(%)": number | null;
  "2달(%)": number | null;
  "3달(%)": number | null;
  "4달(%)": number | null;
  "5달(%)": number | null;
  "6달(%)": number | null;
  "7달(%)": number | null;
  "8달(%)": number | null;
  "9달(%)": number | null;
  "10달(%)": number | null;
  "11달(%)": number | null;
  "12달(%)": number | null;
  고점: number | null;
  RSI: number | null;
};

type RankResponse = {
  ticker_types?: RankTickerType[];
  ticker_type?: string;
  ma_type?: string;
  ma_months?: number;
  ma_type_options?: string[];
  ma_months_max?: number;
  as_of_date?: string | null;
  monthly_return_labels?: string[];
  rows?: RankRow[];
  cache_blocked?: boolean;
  latest_trading_day?: string | null;
  cache_updated_at?: string | null;
  ranking_computed_at?: string | null;
  realtime_fetched_at?: string | null;
  missing_tickers?: string[];
  missing_ticker_labels?: string[];
  stale_tickers?: string[];
  error?: string;
};

type RankGridRow = RankRow & {
  id: string;
  displayTrendRank: string;
};

type RankToolbarCache = {
  ticker_types: RankTickerType[];
  ticker_type: string;
  ma_type: string;
  ma_months: number;
  ma_type_options: string[];
  ma_months_max: number;
};

let rankToolbarCache: RankToolbarCache | null = null;

function getTodayDateInputValue(): string {
  return new Date().toLocaleDateString("en-CA", { timeZone: "Asia/Seoul" });
}

function toDateInputValue(value: string | null | undefined): string {
  if (!value) {
    return getTodayDateInputValue();
  }
  return String(value).slice(0, 10);
}

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
  const router = useRouter();
  const toast = useToast();
  const lastBlockedToastRef = useRef<string | null>(null);
  const [ticker_types, setAccounts] = useState<RankTickerType[]>(rankToolbarCache?.ticker_types ?? []);
  const [selectedTickerType, setSelectedAccountId] = useState(
    rankToolbarCache?.ticker_type ?? readRememberedTickerType() ?? "",
  );
  const [maType, setMaType] = useState(rankToolbarCache?.ma_type ?? "");
  const [maMonths, setMaMonths] = useState(rankToolbarCache?.ma_months ?? 1);
  const [maTypeOptions, setMaTypeOptions] = useState<string[]>(rankToolbarCache?.ma_type_options ?? []);
  const [maMonthsMax, setMaMonthsMax] = useState(rankToolbarCache?.ma_months_max ?? 12);
  const [metricMode, setMetricMode] = useState<"cumulative" | "monthly">("cumulative");
  const [monthlyReturnLabels, setMonthlyReturnLabels] = useState<string[]>([]);
  const [selectedAsOfDate, setSelectedAsOfDate] = useState<string>(getTodayDateInputValue());
  const [nameKeyword, setNameKeyword] = useState("");
  const [page, setPage] = useState(0);
  const [rows, setRows] = useState<RankRow[]>([]);
  const [cacheBlocked, setCacheBlocked] = useState(false);
  const [rankingComputedAt, setRankingComputedAt] = useState<string | null>(null);
  const [realtimeFetchedAt, setRealtimeFetchedAt] = useState<string | null>(null);
  const [missingTickers, setMissingTickers] = useState<string[]>([]);
  const [missingTickerLabels, setMissingTickerLabels] = useState<string[]>([]);
  const [staleTickers, setStaleTickers] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const deferredNameKeyword = useDeferredValue(nameKeyword);

  async function load(next?: { ticker_type?: string; ma_type?: string; ma_months?: number; as_of_date?: string }) {
    setLoading(true);
    setError(null);

    try {
      const search = new URLSearchParams();
      if (next?.ticker_type) {
        search.set("ticker_type", next.ticker_type);
      }
      if (next?.ma_type) {
        search.set("ma_type", next.ma_type);
      }
      if (next?.ma_months) {
        search.set("ma_months", String(next.ma_months));
      }
      if (next?.as_of_date) {
        search.set("as_of_date", next.as_of_date);
      }

      const query = search.size > 0 ? `?${search.toString()}` : "";
      const response = await fetch(`/api/rank${query}`, { cache: "no-store" });
      const payload = (await response.json()) as RankResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "순위 데이터를 불러오지 못했습니다.");
      }

      setAccounts(payload.ticker_types ?? []);
      const nextAccountId = payload.ticker_type ?? "";
      setSelectedAccountId(nextAccountId);
      writeRememberedTickerType(nextAccountId);
      setMaType(payload.ma_type ?? "");
      setMaMonths(payload.ma_months ?? 1);
      setMaTypeOptions(payload.ma_type_options ?? []);
      setMaMonthsMax(payload.ma_months_max ?? 12);
      setSelectedAsOfDate(toDateInputValue(payload.as_of_date));
      setMonthlyReturnLabels(payload.monthly_return_labels ?? []);
      rankToolbarCache = {
        ticker_types: payload.ticker_types ?? [],
        ticker_type: nextAccountId,
        ma_type: payload.ma_type ?? "",
        ma_months: payload.ma_months ?? 1,
        ma_type_options: payload.ma_type_options ?? [],
        ma_months_max: payload.ma_months_max ?? 12,
      };
      setRows(payload.rows ?? []);
      setPage(0);
      setCacheBlocked(Boolean(payload.cache_blocked));
      setRankingComputedAt(payload.ranking_computed_at ?? null);
      setRealtimeFetchedAt(payload.realtime_fetched_at ?? null);
      setMissingTickers(payload.missing_tickers ?? []);
      setMissingTickerLabels(payload.missing_ticker_labels ?? []);
      setStaleTickers(payload.stale_tickers ?? []);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "순위 데이터를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load({ ticker_type: readRememberedTickerType() ?? undefined, as_of_date: getTodayDateInputValue() });
  }, []);

  const selectedTickerTypeItem = useMemo(
    () => ticker_types.find((account) => account.ticker_type === selectedTickerType) ?? null,
    [ticker_types, selectedTickerType],
  );

  const gridRows = useMemo<RankGridRow[]>(() => {
    let holdRank = 0;
    let waitRank = 0;
    return rows.map((row, index) => {
      const isHold = Boolean(row["보유"] && String(row["보유"]).trim() !== "");
      let displayTrendRank = "";
      if (isHold) {
        holdRank++;
        displayTrendRank = `보유 ${holdRank}`;
      } else {
        waitRank++;
        displayTrendRank = `대기 ${waitRank}`;
      }
      return {
        ...row,
        displayTrendRank,
        id: `${row.티커}-${row.순번 || "none"}-${index}`,
      };
    });
  }, [rows]);

  const filteredGridRows = useMemo(() => {
    const keyword = deferredNameKeyword.trim().toLowerCase();
    if (!keyword) {
      return gridRows;
    }
    return gridRows.filter((row) => String(row.종목명 ?? "").toLowerCase().includes(keyword));
  }, [gridRows, deferredNameKeyword]);

  useEffect(() => {
    setPage(0);
  }, [selectedTickerType, maType, maMonths, metricMode, selectedAsOfDate, deferredNameKeyword]);

  const columns = useMemo<GridColDef<RankGridRow>[]>(() => {
    const leadingColumns: GridColDef<RankGridRow>[] = [
      {
        field: "displayTrendRank",
        headerName: "보유",
        minWidth: 72,
        width: 72,
        align: "center",
        headerAlign: "center",
        renderCell: (params) => {
          const isHold = String(params.value || "").startsWith("보유");
          return (
            <span
              style={{
                fontWeight: 700,
                color: isHold ? "inherit" : "#888888",
              }}
            >
              {String(params.value || "")}
            </span>
          );
        },
      },
      {
        field: "버킷",
        headerName: "버킷",
        minWidth: 108,
        width: 108,
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
        renderCell: (params) => formatNumber(params.value ?? null, selectedTickerTypeItem?.country_code === "au" ? 2 : 0),
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
    ];

    const cumulativeColumns: GridColDef<RankGridRow>[] = [
      {
        field: "추세",
        headerName: "추세",
        minWidth: 72,
        width: 72,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatNumber(params.value ?? null, 1),
      },
      ...(selectedTickerTypeItem?.country_code !== "au"
        ? [
            {
              field: "괴리율",
              headerName: "괴리율",
              minWidth: 88,
              width: 88,
              align: "right",
              headerAlign: "right",
              renderCell: (params: GridRenderCellParams<RankGridRow, number | null>) => {
                const val = params.value ?? 0;
                const isExtreme = val > 2.0 || val < -2.0;
                return (
                  <span style={{ color: isExtreme ? "#d63939" : "inherit", fontWeight: isExtreme ? 700 : 400 }}>
                    {formatPercent(params.value ?? null)}
                  </span>
                );
              },
            } as GridColDef<RankGridRow>,
          ]
        : []),
      {
        field: "고점",
        headerName: "고점",
        minWidth: 92,
        width: 92,
        align: "right",
        headerAlign: "right",
        renderCell: (params: GridRenderCellParams<RankGridRow, number | null>) => {
          const value = params.value ?? null;
          const isHighlighted = value !== null && value >= -5;
          return (
            <span style={{ color: isHighlighted ? "#198754" : "inherit", fontWeight: isHighlighted ? 700 : 400 }}>
              {formatPercent(value)}
            </span>
          );
        },
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
        field: "3주(%)",
        headerName: "3주(%)",
        minWidth: 88,
        width: 88,
        align: "right",
        headerAlign: "right",
        renderCell: renderSignedPercentCell,
      },
      {
        field: "4주(%)",
        headerName: "4주(%)",
        minWidth: 88,
        width: 88,
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
      ...[
        "1달(%)",
        "2달(%)",
        "3달(%)",
        "4달(%)",
        "5달(%)",
        "6달(%)",
        "7달(%)",
        "8달(%)",
        "9달(%)",
        "10달(%)",
        "11달(%)",
        "12달(%)",
      ].map(
        (field) =>
          ({
            field,
            headerName: field,
            minWidth: field.length > 6 ? 94 : 88,
            width: field.length > 6 ? 94 : 88,
            align: "right",
            headerAlign: "right",
            renderCell: renderSignedPercentCell,
          }) as GridColDef<RankGridRow>,
      ),
    ];

    const monthlyColumns: GridColDef<RankGridRow>[] = monthlyReturnLabels.map(
      (label) =>
        ({
          field: label,
          headerName: label,
          minWidth: 108,
          width: 108,
          align: "right",
          headerAlign: "right",
          renderCell: renderSignedPercentCell,
        }) as GridColDef<RankGridRow>,
    );

    const monthlyLeadingColumns: GridColDef<RankGridRow>[] = [
      {
        field: "추세",
        headerName: "추세",
        minWidth: 72,
        width: 72,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatNumber(params.value ?? null, 1),
      },
    ];

    return [
      ...leadingColumns,
      ...(metricMode === "cumulative" ? cumulativeColumns : [...monthlyLeadingColumns, ...monthlyColumns]),
    ];
  }, [metricMode, monthlyReturnLabels, selectedTickerTypeItem?.country_code]);

  function handleTickerTypeChange(accountId: string) {
    setSelectedAccountId(accountId);
    writeRememberedTickerType(accountId);
    void load({ ticker_type: accountId, ma_type: maType, ma_months: maMonths, as_of_date: selectedAsOfDate });
  }

  function handleMaTypeChange(nextMaType: string) {
    void load({ ticker_type: selectedTickerType, ma_type: nextMaType, ma_months: maMonths, as_of_date: selectedAsOfDate });
  }

  function handleMaMonthsChange(nextMaMonths: number) {
    void load({ ticker_type: selectedTickerType, ma_type: maType, ma_months: nextMaMonths, as_of_date: selectedAsOfDate });
  }

  function handleAsOfDateChange(nextAsOfDate: string) {
    setSelectedAsOfDate(nextAsOfDate);
    void load({ ticker_type: selectedTickerType, ma_type: maType, ma_months: maMonths, as_of_date: nextAsOfDate });
  }

  const blockedMessage = useMemo(() => {
    if (!cacheBlocked) {
      return null;
    }

    const parts: string[] = [];
    parts.push("일부 종목의 가격 캐시가 없습니다. 종목 관리에서 해당 종목의 메타/캐시 새로고침을 실행하세요.");
    if (missingTickerLabels.length > 0) {
      parts.push(`누락 ${missingTickerLabels.join(", ")}`);
    } else if (missingTickers.length > 0) {
      parts.push(`누락 ${missingTickers.join(", ")}`);
    }
    if (staleTickers.length > 0) {
      parts.push(`오래된 캐시 ${staleTickers.join(", ")}`);
    }
    return parts.join(" | ");
  }, [cacheBlocked, missingTickerLabels, missingTickers, staleTickers]);

  useEffect(() => {
    if (!blockedMessage) {
      lastBlockedToastRef.current = null;
      return;
    }

    if (lastBlockedToastRef.current === blockedMessage) {
      return;
    }

    lastBlockedToastRef.current = blockedMessage;
    toast.error(`[ETF-순위] ${blockedMessage}`);
  }, [blockedMessage, toast]);

  return (
    <div className="appPageStack appPageStackFill">
      {error ? (
        <div className="appBannerStack">
          <div className="bannerError alert alert-danger mb-0">{error}</div>
        </div>
      ) : null}

      <section className="appSection appSectionFill">
        <div className="card appCard">
          <div className="card-header">
            <div className="tickerTypeToolbar w-100" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div className="tickerTypeToolbarLeft" style={{ display: "flex", gap: "0.4rem", alignItems: "center" }}>
                <input
                  className="form-control"
                  type="date"
                  style={{ width: "auto", fontWeight: 600 }}
                  value={selectedAsOfDate}
                  max={getTodayDateInputValue()}
                  onChange={(event) => handleAsOfDateChange(event.target.value)}
                />
                <select
                  className="form-select"
                  style={{ width: "auto", minWidth: "180px", fontWeight: 600 }}
                  value={selectedTickerType}
                  onChange={(event) => handleTickerTypeChange(event.target.value)}
                  disabled={ticker_types.length === 0}
                >
                  {ticker_types.length === 0 ? (
                    <option value="">종목 타입 불러오는 중...</option>
                  ) : (
                    ticker_types.map((account) => (
                      <option key={account.ticker_type} value={account.ticker_type}>
                        {account.name}
                      </option>
                    ))
                  )}
                </select>

                <select
                  className="form-select"
                  style={{ width: "auto", fontWeight: 600 }}
                  value={maType}
                  onChange={(event) => handleMaTypeChange(event.target.value)}
                  disabled={maTypeOptions.length === 0}
                >
                  {maTypeOptions.map((option) => (
                    <option key={option} value={option}>
                      MA: {option}
                    </option>
                  ))}
                </select>

                <select
                  className="form-select"
                  style={{ width: "auto", fontWeight: 600 }}
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
                <div className="btn-group" role="group" aria-label="수익률 보기 방식">
                  <button
                    type="button"
                    className={metricMode === "cumulative" ? "btn btn-primary btn-sm" : "btn btn-outline-primary btn-sm"}
                    onClick={() => setMetricMode("cumulative")}
                  >
                    누적
                  </button>
                  <button
                    type="button"
                    className={metricMode === "monthly" ? "btn btn-primary btn-sm" : "btn btn-outline-primary btn-sm"}
                    onClick={() => setMetricMode("monthly")}
                  >
                    월별
                  </button>
                </div>
                <input
                  className="form-control"
                  type="text"
                  style={{ width: "200px", fontWeight: 600 }}
                  value={nameKeyword}
                  placeholder="종목명 검색"
                  onChange={(event) => setNameKeyword(event.target.value)}
                />
              </div>

              <div className="tickerTypeToolbarRight" style={{ display: "flex", alignItems: "center", gap: "1.25rem" }}>
                <div className="stocksSummary d-flex align-items-center gap-3">
                  {filteredGridRows.length > 0 ? (() => {
                    const upCount = filteredGridRows.filter((r) => (r["추세"] ?? 0) > 0).length;
                    const upPct = Math.round((upCount / filteredGridRows.length) * 100);
                    return (
                      <div className="d-flex align-items-center gap-1">
                        <span style={{ color: "#6c757d", fontSize: "0.85rem", fontWeight: 600 }}>추세 상승:</span>
                        <span style={{ fontWeight: 700, color: "#d63939" }}>{upCount}개 ({upPct}%)</span>
                      </div>
                    );
                  })() : null}
                  <div className="d-flex align-items-center gap-1">
                    <span style={{ color: "#6c757d", fontSize: "0.85rem", fontWeight: 600 }}>총 개수:</span>
                    <span style={{ fontWeight: 700 }}>{new Intl.NumberFormat("ko-KR").format(filteredGridRows.length)}개</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="card-body appCardBodyTight">
            <div className="rankGridWrap">
              <AppDataGrid
                className="rankDataGrid"
                rows={filteredGridRows}
                columns={columns}
                loading={loading}
                hideFooter={false}
                pageSizeOptions={[20]}
                paginationModel={{ page, pageSize: 20 }}
                onPaginationModelChange={(model) => setPage(model.page)}
                getRowClassName={(params) => {
                  const classes: string[] = [];
                  if ((params.row.추세 ?? 0) < 0) {
                    classes.push("rankNegativeTrendRow");
                  }
                  if (String(params.row.displayTrendRank || "").startsWith("보유")) {
                    classes.push("rankHeldRow");
                  }
                  return classes.join(" ");
                }}
                minHeight="70vh"
              />
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
