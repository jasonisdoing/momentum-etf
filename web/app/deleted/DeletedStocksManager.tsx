"use client";

import { useEffect, useMemo, useState, useTransition } from "react";
import { type GridColDef, type GridRowSelectionModel } from "@mui/x-data-grid";

import { AppDataGrid } from "../components/AppDataGrid";
import { AppLoadingState } from "../components/AppLoadingState";
import { useToast } from "../components/ToastProvider";

type DeletedStocksAccountItem = {
  ticker_type: string;
  order: number;
  name: string;
  icon: string;
};

type DeletedStocksRowItem = {
  ticker: string;
  name: string;
  bucket_id: number;
  bucket_name: string;
  listing_date: string;
  week_volume: number | null;
  return_1w: number | null;
  return_2w: number | null;
  return_1m: number | null;
  return_3m: number | null;
  return_6m: number | null;
  return_12m: number | null;
  deleted_date: string;
  deleted_reason: string;
};

type DeletedStocksResponse = {
  ticker_types?: DeletedStocksAccountItem[];
  rows?: DeletedStocksRowItem[];
  ticker_type?: string;
  error?: string;
};

function formatNumber(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ko-KR").format(value);
}

function formatPercent(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${value.toFixed(2)}%`;
}

function getSignedMetricClass(value: number | null): string | undefined {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return undefined;
  }
  if (value > 0) {
    return "metricPositive";
  }
  if (value < 0) {
    return "metricNegative";
  }
  return undefined;
}

type DeletedStocksGridRow = DeletedStocksRowItem & { id: string };

const columns: GridColDef<DeletedStocksGridRow>[] = [
  { field: "bucket_name", headerName: "버킷", minWidth: 110, width: 110 },
  { field: "ticker", headerName: "티커", minWidth: 90, width: 90 },
  { field: "name", headerName: "종목명", minWidth: 130, flex: 1 },
  {
    field: "week_volume",
    headerName: "주간거래량",
    minWidth: 110,
    align: "right",
    headerAlign: "right",
    renderCell: (params) => formatNumber(params.value),
  },
  ...([1, 2] as const).map((n) => ({
    field: `return_${n}w` as const,
    headerName: `${n}주(%)`,
    minWidth: 86,
    align: "right" as const,
    headerAlign: "right" as const,
    renderCell: (params: { row: DeletedStocksGridRow }) => {
      const value = params.row[`return_${n}w` as keyof DeletedStocksGridRow] as number | null;
      return <span className={getSignedMetricClass(value)}>{formatPercent(value)}</span>;
    },
  })),
  ...([1, 3, 6, 12] as const).map((n) => ({
    field: `return_${n}m` as const,
    headerName: `${n}달(%)`,
    minWidth: 86,
    align: "right" as const,
    headerAlign: "right" as const,
    renderCell: (params: { row: DeletedStocksGridRow }) => {
      const value = params.row[`return_${n}m` as keyof DeletedStocksGridRow] as number | null;
      return <span className={getSignedMetricClass(value)}>{formatPercent(value)}</span>;
    },
  })),
  { field: "listing_date", headerName: "상장일", minWidth: 100, width: 100 },
  { field: "deleted_date", headerName: "삭제일", minWidth: 100, width: 100 },
  { field: "deleted_reason", headerName: "삭제 사유", minWidth: 120, flex: 0.8 },
];

export function DeletedStocksManager() {
  const [ticker_types, setAccounts] = useState<DeletedStocksAccountItem[]>([]);
  const [selectedTickerType, setSelectedAccountId] = useState("");
  const [rows, setRows] = useState<DeletedStocksRowItem[]>([]);
  const [selectionModel, setSelectionModel] = useState<GridRowSelectionModel>({ type: "include", ids: new Set() });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [isPending, startTransition] = useTransition();
  const toast = useToast();

  async function load(tickerType?: string) {
    setLoading(true);
    setError(null);

    try {
      const search = tickerType ? `?type=${encodeURIComponent(tickerType)}` : "";
      const response = await fetch(`/api/deleted${search}`, { cache: "no-store" });
      const payload = (await response.json()) as DeletedStocksResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "삭제된 종목 데이터를 불러오지 못했습니다.");
      }

      setAccounts(payload.ticker_types ?? []);
      setSelectedAccountId(payload.ticker_type ?? "");
      setRows(payload.rows ?? []);
      setSelectionModel({ type: "include", ids: new Set() });
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "삭제된 종목 데이터를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  const selectedTickerTypeItem = useMemo(
    () => ticker_types.find((account) => account.ticker_type === selectedTickerType) ?? null,
    [ticker_types, selectedTickerType],
  );

  const selectedTickers = useMemo(
    () => Array.from(selectionModel.ids) as string[],
    [selectionModel],
  );

  const gridRows: DeletedStocksGridRow[] = useMemo(
    () => rows.map((row) => ({ ...row, id: row.ticker })),
    [rows],
  );

  function handleTickerTypeChange(nextAccountId: string) {
    void load(nextAccountId);
  }

  function handleRestore() {
    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/deleted", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ticker_type: selectedTickerType,
            tickers: selectedTickers,
          }),
        });
        const payload = (await response.json()) as { error?: string; restored_count?: number };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 복구에 실패했습니다.");
        }
        const restoredCount = Number(payload.restored_count ?? 0);
        const removed = new Set(selectedTickers);
        setRows((current) => current.filter((row) => !removed.has(row.ticker)));
        setSelectionModel({ type: "include", ids: new Set() });
        toast.success(`[Momentum ETF-삭제된 종목] ${restoredCount}개 종목 복구 완료`);
      } catch (restoreError) {
        setError(restoreError instanceof Error ? restoreError.message : "종목 복구에 실패했습니다.");
      }
    });
  }

  function handleHardDelete() {
    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/deleted", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ticker_type: selectedTickerType,
            tickers: selectedTickers,
          }),
        });
        const payload = (await response.json()) as { error?: string; deleted_count?: number };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 완전 삭제에 실패했습니다.");
        }
        const deletedCount = Number(payload.deleted_count ?? 0);
        const removed = new Set(selectedTickers);
        setRows((current) => current.filter((row) => !removed.has(row.ticker)));
        setSelectionModel({ type: "include", ids: new Set() });
        toast.success(`[Momentum ETF-삭제된 종목] ${deletedCount}개 종목 영구 삭제 완료`);
      } catch (deleteError) {
        setError(deleteError instanceof Error ? deleteError.message : "종목 완전 삭제에 실패했습니다.");
      }
    });
  }

  if (loading) {
    return (
      <div className="appPageStack">
        <div className="appPageLoading">
          <AppLoadingState label="삭제된 종목 데이터를 불러오는 중..." />
        </div>
      </div>
    );
  }

  return (
    <div className="appPageStack">
      {error ? (
        <div className="appBannerStack">
          {error ? <div className="bannerError">{error}</div> : null}
        </div>
      ) : null}
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBody">
            <div className="tableToolbar">
              <div className="toolbarActions">
                <select
                  className="field compactField"
                  value={selectedTickerType}
                  onChange={(event) => handleTickerTypeChange(event.target.value)}
                >
                  {ticker_types.map((account) => (
                    <option key={account.ticker_type} value={account.ticker_type}>
                      {account.order}. {account.name}
                    </option>
                  ))}
                </select>
                <button
                  className="btn btn-outline-secondary btn-sm"
                  type="button"
                  onClick={handleRestore}
                  disabled={selectedTickers.length === 0 || isPending}
                >
                  선택 복구
                </button>
                <button
                  className="btn btn-outline-danger btn-sm"
                  type="button"
                  onClick={handleHardDelete}
                  disabled={selectedTickers.length === 0 || isPending}
                >
                  완전 삭제
                </button>
              </div>
              <div className="tableMeta">
                {selectedTickerTypeItem ? (
                  <span>
                    {selectedTickerTypeItem.icon} {selectedTickerTypeItem.name}
                  </span>
                ) : null}
                <span>총 {new Intl.NumberFormat("ko-KR").format(rows.length)}개 종목</span>
                <span>선택 {new Intl.NumberFormat("ko-KR").format(selectedTickers.length)}개</span>
              </div>
            </div>
            <AppDataGrid
              rows={gridRows}
              columns={columns}
              loading={loading}
              checkboxSelection
              rowSelectionModel={selectionModel}
              onRowSelectionModelChange={setSelectionModel}
              minHeight="68vh"
            />
          </div>
        </div>
      </section>
    </div>
  );
}
