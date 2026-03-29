"use client";

import {
  IconCircleCheck,
  IconArrowBackUp,
  IconPlus,
  IconChecks,
  IconLayoutGrid,
  IconPlaylistX,
  IconTrash,
  IconSearch,
  IconRefresh,
} from "@tabler/icons-react";
import { useEffect, useMemo, useState, useTransition } from "react";
import { type GridColDef, type GridRowSelectionModel } from "@mui/x-data-grid";

import { BUCKET_OPTIONS } from "@/lib/bucket-theme";
import { AppDataGrid } from "../components/AppDataGrid";
import { AppModal } from "../components/AppModal";
import {
  readRememberedMomentumEtfAccountId,
  writeRememberedMomentumEtfAccountId,
} from "../components/account-selection";
import { useToast } from "../components/ToastProvider";

type StocksAccountItem = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
};

type ActiveStocksRowItem = {
  ticker: string;
  name: string;
  bucket_id: number;
  bucket_name: string;
  added_date: string;
  listing_date: string;
  week_volume: number | null;
  return_1w: number | null;
  return_2w: number | null;
  return_1m: number | null;
  return_3m: number | null;
  return_6m: number | null;
  return_12m: number | null;
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

type StocksResponse = {
  accounts?: StocksAccountItem[];
  rows?: ActiveStocksRowItem[];
  account_id?: string;
  error?: string;
};

type DeletedStocksResponse = {
  accounts?: StocksAccountItem[];
  rows?: DeletedStocksRowItem[];
  account_id?: string;
  error?: string;
};

type ViewMode = "active" | "deleted";

type ActiveStockGridRow = ActiveStocksRowItem & { id: string };
type DeletedStockGridRow = DeletedStocksRowItem & { id: string };
type StockValidationState = {
  ticker: string;
  name: string;
  listing_date: string;
  status: "active" | "deleted" | "new";
  is_deleted: boolean;
  deleted_reason: string;
  bucket_id: number;
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

function getSignedMetricClass(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value) || value === 0) {
    return "";
  }
  return value > 0 ? "stocksPositive" : "stocksNegative";
}

function getBucketClass(bucketId: number): string {
  return `stocksBucket stocksBucket${bucketId}`;
}

export function StocksManager() {
  const [accounts, setAccounts] = useState<StocksAccountItem[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("active");
  const [activeRows, setActiveRows] = useState<ActiveStocksRowItem[]>([]);
  const [deletedRows, setDeletedRows] = useState<DeletedStocksRowItem[]>([]);
  const [selectedDeletedTickers, setSelectedDeletedTickers] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [isPending, startTransition] = useTransition();
  const [editingRow, setEditingRow] = useState<ActiveStocksRowItem | null>(null);
  const [editingBucketId, setEditingBucketId] = useState<number>(1);
  const [editingDeleteReason, setEditingDeleteReason] = useState("");
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [addTickerInput, setAddTickerInput] = useState("");
  const [validatedCandidate, setValidatedCandidate] = useState<StockValidationState | null>(null);
  const [isValidatingTicker, setIsValidatingTicker] = useState(false);
  const [addBucketId, setAddBucketId] = useState<number>(1);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const toast = useToast();

  async function load(mode: ViewMode, accountId?: string) {
    setLoading(true);
    setError(null);

    try {
      const search = accountId ? `?account=${encodeURIComponent(accountId)}` : "";
      const apiPath = mode === "active" ? `/api/stocks${search}` : `/api/deleted${search}`;
      const response = await fetch(apiPath, { cache: "no-store" });
      const payload = (await response.json()) as StocksResponse | DeletedStocksResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "종목 관리 데이터를 불러오지 못했습니다.");
      }

      setAccounts(payload.accounts ?? []);
      const nextAccountId = payload.account_id ?? "";
      setSelectedAccountId(nextAccountId);
      writeRememberedMomentumEtfAccountId(nextAccountId);
      setSelectedDeletedTickers([]);

      if (mode === "active") {
        setActiveRows((payload.rows as ActiveStocksRowItem[] | undefined) ?? []);
      } else {
        setDeletedRows((payload.rows as DeletedStocksRowItem[] | undefined) ?? []);
      }
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "종목 관리 데이터를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load(viewMode, readRememberedMomentumEtfAccountId() ?? undefined);
  }, []);

  const selectedAccount = useMemo(
    () => accounts.find((account) => account.account_id === selectedAccountId) ?? null,
    [accounts, selectedAccountId],
  );

  const selectedDeletedTickerSet = useMemo(() => new Set(selectedDeletedTickers), [selectedDeletedTickers]);
  const allDeletedSelected = deletedRows.length > 0 && selectedDeletedTickers.length === deletedRows.length;
  const activeGridRows = useMemo<ActiveStockGridRow[]>(
    () => activeRows.map((row) => ({ ...row, id: row.ticker.trim().toUpperCase() })),
    [activeRows],
  );
  const deletedGridRows = useMemo<DeletedStockGridRow[]>(
    () => deletedRows.map((row) => ({ ...row, id: row.ticker.trim().toUpperCase() })),
    [deletedRows],
  );
  const activeColumns = useMemo<GridColDef<ActiveStockGridRow>[]>(
    () => [
      {
        field: "__edit__",
        headerName: "",
        width: 58,
        sortable: false,
        filterable: false,
        disableColumnMenu: true,
        renderCell: (params) => (
          <button className="btn btn-link btn-sm p-0 appEditLink" type="button" onClick={() => openEditModal(params.row)}>
            Edit
          </button>
        ),
      },
      {
        field: "bucket_name",
        headerName: "버킷",
        width: 112,
        minWidth: 112,
        sortable: false,
        cellClassName: (params) => `appBucketCell appBucketCell${params.row.bucket_id}`,
      },
      {
        field: "ticker",
        headerName: "티커",
        width: 98,
        minWidth: 98,
        renderCell: (params) => <span className="appCodeText">{params.row.ticker}</span>,
      },
      { field: "name", headerName: "종목명", minWidth: 220, flex: 1 },
      {
        field: "week_volume",
        headerName: "주간거래량",
        width: 116,
        minWidth: 116,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatNumber(params.row.week_volume),
      },
      ...(["return_1w", "return_2w", "return_1m", "return_3m", "return_6m", "return_12m"] as const).map(
        (field) => ({
          field,
          headerName:
            field === "return_1w"
              ? "1주(%)"
              : field === "return_2w"
                ? "2주(%)"
                : field === "return_1m"
                  ? "1달(%)"
                  : field === "return_3m"
                    ? "3달(%)"
                    : field === "return_6m"
                      ? "6달(%)"
                      : "12달(%)",
          width: 88,
          minWidth: 88,
          align: "right" as const,
          headerAlign: "right" as const,
          renderCell: (params: { row: ActiveStockGridRow }) => (
            <span className={getSignedMetricClass(params.row[field])}>{formatPercent(params.row[field])}</span>
          ),
        }),
      ),
      { field: "listing_date", headerName: "상장일", width: 112, minWidth: 112 },
      { field: "added_date", headerName: "추가일자", width: 112, minWidth: 112 },
    ],
    [],
  );
  const deletedColumns = useMemo<GridColDef<DeletedStockGridRow>[]>(
    () => [
      {
        field: "bucket_name",
        headerName: "버킷",
        width: 112,
        minWidth: 112,
        sortable: false,
        cellClassName: (params) => `appBucketCell appBucketCell${params.row.bucket_id}`,
      },
      {
        field: "ticker",
        headerName: "티커",
        width: 98,
        minWidth: 98,
        renderCell: (params) => <span className="appCodeText">{params.row.ticker}</span>,
      },
      { field: "name", headerName: "종목명", minWidth: 220, flex: 1 },
      {
        field: "week_volume",
        headerName: "주간거래량",
        width: 116,
        minWidth: 116,
        align: "right",
        headerAlign: "right",
        renderCell: (params) => formatNumber(params.row.week_volume),
      },
      ...(["return_1w", "return_2w", "return_1m", "return_3m", "return_6m", "return_12m"] as const).map(
        (field) => ({
          field,
          headerName:
            field === "return_1w"
              ? "1주(%)"
              : field === "return_2w"
                ? "2주(%)"
                : field === "return_1m"
                  ? "1달(%)"
                  : field === "return_3m"
                    ? "3달(%)"
                    : field === "return_6m"
                      ? "6달(%)"
                      : "12달(%)",
          width: 88,
          minWidth: 88,
          align: "right" as const,
          headerAlign: "right" as const,
          renderCell: (params: { row: DeletedStockGridRow }) => (
            <span className={getSignedMetricClass(params.row[field])}>{formatPercent(params.row[field])}</span>
          ),
        }),
      ),
      { field: "listing_date", headerName: "상장일", width: 112, minWidth: 112 },
      { field: "deleted_date", headerName: "삭제일", width: 112, minWidth: 112 },
      { field: "deleted_reason", headerName: "삭제 사유", minWidth: 160, flex: 0.7 },
    ],
    [],
  );

  function handleAccountChange(nextAccountId: string) {
    writeRememberedMomentumEtfAccountId(nextAccountId);
    void load(viewMode, nextAccountId);
  }

  function handleViewModeChange(nextMode: ViewMode) {
    if (nextMode === viewMode) {
      return;
    }
    setViewMode(nextMode);
    void load(nextMode, selectedAccountId);
  }

  function handleBucketChange(ticker: string, bucketId: number) {
    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/stocks", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            ticker,
            bucket_id: bucketId,
          }),
        });
        const payload = (await response.json()) as { error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "버킷 변경에 실패했습니다.");
        }
        setActiveRows((current) =>
          current.map((row) =>
            row.ticker === ticker
              ? {
                  ...row,
                  bucket_id: bucketId,
                  bucket_name: BUCKET_OPTIONS.find((bucket) => bucket.id === bucketId)?.name ?? row.bucket_name,
                }
              : row,
          ),
        );
        const targetRow = activeRows.find((row) => row.ticker === ticker);
        const label = targetRow ? `${targetRow.name}(${targetRow.ticker})` : ticker;
        toast.success(`[Momentum ETF-종목 관리] ${label} 변경 완료`);
      } catch (updateError) {
        setError(updateError instanceof Error ? updateError.message : "버킷 변경에 실패했습니다.");
      }
    });
  }

  function handleDelete(ticker: string) {
    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/stocks", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            ticker,
          }),
        });
        const payload = (await response.json()) as { error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 삭제에 실패했습니다.");
        }
        setActiveRows((current) => current.filter((row) => row.ticker !== ticker));
        const targetRow = activeRows.find((row) => row.ticker === ticker);
        const label = targetRow ? `${targetRow.name}(${targetRow.ticker})` : ticker;
        toast.success(`[Momentum ETF-종목 관리] ${label} 삭제 완료`);
      } catch (deleteError) {
        setError(deleteError instanceof Error ? deleteError.message : "종목 삭제에 실패했습니다.");
      }
    });
  }

  function openEditModal(row: ActiveStocksRowItem) {
    setEditingRow(row);
    setEditingBucketId(row.bucket_id);
    setEditingDeleteReason("");
  }

  function openAddModal() {
    setIsAddModalOpen(true);
    setAddTickerInput("");
    setValidatedCandidate(null);
    setAddBucketId(1);
  }

  function closeAddModal() {
    if (isPending || isValidatingTicker) {
      return;
    }
    setIsAddModalOpen(false);
    setAddTickerInput("");
    setValidatedCandidate(null);
    setAddBucketId(1);
  }

  function closeEditModal() {
    if (isPending) {
      return;
    }
    setEditingRow(null);
    setEditingDeleteReason("");
  }

  async function handleValidateTicker() {
    try {
      setError(null);
      setIsValidatingTicker(true);
      const response = await fetch("/api/stocks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "validate",
          account_id: selectedAccountId,
          ticker: addTickerInput,
        }),
      });
      const payload = (await response.json()) as
        | ({
            error?: string;
          } & StockValidationState)
        | { error?: string };
      if (!response.ok) {
        throw new Error(payload.error ?? "티커 확인에 실패했습니다.");
      }
      setValidatedCandidate({
        ticker: String((payload as StockValidationState).ticker ?? "").trim().toUpperCase(),
        name: String((payload as StockValidationState).name ?? "").trim(),
        listing_date: String((payload as StockValidationState).listing_date ?? "-").trim() || "-",
        status: (payload as StockValidationState).status,
        is_deleted: Boolean((payload as StockValidationState).is_deleted),
        deleted_reason: String((payload as StockValidationState).deleted_reason ?? "").trim(),
        bucket_id: Number((payload as StockValidationState).bucket_id ?? 1),
      });
      setAddBucketId(Number((payload as StockValidationState).bucket_id ?? 1));
    } catch (validationError) {
      setValidatedCandidate(null);
      setError(validationError instanceof Error ? validationError.message : "티커 확인에 실패했습니다.");
    } finally {
      setIsValidatingTicker(false);
    }
  }

  function handleAddTickerInputChange(value: string) {
    setAddTickerInput(value);
    if (validatedCandidate) {
      setValidatedCandidate(null);
    }
  }

  function handleCreateStock() {
    if (!validatedCandidate) {
      return;
    }

    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/stocks", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "create",
            account_id: selectedAccountId,
            ticker: validatedCandidate.ticker,
            bucket_id: addBucketId,
          }),
        });
        const payload = (await response.json()) as {
          error?: string;
          ticker?: string;
          name?: string;
          listing_date?: string;
          bucket_id?: number;
          bucket_name?: string;
        };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 추가에 실패했습니다.");
        }
        setActiveRows((current) => [
          {
            ticker: String(payload.ticker ?? validatedCandidate.ticker).trim().toUpperCase(),
            name: String(payload.name ?? validatedCandidate.name).trim(),
            bucket_id: Number(payload.bucket_id ?? addBucketId),
            bucket_name:
              String(payload.bucket_name ?? "").trim() ||
              BUCKET_OPTIONS.find((bucket) => bucket.id === addBucketId)?.name ||
              "1. 모멘텀",
            added_date: new Date().toISOString().slice(0, 10),
            listing_date: String(payload.listing_date ?? validatedCandidate.listing_date ?? "-").trim() || "-",
            week_volume: null,
            return_1w: null,
            return_2w: null,
            return_1m: null,
            return_3m: null,
            return_6m: null,
            return_12m: null,
          },
          ...current,
        ]);
        toast.success(
          `[ETF-종목 관리] ${validatedCandidate.name}(${validatedCandidate.ticker}) ${
            validatedCandidate.is_deleted ? "복구 완료" : "추가 완료"
          }`,
        );
        closeAddModal();
      } catch (createError) {
        setError(createError instanceof Error ? createError.message : "종목 추가에 실패했습니다.");
      }
    });
  }

  function handleSaveFromModal() {
    if (!editingRow) {
      return;
    }

    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/stocks", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            ticker: editingRow.ticker,
            bucket_id: editingBucketId,
          }),
        });
        const payload = (await response.json()) as { error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "버킷 변경에 실패했습니다.");
        }
        setActiveRows((current) =>
          current.map((row) =>
            row.ticker === editingRow.ticker
              ? {
                  ...row,
                  bucket_id: editingBucketId,
                  bucket_name: BUCKET_OPTIONS.find((bucket) => bucket.id === editingBucketId)?.name ?? row.bucket_name,
                }
              : row,
          ),
        );
        toast.success(`[Momentum ETF-종목 관리] ${editingRow.name}(${editingRow.ticker}) 변경 완료`);
        setEditingRow(null);
      } catch (updateError) {
        setError(updateError instanceof Error ? updateError.message : "버킷 변경에 실패했습니다.");
      }
    });
  }

  function handleDeleteFromModal() {
    if (!editingRow) {
      return;
    }

    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/stocks", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            ticker: editingRow.ticker,
            reason: editingDeleteReason.trim() || undefined,
          }),
        });
        const payload = (await response.json()) as { error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 삭제에 실패했습니다.");
        }
        setActiveRows((current) => current.filter((row) => row.ticker !== editingRow.ticker));
        toast.success(`[Momentum ETF-종목 관리] ${editingRow.name}(${editingRow.ticker}) 삭제 완료`);
        setEditingRow(null);
        setEditingDeleteReason("");
      } catch (deleteError) {
        setError(deleteError instanceof Error ? deleteError.message : "종목 삭제에 실패했습니다.");
      }
    });
  }

  async function handleRefreshFromModal() {
    if (!editingRow) {
      return;
    }

    try {
      setIsRefreshing(true);
      setError(null);
      const response = await fetch("/api/stocks/refresh", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          account_id: selectedAccountId,
          ticker: editingRow.ticker,
        }),
      });
      const payload = (await response.json()) as { error?: string };
      if (!response.ok) {
        throw new Error(payload.error ?? "종목 새로고침에 실패했습니다.");
      }
      toast.success(`[Momentum ETF-종목 관리] ${editingRow.name}(${editingRow.ticker}) 새로고침 완료`);
    } catch (refreshError) {
      setError(refreshError instanceof Error ? refreshError.message : "종목 새로고침에 실패했습니다.");
    } finally {
      setIsRefreshing(false);
    }
  }

  function toggleAllDeleted() {
    if (allDeletedSelected) {
      setSelectedDeletedTickers([]);
      return;
    }
    setSelectedDeletedTickers(deletedRows.map((row) => row.ticker.trim().toUpperCase()));
  }

  function handleDeletedSelectionChange(model: GridRowSelectionModel) {
    setSelectedDeletedTickers(Array.from(model.ids, (item) => String(item).trim().toUpperCase()));
  }

  function handleRestoreDeleted() {
    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/deleted", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            tickers: selectedDeletedTickers,
          }),
        });
        const payload = (await response.json()) as { error?: string; restored_count?: number };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 복구에 실패했습니다.");
        }
        const restoredCount = Number(payload.restored_count ?? 0);
        setDeletedRows((current) =>
          current.filter((row) => !selectedDeletedTickerSet.has(row.ticker.trim().toUpperCase())),
        );
        setSelectedDeletedTickers([]);
        toast.success(`[Momentum ETF-종목 관리] ${restoredCount}개 종목 복구 완료`);
      } catch (restoreError) {
        setError(restoreError instanceof Error ? restoreError.message : "종목 복구에 실패했습니다.");
      }
    });
  }

  function handleHardDeleteDeleted() {
    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/deleted", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: selectedAccountId,
            tickers: selectedDeletedTickers,
          }),
        });
        const payload = (await response.json()) as { error?: string; deleted_count?: number };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 완전 삭제에 실패했습니다.");
        }
        const deletedCount = Number(payload.deleted_count ?? 0);
        setDeletedRows((current) =>
          current.filter((row) => !selectedDeletedTickerSet.has(row.ticker.trim().toUpperCase())),
        );
        setSelectedDeletedTickers([]);
        toast.success(`[Momentum ETF-종목 관리] ${deletedCount}개 종목 영구 삭제 완료`);
      } catch (deleteError) {
        setError(deleteError instanceof Error ? deleteError.message : "종목 완전 삭제에 실패했습니다.");
      }
    });
  }

  return (
    <div className="appPageStack appPageStackFill">
      {error ? (
        <div className="appBannerStack">
          {error ? <div className="alert alert-danger mb-0">{error}</div> : null}
        </div>
      ) : null}

      <section className="appSection appSectionFill stocksPage">
        <div className="card appCard stocksCard">
          <div className="card-header">
            <div className="accountToolbar w-100">
              <div className="accountToolbarLeft">
                <div className="accountSelect">
                  <select
                    className="form-select"
                    aria-label="계좌 선택"
                    value={selectedAccountId}
                    onChange={(event) => handleAccountChange(event.target.value)}
                    disabled={loading}
                  >
                    {accounts.map((account) => (
                      <option key={account.account_id} value={account.account_id}>
                        {account.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="accountToolbarOptions stocksToolbarModes">
                  <button className="btn btn-primary" type="button" onClick={openAddModal} disabled={loading || viewMode !== "active"}>
                    <IconPlus size={16} stroke={1.75} />
                    <span>종목 추가</span>
                  </button>
                  <button
                    className={
                      viewMode === "active" ? "btn stocksModeButton is-active" : "btn btn-outline-secondary stocksModeButton"
                    }
                    type="button"
                    onClick={() => handleViewModeChange("active")}
                  >
                    <IconLayoutGrid size={16} stroke={1.75} />
                    <span>등록된 종목</span>
                  </button>
                  <button
                    className={
                      viewMode === "deleted" ? "btn stocksModeButton is-active" : "btn btn-outline-secondary stocksModeButton"
                    }
                    type="button"
                    onClick={() => handleViewModeChange("deleted")}
                  >
                    <IconPlaylistX size={16} stroke={1.75} />
                    <span>삭제된 종목</span>
                  </button>
                </div>
              </div>

              <div className="accountToolbarRight">
                <div className="stocksSummary">
                  {selectedAccount ? (
                    <span className="badge stocksMetricBadge">
                      {selectedAccount.icon} {selectedAccount.name}
                    </span>
                  ) : null}
                  <span className="badge stocksMetricBadge">
                    {viewMode === "active"
                      ? `총 ${new Intl.NumberFormat("ko-KR").format(activeRows.length)}개`
                      : `총 ${new Intl.NumberFormat("ko-KR").format(deletedRows.length)}개`}
                  </span>
                  {viewMode === "deleted" ? (
                    <span className="badge stocksMetricBadge">
                      선택 {new Intl.NumberFormat("ko-KR").format(selectedDeletedTickers.length)}개
                    </span>
                  ) : null}
                </div>

                {viewMode === "deleted" ? (
                  <div className="btn-list">
                    <button
                      className="btn btn-outline-secondary"
                      type="button"
                      onClick={toggleAllDeleted}
                      disabled={deletedRows.length === 0 || isPending}
                    >
                      <IconChecks size={16} stroke={1.75} />
                      <span>{allDeletedSelected ? "전체 해제" : "전체 선택"}</span>
                    </button>
                    <button
                      className="btn btn-primary"
                      type="button"
                      onClick={handleRestoreDeleted}
                      disabled={selectedDeletedTickers.length === 0 || isPending}
                    >
                      <IconArrowBackUp size={16} stroke={1.75} />
                      <span>선택 복구</span>
                    </button>
                    <button
                      className="btn btn-outline-danger"
                      type="button"
                      onClick={handleHardDeleteDeleted}
                      disabled={selectedDeletedTickers.length === 0 || isPending}
                    >
                      <IconTrash size={16} stroke={1.75} />
                      <span>완전 삭제</span>
                    </button>
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          <div className="stocksTableWrap">
            {viewMode === "active" ? (
              <AppDataGrid
                rows={activeGridRows}
                columns={activeColumns}
                loading={loading}
                minHeight="68vh"
              />
            ) : (
              <AppDataGrid
                rows={deletedGridRows}
                columns={deletedColumns}
                loading={loading}
                minHeight="68vh"
                checkboxSelection
                rowSelectionModel={{ type: "include", ids: new Set(selectedDeletedTickers) }}
                onRowSelectionModelChange={handleDeletedSelectionChange}
              />
            )}
          </div>
        </div>
      </section>
      <AppModal
        open={isAddModalOpen}
        title="종목 추가"
        onClose={closeAddModal}
        footer={
          <>
            <button type="button" className="btn btn-link link-secondary" onClick={closeAddModal} disabled={isPending || isValidatingTicker}>
              취소
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleCreateStock}
              disabled={isPending || isValidatingTicker || !validatedCandidate || !addBucketId}
            >
              저장
            </button>
          </>
        }
      >
        <div className="mb-3">
          <label className="form-label">티커</label>
          <div className="d-flex gap-2">
            <input
              className="form-control appCodeText"
              value={addTickerInput}
              onChange={(event) => handleAddTickerInputChange(event.target.value)}
              placeholder="예: 069500 / VGS / ASX:VGS"
              disabled={isPending || isValidatingTicker}
            />
            <button
              type="button"
              className="btn btn-outline-secondary"
              onClick={() => void handleValidateTicker()}
              disabled={isPending || isValidatingTicker || !addTickerInput.trim()}
            >
              <IconSearch size={16} stroke={1.9} />
              <span>확인</span>
            </button>
          </div>
        </div>

        {validatedCandidate ? (
          <div className="mb-3">
            <div className="alert alert-success d-flex flex-column gap-2 mb-0">
              <div className="d-flex align-items-center gap-2">
                <IconCircleCheck size={18} stroke={1.9} />
                <strong>{validatedCandidate.name}</strong>
              </div>
              <div className="small">티커: <span className="appCodeText">{validatedCandidate.ticker}</span></div>
              <div className="small">상장일: {validatedCandidate.listing_date}</div>
              {validatedCandidate.is_deleted ? (
                <div className="small text-warning-emphasis">
                  삭제된 종목 입니다{validatedCandidate.deleted_reason ? ` (${validatedCandidate.deleted_reason})` : ""}.
                </div>
              ) : null}
              {validatedCandidate.status === "active" ? (
                <div className="small text-danger">이미 등록된 종목입니다. 저장할 수 없습니다.</div>
              ) : null}
            </div>
          </div>
        ) : null}

        <div className="mb-0">
          <label className="form-label">버킷</label>
          <select
            className="form-select"
            value={addBucketId}
            onChange={(event) => setAddBucketId(Number(event.target.value))}
            disabled={isPending || isValidatingTicker || !validatedCandidate || validatedCandidate.status === "active"}
          >
            {BUCKET_OPTIONS.map((bucket) => (
              <option key={bucket.id} value={bucket.id}>
                {bucket.name}
              </option>
            ))}
          </select>
        </div>
      </AppModal>
      <AppModal
        open={Boolean(editingRow)}
        title="종목 편집"
        onClose={closeEditModal}
        footer={
          <>
            <button type="button" className="btn me-auto btn-outline-danger" onClick={handleDeleteFromModal} disabled={isPending || isRefreshing}>
              <IconTrash size={16} stroke={1.9} />
              <span>삭제</span>
            </button>
            <button
              type="button"
              className="btn btn-outline-secondary"
              onClick={() => void handleRefreshFromModal()}
              disabled={isPending || isRefreshing}
            >
              <IconRefresh size={16} stroke={1.9} />
              <span>{isRefreshing ? "새로고침 중..." : "메타/캐시 새로고침"}</span>
            </button>
            <button type="button" className="btn btn-link link-secondary" onClick={closeEditModal} disabled={isPending || isRefreshing}>
              취소
            </button>
            <button type="button" className="btn btn-primary" onClick={handleSaveFromModal} disabled={isPending || isRefreshing}>
              저장
            </button>
          </>
        }
      >
        {editingRow ? (
          <>
            <div className="mb-3">
              <label className="form-label">티커</label>
              <div className="form-control-plaintext appCodeText">{editingRow.ticker}</div>
            </div>
            <div className="mb-3">
              <label className="form-label">종목명</label>
              <div className="form-control-plaintext">{editingRow.name}</div>
            </div>
            <div className="mb-3">
              <label className="form-label">버킷</label>
              <select
                className="form-select"
                value={editingBucketId}
                onChange={(event) => setEditingBucketId(Number(event.target.value))}
                disabled={isPending}
              >
                {BUCKET_OPTIONS.map((bucket) => (
                  <option key={bucket.id} value={bucket.id}>
                    {bucket.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="mb-3">
              <label className="form-label">삭제 사유</label>
              <input
                className="form-control"
                value={editingDeleteReason}
                onChange={(event) => setEditingDeleteReason(event.target.value)}
                placeholder="선택 입력"
                disabled={isPending}
              />
            </div>
            <div className="row g-2 text-secondary small">
              <div className="col-6">상장일: {editingRow.listing_date}</div>
              <div className="col-6">추가일자: {editingRow.added_date}</div>
            </div>
          </>
        ) : null}
      </AppModal>
    </div>
  );
}
