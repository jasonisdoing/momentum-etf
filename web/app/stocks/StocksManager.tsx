"use client";

import {
  IconArrowBackUp,
  IconPlus,
  IconLayoutGrid,
  IconPlaylistX,
  IconSearch,
} from "@tabler/icons-react";
import { useEffect, useMemo, useState, useTransition } from "react";
import { type GridColDef, type GridRenderCellParams } from "@mui/x-data-grid";

import { BUCKET_OPTIONS } from "@/lib/bucket-theme";
import { AppDataGrid } from "../components/AppDataGrid";
import { AppModal } from "../components/AppModal";
import {
  readRememberedTickerType,
  writeRememberedTickerType,
} from "../components/account-selection";
import { useToast } from "../components/ToastProvider";

type StocksAccountItem = {
  ticker_type: string;
  order: number;
  name: string;
  icon: string;
  country_code: string;
};

type ActiveStocksRowItem = {
  ticker: string;
  name: string;
  bucket_id: number;
  bucket_name: string;
  added_date: string;
  listing_date: string;
  week_volume: number | null;
  return_1d: number | null;
  return_1w: number | null;
  return_2w: number | null;
  return_1m: number | null;
  return_3m: number | null;
  return_6m: number | null;
  return_12m: number | null;
  "괴리율": number | null;
};

type DeletedStocksRowItem = {
  ticker: string;
  name: string;
  bucket_id: number;
  bucket_name: string;
  added_date: string;
  listing_date: string;
  week_volume: number | null;
  return_1d: number | null;
  return_1w: number | null;
  return_2w: number | null;
  return_1m: number | null;
  return_3m: number | null;
  return_6m: number | null;
  return_12m: number | null;
  "괴리율": number | null;
  deleted_date: string;
  deleted_reason: string;
};

type StocksResponse = {
  ticker_types?: StocksAccountItem[];
  rows?: ActiveStocksRowItem[];
  ticker_type?: string;
  error?: string;
};

type DeletedStocksResponse = {
  ticker_types?: StocksAccountItem[];
  rows?: DeletedStocksRowItem[];
  ticker_type?: string;
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

export function StocksManager() {
  const [ticker_types, setAccounts] = useState<StocksAccountItem[]>([]);
  const [selectedTickerType, setSelectedAccountId] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("active");
  const [activeRows, setActiveRows] = useState<ActiveStocksRowItem[]>([]);
  const [deletedRows, setDeletedRows] = useState<DeletedStocksRowItem[]>([]);
  const [selectedDeletedTickers, setSelectedDeletedTickers] = useState<string[]>([]);
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
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const toast = useToast();

  function showErrorToast(message: string) {
    toast.error(`[ETF-종목 관리] ${message}`);
  }

  async function load(mode: ViewMode, tickerType?: string) {
    setLoading(true);

    try {
      const search = tickerType ? `?ticker_type=${encodeURIComponent(tickerType)}` : "";
      const apiPath = mode === "active" ? `/api/stocks${search}` : `/api/deleted${search}`;
      const response = await fetch(apiPath, { cache: "no-store" });
      const payload = (await response.json()) as StocksResponse | DeletedStocksResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "종목 관리 데이터를 불러오지 못했습니다.");
      }

      setAccounts(payload.ticker_types ?? []);
      const nextAccountId = payload.ticker_type ?? "";
      setSelectedAccountId(nextAccountId);
      writeRememberedTickerType(nextAccountId);
      setSelectedDeletedTickers([]);

      if (mode === "active") {
        setActiveRows((payload.rows as ActiveStocksRowItem[] | undefined) ?? []);
      } else {
        setDeletedRows((payload.rows as DeletedStocksRowItem[] | undefined) ?? []);
      }
    } catch (loadError) {
      showErrorToast(loadError instanceof Error ? loadError.message : "종목 관리 데이터를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load(viewMode, readRememberedTickerType() ?? undefined);
  }, []);

  const selectedTickerTypeItem = useMemo(
    () => ticker_types.find((account) => account.ticker_type === selectedTickerType) ?? null,
    [ticker_types, selectedTickerType],
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

  const toggleSelection = (ticker: string) => {
    const t = ticker.trim().toUpperCase();
    setSelectedDeletedTickers(prev => 
      prev.includes(t) ? prev.filter(x => x !== t) : [...prev, t]
    );
  };

  const toggleAllSelection = () => {
    if (allDeletedSelected) {
      setSelectedDeletedTickers([]);
    } else {
      setSelectedDeletedTickers(deletedRows.map(r => r.ticker.trim().toUpperCase()));
    }
  };

  const activeColumns = useMemo<GridColDef<ActiveStockGridRow>[]>(
    () => [
      {
        field: "__edit__",
        headerName: "",
        width: 58,
        sortable: false,
        filterable: false,
        renderCell: (params: GridRenderCellParams<ActiveStockGridRow>) => (
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
        renderCell: (params: GridRenderCellParams<ActiveStockGridRow>) => <span className="appCodeText">{params.row.ticker}</span>,
      },
      { field: "name", headerName: "종목명", minWidth: 220, flex: 1 },
      {
        field: "week_volume",
        headerName: "주간거래량",
        width: 116,
        minWidth: 116,
        align: "right",
        headerAlign: "right",
        renderCell: (params: GridRenderCellParams<ActiveStockGridRow>) => formatNumber(params.row.week_volume),
      },
      {
        field: "return_1d",
        headerName: "일간(%)",
        width: 88,
        minWidth: 88,
        align: "right",
        headerAlign: "right",
        renderCell: (params: GridRenderCellParams<ActiveStockGridRow>) => (
          <span className={getSignedMetricClass(params.row.return_1d)} style={{ fontWeight: 600 }}>
             {formatPercent(params.row.return_1d)}
          </span>
        ),
      },
      ...(selectedTickerTypeItem?.country_code !== "au" ? [
        {
          field: "괴리율",
          headerName: "괴리율",
          width: 88,
          minWidth: 88,
          align: "right",
          headerAlign: "right",
          renderCell: (params: GridRenderCellParams<ActiveStockGridRow, number | null>) => {
            const val = params.value ?? 0;
            const isExtreme = val > 2.0 || val < -2.0;
            return (
              <span style={{ color: isExtreme ? "#d63939" : "inherit", fontWeight: isExtreme ? 700 : 400 }}>
                {formatPercent(params.value ?? null)}
              </span>
            );
          },
        } as GridColDef<ActiveStockGridRow>
      ] : []),
      ...(["return_1w", "return_1m", "return_3m", "return_6m", "return_12m"] as const).map(
        (field) => ({
          field,
          headerName:
            field === "return_1w"
              ? "1주(%)"
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
          renderCell: (params: GridRenderCellParams<ActiveStockGridRow>) => (
            <span className={getSignedMetricClass(params.row[field])}>{formatPercent(params.row[field])}</span>
          ),
        }),
      ),
      { field: "listing_date", headerName: "상장일", width: 112, minWidth: 112 },
      { field: "added_date", headerName: "등록일", width: 112, minWidth: 112 },
    ],
    [],
  );
  const deletedColumns = useMemo<GridColDef<DeletedStockGridRow>[]>(
    () => [
      {
        field: "__selection__",
        headerName: "",
        width: 42,
        sortable: false,
        filterable: false,
        renderHeader: () => (
          <input 
            type="checkbox" 
            className="form-check-input"
            checked={allDeletedSelected} 
            onChange={toggleAllSelection} 
          />
        ),
        renderCell: (params: GridRenderCellParams<DeletedStockGridRow>) => (
          <input 
            type="checkbox" 
            className="form-check-input"
            checked={selectedDeletedTickerSet.has(params.row.ticker.trim().toUpperCase())}
            onChange={() => toggleSelection(params.row.ticker)}
          />
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
        renderCell: (params: GridRenderCellParams<DeletedStockGridRow>) => <span className="appCodeText">{params.row.ticker}</span>,
      },
      { field: "name", headerName: "종목명", minWidth: 220, flex: 1 },
      {
        field: "week_volume",
        headerName: "주간거래량",
        width: 116,
        minWidth: 116,
        align: "right",
        headerAlign: "right",
        renderCell: (params: GridRenderCellParams<DeletedStockGridRow>) => formatNumber(params.row.week_volume),
      },
      {
        field: "return_1d",
        headerName: "일간(%)",
        width: 88,
        minWidth: 88,
        align: "right",
        headerAlign: "right",
        renderCell: (params: GridRenderCellParams<DeletedStockGridRow>) => (
          <span className={getSignedMetricClass(params.row.return_1d)} style={{ fontWeight: 600 }}>
             {formatPercent(params.row.return_1d)}
          </span>
        ),
      },
      ...(selectedTickerTypeItem?.country_code !== "au" ? [
        {
          field: "괴리율",
          headerName: "괴리율",
          width: 88,
          minWidth: 88,
          align: "right",
          headerAlign: "right",
          renderCell: (params: GridRenderCellParams<DeletedStockGridRow, number | null>) => {
            const val = params.value ?? 0;
            const isExtreme = val > 2.0 || val < -2.0;
            return (
              <span style={{ color: isExtreme ? "#d63939" : "inherit", fontWeight: isExtreme ? 700 : 400 }}>
                {formatPercent(params.value ?? null)}
              </span>
            );
          },
        } as GridColDef<DeletedStockGridRow>
      ] : []),
      ...(["return_1w", "return_1m", "return_3m", "return_6m", "return_12m"] as const).map(
        (field) => ({
          field,
          headerName:
            field === "return_1w"
              ? "1주(%)"
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
          renderCell: (params: GridRenderCellParams<DeletedStockGridRow>) => (
            <span className={getSignedMetricClass(params.row[field])}>{formatPercent(params.row[field])}</span>
          ),
        }),
      ),
      { field: "listing_date", headerName: "상장일", width: 112, minWidth: 112 },
      { field: "added_date", headerName: "등록일", width: 112, minWidth: 112 },
      { field: "deleted_date", headerName: "삭제일", width: 112, minWidth: 112 },
      { field: "deleted_reason", headerName: "삭제 사유", minWidth: 160, flex: 0.7 },
    ],
    [selectedDeletedTickerSet, allDeletedSelected],
  );

  function handleTickerTypeChange(nextAccountId: string) {
    setSelectedAccountId(nextAccountId);
    writeRememberedTickerType(nextAccountId);
    void load(viewMode, nextAccountId);
  }

  function handleViewModeChange(nextMode: ViewMode) {
    if (nextMode === viewMode) {
      return;
    }
    setViewMode(nextMode);
    void load(nextMode, selectedTickerType);
  }

  function openEditModal(row: ActiveStocksRowItem) {
    setEditingRow(row);
    setEditingBucketId(row.bucket_id);
    setEditingDeleteReason("");
    setIsEditModalOpen(true);
  }

  function closeEditModal() {
    setIsEditModalOpen(false);
    setEditingRow(null);
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

  async function handleValidateTicker() {
    try {
      setIsValidatingTicker(true);
      const response = await fetch("/api/stocks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "validate",
          ticker_type: selectedTickerType,
          ticker: addTickerInput,
        }),
      });
      const payload = (await response.json()) as
        | ({ error?: string } & StockValidationState)
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
      showErrorToast(validationError instanceof Error ? validationError.message : "티커 확인에 실패했습니다.");
    } finally {
      setIsValidatingTicker(false);
    }
  }

  function handleEditSave(isForceDelete = false) {
    if (!editingRow) return;

    startTransition(async () => {
      try {
        const isDelete = isForceDelete || !!editingDeleteReason.trim();
        const response = await fetch("/api/stocks", {
          method: isDelete ? "DELETE" : "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(
            isDelete
              ? {
                  ticker_type: selectedTickerType,
                  ticker: editingRow.ticker,
                  reason: editingDeleteReason.trim(),
                }
              : {
                  ticker_type: selectedTickerType,
                  ticker: editingRow.ticker,
                  bucket_id: editingBucketId,
                },
          ),
        });

        const payload = (await response.json()) as { error?: string; name?: string; ticker?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 수정에 실패했습니다.");
        }

        const msgPrefix = isDelete ? "삭제" : "수정";
        toast.success(`[ETF-종목 관리] ${payload.name || editingRow.name}(${payload.ticker || editingRow.ticker}) ${msgPrefix} 완료`);
        closeEditModal();
        void load(viewMode, selectedTickerType);
      } catch (saveError) {
        showErrorToast(saveError instanceof Error ? saveError.message : "종목 수정에 실패했습니다.");
      }
    });
  }

  function handleCreateStock() {
    if (!validatedCandidate) {
      return;
    }

    startTransition(async () => {
      try {
        const response = await fetch("/api/stocks", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "create",
            ticker_type: selectedTickerType,
            ticker: validatedCandidate.ticker,
            bucket_id: addBucketId,
          }),
        });
        const payload = (await response.json()) as {
          error?: string;
          ticker?: string;
          name?: string;
        };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 추가에 실패했습니다.");
        }
        toast.success(`[ETF-종목 관리] ${payload.name || validatedCandidate.name}(${payload.ticker || validatedCandidate.ticker}) 추가 완료`);
        closeAddModal();
        void load(viewMode, selectedTickerType);
      } catch (createError) {
        showErrorToast(createError instanceof Error ? createError.message : "종목 추가에 실패했습니다.");
      }
    });
  }

  function handleRestoreDeleted() {
    startTransition(async () => {
      try {
        const response = await fetch("/api/deleted", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ticker_type: selectedTickerType,
            tickers: selectedDeletedTickers,
          }),
        });
        const payload = (await response.json()) as { error?: string; restored_count?: number };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 복구에 실패했습니다.");
        }
        toast.success(`[Momentum ETF-종목 관리] ${payload.restored_count ?? 0}개 종목 복구 완료`);
        setSelectedDeletedTickers([]);
        void load(viewMode, selectedTickerType);
      } catch (restoreError) {
        showErrorToast(restoreError instanceof Error ? restoreError.message : "종목 복구에 실패했습니다.");
      }
    });
  }

  function handleHardDeleteDeleted() {
    startTransition(async () => {
      try {
        const response = await fetch("/api/deleted", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ticker_type: selectedTickerType,
            tickers: selectedDeletedTickers,
          }),
        });
        const payload = (await response.json()) as { error?: string; deleted_count?: number };
        if (!response.ok) {
          throw new Error(payload.error ?? "종목 완전 삭제에 실패했습니다.");
        }
        toast.success(`[Momentum ETF-종목 관리] ${payload.deleted_count ?? 0}개 종목 영구 삭제 완료`);
        setSelectedDeletedTickers([]);
        void load(viewMode, selectedTickerType);
      } catch (deleteError) {
        showErrorToast(deleteError instanceof Error ? deleteError.message : "종목 완전 삭제에 실패했습니다.");
      }
    });
  }

  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection appSectionFill stocksPage">
        <div className="card appCard stocksCard appTableCardFill">
          <div className="card-header">
            <div className="tickerTypeToolbar w-100" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div className="tickerTypeToolbarLeft" style={{ display: "flex", gap: "0.4rem", alignItems: "center" }}>
                <div className="accountSelect">
                  <select
                    className="form-select"
                    style={{ width: "auto", minWidth: "180px", fontWeight: 600 }}
                    aria-label="계좌 선택"
                    value={selectedTickerType}
                    onChange={(event) => handleTickerTypeChange(event.target.value)}
                    disabled={loading}
                  >
                    {ticker_types.map((account) => (
                      <option key={account.ticker_type} value={account.ticker_type}>
                        {account.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="stocksToolbarModes d-flex gap-1">
                  <button className="btn btn-primary d-flex align-items-center gap-1" type="button" onClick={openAddModal} disabled={loading || viewMode !== "active"}>
                    <IconPlus size={18} stroke={2} />
                    <span style={{ fontWeight: 600 }}>종목 추가</span>
                  </button>
                  <button
                    className={viewMode === "active" ? "btn stocksModeButton is-active" : "btn btn-outline-secondary stocksModeButton"}
                    type="button"
                    onClick={() => handleViewModeChange("active")}
                  >
                    <IconLayoutGrid size={16} stroke={1.75} />
                    <span>등록된 종목</span>
                  </button>
                  <button
                    className={viewMode === "deleted" ? "btn stocksModeButton is-active" : "btn btn-outline-secondary stocksModeButton"}
                    type="button"
                    onClick={() => handleViewModeChange("deleted")}
                  >
                    <IconPlaylistX size={16} stroke={1.75} />
                    <span>삭제된 종목</span>
                  </button>
                </div>
              </div>

              <div className="tickerTypeToolbarRight" style={{ display: "flex", alignItems: "center", gap: "1.25rem" }}>
                <div className="stocksSummary d-flex align-items-center gap-3">
                  <div className="d-flex align-items-center gap-1">
                    <span style={{ color: "#6c757d", fontSize: "0.85rem", fontWeight: 600 }}>총 개수:</span>
                    <span style={{ fontWeight: 700 }}>
                      {viewMode === "active"
                        ? `${new Intl.NumberFormat("ko-KR").format(activeRows.length)}개`
                        : `${new Intl.NumberFormat("ko-KR").format(deletedRows.length)}개`}
                    </span>
                  </div>
                  {viewMode === "deleted" && selectedDeletedTickers.length > 0 ? (
                    <div className="d-flex align-items-center gap-1">
                      <span style={{ color: "#6c757d", fontSize: "0.85rem", fontWeight: 600 }}>선택:</span>
                      <span style={{ fontWeight: 700, color: "var(--tblr-primary)" }}>
                        {new Intl.NumberFormat("ko-KR").format(selectedDeletedTickers.length)}개
                      </span>
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          </div>

          {viewMode === "deleted" && deletedRows.length > 0 && (
            <div className="card-header bg-light-subtle border-top py-2">
              <div className="d-flex justify-content-between align-items-center">
                <div></div>
                <div className="d-flex gap-2">
                  <button className="btn btn-sm btn-primary d-flex align-items-center gap-1" onClick={handleRestoreDeleted} disabled={selectedDeletedTickers.length === 0}>
                    <IconArrowBackUp size={14} />
                    <span>선택 복구</span>
                  </button>
                  <button className="btn btn-sm btn-outline-danger d-flex align-items-center gap-1" onClick={handleHardDeleteDeleted} disabled={selectedDeletedTickers.length === 0}>
                    <IconSearch size={14} />
                    <span>영구 삭제</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          <div className="stocksTableWrap">
            <AppDataGrid
              className="stocksTable"
              rows={(viewMode === "active" ? activeGridRows : deletedGridRows) as any}
              columns={(viewMode === "active" ? activeColumns : deletedColumns) as any}
              loading={loading}
              minHeight="100%"
            />
          </div>
        </div>
      </section>

      <AppModal
        open={isAddModalOpen}
        onClose={closeAddModal}
        title="종목 추가"
        footer={
          <div className="d-flex justify-content-end gap-2 w-100">
            <button className="btn btn-link link-secondary" type="button" onClick={closeAddModal}>
              취소
            </button>
            <button 
              className="btn btn-primary" 
              type="button" 
              onClick={handleCreateStock} 
              disabled={isPending || !validatedCandidate || validatedCandidate.status === "active"}
              style={{ minWidth: "100px" }}
            >
              저장
            </button>
          </div>
        }
      >
        <div className="appModalBody">
          <div className="mb-3">
            <label className="form-label">티커</label>
            <div className="row g-2">
              <div className="col">
                <input
                  type="text"
                  className="form-control"
                  placeholder="예: 069500 / VGS / ASX:VGS"
                  value={addTickerInput}
                  onChange={(e) => {
                    setAddTickerInput(e.target.value);
                    if (validatedCandidate) setValidatedCandidate(null);
                  }}
                  onKeyDown={(e) => e.key === "Enter" && handleValidateTicker()}
                />
              </div>
              <div className="col-auto">
                <button
                  className="btn btn-outline-secondary d-flex align-items-center gap-1"
                  type="button"
                  onClick={handleValidateTicker}
                  disabled={isValidatingTicker || !addTickerInput.trim()}
                >
                  <IconSearch size={16} stroke={1.75} />
                  <span>확인</span>
                </button>
              </div>
            </div>
            {isValidatingTicker ? (
              <div className="mt-1 small text-muted">티커 확인 중...</div>
            ) : validatedCandidate ? (
              <div className={`mt-2 p-2 rounded border ${validatedCandidate.status === "active" ? "bg-danger-lt border-danger-subtle" : "bg-success-lt border-success-subtle"}`}>
                <div className={`d-flex align-items-center gap-1 fw-bold ${validatedCandidate.status === "active" ? "text-danger" : "text-success"}`}>
                  <span className="appCodeText">{validatedCandidate.ticker}</span>
                  <span>-</span>
                  <span>{validatedCandidate.name}</span>
                </div>
                {validatedCandidate.status === "active" && (
                  <div className="mt-1 small text-danger fw-bold">
                    이미 등록된 종목입니다.
                  </div>
                )}
                {validatedCandidate.status === "deleted" && (
                  <div className="mt-1 small text-success fw-bold">
                    삭제된 종목입니다. 다시 추가하시겠습니까?
                  </div>
                )}
              </div>
            ) : null}
          </div>

          <div className="mb-3">
            <label className="form-label">버킷</label>
            <select 
              className="form-select" 
              value={addBucketId} 
              onChange={(e) => setAddBucketId(Number(e.target.value))}
            >
              {BUCKET_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id}>{opt.name}</option>
              ))}
            </select>
          </div>
        </div>
      </AppModal>

      {/* 종목 수정 모달 */}
      <AppModal
        open={isEditModalOpen}
        onClose={closeEditModal}
        title="종목 수정"
        footer={
          <div className="d-flex justify-content-between w-100">
            <button
              className="btn btn-outline-danger"
              type="button"
              onClick={() => {
                handleEditSave(true);
              }}
              disabled={isPending}
            >
              종목 삭제
            </button>
            <div className="d-flex gap-2">
              <button className="btn btn-link link-secondary" type="button" onClick={closeEditModal}>
                취소
              </button>
              <button 
                className="btn btn-primary" 
                type="button" 
                onClick={() => handleEditSave()}
                style={{ minWidth: "100px" }}
                disabled={isPending}
              >
                저장
              </button>
            </div>
          </div>
        }
      >
        <div className="appModalBody">
          {editingRow && (
            <>
              <div className="mb-3">
                <div className="fw-bold text-secondary mb-1">대상 종목</div>
                <div className="appCodeText" style={{ fontSize: "1.2rem" }}>
                  {editingRow.ticker}
                </div>
                <div style={{ fontSize: "1.1rem" }}>{editingRow.name}</div>
              </div>
              <div className="mb-3">
                <label className="form-label">버킷 변경</label>
                <select
                  className="form-select"
                  value={editingBucketId}
                  onChange={(e) => setEditingBucketId(Number(e.target.value))}
                >
                  {BUCKET_OPTIONS.map((opt) => (
                    <option key={opt.id} value={opt.id}>
                      {opt.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="mb-3">
                <label className="form-label text-secondary">삭제 사유(옵션)</label>
                <textarea
                  className="form-control"
                  rows={2}
                  placeholder="삭제 시 참고할 사유가 있다면 입력해주세요 (필수 아님)."
                  value={editingDeleteReason}
                  onChange={(e) => setEditingDeleteReason(e.target.value)}
                />
              </div>
            </>
          )}
        </div>
      </AppModal>
    </div>
  );
}
