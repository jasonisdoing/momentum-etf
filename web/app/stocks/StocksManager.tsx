"use client";

import {
  IconArrowBackUp,
  IconPlus,
  IconTrash,
  IconDeviceFloppy,
  IconLayoutGrid,
  IconPlaylistX,
  IconSearch,
} from "@tabler/icons-react";
import { iconSetQuartzBold, themeQuartz } from "ag-grid-community";
import { useEffect, useMemo, useState, useTransition } from "react";
import type { ColDef } from "ag-grid-community";
import { useRouter } from "next/navigation";

import { BUCKET_OPTIONS } from "@/lib/bucket-theme";
import { AppAgGrid } from "../components/AppAgGrid";
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
  bucket_id: number;
};
type AddingStockRowState = {
  ticker: string;
  name: string;
  listing_date: string;
  bucket_id: number;
  status: "active" | "deleted" | "new" | null;
  is_validating: boolean;
  is_validated: boolean;
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
  return value > 0 ? "metricPositive" : "metricNegative";
}

function getBucketCellClass(bucketId: number): string {
  return `appBucketCell appBucketCell${bucketId}`;
}

function buildDirtyCellKey(rowId: string, field: string): string {
  return `${rowId}::${field}`;
}

function normalizeTicker(value: string): string {
  return String(value || "").trim().toUpperCase();
}

function getBucketName(bucketId: number): string {
  return BUCKET_OPTIONS.find((option) => option.id === bucketId)?.name ?? "-";
}

function getBucketIdByName(bucketName: string): number {
  return BUCKET_OPTIONS.find((option) => option.name === bucketName)?.id ?? 1;
}

const stocksGridTheme = themeQuartz
  .withPart(iconSetQuartzBold)
  .withParams({
    accentColor: "#206bc4",
    backgroundColor: "#ffffff",
    foregroundColor: "#182433",
    headerBackgroundColor: "#f8fafc",
    headerTextColor: "#5b6778",
    spacing: 8,
    fontSize: 14,
    wrapperBorderRadius: 10,
    rowHeight: 38,
    headerHeight: 38,
    cellHorizontalPadding: 12,
    headerColumnBorder: true,
    headerColumnBorderHeight: "70%",
    columnBorder: true,
    oddRowBackgroundColor: "#fbfdff",
    headerCellHoverBackgroundColor: "#eef4fb",
    headerCellMovingBackgroundColor: "#e8f0fb",
    iconButtonHoverBackgroundColor: "#eef4fb",
    iconButtonHoverColor: "#206bc4",
    iconSize: 18,
  });

export function StocksManager({
  onHeaderSummaryChange,
}: {
  onHeaderSummaryChange?: (summary: {
    tickerTypeName: string;
    viewLabel: string;
    totalCount: number;
    selectedCount: number;
    dirtyCount: number;
  }) => void;
}) {
  const router = useRouter();
  const [ticker_types, setAccounts] = useState<StocksAccountItem[]>([]);
  const [selectedTickerType, setSelectedAccountId] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("active");
  const [activeRows, setActiveRows] = useState<ActiveStocksRowItem[]>([]);
  const [deletedRows, setDeletedRows] = useState<DeletedStocksRowItem[]>([]);
  const [selectedActiveTickers, setSelectedActiveTickers] = useState<string[]>([]);
  const [selectedDeletedTickers, setSelectedDeletedTickers] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [isPending, startTransition] = useTransition();
  const [addingRow, setAddingRow] = useState<AddingStockRowState | null>(null);
  const [dirtyActiveRowIds, setDirtyActiveRowIds] = useState<string[]>([]);
  const [dirtyActiveCellKeys, setDirtyActiveCellKeys] = useState<string[]>([]);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const toast = useToast();
  const selectedTickerTypeMeta = useMemo(
    () => ticker_types.find((account) => account.ticker_type === selectedTickerType) ?? null,
    [selectedTickerType, ticker_types],
  );

  useEffect(() => {
    onHeaderSummaryChange?.({
      tickerTypeName: selectedTickerTypeMeta?.name ?? "-",
      viewLabel: viewMode === "active" ? "등록된 종목" : "삭제된 종목",
      totalCount: viewMode === "active" ? activeRows.length : deletedRows.length,
      selectedCount: viewMode === "active" ? selectedActiveTickers.length : selectedDeletedTickers.length,
      dirtyCount: viewMode === "active" ? dirtyActiveRowIds.length : 0,
    });
  }, [
    activeRows.length,
    deletedRows.length,
    dirtyActiveRowIds.length,
    onHeaderSummaryChange,
    selectedActiveTickers.length,
    selectedDeletedTickers.length,
    selectedTickerTypeMeta?.name,
    viewMode,
  ]);

  function moveToTickerDetail(ticker: string | null | undefined) {
    const normalizedTicker = String(ticker ?? "").trim().toUpperCase();
    if (!normalizedTicker) {
      return;
    }
    router.push(`/ticker?ticker=${encodeURIComponent(normalizedTicker)}`);
  }

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
      setSelectedActiveTickers([]);
      setSelectedDeletedTickers([]);
      setAddingRow(null);
      setDirtyActiveRowIds([]);
      setDirtyActiveCellKeys([]);
      setDeleteConfirmOpen(false);

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

  const activeGridRows = useMemo<ActiveStockGridRow[]>(() => {
    const baseRows = activeRows.map((row) => ({
      ...row,
      id: normalizeTicker(row.ticker),
    }));
    if (!addingRow) {
      return baseRows;
    }
    return [
      {
        id: "__adding__",
        ticker: addingRow.ticker,
        name: addingRow.name,
        bucket_id: addingRow.bucket_id,
        bucket_name: getBucketName(addingRow.bucket_id),
        added_date: "-",
        listing_date: addingRow.listing_date || "-",
        week_volume: null,
        return_1d: null,
        return_1w: null,
        return_2w: null,
        return_1m: null,
        return_3m: null,
        return_6m: null,
        return_12m: null,
        괴리율: null,
      },
      ...baseRows,
    ];
  }, [activeRows, addingRow]);
  const deletedGridRows = useMemo<DeletedStockGridRow[]>(
    () => deletedRows.map((row) => ({ ...row, id: row.ticker.trim().toUpperCase() })),
    [deletedRows],
  );

  const activeColumns = useMemo<ColDef<ActiveStockGridRow>[]>(
    () => [
      {
        field: "bucket_id",
        headerName: "버킷",
        width: 118,
        minWidth: 118,
        sortable: false,
        cellClass: (params) => {
          if (!params.data) {
            return "";
          }
          const dirtyClass = dirtyActiveCellKeys.includes(buildDirtyCellKey(params.data.id, "bucket_name"))
            ? " stocksDirtyCell"
            : "";
          return `${getBucketCellClass(params.data.bucket_id)}${dirtyClass}`;
        },
        editable: (params) => params.data?.id !== "__adding__",
        cellEditor: "agSelectCellEditor",
        cellEditorParams: {
          values: BUCKET_OPTIONS.map((option) => option.name),
        },
        valueGetter: (params) => getBucketName(params.data?.bucket_id ?? 1),
        valueSetter: (params) => {
          if (!params.data || params.data.id === "__adding__") {
            return false;
          }
          const nextBucketId = getBucketIdByName(String(params.newValue ?? ""));
          params.data.bucket_id = nextBucketId;
          params.data.bucket_name = getBucketName(nextBucketId);
          return true;
        },
        cellRenderer: (params: { data?: ActiveStockGridRow }) => {
          if (params.data?.id === "__adding__") {
            return (
              <select
                className="form-select form-select-sm"
                value={addingRow?.bucket_id ?? 1}
                onChange={(event) =>
                  setAddingRow((prev) =>
                    prev
                      ? {
                          ...prev,
                          bucket_id: Number(event.target.value),
                        }
                      : null,
                  )
                }
              >
                {BUCKET_OPTIONS.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.name}
                  </option>
                ))}
              </select>
            );
          }
          return getBucketName(params.data?.bucket_id ?? 1);
        },
      },
      {
        field: "ticker",
        headerName: "티커",
        width: 124,
        minWidth: 124,
        cellRenderer: (params: { data?: ActiveStockGridRow }) => (
          params.data?.id === "__adding__" ? (
            <div className="stocksTickerLookup">
              <input
                type="text"
                className="form-control form-control-sm"
                value={addingRow?.ticker ?? ""}
                onChange={(event) =>
                  setAddingRow((prev) =>
                    prev
                      ? {
                          ...prev,
                          ticker: event.target.value,
                          name: "",
                          listing_date: "-",
                          status: null,
                          is_validated: false,
                        }
                      : null,
                  )
                }
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    void handleValidateAddingTicker();
                  }
                }}
              />
              <button
                className="btn btn-outline-primary btn-sm"
                type="button"
                onClick={() => void handleValidateAddingTicker()}
                disabled={!addingRow?.ticker.trim() || addingRow.is_validating}
              >
                확인
              </button>
            </div>
          ) : (
            <button
              type="button"
              className="btn btn-link p-0 appCodeText stocksTickerLink"
              onClick={() => moveToTickerDetail(params.data?.ticker)}
            >
              {params.data?.ticker ?? "-"}
            </button>
          )
        ),
      },
      {
        field: "name",
        headerName: "종목명",
        minWidth: 220,
        flex: 1,
        cellClass: "stocksNameColumn",
        cellRenderer: (params: { data?: ActiveStockGridRow }) => {
          if (params.data?.id !== "__adding__") {
            return (
              <div className="stocksNameCell" title={params.data?.name ?? "-"}>
                {params.data?.name ?? "-"}
              </div>
            );
          }
          if (addingRow?.is_validating) {
            return <span className="text-muted">티커 확인 중...</span>;
          }
          if (addingRow?.status === "active") {
            return <span className="text-danger fw-bold">이미 등록된 종목입니다.</span>;
          }
          if (addingRow?.is_validated) {
            return (
              <span className="fw-semibold stocksNameCell" title={addingRow.name}>
                {addingRow.name}
                {addingRow.status === "deleted" ? " (삭제된 종목 재추가)" : ""}
              </span>
            );
          }
          return <span className="text-muted">티커 확인 후 종목명이 표시됩니다.</span>;
        },
      },
      {
        field: "week_volume",
        headerName: "주간거래량",
        width: 116,
        minWidth: 116,
        type: "rightAligned",
        valueFormatter: (params) => formatNumber((params.data?.week_volume as number | null) ?? null),
      },
      {
        field: "return_1d",
        headerName: "일간(%)",
        width: 88,
        minWidth: 88,
        type: "rightAligned",
        cellRenderer: (params: { data?: ActiveStockGridRow }) => (
          <span className={getSignedMetricClass(params.data?.return_1d ?? null)} style={{ fontWeight: 600 }}>
             {formatPercent(params.data?.return_1d ?? null)}
          </span>
        ),
      },
      ...(selectedTickerTypeItem?.country_code !== "au" ? [
        {
          field: "괴리율",
          headerName: "괴리율",
          width: 88,
          minWidth: 88,
          type: "rightAligned",
          cellRenderer: (params: { value?: number | null }) => {
            const val = params.value ?? 0;
            const isExtreme = val > 2.0 || val < -2.0;
            return (
              <span style={{ color: isExtreme ? "#d63939" : "inherit", fontWeight: isExtreme ? 700 : 400 }}>
                {formatPercent(params.value ?? null)}
              </span>
            );
          },
        } as ColDef<ActiveStockGridRow>
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
          type: "rightAligned" as const,
          cellRenderer: (params: { data?: ActiveStockGridRow }) => (
            <span className={getSignedMetricClass((params.data?.[field] as number | null) ?? null)}>
              {formatPercent((params.data?.[field] as number | null) ?? null)}
            </span>
          ),
        }),
      ),
      { field: "listing_date", headerName: "상장일", width: 112, minWidth: 112 },
      { field: "added_date", headerName: "등록일", width: 112, minWidth: 112 },
    ],
    [addingRow, dirtyActiveCellKeys, selectedTickerTypeItem?.country_code],
  );
  const deletedColumns = useMemo<ColDef<DeletedStockGridRow>[]>(
    () => [
      {
        field: "bucket_name",
        headerName: "버킷",
        width: 118,
        minWidth: 118,
        sortable: false,
        cellClass: (params) => (params.data ? getBucketCellClass(params.data.bucket_id) : ""),
      },
      {
        field: "ticker",
        headerName: "티커",
        width: 98,
        minWidth: 98,
        cellRenderer: (params: { data?: DeletedStockGridRow }) => (
          <button
            type="button"
            className="btn btn-link p-0 appCodeText stocksTickerLink"
            onClick={() => moveToTickerDetail(params.data?.ticker)}
          >
            {params.data?.ticker ?? "-"}
          </button>
        ),
      },
      {
        field: "name",
        headerName: "종목명",
        minWidth: 220,
        flex: 1,
        cellClass: "stocksNameColumn",
        cellRenderer: (params: { data?: DeletedStockGridRow }) => (
          <div className="stocksNameCell" title={params.data?.name ?? "-"}>
            {params.data?.name ?? "-"}
          </div>
        ),
      },
      {
        field: "week_volume",
        headerName: "주간거래량",
        width: 116,
        minWidth: 116,
        type: "rightAligned",
        valueFormatter: (params) => formatNumber((params.data?.week_volume as number | null) ?? null),
      },
      {
        field: "return_1d",
        headerName: "일간(%)",
        width: 88,
        minWidth: 88,
        type: "rightAligned",
        cellRenderer: (params: { data?: DeletedStockGridRow }) => (
          <span className={getSignedMetricClass(params.data?.return_1d ?? null)} style={{ fontWeight: 600 }}>
             {formatPercent(params.data?.return_1d ?? null)}
          </span>
        ),
      },
      ...(selectedTickerTypeItem?.country_code !== "au" ? [
        {
          field: "괴리율",
          headerName: "괴리율",
          width: 88,
          minWidth: 88,
          type: "rightAligned",
          cellRenderer: (params: { value?: number | null }) => {
            const val = params.value ?? 0;
            const isExtreme = val > 2.0 || val < -2.0;
            return (
              <span style={{ color: isExtreme ? "#d63939" : "inherit", fontWeight: isExtreme ? 700 : 400 }}>
                {formatPercent(params.value ?? null)}
              </span>
            );
          },
        } as ColDef<DeletedStockGridRow>
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
          type: "rightAligned" as const,
          cellRenderer: (params: { data?: DeletedStockGridRow }) => (
            <span className={getSignedMetricClass((params.data?.[field] as number | null) ?? null)}>
              {formatPercent((params.data?.[field] as number | null) ?? null)}
            </span>
          ),
        }),
      ),
      { field: "listing_date", headerName: "상장일", width: 112, minWidth: 112 },
      { field: "added_date", headerName: "등록일", width: 112, minWidth: 112 },
    ],
    [selectedTickerTypeItem?.country_code],
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
    setSelectedActiveTickers([]);
    void load(nextMode, selectedTickerType);
  }

  function handleAddRow() {
    if (addingRow) {
      return;
    }
    setAddingRow({
      ticker: "",
      name: "",
      listing_date: "-",
      bucket_id: 1,
      status: null,
      is_validating: false,
      is_validated: false,
    });
  }

  function handleActiveBucketChanged(row: ActiveStockGridRow | undefined, bucketName: string) {
    if (!row || row.id === "__adding__") {
      return;
    }
    const nextBucketId = getBucketIdByName(bucketName);
    setActiveRows((prev) =>
      prev.map((currentRow) =>
        normalizeTicker(currentRow.ticker) === row.id
          ? {
              ...currentRow,
              bucket_id: nextBucketId,
              bucket_name: bucketName,
            }
          : currentRow,
      ),
    );
    setDirtyActiveRowIds((prev) => (prev.includes(row.id) ? prev : [...prev, row.id]));
    const dirtyCellKey = buildDirtyCellKey(row.id, "bucket_name");
    setDirtyActiveCellKeys((prev) => (prev.includes(dirtyCellKey) ? prev : [...prev, dirtyCellKey]));
  }

  async function handleValidateAddingTicker() {
    const ticker = normalizeTicker(addingRow?.ticker ?? "");
    if (!ticker || !selectedTickerType || !addingRow || addingRow.is_validating) {
      return;
    }

    try {
      setAddingRow((prev) => (prev ? { ...prev, ticker, is_validating: true } : null));
      const response = await fetch("/api/stocks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "validate",
          ticker_type: selectedTickerType,
          ticker,
        }),
      });
      const payload = (await response.json()) as
        | ({ error?: string } & StockValidationState)
        | { error?: string };
      if (!response.ok) {
        throw new Error(payload.error ?? "티커 확인에 실패했습니다.");
      }
      const validated = payload as StockValidationState;
      setAddingRow((prev) =>
        prev
          ? {
              ...prev,
              ticker: normalizeTicker(validated.ticker),
              name: String(validated.name ?? "").trim(),
              listing_date: String(validated.listing_date ?? "-").trim() || "-",
              bucket_id: Number(validated.bucket_id ?? prev.bucket_id ?? 1),
              status: validated.status,
              is_validating: false,
              is_validated: validated.status !== "active",
            }
          : null,
      );
      if (validated.status === "active") {
        showErrorToast("이미 등록된 종목입니다.");
      } else {
        toast.success(`[ETF-종목 관리] ${validated.name}(${validated.ticker}) 확인 완료`);
      }
    } catch (validationError) {
      setAddingRow((prev) =>
        prev
          ? {
              ...prev,
              is_validating: false,
              is_validated: false,
            }
          : null,
      );
      showErrorToast(validationError instanceof Error ? validationError.message : "티커 확인에 실패했습니다.");
    }
  }

  async function processAddingRow() {
    if (!addingRow || !addingRow.is_validated) {
      throw new Error("추가할 종목을 먼저 확인해주세요.");
    }

    const response = await fetch("/api/stocks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "create",
        ticker_type: selectedTickerType,
        ticker: addingRow.ticker,
        bucket_id: addingRow.bucket_id,
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
    toast.success(`[ETF-종목 관리] ${payload.name || addingRow.name}(${payload.ticker || addingRow.ticker}) 추가 완료`);
  }

  async function processDirtyRows() {
    const dirtyRows = activeRows.filter((row) => dirtyActiveRowIds.includes(normalizeTicker(row.ticker)));
    for (const row of dirtyRows) {
      const response = await fetch("/api/stocks", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker_type: selectedTickerType,
          ticker: row.ticker,
          bucket_id: row.bucket_id,
        }),
      });
      const payload = (await response.json()) as { error?: string };
      if (!response.ok) {
        throw new Error(payload.error ?? `${row.name} 버킷 저장에 실패했습니다.`);
      }
    }
  }

  function handleSaveActiveChanges() {
    if (!selectedTickerType || (dirtyActiveRowIds.length === 0 && !addingRow)) {
      return;
    }

    startTransition(async () => {
      try {
        if (addingRow) {
          await processAddingRow();
        }
        if (dirtyActiveRowIds.length > 0) {
          await processDirtyRows();
        }
        setAddingRow(null);
        setDirtyActiveRowIds([]);
        setDirtyActiveCellKeys([]);
        setSelectedActiveTickers([]);
        void load(viewMode, selectedTickerType);
        toast.success("[ETF-종목 관리] 변경사항 저장 완료");
      } catch (saveError) {
        showErrorToast(saveError instanceof Error ? saveError.message : "변경사항 저장에 실패했습니다.");
      }
    });
  }

  function handleDeleteActiveSelected() {
    if (!selectedActiveTickers.length) {
      return;
    }
    const selectedRows = activeRows.filter((row) => selectedActiveTickers.includes(normalizeTicker(row.ticker)));
    if (!selectedRows.length) {
      return;
    }
    setDeleteConfirmOpen(true);
  }

  function handleCloseDeleteConfirm() {
    if (isPending) {
      return;
    }
    setDeleteConfirmOpen(false);
  }

  function handleConfirmDeleteActiveSelected() {
    const selectedRows = activeRows.filter((row) => selectedActiveTickers.includes(normalizeTicker(row.ticker)));
    if (!selectedRows.length) {
      setDeleteConfirmOpen(false);
      return;
    }

    startTransition(async () => {
      try {
        for (const row of selectedRows) {
          const response = await fetch("/api/stocks", {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ticker_type: selectedTickerType,
              ticker: row.ticker,
            }),
          });
          const payload = (await response.json()) as { error?: string };
          if (!response.ok) {
            throw new Error(payload.error ?? `${row.name} 삭제에 실패했습니다.`);
          }
        }
        setSelectedActiveTickers([]);
        setDirtyActiveRowIds([]);
        setDirtyActiveCellKeys([]);
        setDeleteConfirmOpen(false);
        void load(viewMode, selectedTickerType);
        toast.success(`[ETF-종목 관리] ${selectedRows.length}개 종목 삭제 완료`);
      } catch (deleteError) {
        showErrorToast(deleteError instanceof Error ? deleteError.message : "종목 삭제에 실패했습니다.");
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
            <div className="appMainHeader">
              <div className="appMainHeaderLeft stocksMainHeaderLeft">
                <label className="appLabeledField accountSelect">
                  <span className="appLabeledFieldLabel">종목 타입</span>
                  <select
                    className="form-select"
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
                </label>

                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">보기 방식</span>
                  <div className="appSegmentedToggle appSegmentedToggleCompact">
                    <button
                      className={viewMode === "active" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                      type="button"
                      onClick={() => handleViewModeChange("active")}
                    >
                      <IconLayoutGrid size={16} stroke={1.75} />
                      <span>등록된 종목</span>
                    </button>
                    <button
                      className={viewMode === "deleted" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                      type="button"
                      onClick={() => handleViewModeChange("deleted")}
                    >
                      <IconPlaylistX size={16} stroke={1.75} />
                      <span>삭제된 종목</span>
                    </button>
                  </div>
                </label>
              </div>
            </div>
          </div>

          {viewMode === "deleted" && deletedRows.length > 0 && (
            <div className="card-header appActionHeader bg-light-subtle border-top">
              <div className="appActionHeaderInner">
                <button
                  className="btn btn-primary btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                  onClick={handleRestoreDeleted}
                  disabled={selectedDeletedTickers.length === 0}
                >
                    <IconArrowBackUp size={14} />
                    <span>선택 복구</span>
                </button>
                <button
                  className="btn btn-outline-danger btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                  onClick={handleHardDeleteDeleted}
                  disabled={selectedDeletedTickers.length === 0}
                >
                    <IconSearch size={14} />
                    <span>영구 삭제</span>
                </button>
              </div>
            </div>
          )}

          {viewMode === "active" && (
            <div className="card-header appActionHeader bg-light-subtle border-top">
              <div className="appActionHeaderInner">
                <button
                  className="btn btn-primary btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                  type="button"
                  onClick={handleAddRow}
                  disabled={loading || Boolean(addingRow)}
                >
                  <IconPlus size={16} stroke={2} />
                  <span>추가</span>
                </button>
                <button
                  className="btn btn-success btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                  type="button"
                  onClick={handleSaveActiveChanges}
                  disabled={loading || isPending || (!addingRow && dirtyActiveRowIds.length === 0)}
                >
                  <IconDeviceFloppy size={16} stroke={2} />
                  <span>저장</span>
                </button>
                <button
                  className="btn btn-outline-danger btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                  type="button"
                  onClick={handleDeleteActiveSelected}
                  disabled={loading || isPending || selectedActiveTickers.length === 0}
                >
                  <IconTrash size={16} stroke={2} />
                  <span>삭제</span>
                </button>
              </div>
            </div>
          )}

          <div className="card-body p-0 appTableCardBodyFill">
            {viewMode === "active" ? (
              <AppAgGrid<ActiveStockGridRow>
                className="stocksAgGrid"
                rowData={activeGridRows}
                columnDefs={activeColumns}
                loading={loading || isPending}
                minHeight="100%"
                theme={stocksGridTheme}
                gridOptions={{
                  suppressMovableColumns: true,
                  rowSelection: {
                    mode: "multiRow",
                    checkboxes: (params) => params.data?.id !== "__adding__",
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
                    suppressMovable: true,
                    headerName: "",
                    cellClass: "stocksSelectCell",
                  },
                  onSelectionChanged: (params: {
                    api: { getSelectedRows: () => ActiveStockGridRow[] };
                  }) => {
                    setSelectedActiveTickers(
                      params.api
                        .getSelectedRows()
                        .map((row) => row.id)
                        .filter((rowId) => rowId !== "__adding__"),
                    );
                  },
                  onCellValueChanged: (params: {
                    data?: ActiveStockGridRow;
                    newValue?: unknown;
                    oldValue?: unknown;
                    colDef: { field?: string };
                  }) => {
                    if (!params.data || params.data.id === "__adding__" || params.newValue === params.oldValue) {
                      return;
                    }
                    handleActiveBucketChanged(params.data, String(params.newValue ?? ""));
                  },
                }}
              />
            ) : (
              <AppAgGrid<DeletedStockGridRow>
                className="stocksAgGrid"
                rowData={deletedGridRows}
                columnDefs={deletedColumns}
                loading={loading || isPending}
                minHeight="100%"
                theme={stocksGridTheme}
                gridOptions={{
                  suppressMovableColumns: true,
                  rowSelection: {
                    mode: "multiRow",
                    checkboxes: true,
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
                    suppressMovable: true,
                    headerName: "",
                    cellClass: "stocksSelectCell",
                  },
                  onSelectionChanged: (params: {
                    api: { getSelectedRows: () => DeletedStockGridRow[] };
                  }) => {
                    setSelectedDeletedTickers(
                      params.api
                        .getSelectedRows()
                        .map((row) => row.ticker.trim().toUpperCase()),
                    );
                  },
                }}
              />
            )}
          </div>
        </div>
      </section>
      <AppModal
        open={deleteConfirmOpen}
        title="종목 삭제 확인"
        subtitle="선택 종목은 즉시 영구 삭제됩니다."
        onClose={handleCloseDeleteConfirm}
        footer={(
          <>
            <button type="button" className="btn btn-outline-secondary" onClick={handleCloseDeleteConfirm} disabled={isPending}>
              취소
            </button>
            <button type="button" className="btn btn-danger" onClick={handleConfirmDeleteActiveSelected} disabled={isPending}>
              삭제
            </button>
          </>
        )}
      >
        <div className="d-flex flex-column gap-3">
          <div className="fw-semibold">
            {selectedActiveTickers.length === 1
              ? `${activeRows.find((row) => selectedActiveTickers.includes(normalizeTicker(row.ticker)))?.name ?? ""}(${selectedActiveTickers[0]}) 종목을 삭제합니다.`
              : `${selectedActiveTickers.length}개 종목을 삭제합니다.`}
          </div>
          <div className="text-secondary small">
            {selectedActiveTickers.length > 1
              ? selectedActiveTickers.join(", ")
              : "삭제된 종목은 복구되지 않으며 즉시 제거됩니다."}
          </div>
        </div>
      </AppModal>
    </div>
  );
}
