"use client";

import { useEffect, useMemo, useState, useTransition } from "react";
import type { ColDef } from "ag-grid-community";
import { IconCheck } from "@tabler/icons-react";

import { AppAgGrid } from "../components/AppAgGrid";
import { ResponsiveFiltersSection } from "../components/ResponsiveFiltersSection";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";

type DailyEditableField = {
  key: string;
  label: string;
  type: "int" | "float" | "text";
};

type DailyRow = {
  date: string;
  date_display: string;
  withdrawal_personal: number;
  withdrawal_mom: number;
  nh_principal_interest: number;
  total_expense: number;
  deposit_withdrawal: number;
  total_principal: number;
  total_assets: number;
  purchase_amount: number;
  valuation_amount: number;
  profit_loss: number;
  daily_profit: number;
  cumulative_profit: number;
  daily_return_pct: number;
  cumulative_return_pct: number;
  memo: string;
  exchange_rate: number;
  exchange_rate_change_pct: number;
  bucket_pct_momentum: number;
  bucket_pct_market: number;
  bucket_pct_dividend: number;
  bucket_pct_alternative: number;
  bucket_pct_cash: number;
  total_stocks: number;
  profit_count: number;
  loss_count: number;
  updated_at: string | null;
};

type DailyResponse = {
  latest_date?: string;
  rows?: DailyRow[];
  editable_fields?: DailyEditableField[];
  read_only_keys?: string[];
  core_hidden_keys?: string[];
  error?: string;
};

type DailyGridRow = DailyRow & {
  id: string;
};

type ViewMode = "core" | "full";

const MONEY_KEYS = new Set([
  "withdrawal_personal",
  "withdrawal_mom",
  "nh_principal_interest",
  "total_expense",
  "deposit_withdrawal",
  "total_principal",
  "total_assets",
  "purchase_amount",
  "valuation_amount",
  "profit_loss",
  "daily_profit",
  "cumulative_profit",
]);

const PERCENT_KEYS = new Set([
  "daily_return_pct",
  "cumulative_return_pct",
  "exchange_rate_change_pct",
  "bucket_pct_momentum",
  "bucket_pct_market",
  "bucket_pct_dividend",
  "bucket_pct_alternative",
  "bucket_pct_cash",
]);

const COLUMN_DEFS = [
  { key: "date_display", label: "일자" },
  { key: "memo", label: "비고" },
  { key: "withdrawal_personal", label: "개인 인출" },
  { key: "withdrawal_mom", label: "엄마" },
  { key: "nh_principal_interest", label: "농협원리금" },
  { key: "total_expense", label: "지출 합계" },
  { key: "deposit_withdrawal", label: "입출금" },
  { key: "total_principal", label: "총 원금" },
  { key: "total_assets", label: "총 자산" },
  { key: "purchase_amount", label: "매입 금액" },
  { key: "valuation_amount", label: "평가 금액" },
  { key: "profit_loss", label: "평가 손익" },
  { key: "daily_profit", label: "금일 손익" },
  { key: "cumulative_profit", label: "누적 손익" },
  { key: "daily_return_pct", label: "일수익률" },
  { key: "cumulative_return_pct", label: "누적 수익률" },
  { key: "exchange_rate_change_pct", label: "환율(변동)" },
  { key: "exchange_rate", label: "환율" },
  { key: "bucket_pct_momentum", label: "1. 모멘텀" },
  { key: "bucket_pct_market", label: "2. 시장지수" },
  { key: "bucket_pct_dividend", label: "3. 배당방어" },
  { key: "bucket_pct_alternative", label: "4. 대체헷지" },
  { key: "bucket_pct_cash", label: "5. 현금" },
  { key: "total_stocks", label: "총 종목 수" },
  { key: "profit_count", label: "수익 종목 수" },
  { key: "loss_count", label: "손실 종목 수" },
] as const;

const dailyGridTheme = createAppGridTheme();

function formatMoney(value: number): string {
  return new Intl.NumberFormat("ko-KR").format(Math.round(value));
}

function formatPercent(value: number): string {
  return `${value.toFixed(2)}%`;
}

function formatExchangeRate(value: number): string {
  return value.toFixed(2);
}

function getSignedClass(value: number): string {
  if (!Number.isFinite(value) || value === 0) {
    return "";
  }
  return value > 0 ? "metricPositive" : "metricNegative";
}

function formatCellValue(row: DailyRow, key: (typeof COLUMN_DEFS)[number]["key"]): string {
  const value = row[key as keyof DailyRow];
  if (key === "date_display" || key === "memo") {
    return String(value ?? "-");
  }
  if (key === "exchange_rate") {
    return formatExchangeRate(Number(value ?? 0));
  }
  if (PERCENT_KEYS.has(key)) {
    return formatPercent(Number(value ?? 0));
  }
  return formatMoney(Number(value ?? 0));
}

function getColumnCellClass(key: string, value: number | string): string {
  const classes: string[] = [];
  if (key === "date_display") {
    classes.push("weeklyDateCell");
  }
  if (key === "memo") {
    classes.push("weeklyMemoCell");
  }
  if (MONEY_KEYS.has(key) || PERCENT_KEYS.has(key) || key === "exchange_rate") {
    classes.push("tableAlignRight");
  }
  if (typeof value === "number" && (key === "profit_loss" || key === "cumulative_profit" || PERCENT_KEYS.has(key))) {
    const signedClass = getSignedClass(value);
    if (signedClass) {
      classes.push(signedClass);
    }
  }
  return classes.join(" ");
}

function buildDirtyCellKey(rowId: string, field: string): string {
  return `${rowId}::${field}`;
}

function parseDailyCellValue(field: DailyEditableField | undefined, newValue: unknown, oldValue: unknown) {
  if (!field) {
    return newValue;
  }
  if (field.type === "text") {
    return String(newValue ?? "");
  }
  const parsed = Number(newValue);
  if (!Number.isFinite(parsed)) {
    return oldValue;
  }
  if (field.type === "int") {
    return Math.trunc(parsed);
  }
  return parsed;
}

function getDailyColumnWidth(key: (typeof COLUMN_DEFS)[number]["key"]): { width?: number; minWidth: number; flex?: number } {
  if (key === "date_display") {
    return { width: 132, minWidth: 132 };
  }
  if (key === "memo") {
    return { width: 156, minWidth: 140, flex: 1 };
  }
  if (
    key === "total_principal" ||
    key === "total_assets" ||
    key === "purchase_amount" ||
    key === "valuation_amount" ||
    key === "profit_loss" ||
    key === "daily_profit" ||
    key === "cumulative_profit"
  ) {
    return { width: 102, minWidth: 98 };
  }
  if (key === "exchange_rate") {
    return { width: 82, minWidth: 82 };
  }
  if (key === "exchange_rate_change_pct") {
    return { width: 94, minWidth: 94 };
  }
  if (key === "total_stocks" || key === "profit_count" || key === "loss_count") {
    return { width: 84, minWidth: 80 };
  }
  if (
    key === "withdrawal_mom" ||
    key === "total_expense" ||
    key === "deposit_withdrawal" ||
    key === "bucket_pct_cash"
  ) {
    return { width: 88, minWidth: 84 };
  }
  if (key === "daily_return_pct" || key === "cumulative_return_pct") {
    return { width: 96, minWidth: 92 };
  }
  if (
    key === "bucket_pct_momentum" ||
    key === "bucket_pct_market" ||
    key === "bucket_pct_dividend" ||
    key === "bucket_pct_alternative"
  ) {
    return { width: 96, minWidth: 92 };
  }
  if (PERCENT_KEYS.has(key)) {
    return { width: 84, minWidth: 80 };
  }
  if (MONEY_KEYS.has(key)) {
    return { width: 88, minWidth: 84 };
  }
  return { width: 78, minWidth: 72 };
}

export function DailyManager({
  onHeaderSummaryChange,
}: {
  onHeaderSummaryChange?: (summary: { latestDate: string; rowCount: number; dirtyCount: number }) => void;
}) {
  const [rows, setRows] = useState<DailyRow[]>([]);
  const [editableFields, setEditableFields] = useState<DailyEditableField[]>([]);
  const [readOnlyKeys, setReadOnlyKeys] = useState<Set<string>>(new Set());
  const [latestDate, setLatestDate] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("core");
  const [coreHiddenKeys, setCoreHiddenKeys] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [dirtyRowIds, setDirtyRowIds] = useState<string[]>([]);
  const [dirtyCellKeys, setDirtyCellKeys] = useState<string[]>([]);
  const [isPending, startTransition] = useTransition();
  const toast = useToast();

  async function load(options?: { silent?: boolean }) {
    const silent = options?.silent ?? false;
    if (!silent) {
      setLoading(true);
    }
    setError(null);

    try {
      const response = await fetch("/api/daily", { cache: "no-store" });
      const payload = (await response.json()) as DailyResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "일별 데이터를 불러오지 못했습니다.");
      }
      setRows(payload.rows ?? []);
      setEditableFields(payload.editable_fields ?? []);
      setReadOnlyKeys(new Set(payload.read_only_keys ?? []));
      setCoreHiddenKeys(payload.core_hidden_keys ?? []);
      setLatestDate(payload.latest_date ?? "");
      setDirtyRowIds([]);
      setDirtyCellKeys([]);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "일별 데이터를 불러오지 못했습니다.");
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }

  useEffect(() => {
    void load();
  }, []);

  const visibleColumns = useMemo(() => {
    if (viewMode === "full") {
      return COLUMN_DEFS;
    }
    return COLUMN_DEFS.filter((column) => !coreHiddenKeys.includes(column.key));
  }, [coreHiddenKeys, viewMode]);

  const gridRows = useMemo<DailyGridRow[]>(
    () =>
      rows.map((row) => ({
        ...row,
        id: row.date,
      })),
    [rows],
  );

  useEffect(() => {
    onHeaderSummaryChange?.({
      latestDate: latestDate || "-",
      rowCount: rows.length,
      dirtyCount: dirtyRowIds.length,
    });
  }, [dirtyRowIds.length, latestDate, onHeaderSummaryChange, rows.length]);

  const editableFieldMap = useMemo(
    () => new Map(editableFields.map((field) => [field.key, field])),
    [editableFields],
  );

  const gridColumns = useMemo<ColDef<DailyGridRow>[]>(
    () => [
      ...visibleColumns.map<ColDef<DailyGridRow>>((column) => ({
        ...getDailyColumnWidth(column.key),
        field: column.key,
        headerName: column.label,
        type:
          MONEY_KEYS.has(column.key) || PERCENT_KEYS.has(column.key) || column.key === "exchange_rate"
            ? "rightAligned"
            : undefined,
        sortable: false,
        editable: () => editableFieldMap.has(column.key) && !readOnlyKeys.has(column.key),
        valueParser: (params) => parseDailyCellValue(editableFieldMap.get(column.key), params.newValue, params.oldValue),
        cellClass: (params) => {
          const classes = getColumnCellClass(column.key, params.value as number | string);
          const editableClasses =
            editableFieldMap.has(column.key) && !readOnlyKeys.has(column.key)
              ? dirtyCellKeys.includes(buildDirtyCellKey(params.data?.id ?? "", column.key))
                ? " weeklyEditableCell weeklyDirtyCell"
                : " weeklyEditableCell"
              : "";
          return `${classes}${editableClasses}`.trim();
        },
        cellRenderer: (params: { data?: DailyGridRow; value?: unknown }) => (
          <span title={column.key === "memo" ? String(params.value ?? "") : undefined}>
            {params.data ? formatCellValue(params.data, column.key) : "-"}
          </span>
        ),
      })),
    ],
    [dirtyCellKeys, editableFieldMap, readOnlyKeys, visibleColumns],
  );

  function handleSave() {
    if (dirtyRowIds.length === 0) {
      return;
    }

    startTransition(async () => {
      try {
        setError(null);
        const dirtyRows = rows.filter((row) => dirtyRowIds.includes(row.date));
        for (const row of dirtyRows) {
          const response = await fetch("/api/daily", {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              date: row.date,
              ...Object.fromEntries(
                editableFields
                  .filter((field) => !readOnlyKeys.has(field.key))
                  .map((field) => [field.key, row[field.key as keyof DailyRow]]),
              ),
            }),
          });
          const payload = (await response.json()) as { error?: string };
          if (!response.ok) {
            throw new Error(payload.error ?? "일별 데이터 저장에 실패했습니다.");
          }
        }
        await load({ silent: true });
        toast.success("[자산-일별] 변경사항 저장 완료");
      } catch (saveError) {
        setError(saveError instanceof Error ? saveError.message : "일별 데이터 저장에 실패했습니다.");
      }
    });
  }

  return (
    <div className="appPageStack appPageStackFill">
      {error ? (
        <div className="appBannerStack">
          <div className="bannerError">{error}</div>
        </div>
      ) : null}

      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <ResponsiveFiltersSection>
              <div className="appMainHeader">
                <div className="appMainHeaderLeft weeklyMainHeaderLeft">
                  <label className="appLabeledField">
                    <span className="appLabeledFieldLabel">보기 방식</span>
                    <div className="appSegmentedToggle" role="group" aria-label="일별 보기 방식">
                      <button
                        type="button"
                        className={viewMode === "core" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                        onClick={() => setViewMode("core")}
                      >
                        핵심만 보기
                      </button>
                      <button
                        type="button"
                        className={viewMode === "full" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                        onClick={() => setViewMode("full")}
                      >
                        전체 보기
                      </button>
                    </div>
                  </label>
                  <label className="appLabeledField">
                    <span className="appLabeledFieldLabel">원장 상태</span>
                    <span className="form-control form-control-sm bg-light text-secondary d-flex align-items-center">
                      기존 주별 종료일 스냅샷을 일별 원장 시드로 조회
                    </span>
                  </label>
                </div>
              </div>
            </ResponsiveFiltersSection>
          </div>
          <div className="card-header appActionHeader bg-light-subtle border-top">
            <div className="appActionHeaderInner">
              <button
                type="button"
                className="btn btn-success btn-sm px-3 fw-bold d-flex align-items-center gap-1"
                onClick={handleSave}
                disabled={isPending || dirtyRowIds.length === 0}
              >
                <IconCheck size={16} />
                <span>저장</span>
              </button>
            </div>
          </div>
          <div className="card-body appCardBodyTight appTableCardBodyFill">
            <AppAgGrid<DailyGridRow>
              rowData={gridRows}
              columnDefs={gridColumns}
              loading={loading || isPending}
              minHeight="100%"
              className="weeklyAgGrid"
              theme={dailyGridTheme}
              getRowClass={(params) => (params.data?.date === latestDate ? "tableRowSelected" : "")}
              gridOptions={{
                suppressMovableColumns: true,
                onCellValueChanged: (params: {
                  data?: DailyGridRow;
                  colDef: { field?: string };
                  newValue?: unknown;
                  oldValue?: unknown;
                }) => {
                  if (!params.data || !params.colDef.field || params.newValue === params.oldValue) {
                    return;
                  }
                  const rowId = params.data.id;
                  const field = params.colDef.field;
                  setRows((current) =>
                    current.map((row) =>
                      row.date === rowId
                        ? {
                            ...row,
                            [field]: params.data?.[field as keyof DailyGridRow] ?? row[field as keyof DailyRow],
                          }
                        : row,
                    ),
                  );
                  setDirtyRowIds((prev) => (prev.includes(rowId) ? prev : [...prev, rowId]));
                  const dirtyCellKey = buildDirtyCellKey(rowId, field);
                  setDirtyCellKeys((prev) => (prev.includes(dirtyCellKey) ? prev : [...prev, dirtyCellKey]));
                },
              }}
            />
          </div>
        </div>
      </section>
    </div>
  );
}
