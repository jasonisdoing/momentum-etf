"use client";

import { useEffect, useMemo, useState, useTransition } from "react";
import { type GridColDef } from "@mui/x-data-grid";

import { AppDataGrid } from "../components/AppDataGrid";
import { AppModal } from "../components/AppModal";
import { useToast } from "../components/ToastProvider";

type WeeklyEditableField = {
  key: string;
  label: string;
  type: "int" | "float" | "text";
};

type WeeklyRow = {
  week_date: string;
  week_date_display: string;
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
  cumulative_profit: number;
  weekly_profit: number;
  weekly_return_pct: number;
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

type WeeklyResponse = {
  active_week_date?: string;
  rows?: WeeklyRow[];
  editable_fields?: WeeklyEditableField[];
  core_hidden_keys?: string[];
  error?: string;
};

type WeeklyGridRow = WeeklyRow & {
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
  "cumulative_profit",
  "weekly_profit",
]);

const PERCENT_KEYS = new Set([
  "weekly_return_pct",
  "cumulative_return_pct",
  "exchange_rate_change_pct",
  "bucket_pct_momentum",
  "bucket_pct_market",
  "bucket_pct_dividend",
  "bucket_pct_alternative",
  "bucket_pct_cash",
]);

const COLUMN_DEFS = [
  { key: "week_date_display", label: "날짜" },
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
  { key: "cumulative_profit", label: "누적 손익" },
  { key: "weekly_profit", label: "금주 손익" },
  { key: "weekly_return_pct", label: "주수익률" },
  { key: "cumulative_return_pct", label: "누적 수익률" },
  { key: "memo", label: "비고" },
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

function formatCellValue(row: WeeklyRow, key: (typeof COLUMN_DEFS)[number]["key"]): string {
  const value = row[key as keyof WeeklyRow];
  if (key === "week_date_display" || key === "memo") {
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
  if (key === "week_date_display") {
    classes.push("weeklyDateCell");
  }
  if (key === "memo") {
    classes.push("weeklyMemoCell");
  }
  if (MONEY_KEYS.has(key) || PERCENT_KEYS.has(key) || key === "exchange_rate") {
    classes.push("tableAlignRight");
  }
  if (typeof value === "number" && (key === "profit_loss" || key === "cumulative_profit" || key === "weekly_profit" || PERCENT_KEYS.has(key))) {
    const signedClass = getSignedClass(value);
    if (signedClass) {
      classes.push(signedClass);
    }
  }
  return classes.join(" ");
}

function toInputText(field: WeeklyEditableField, value: WeeklyRow | null): string {
  if (!value) {
    return "";
  }
  const raw = value[field.key as keyof WeeklyRow];
  if (field.type === "text") {
    return String(raw ?? "");
  }
  return String(raw ?? 0);
}

export function WeeklyManager() {
  const [rows, setRows] = useState<WeeklyRow[]>([]);
  const [editableFields, setEditableFields] = useState<WeeklyEditableField[]>([]);
  const [activeWeekDate, setActiveWeekDate] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("core");
  const [coreHiddenKeys, setCoreHiddenKeys] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [editingRow, setEditingRow] = useState<WeeklyRow | null>(null);
  const [editingValues, setEditingValues] = useState<Record<string, string>>({});
  const [isPending, startTransition] = useTransition();
  const toast = useToast();

  async function load(options?: { silent?: boolean }) {
    const silent = options?.silent ?? false;
    if (!silent) {
      setLoading(true);
    }
    setError(null);

    try {
      const response = await fetch("/api/weekly", { cache: "no-store" });
      const payload = (await response.json()) as WeeklyResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "주별 데이터를 불러오지 못했습니다.");
      }
      setRows(payload.rows ?? []);
      setEditableFields(payload.editable_fields ?? []);
      setCoreHiddenKeys(payload.core_hidden_keys ?? []);
      setActiveWeekDate(payload.active_week_date ?? "");
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "주별 데이터를 불러오지 못했습니다.");
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

  const gridRows = useMemo<WeeklyGridRow[]>(
    () =>
      rows.map((row) => ({
        ...row,
        id: row.week_date,
      })),
    [rows],
  );

  const gridColumns = useMemo<GridColDef<WeeklyGridRow>[]>(
    () => [
      {
        field: "__edit__",
        headerName: "",
        width: 58,
        sortable: false,
        filterable: false,
        disableColumnMenu: true,
        renderCell: (params) => (
          <button
            className="btn btn-link btn-sm p-0 appEditLink"
            type="button"
            onClick={() => openEditModal(params.row)}
          >
            Edit
          </button>
        ),
      },
      ...visibleColumns.map<GridColDef<WeeklyGridRow>>((column) => ({
        field: column.key,
        headerName: column.label,
        type: "string",
        minWidth:
          column.key === "week_date_display"
            ? 132
            : column.key === "memo"
              ? 320
              : MONEY_KEYS.has(column.key) || PERCENT_KEYS.has(column.key) || column.key === "exchange_rate"
                ? 108
                : 96,
        flex: column.key === "memo" ? 1.4 : column.key === "week_date_display" ? 0 : undefined,
        align:
          MONEY_KEYS.has(column.key) || PERCENT_KEYS.has(column.key) || column.key === "exchange_rate"
            ? "right"
            : "left",
        headerAlign:
          MONEY_KEYS.has(column.key) || PERCENT_KEYS.has(column.key) || column.key === "exchange_rate"
            ? "right"
            : "left",
        sortable: false,
        cellClassName: (params) => getColumnCellClass(column.key, params.value as number | string),
        renderCell: (params) => (
          <span title={column.key === "memo" ? String(params.value ?? "") : undefined}>
            {formatCellValue(params.row, column.key)}
          </span>
        ),
      })),
    ],
    [visibleColumns],
  );

  function openEditModal(row: WeeklyRow) {
    setEditingRow(row);
    setEditingValues(
      Object.fromEntries(editableFields.map((field) => [field.key, toInputText(field, row)])),
    );
  }

  function closeEditModal() {
    if (isPending) {
      return;
    }
    setEditingRow(null);
    setEditingValues({});
  }

  function updateFieldValue(key: string, value: string) {
    setEditingValues((current) => ({ ...current, [key]: value }));
  }

  function handleAggregate() {
    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/weekly", { method: "POST" });
        const payload = (await response.json()) as { week_date?: string; error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "이번주 데이터 집계에 실패했습니다.");
        }
        await load({ silent: true });
        toast.success(`[자산-주별] ${payload.week_date ?? activeWeekDate} 집계 완료`);
      } catch (aggregateError) {
        setError(aggregateError instanceof Error ? aggregateError.message : "이번주 데이터 집계에 실패했습니다.");
      }
    });
  }

  function handleSave() {
    if (!editingRow) {
      return;
    }

    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/weekly", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            week_date: editingRow.week_date,
            ...Object.fromEntries(
              editableFields.map((field) => {
                if (field.type === "text") {
                  return [field.key, editingValues[field.key] ?? ""];
                }
                const normalized = Number(editingValues[field.key] ?? 0);
                return [field.key, Number.isFinite(normalized) ? normalized : 0];
              }),
            ),
          }),
        });
        const payload = (await response.json()) as { error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "주별 데이터 저장에 실패했습니다.");
        }
        await load({ silent: true });
        toast.success(`[자산-주별] ${editingRow.week_date_display} 저장 완료`);
        closeEditModal();
      } catch (saveError) {
        setError(saveError instanceof Error ? saveError.message : "주별 데이터 저장에 실패했습니다.");
      }
    });
  }

  const halfIndex = Math.ceil(editableFields.length / 2);
  const leftFields = editableFields.slice(0, halfIndex);
  const rightFields = editableFields.slice(halfIndex);

  return (
    <div className="appPageStack">
      {error ? (
        <div className="appBannerStack">
          {error ? <div className="bannerError">{error}</div> : null}
        </div>
      ) : null}

      <section className="appSection">
        <div className="card appCard">
          <div className="card-header appCardHeader">
            <div className="sectionHeaderCompact w-100">
              <div className="tableToolbar">
                <div className="toolbarActions">
                  <button
                    type="button"
                    className={`btn btn-sm ${viewMode === "core" ? "btn-primary" : "btn-outline-secondary"}`}
                    onClick={() => setViewMode("core")}
                  >
                    핵심만 보기
                  </button>
                  <button
                    type="button"
                    className={`btn btn-sm ${viewMode === "full" ? "btn-primary" : "btn-outline-secondary"}`}
                    onClick={() => setViewMode("full")}
                  >
                    전체 보기
                  </button>
                  <button type="button" className="btn btn-sm btn-success" onClick={handleAggregate} disabled={isPending}>
                    {isPending ? "집계 중..." : "이번주 데이터 집계"}
                  </button>
                </div>
                <div className="tableMeta">
                  <span>활성 주차 {activeWeekDate || "-"}</span>
                  <span>행 {rows.length}</span>
                </div>
              </div>
            </div>
          </div>
          <div className="card-body appCardBodyTight">
            <AppDataGrid
              rows={gridRows}
              columns={gridColumns}
              loading={loading}
              minHeight="68vh"
              getRowClassName={(params) => (params.row.week_date === activeWeekDate ? "tableRowSelected" : "")}
            />
          </div>
        </div>
      </section>

      <AppModal
        open={Boolean(editingRow)}
        title={editingRow?.week_date_display ?? ""}
        subtitle="주별 데이터 수정"
        size="xl"
        onClose={closeEditModal}
        footer={
          <>
            <button type="button" className="btn btn-outline-secondary" onClick={closeEditModal} disabled={isPending}>
              취소
            </button>
            <button type="button" className="btn btn-primary" onClick={handleSave} disabled={isPending}>
              {isPending ? "저장 중..." : "저장"}
            </button>
          </>
        }
      >
        {editingRow ? (
          <div className="row g-3">
            <div className="col-md-6">
              {leftFields.map((field) => (
                <div key={field.key} className="mb-3">
                  <label className="form-label">{field.label}</label>
                  {field.type === "text" ? (
                    <input
                      className="form-control"
                      value={editingValues[field.key] ?? ""}
                      onChange={(event) => updateFieldValue(field.key, event.target.value)}
                    />
                  ) : (
                    <input
                      className="form-control"
                      type="number"
                      step={field.type === "float" ? "0.01" : "1"}
                      value={editingValues[field.key] ?? "0"}
                      onChange={(event) => updateFieldValue(field.key, event.target.value)}
                    />
                  )}
                </div>
              ))}
            </div>
            <div className="col-md-6">
              {rightFields.map((field) => (
                <div key={field.key} className="mb-3">
                  <label className="form-label">{field.label}</label>
                  {field.type === "text" ? (
                    <input
                      className="form-control"
                      value={editingValues[field.key] ?? ""}
                      onChange={(event) => updateFieldValue(field.key, event.target.value)}
                    />
                  ) : (
                    <input
                      className="form-control"
                      type="number"
                      step={field.type === "float" ? "0.01" : "1"}
                      value={editingValues[field.key] ?? "0"}
                      onChange={(event) => updateFieldValue(field.key, event.target.value)}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </AppModal>
    </div>
  );
}
