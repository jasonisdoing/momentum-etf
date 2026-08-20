"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import type { ColDef, ColumnState, GridApi, GridOptions, RowClassParams } from "ag-grid-community";
import { IconLoader2 } from "@tabler/icons-react";
import { AppAgGrid } from "../components/AppAgGrid";
import { GridToolbarButton } from "../components/GridToolbarButton";
import { StableInlineInput } from "../components/StableInlineInput";
import { AppLoadingState } from "../components/AppLoadingState";
import { AppModal } from "../components/AppModal";
import { TickerDetailLink } from "../components/TickerDetailLink";
import { renderStockNameCell } from "@/lib/name-highlight";
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";
import { reorderHoldings } from "@/lib/holdings-store";
import { fetchAlertBadges, normalizeBadgeTicker, type AlertBadges } from "@/lib/alert-badges";
import { formatKstDateTime } from "@/lib/datetime";

import {
  AccountSummary,
  AddingRowState,
  CASH_ROW_TICKER,
  GridRow,
  HoldingEditableSnapshot,
  HoldingsRow,
  SavedCashAccount,
  assetsGridTheme,
  buildAutoSaveToastMessage,
  buildCashGridRow,
  formatUpdatedBy,
  buildDirtyCellKey,
  buildGridRowId,
  buildHoldingEditableSnapshot,
  buildSyncedHoldingRows,
  formatHiddenAmount,
  formatKrw,
  formatPrice,
  getBucketCellClass,
  getPreviewValuationKrw,
  getPreviewWeightPct,
  getSignedClass,
  parseEditableQuantity,
  parseRawPrice,
  reorderRowsByTickers,
  safeParseFloat,
  stopActionButtonClick,
  stopActionButtonMouseDown,
} from "./assets-helpers";

export function AccountHoldingsDetailPanel({
  summary,
  initialRows,
  onRowsSync,
  onCashSync,
  onSortStateChange,
  onReload,
  showAmounts,
}: {
  summary: AccountSummary;
  initialRows: HoldingsRow[];
  onRowsSync: (accountId: string, rows: HoldingsRow[]) => void;
  onCashSync: (accountId: string, balance: number, targetRatio: number, saved?: SavedCashAccount) => void;
  onSortStateChange: (accountId: string, sortState: ColumnState[]) => void;
  onReload: (options?: { silent?: boolean }) => Promise<void>;
  showAmounts: boolean;
}) {
  const toast = useToast();

  const hydrateRows = useCallback(
    (sourceRows: HoldingsRow[]) =>
      sourceRows.map((row) => ({
        ...row,
        original_quantity: Number(row.original_quantity ?? row.quantity ?? 0),
        original_average_buy_price: Number(row.original_average_buy_price ?? safeParseFloat(row.average_buy_price)),
        original_buy_amount_krw: Number(row.original_buy_amount_krw ?? row.buy_amount_krw ?? 0),
        original_valuation_krw: Number(row.original_valuation_krw ?? row.valuation_krw ?? 0),
      })),
    [],
  );

  const [rows, setRows] = useState<HoldingsRow[]>(() => hydrateRows(initialRows));
  const [addingRow, setAddingRow] = useState<AddingRowState | null>(null);
  // 입력 중인 티커 초안 — 타이핑마다 setState 하면 그리드 셀이 리마운트돼 포커스가 튄다(1글자만 입력됨).
  // draft 는 ref 에만 쌓고, 검증 시점에만 읽는다.
  const addingTickerDraftRef = useRef("");
  // 종목명 알람 배지(이동선 이탈·손절 아이콘) — /alarms 설정·판정 그대로. 보조 정보라 실패 시 빈 맵.
  const [alertBadges, setAlertBadges] = useState<AlertBadges>({});
  // 이동선 이탈 종목 — 배지와 같은 조건으로 행을 회색 처리한다.
  const [maBrokenTickers, setMaBrokenTickers] = useState<Set<string>>(new Set());
  useEffect(() => {
    let alive = true;
    void fetchAlertBadges(summary.account_id).then((info) => {
      if (!alive) return;
      setAlertBadges(info.badgeByTicker);
      setMaBrokenTickers(new Set(info.maTickers));
    });
    return () => {
      alive = false;
    };
  }, [summary.account_id]);
  const [editingRowId, setEditingRowId] = useState<string | null>(null);
  const [dirtyRowIds, setDirtyRowIds] = useState<string[]>([]);
  const [dirtyCellKeys, setDirtyCellKeys] = useState<string[]>([]);
  const [selectedRowIds, setSelectedRowIds] = useState<string[]>([]);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [isReorderDirty, setIsReorderDirty] = useState(false);
  const qtyRef = useRef<HTMLInputElement>(null);
  const priceRef = useRef<HTMLInputElement>(null);
  const rowsRef = useRef<HoldingsRow[]>(initialRows);
  const summaryRef = useRef(summary);
  const dirtyRowIdsRef = useRef<string[]>([]);
  const isReorderDirtyRef = useRef(false);
  const gridApiRef = useRef<GridApi<GridRow> | null>(null);
  const lastSavedSnapshotsRef = useRef<Map<string, HoldingEditableSnapshot>>(new Map());
  const childSaveTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const childSavingRowIdsRef = useRef<Set<string>>(new Set());
  const childQueuedRowIdsRef = useRef<Set<string>>(new Set());
  const reorderSaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reorderSavingRef = useRef(false);
  const reorderQueuedRef = useRef(false);
  const cashDraftRef = useRef({
    cashBalanceKrw: Number(summary.cash_balance_krw ?? 0),
    cashTargetRatio: Number(summary.cash_target_ratio ?? 0),
  });
  const isAusAccount = String(summary.currency || "KRW").toUpperCase() === "AUD";
  const intlDraftRef = useRef({
    intlSharesValue: Number(summary.intl_shares_value ?? 0),
    intlSharesChange: Number(summary.intl_shares_change ?? 0),
    cashNative: Number(summary.cash_balance_native ?? 0),
  });
  const [intlDirtyFields, setIntlDirtyFields] = useState<string[]>([]);
  // 통화별 현금 native 입력 초안(상단 박스). summary.cash 로 시드하고 입력 시 갱신.
  const cashMapDraftRef = useRef<Record<string, number>>({ ...(summary.cash ?? {}) });
  const [cashMapDirty, setCashMapDirty] = useState(false);
  const [cashMapSaving, setCashMapSaving] = useState(false);
  // 이 계좌가 보유하는 현금 통화 목록(설정). 없으면 주 통화 1개.
  const cashCurrencyList =
    summary.cash_currencies && summary.cash_currencies.length > 0
      ? summary.cash_currencies.map((c) => c.toUpperCase())
      : [String(summary.currency || "KRW").toUpperCase()];
  useEffect(() => {
    const nextRows = hydrateRows(initialRows);
    setRows(nextRows);
    rowsRef.current = nextRows;
    dirtyRowIdsRef.current = [];
    isReorderDirtyRef.current = false;
    lastSavedSnapshotsRef.current = new Map(
      nextRows.map((row) => [buildGridRowId(row), buildHoldingEditableSnapshot(row)]),
    );
    setDirtyRowIds([]);
    setDirtyCellKeys([]);
    setSelectedRowIds([]);
    setEditingRowId(null);
    setAddingRow(null);
    setIsReorderDirty(false);
    queueMicrotask(() => {
      if (!gridApiRef.current) {
        return;
      }
      gridApiRef.current.applyColumnState({
        state: [],
        applyOrder: false,
      });
    });
  }, [hydrateRows, initialRows]);

  useEffect(() => {
    rowsRef.current = rows;
  }, [rows]);

  useEffect(() => {
    summaryRef.current = summary;
    cashDraftRef.current = {
      cashBalanceKrw: Number(summary.cash_balance_krw ?? 0),
      cashTargetRatio: Number(summary.cash_target_ratio ?? 0),
    };
    cashMapDraftRef.current = { ...(summary.cash ?? {}) };
    setCashMapDirty(false);
  }, [summary]);

  useEffect(() => {
    dirtyRowIdsRef.current = dirtyRowIds;
  }, [dirtyRowIds]);

  useEffect(() => {
    isReorderDirtyRef.current = isReorderDirty;
  }, [isReorderDirty]);

  const isEditableHoldingRow = useCallback(
    (row: GridRow | undefined | null) =>
      Boolean(row && row.id !== "__adding__" && row.ticker !== "IS" && row.ticker !== CASH_ROW_TICKER),
    [],
  );
  const isCashGridRow = useCallback(
    (row: GridRow | undefined | null) => Boolean(row && row.ticker === CASH_ROW_TICKER),
    [],
  );
  const gridRows = useMemo<GridRow[]>(() => {
    const cashRow = buildCashGridRow(summary);
    const baseRows = rows
      .map((row, index) => ({
        ...row,
        id: buildGridRowId(row),
        quantity: typeof row.quantity === "number" ? row.quantity : parseInt(String(row.quantity), 10) || 0,
        average_buy_price: safeParseFloat(row.average_buy_price),
        target_ratio: row.target_ratio ?? 0,
        memo: row.memo ?? "",
      }));

    if (!addingRow) {
      return [cashRow, ...baseRows];
    }

    return [
      cashRow,
      {
        id: "__adding__",
        account_id: summary.account_id,
        account_name: summary.name,
        currency: summary.currency,
        bucket: "",
        bucket_id: 0,
        ticker: addingRow.ticker,
        name: addingRow.name || "",
        quantity: 0,
        average_buy_price: 0,
        current_price: "-",
        pnl_krw: 0,
        return_pct: 0,
        weight_pct: 0,
        buy_amount_krw: 0,
        valuation_krw: 0,
        target_ratio: 0,
        memo: "",
      } as GridRow,
      ...baseRows,
    ];
  }, [addingRow, rows, summary]);

  const hasPendingAdd = Boolean(addingRow);
  const hasSelectedRows = selectedRowIds.length > 0;
  const hasPendingSave = hasPendingAdd || dirtyRowIds.length > 0 || isReorderDirty;
  const selectedDeletableRows = useMemo(
    () => gridRows.filter((row) => selectedRowIds.includes(row.id) && row.id !== "__adding__"),
    [gridRows, selectedRowIds],
  );

  const isDirtyEditableCell = useCallback(
    (rowId: string | undefined, field: string) => Boolean(rowId && dirtyCellKeys.includes(buildDirtyCellKey(rowId, field))),
    [dirtyCellKeys],
  );

  const handleValidateTicker = useCallback(async (tickerToUse?: string) => {
    const ticker = String(tickerToUse || addingTickerDraftRef.current || addingRow?.ticker || "").trim().toUpperCase();
    if (!ticker || addingRow?.isValidatingTicker) {
      return;
    }

    const normalizedTicker = ticker.replace(/^ASX:/, "");
    const hasDuplicate = rows.some(
      (row) => String(row.ticker || "").trim().toUpperCase().replace(/^ASX:/, "") === normalizedTicker,
    );

    if (hasDuplicate) {
      const message = "이미 해당 계좌에 추가된 종목입니다.";
      setAddingRow((previous) =>
        previous
          ? {
            ...previous,
            ticker,
            name: message,
            bucketId: undefined,
            isValidated: false,
            isValidatingTicker: false,
          }
          : null,
      );
      toast.error(message);
      return;
    }

    setAddingRow((previous) =>
      previous
        ? {
          ...previous,
          ticker,
          name: "",
          bucketId: undefined,
          isValidated: false,
          isValidatingTicker: true,
        }
        : null,
    );

    try {
      const response = await fetch("/api/assets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "validate",
          account_id: summary.account_id,
          ticker,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "검증 실패");
      }
      addingTickerDraftRef.current = payload.ticker;
      setAddingRow((previous) =>
        previous
          ? {
            ...previous,
            ticker: payload.ticker,
            name: payload.name,
            bucketId: payload.bucket_id,
            isValidated: true,
            isValidatingTicker: false,
          }
          : null,
      );
      toast.success(`조회 성공: ${payload.name}`);
    } catch (error) {
      setAddingRow((previous) =>
        previous
          ? {
            ...previous,
            name: previous.name || "",
            isValidatingTicker: false,
          }
          : null,
      );
      toast.error(error instanceof Error ? error.message : "검증 실패");
    }
  }, [addingRow?.isValidatingTicker, addingRow?.ticker, rows, summary.account_id, toast]);

  const processAddingRow = useCallback(async () => {
    if (!addingRow?.isValidated) {
      throw new Error("먼저 종목 확인을 완료해 주세요.");
    }

    const rawQuantity = qtyRef.current?.value ?? "";
    const rawPrice = priceRef.current?.value ?? "";

    const quantity = parseInt(parseRawPrice(rawQuantity), 10);
    const averageBuyPrice = safeParseFloat(rawPrice);

    if (Number.isNaN(quantity) || quantity < 0 || Number.isNaN(averageBuyPrice) || averageBuyPrice < 0) {
      throw new Error("수량과 매입 단가를 확인해 주세요.");
    }

    const response = await fetch("/api/assets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        account_id: summary.account_id,
        ticker: addingRow.ticker,
        quantity,
        average_buy_price: averageBuyPrice,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "종목 추가에 실패했습니다.");
    }
  }, [addingRow, summary.account_id]);

  const processRowUpdate = useCallback(async (row: GridRow) => {
    const quantity = parseEditableQuantity(row.quantity);
    const averageBuyPrice = safeParseFloat(row.average_buy_price);

    if (Number.isNaN(quantity) || quantity < 0 || Number.isNaN(averageBuyPrice) || averageBuyPrice < 0) {
      throw new Error("입력값이 올바르지 않습니다.");
    }

    const response = await fetch("/api/assets", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        account_id: summary.account_id,
        ticker: row.ticker,
        quantity,
        average_buy_price: averageBuyPrice,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "종목 수정에 실패했습니다.");
    }
  }, [summary.account_id]);

  const processCashUpdate = useCallback(async (cashBalanceKrw: number, cashTargetRatio: number) => {
    const isAud = String(summary.currency || "KRW").toUpperCase() === "AUD";
    const currentCashKrw = Number(summary.cash_balance_krw ?? 0);
    const currentCashNative = Number(summary.cash_balance_native ?? 0);
    const nextCashNative =
      isAud && currentCashKrw > 0 && currentCashNative > 0
        ? (cashBalanceKrw / currentCashKrw) * currentCashNative
        : (isAud ? currentCashNative : cashBalanceKrw);
    const response = await fetch("/api/assets", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        account_id: summary.account_id,
        total_principal: summary.total_principal,
        cash_balance_krw: cashBalanceKrw,
        cash_balance_native: nextCashNative,
        cash_currency: summary.cash_currency,
        cash_target_ratio: cashTargetRatio,
        intl_shares_value: summary.account_id === "aus_account" ? summary.intl_shares_value : null,
        intl_shares_change: summary.account_id === "aus_account" ? summary.intl_shares_change : null,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "현금 저장에 실패했습니다.");
    }
  }, [summary]);

  // Intl Value/Change 만 저장한다. 현금 키를 함께 보내면 백엔드가 통화별 `cash` 맵을
  // 레거시 필드로 재합성해 잔액이 0 으로 덮이므로, 현금 관련 필드는 보내지 않는다.
  const processIntlUpdate = useCallback(async (intlSharesValue: number, intlSharesChange: number) => {
    const response = await fetch("/api/assets", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        account_id: summary.account_id,
        total_principal: summary.total_principal,
        cash_currency: summary.cash_currency,
        cash_target_ratio: summary.cash_target_ratio,
        intl_shares_value: intlSharesValue,
        intl_shares_change: intlSharesChange,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "호주 계좌 저장에 실패했습니다.");
    }
  }, [summary]);

  // 통화별 native 현금 맵 저장. 원화 합계(cash_balance)는 백엔드가 환율로 계산한다.
  const processCashMapUpdate = useCallback(async (cashMap: Record<string, number>): Promise<SavedCashAccount | null> => {
    const response = await fetch("/api/assets", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        account_id: summary.account_id,
        total_principal: summary.total_principal,
        cash: cashMap,
        cash_currency: summary.cash_currency,
        cash_target_ratio: summary.cash_target_ratio,
        intl_shares_value: summary.account_id === "aus_account" ? summary.intl_shares_value : null,
        intl_shares_change: summary.account_id === "aus_account" ? summary.intl_shares_change : null,
      }),
    });
    const payload = (await response.json()) as { accounts?: SavedCashAccount[]; error?: string };
    if (!response.ok) {
      throw new Error(payload.error || "현금 저장에 실패했습니다.");
    }
    // 통화별 native 를 원화로 합친 값은 환율이 필요해 화면이 못 만든다 — 서버 계산값을 쓴다.
    return payload.accounts?.find((item) => item.account_id === summary.account_id) ?? null;
  }, [summary]);

  const clearDirtyRowState = useCallback((rowId: string) => {
    setDirtyRowIds((previous) => {
      const next = previous.filter((id) => id !== rowId);
      dirtyRowIdsRef.current = next;
      return next;
    });
    setDirtyCellKeys((previous) => previous.filter((key) => !key.startsWith(`${rowId}::`)));
  }, []);

  const processReorderUpdate = useCallback(async (orderedRows: HoldingsRow[]) => {
    const orderedTickers = orderedRows
      .map((row) => String(row.ticker || "").trim().toUpperCase())
      .filter((ticker) => ticker && ticker !== CASH_ROW_TICKER);
    await reorderHoldings(summary.account_id, orderedTickers);
  }, [summary.account_id]);

  const silentlySaveRow = useCallback(async (rowId: string) => {
    if (childSavingRowIdsRef.current.has(rowId)) {
      childQueuedRowIdsRef.current.add(rowId);
      return;
    }

    const sourceRow = rowsRef.current.find((row) => buildGridRowId(row) === rowId);
    if (!sourceRow) {
      clearDirtyRowState(rowId);
      return;
    }

    childSavingRowIdsRef.current.add(rowId);
    try {
      const previousSnapshot = lastSavedSnapshotsRef.current.get(rowId) ?? buildHoldingEditableSnapshot(sourceRow);
      const nextSnapshot = buildHoldingEditableSnapshot(sourceRow);
      await processRowUpdate({
        ...sourceRow,
        id: rowId,
        quantity: typeof sourceRow.quantity === "number" ? sourceRow.quantity : parseInt(String(sourceRow.quantity), 10) || 0,
        average_buy_price: safeParseFloat(sourceRow.average_buy_price),
      });
      lastSavedSnapshotsRef.current.set(rowId, nextSnapshot);
      const message = buildAutoSaveToastMessage(sourceRow, previousSnapshot, nextSnapshot);
      if (message) {
        toast.success(message);
      }
      clearDirtyRowState(rowId);
    } catch (error) {
      await onReload();
      toast.error(error instanceof Error ? error.message : "변경사항 저장에 실패했습니다.");
    } finally {
      childSavingRowIdsRef.current.delete(rowId);
      if (childQueuedRowIdsRef.current.has(rowId)) {
        childQueuedRowIdsRef.current.delete(rowId);
        const nextTimer = setTimeout(() => {
          childSaveTimersRef.current.delete(rowId);
          void silentlySaveRow(rowId);
        }, 400);
        childSaveTimersRef.current.set(rowId, nextTimer);
      }
    }
  }, [clearDirtyRowState, onReload, processRowUpdate, toast]);

  const scheduleSilentRowSave = useCallback((rowId: string) => {
    const currentTimer = childSaveTimersRef.current.get(rowId);
    if (currentTimer) {
      clearTimeout(currentTimer);
    }
    const nextTimer = setTimeout(() => {
      childSaveTimersRef.current.delete(rowId);
      void silentlySaveRow(rowId);
    }, 700);
    childSaveTimersRef.current.set(rowId, nextTimer);
  }, [silentlySaveRow]);

  const persistRowOrder = useCallback(async (orderedRows: HoldingsRow[]) => {
    if (reorderSavingRef.current) {
      reorderQueuedRef.current = true;
      return;
    }

    reorderSavingRef.current = true;
    try {
      await processReorderUpdate(orderedRows);
      setIsReorderDirty(false);
      isReorderDirtyRef.current = false;
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "순서 저장에 실패했습니다.");
    } finally {
      reorderSavingRef.current = false;
      if (reorderQueuedRef.current) {
        reorderQueuedRef.current = false;
        const nextRows = rowsRef.current;
        const nextTimer = setTimeout(() => {
          reorderSaveTimerRef.current = null;
          void persistRowOrder(nextRows);
        }, 400);
        reorderSaveTimerRef.current = nextTimer;
      }
    }
  }, [processReorderUpdate, toast]);

  const scheduleSilentReorderSave = useCallback((orderedRows: HoldingsRow[]) => {
    if (reorderSaveTimerRef.current) {
      clearTimeout(reorderSaveTimerRef.current);
    }
    const nextTimer = setTimeout(() => {
      reorderSaveTimerRef.current = null;
      void persistRowOrder(orderedRows);
    }, 700);
    reorderSaveTimerRef.current = nextTimer;
  }, [persistRowOrder]);

  const flushPendingSaves = useCallback(() => {
    childSaveTimersRef.current.forEach((timer) => clearTimeout(timer));
    childSaveTimersRef.current.clear();
    if (reorderSaveTimerRef.current) {
      clearTimeout(reorderSaveTimerRef.current);
      reorderSaveTimerRef.current = null;
    }
  }, []);

  const flushPendingSavesOnUnmount = useCallback(() => {
    childSaveTimersRef.current.forEach((timer) => clearTimeout(timer));
    childSaveTimersRef.current.clear();
    if (reorderSaveTimerRef.current) {
      clearTimeout(reorderSaveTimerRef.current);
      reorderSaveTimerRef.current = null;
    }

    const cashRowId = `${summary.account_id}-${CASH_ROW_TICKER}`;
    if (dirtyRowIdsRef.current.includes(cashRowId)) {
      void processCashUpdate(
        cashDraftRef.current.cashBalanceKrw,
        cashDraftRef.current.cashTargetRatio,
      ).catch(() => undefined);
    }

    const dirtyRows = rowsRef.current
      .map((row) => ({ ...row, id: buildGridRowId(row) }))
      .filter((row) => dirtyRowIdsRef.current.includes(row.id));

    for (const row of dirtyRows) {
      void processRowUpdate(row).catch(() => undefined);
    }

    if (isReorderDirtyRef.current) {
      void processReorderUpdate(rowsRef.current).catch(() => undefined);
    }

  }, [processCashUpdate, processReorderUpdate, processRowUpdate, summary.account_id]);

  useEffect(() => {
    return () => {
      flushPendingSavesOnUnmount();
    };
  }, [flushPendingSavesOnUnmount]);

  const handleSaveChanges = useCallback(async () => {
    if (processingId === "__adding__" || processingId === "__deleting__") {
      return;
    }

    gridApiRef.current?.stopEditing();
    flushPendingSaves();

    try {
      const cashRowId = `${summary.account_id}-${CASH_ROW_TICKER}`;
      if (dirtyRowIds.includes(cashRowId)) {
        await processCashUpdate(
          cashDraftRef.current.cashBalanceKrw,
          cashDraftRef.current.cashTargetRatio,
        );
        clearDirtyRowState(cashRowId);
      }

      const dirtyRows = rowsRef.current
        .map((row) => ({
          ...row,
          id: buildGridRowId(row),
          quantity: typeof row.quantity === "number" ? row.quantity : parseInt(String(row.quantity), 10) || 0,
          average_buy_price: safeParseFloat(row.average_buy_price),
          target_ratio: row.target_ratio ?? 0,
        }))
        .filter((row) => dirtyRowIds.includes(row.id));

      for (const row of dirtyRows) {
        await processRowUpdate(row);
        lastSavedSnapshotsRef.current.set(row.id, buildHoldingEditableSnapshot(row));
        clearDirtyRowState(row.id);
      }

      if (isReorderDirty) {
        await persistRowOrder(rowsRef.current);
      }

      if (addingRow) {
        setProcessingId("__adding__");
        await processAddingRow();
        await onReload();
        toast.success("종목 추가 완료");
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "변경사항 저장에 실패했습니다.");
    } finally {
      setProcessingId(null);
    }
  }, [
    addingRow,
    clearDirtyRowState,
    dirtyRowIds,
    flushPendingSaves,
    isReorderDirty,
    onReload,
    persistRowOrder,
    processAddingRow,
    processCashUpdate,
    processRowUpdate,
    processingId,
    summary,
    toast,
  ]);

  const handleDeleteSelected = useCallback(() => {
    if (!selectedDeletableRows.length) {
      return;
    }
    setDeleteConfirmOpen(true);
  }, [selectedDeletableRows.length]);

  const handleCloseDeleteConfirm = useCallback(() => {
    if (processingId === "__deleting__") {
      return;
    }
    setDeleteConfirmOpen(false);
  }, [processingId]);

  const handleConfirmDeleteSelected = useCallback(async () => {
    if (!selectedDeletableRows.length) {
      setDeleteConfirmOpen(false);
      return;
    }

    setProcessingId("__deleting__");
    try {
      for (const row of selectedDeletableRows) {
        const params = new URLSearchParams({
          account: summary.account_id,
          ticker: row.ticker,
        });
        const response = await fetch(`/api/assets?${params.toString()}`, { method: "DELETE" });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "삭제 실패");
        }
      }
      setDeleteConfirmOpen(false);
      setSelectedRowIds([]);
      await onReload();
      toast.success("삭제 완료");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "삭제 실패");
    } finally {
      setProcessingId(null);
    }
  }, [onReload, selectedDeletableRows, summary.account_id, toast]);

  const handleCellValueChanged = useCallback((row: GridRow | undefined, field: string | undefined) => {
    if (!row) {
      return;
    }
    if (isCashGridRow(row)) {
      const nextCashBalance = Math.max(0, Number(row.valuation_krw ?? 0));
      const nextCashTargetRatio = Number(row.target_ratio ?? 0);
      cashDraftRef.current = {
        cashBalanceKrw: nextCashBalance,
        cashTargetRatio: nextCashTargetRatio,
      };
      setDirtyRowIds((previous) => {
        const next = previous.includes(row.id) ? previous : [...previous, row.id];
        dirtyRowIdsRef.current = next;
        return next;
      });
      if (field) {
        const dirtyCellKey = buildDirtyCellKey(row.id, field);
        setDirtyCellKeys((previous) => (previous.includes(dirtyCellKey) ? previous : [...previous, dirtyCellKey]));
      }
      onCashSync(summary.account_id, nextCashBalance, nextCashTargetRatio);
      const timerKey = row.id;
      const currentTimer = childSaveTimersRef.current.get(timerKey);
      if (currentTimer) {
        clearTimeout(currentTimer);
      }
      const nextTimer = setTimeout(async () => {
        childSaveTimersRef.current.delete(timerKey);
        try {
          await processCashUpdate(nextCashBalance, nextCashTargetRatio);
          clearDirtyRowState(timerKey);
          toast.success("현금 저장 완료");
        } catch (error) {
          await onReload();
          toast.error(error instanceof Error ? error.message : "현금 저장에 실패했습니다.");
        }
      }, 700);
      childSaveTimersRef.current.set(timerKey, nextTimer);
      return;
    }
    if (!isEditableHoldingRow(row)) {
      return;
    }

    const nextRows = rowsRef.current.map((currentRow) => {
      if (buildGridRowId(currentRow) !== row.id) {
        return currentRow;
      }
      return {
        ...currentRow,
        quantity: parseEditableQuantity(row.quantity),
        average_buy_price: safeParseFloat(row.average_buy_price),
      };
    });
    rowsRef.current = nextRows;
    setRows(nextRows);
    onRowsSync(summary.account_id, buildSyncedHoldingRows(nextRows, summary));
    setDirtyRowIds((previous) => {
      const next = previous.includes(row.id) ? previous : [...previous, row.id];
      dirtyRowIdsRef.current = next;
      return next;
    });
    if (field) {
      const dirtyCellKey = buildDirtyCellKey(row.id, field);
      setDirtyCellKeys((previous) => (previous.includes(dirtyCellKey) ? previous : [...previous, dirtyCellKey]));
    }
    scheduleSilentRowSave(row.id);
  }, [clearDirtyRowState, isCashGridRow, isEditableHoldingRow, onCashSync, onReload, onRowsSync, processCashUpdate, scheduleSilentRowSave, summary, toast]);

  const columns = useMemo<ColDef<GridRow>[]>(() => [
    {
      colId: "drag",
      headerName: "",
      width: 42,
      maxWidth: 42,
      pinned: "left",
      sortable: false,
      resizable: false,
      suppressMovable: true,
      rowDrag: (params) =>
        Boolean(params.data && params.data.id !== "__adding__" && params.data.ticker !== CASH_ROW_TICKER),
      cellClass: "assetsDragCell",
      valueGetter: () => "",
    },
    {
      field: "bucket",
      headerName: "버킷",
      width: 96,
      cellClass: (params) => getBucketCellClass(params.data?.bucket_id ?? 0),
    },
    {
      field: "ticker",
      headerName: "티커",
      width: 98,
      cellRenderer: (params: { data?: GridRow; value?: string }) => {
        const row = params.data;
        if (!row) {
          return null;
        }
        if (row.id === "__adding__") {
          return (
            <StableInlineInput
              className="form-control form-control-sm assetsInlineInput assetsInlineInputTicker"
              initialValue={addingRow?.ticker ?? ""}
              disabled={addingRow?.isValidatingTicker || addingRow?.isValidated}
              submitOnBlur={false}
              onChange={(value) => {
                addingTickerDraftRef.current = value; // ref 만 갱신(리렌더 없음) — 포커스 유지
              }}
              onSave={handleValidateTicker}
            />
          );
        }
        if (row.ticker === CASH_ROW_TICKER) return <span>-</span>;
        // IS 고정자산은 가격 프록시(VGS) 티커로 표시하고 상세도 VGS 로 연결한다(자산 헬퍼와 동일).
        if (row.ticker === "IS") {
          return <TickerDetailLink ticker="ASX:VGS" displayTicker="ASX:VGS" className="assetsTickerLink" />;
        }
        return (
          <TickerDetailLink
            ticker={row.ticker}
            displayTicker={String(params.value ?? row.ticker)}
            className="assetsTickerLink"
          />
        );
      },
    },
    {
      field: "name",
      headerName: "종목명",
      minWidth: 248,
      flex: 1.35,
      cellRenderer: (params: { data?: GridRow; value?: string | null }) => {
        if (params.data?.id === "__adding__") {
          return (
            <div className="assetsNameLookup">
              <span className="assetsNameLookupStatus">
                {addingRow?.isValidated
                  ? String(addingRow.name || "-")
                  : String(addingRow?.name || "종목코드를 입력한 뒤 확인하세요.")}
              </span>
              <button
                className={
                  addingRow?.isValidated
                    ? "btn btn-sm btn-link p-0 assetsInlineLinkButton"
                    : "btn btn-outline-primary btn-sm assetsInlineButton d-inline-flex align-items-center gap-1"
                }
                disabled={addingRow?.isValidatingTicker}
                onMouseDown={(event) => {
                  event.stopPropagation();
                }}
                onClick={(event) => {
                  event.stopPropagation();
                  if (addingRow?.isValidated) {
                    setAddingRow((previous) =>
                      previous
                        ? {
                          ...previous,
                          ticker: "",
                          isValidated: false,
                          name: "",
                        }
                        : null,
                    );
                    return;
                  }
                  void handleValidateTicker();
                }}
              >
                {!addingRow?.isValidated && addingRow?.isValidatingTicker ? (
                  <IconLoader2 size={14} style={{ animation: "spin 1s linear infinite" }} />
                ) : null}
                {addingRow?.isValidated ? "변경" : addingRow?.isValidatingTicker ? "확인중" : "확인"}
              </button>
            </div>
          );
        }

        // 종목명 표기 규칙은 전 화면 공통(`@/lib/name-highlight`).
        return renderStockNameCell(params.value, {
          badge: alertBadges[normalizeBadgeTicker(params.data?.ticker ?? "")] ?? "",
        });
      },
    },
    {
      field: "daily_change_pct",
      headerName: "일간(%)",
      width: 92,
      type: "rightAligned",
      cellRenderer: (params: { value?: number | null }) => (
        <span className={getSignedClass(params.value ?? 0)}>
          {params.value === null || params.value === undefined
            ? "-"
            : `${params.value.toFixed(2)}%`}
        </span>
      ),
    },
    {
      field: "current_price",
      headerName: "현재가",
      width: 104,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: string }) => (
        <span>{formatPrice(safeParseFloat(params.value), params.data?.currency || "KRW")}</span>
      ),
    },
    {
      field: "weight_pct",
      headerName: "비중",
      width: 80,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow }) => {
        if (!params.data) {
          return "-";
        }

        const weightPct = getPreviewWeightPct(params.data, rowsRef.current, summaryRef.current);
        return (
          <span style={{ color: "#000000", fontWeight: 700 }}>
            {weightPct.toFixed(2)}%
          </span>
        );
      },
    },
    {
      colId: "target_weight_pct",
      headerName: "목표비중",
      width: 88,
      type: "rightAligned",
      sortable: false,
      cellStyle: { backgroundColor: "#f1f3f5" },
      cellRenderer: (params: { data?: GridRow }) => {
        const row = params.data;
        if (!row || row.id === "__adding__") return <span style={{ color: "var(--text-muted)" }}>-</span>;
        // 현금 행: 자산 헬퍼에서 저장한 현금 목표 비중만 표시한다(미저장 = '-', 파생·기본값 없음).
        if (row.ticker === CASH_ROW_TICKER) {
          const saved = summary.helper_cash_weight_pct;
          if (saved == null || !Number.isFinite(Number(saved))) {
            return <span style={{ color: "var(--text-muted)" }}>-</span>;
          }
          return <span style={{ color: "#000000", fontWeight: 700 }}>{Number(saved).toFixed(2)}%</span>;
        }
        // IS 행: 자동값(현재 비중)이 곧 목표 비중. 나머지는 저장된 target_ratio 그대로.
        const w = row.ticker === "IS" ? row.weight_pct : row.target_ratio;
        return <span style={{ color: w == null ? "var(--text-muted)" : "#000000", fontWeight: 700 }}>{w == null ? "-" : `${Number(w).toFixed(2)}%`}</span>;
      },
    },
    {
      colId: "target_quantity",
      headerName: "목표수량",
      width: 88,
      type: "rightAligned",
      sortable: false,
      cellStyle: { backgroundColor: "#f1f3f5" },
      headerTooltip: "목표비중 × 총자산 ÷ 현재가 (단주거래 전제, 정수 반올림)",
      cellRenderer: (params: { data?: GridRow }) => {
        const row = params.data;
        if (!row || row.id === "__adding__" || row.ticker === CASH_ROW_TICKER || row.ticker === "IS") {
          return <span style={{ color: "var(--text-muted)" }}>-</span>;
        }
        const quantity = row.target_quantity;
        if (quantity == null || !Number.isFinite(Number(quantity))) {
          return <span style={{ color: "var(--text-muted)" }}>-</span>;
        }
        return <span style={{ fontWeight: 700 }}>{Math.round(Number(quantity)).toLocaleString()}</span>;
      },
    },
    {
      field: "quantity",
      headerName: "수량",
      width: 64,
      type: "rightAligned",
      editable: (params) => isEditableHoldingRow(params.data) && processingId !== params.data?.id,
      cellClass: (params) => {
        if (!isEditableHoldingRow(params.data)) {
          return undefined;
        }
        return isDirtyEditableCell(params.data?.id, "quantity")
          ? "assetsEditableCell assetsDirtyCell"
          : "assetsEditableCell";
      },
      valueParser: (params) => {
        const parsed = parseInt(parseRawPrice(params.newValue), 10);
        if (Number.isNaN(parsed) || parsed < 0) {
          return params.oldValue;
        }
        return parsed;
      },
      cellRenderer: (params: { data?: GridRow; value?: number }) => {
        const row = params.data;
        if (!row) {
          return null;
        }
        if (row.ticker === CASH_ROW_TICKER) {
          return <span>-</span>;
        }
        if (row.id === "__adding__") {
          return (
            <input
              type="number"
              step="1"
              ref={qtyRef}
              className="form-control form-control-sm assetsInlineInput"
              defaultValue="0"
              disabled={!addingRow?.isValidated}
            />
          );
        }
        return <span>{new Intl.NumberFormat("ko-KR").format(params.value ?? 0)}</span>;
      },
    },
    {
      field: "average_buy_price",
      headerName: "매입 단가",
      width: 112,
      type: "rightAligned",
      editable: (params) => isEditableHoldingRow(params.data) && processingId !== params.data?.id,
      cellClass: (params) => {
        if (!isEditableHoldingRow(params.data)) {
          return undefined;
        }
        return isDirtyEditableCell(params.data?.id, "average_buy_price")
          ? "assetsEditableCell assetsDirtyCell"
          : "assetsEditableCell";
      },
      valueParser: (params) => {
        const parsed = parseFloat(parseRawPrice(params.newValue));
        if (Number.isNaN(parsed) || parsed < 0) {
          return params.oldValue;
        }
        return parsed;
      },
      cellRenderer: (params: { data?: GridRow; value?: string | number }) => {
        const row = params.data;
        if (!row) {
          return null;
        }
        if (row.ticker === CASH_ROW_TICKER) {
          return <span>-</span>;
        }
        if (row.id === "__adding__") {
          return (
            <input
              type="number"
              step="any"
              ref={priceRef}
              className="form-control form-control-sm assetsInlineInput"
              defaultValue="0"
              disabled={!addingRow?.isValidated}
            />
          );
        }
        return <span>{formatPrice(safeParseFloat(params.value), row.currency || "KRW")}</span>;
      },
    },
    {
      field: "return_pct",
      headerName: "수익률",
      width: 88,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: number }) => (
        params.data?.ticker === CASH_ROW_TICKER ? <span>-</span> :
          <span className={getSignedClass(params.value ?? 0)}>
            {(params.value ?? 0).toFixed(2)}%
          </span>
      ),
    },
    {
      field: "pnl_krw",
      headerName: "평가손익",
      width: 124,
      type: "rightAligned",
      cellRenderer: (params: { data?: GridRow; value?: number }) => (
        params.data?.ticker === CASH_ROW_TICKER ? <span>-</span> :
          <span className={getSignedClass(params.value ?? 0)}>{formatHiddenAmount(showAmounts, formatKrw(params.value ?? 0))}</span>
      ),
    },
    {
      field: "valuation_krw",
      headerName: "평가 금액",
      width: 124,
      type: "rightAligned",
      // 수량/평균매입가 변경 시 평가금액 셀이 즉시 재렌더되도록 valueGetter 로 라이브 계산.
      valueGetter: (params) => (params.data ? getPreviewValuationKrw(params.data) : null),
      // 현금 입력은 상단 통화별 박스로 이전됨 — 그리드 현금 셀 편집은 막고 편집 스타일도 제거한다.
      editable: false,
      valueParser: (params) => {
        const parsed = parseFloat(parseRawPrice(params.newValue));
        if (Number.isNaN(parsed) || parsed < 0) {
          return params.oldValue;
        }
        return parsed;
      },
      cellRenderer: (params: { data?: GridRow }) => (
        <span className="appGridNumericValue">{params.data ? formatHiddenAmount(showAmounts, formatKrw(getPreviewValuationKrw(params.data))) : "-"}</span>
      ),
    },
  ], [addingRow, alertBadges, handleValidateTicker, isAusAccount, isCashGridRow, isDirtyEditableCell, isEditableHoldingRow, processingId, showAmounts]);

  return (
    <div className="assetsDetailPanel">
      <div className="appActionHeader">
        <div className="appActionHeaderInner">
          {isAusAccount && (
            <div className="d-flex align-items-center gap-2">
              <label className="mb-0 text-muted small fw-bold">Intl Value</label>
              <input
                type="text"
                className={`form-control form-control-sm ${intlDirtyFields.includes("intl_shares_value") ? "assetsDirtyInput" : ""}`}
                style={{ width: 120, textAlign: "right" }}
                defaultValue={Number(summary.intl_shares_value ?? 0).toLocaleString("en-AU", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                onChange={(event) => {
                  const parsed = parseFloat(event.target.value.replace(/,/g, ""));
                  if (!Number.isNaN(parsed)) {
                    intlDraftRef.current.intlSharesValue = parsed;
                    setIntlDirtyFields((prev) => (prev.includes("intl_shares_value") ? prev : [...prev, "intl_shares_value"]));
                  }
                }}
              />
              <label className="mb-0 text-muted small fw-bold">Intl Change</label>
              <input
                type="text"
                className={`form-control form-control-sm ${intlDirtyFields.includes("intl_shares_change") ? "assetsDirtyInput" : ""}`}
                style={{ width: 120, textAlign: "right" }}
                defaultValue={Number(summary.intl_shares_change ?? 0).toLocaleString("en-AU", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                onChange={(event) => {
                  const parsed = parseFloat(event.target.value.replace(/,/g, ""));
                  if (!Number.isNaN(parsed)) {
                    intlDraftRef.current.intlSharesChange = parsed;
                    setIntlDirtyFields((prev) => (prev.includes("intl_shares_change") ? prev : [...prev, "intl_shares_change"]));
                  }
                }}
              />
              <button
                type="button"
                className="btn btn-success btn-sm px-2"
                disabled={intlDirtyFields.length === 0}
                onMouseDown={stopActionButtonMouseDown}
                onClick={async () => {
                  try {
                    await processIntlUpdate(
                      intlDraftRef.current.intlSharesValue,
                      intlDraftRef.current.intlSharesChange,
                    );
                    setIntlDirtyFields([]);
                    await onReload();
                    toast.success("호주 계좌 저장 완료");
                  } catch (error) {
                    await onReload();
                    toast.error(error instanceof Error ? error.message : "호주 계좌 저장에 실패했습니다.");
                  }
                }}
              >
                저장
              </button>
            </div>
          )}
          <div className="d-flex align-items-center gap-2 flex-wrap">
            {cashCurrencyList.map((code) => (
              <div key={code} className="d-flex align-items-center gap-1">
                <label className="mb-0 text-muted small fw-bold">{code} 현금</label>
                <input
                  type="text"
                  className={`form-control form-control-sm ${cashMapDirty ? "assetsDirtyInput" : ""}`}
                  style={{ width: 120, textAlign: "right" }}
                  defaultValue={Number(summary.cash?.[code] ?? 0).toLocaleString("en-US", code === "KRW" ? { maximumFractionDigits: 0 } : { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  onChange={(event) => {
                    const parsed = parseFloat(event.target.value.replace(/,/g, ""));
                    if (!Number.isNaN(parsed)) {
                      cashMapDraftRef.current[code] = parsed;
                      setCashMapDirty(true);
                    }
                  }}
                />
              </div>
            ))}
            <button
              type="button"
              className="btn btn-success btn-sm px-2"
              disabled={!cashMapDirty || cashMapSaving}
              onMouseDown={stopActionButtonMouseDown}
              onClick={async () => {
                setCashMapSaving(true);
                try {
                  const saved = await processCashMapUpdate({ ...(summary.cash ?? {}), ...cashMapDraftRef.current });
                  setCashMapDirty(false);
                  if (saved) {
                    // 저장 응답으로 화면을 바로 맞춘다. 전체 리로드(≈2초)를 기다리지 않는다.
                    onCashSync(summary.account_id, saved.cash_balance_krw, saved.cash_target_ratio, saved);
                    toast.success("현금 저장 완료");
                    // 스냅샷 기준으로 계산되는 금일·주간 손익만 뒤에서 따라오게 한다(화면은 덮지 않는다).
                    void onReload({ silent: true });
                  } else {
                    // 응답에 저장 결과가 없으면 화면을 임의로 맞추지 않고 정상 리로드로 확인한다.
                    await onReload();
                    toast.success("현금 저장 완료");
                  }
                } catch (error) {
                  await onReload();
                  toast.error(error instanceof Error ? error.message : "현금 저장에 실패했습니다.");
                } finally {
                  setCashMapSaving(false);
                }
              }}
            >
              {cashMapSaving ? "저장 중…" : "현금 저장"}
            </button>
            {summary.updated_at ? (
              <span className="text-muted small">
                최종 변경 {formatKstDateTime(summary.updated_at)} · {formatUpdatedBy(summary.updated_by)}
              </span>
            ) : null}
          </div>
          <div className="d-flex align-items-center gap-2 ms-auto">
            <GridToolbarButton
              variant="add"
              onMouseDown={stopActionButtonMouseDown}
              onClick={() => {
                addingTickerDraftRef.current = "";
                setAddingRow({
                  ticker: "",
                  quantity: "",
                  average_buy_price: "",
                  isValidated: false,
                });
              }}
              disabled={hasPendingAdd}
            />
            <GridToolbarButton
              variant="save"
              onMouseDown={stopActionButtonMouseDown}
              onClick={() => void handleSaveChanges()}
              disabled={!hasPendingSave || processingId === "__adding__" || processingId === "__deleting__"}
            />
            <GridToolbarButton
              variant="delete"
              onMouseDown={stopActionButtonMouseDown}
              onClick={(event) => {
                stopActionButtonClick(event);
                handleDeleteSelected();
              }}
              disabled={!hasSelectedRows || processingId === "__adding__" || processingId === "__deleting__"}
            />
          </div>
        </div>
      </div>
      <AppModal
        open={deleteConfirmOpen}
        title="종목 삭제 확인"
        subtitle="선택 종목은 즉시 영구 삭제됩니다."
        onClose={handleCloseDeleteConfirm}
        footer={(
          <>
            <button
              type="button"
              className="btn btn-outline-secondary"
              onClick={handleCloseDeleteConfirm}
              disabled={processingId === "__deleting__"}
            >
              취소
            </button>
            <button
              type="button"
              className="btn btn-danger"
              onClick={() => void handleConfirmDeleteSelected()}
              disabled={processingId === "__deleting__"}
            >
              삭제
            </button>
          </>
        )}
      >
        <div className="d-flex flex-column gap-2">
          <div className="fw-semibold">
            {selectedDeletableRows.length === 1
              ? `${selectedDeletableRows[0].name}(${selectedDeletableRows[0].ticker}) 종목을 삭제합니다.`
              : `${selectedDeletableRows.length}개 종목을 삭제합니다.`}
          </div>
          <div className="text-secondary small">삭제된 종목은 복구되지 않으며 즉시 제거됩니다.</div>
        </div>
      </AppModal>
      <div className="assetsDetailGridWrap">
        <AppAgGrid
          rowData={gridRows}
          columnDefs={columns}
          loading={processingId === "__adding__" || processingId === "__deleting__"}
          minHeight="100%"
          className="assetsAgGrid assetsChildAgGrid"
          theme={assetsGridTheme}
          getRowClass={(params: RowClassParams<GridRow>) => {
            const classes: string[] = [];
            if (params.data?.ticker === "IS") {
              return classes.join(" ");
            }
            if (Number(params.data?.quantity ?? 0) > 0) {
              classes.push("appHeldRow");
            }
            if (maBrokenTickers.has(normalizeBadgeTicker(params.data?.ticker ?? ""))) {
              classes.push("appTrendBrokenRow");
            }
            return classes.join(" ");
          }}
          gridOptions={{
            suppressMovableColumns: true,
            ensureDomOrder: true,
            stopEditingWhenCellsLoseFocus: true,
            rowDragManaged: true,
            animateRows: true,
            rowSelection: {
              mode: "multiRow",
              checkboxes: (params) =>
                Boolean(params.data && params.data.id !== "__adding__" && params.data.ticker !== "IS" && params.data.ticker !== CASH_ROW_TICKER),
              headerCheckbox: true,
              hideDisabledCheckboxes: true,
              enableClickSelection: false,
              isRowSelectable: (params) =>
                Boolean(params.data && params.data.id !== "__adding__" && params.data.ticker !== "IS" && params.data.ticker !== CASH_ROW_TICKER),
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
              cellClass: "assetsSelectCell",
            },
            onSelectionChanged: (params) => {
              setSelectedRowIds(
                params.api
                  .getSelectedRows()
                  .map((row) => row.id)
                  .filter((rowId): rowId is string => Boolean(rowId)),
              );
            },
            onCellEditingStarted: (params) => {
              if (params.data && isEditableHoldingRow(params.data)) {
                setEditingRowId(params.data.id);
              }
            },
            onCellEditingStopped: () => {
              setEditingRowId(null);
            },
            onCellValueChanged: (params) => {
              if (params.newValue === params.oldValue) {
                return;
              }
              handleCellValueChanged(params.data, params.colDef.field);
            },
            onRowDragEnd: (params) => {
              const orderedTickers: string[] = [];
              params.api.forEachNode((node) => {
                const ticker = String(node.data?.ticker || "").trim().toUpperCase();
                if (!ticker || ticker === "IS" || ticker === CASH_ROW_TICKER) {
                  return;
                }
                orderedTickers.push(ticker);
              });
              if (!orderedTickers.length) {
                return;
              }
              // setState updater 안에서 부모 setState (onRowsSync) 를 트리거하면
              // React 가 "render 중 다른 컴포넌트 업데이트" 로 경고한다. updater 밖에서
              // 순차 호출하도록 분리한다.
              const nextRows = reorderRowsByTickers(rowsRef.current, orderedTickers);
              rowsRef.current = nextRows;
              setRows(nextRows);
              onRowsSync(summary.account_id, buildSyncedHoldingRows(nextRows, summary));
              setIsReorderDirty(true);
              isReorderDirtyRef.current = true;
              scheduleSilentReorderSave(nextRows);
            },
            onGridReady: (params) => {
              gridApiRef.current = params.api;
            },
            getRowId: (params) => String(params.data.id),
            rowClassRules: {
              assetsAddingRow: (params) => params.data?.id === "__adding__",
              assetsEditingRow: (params) => Boolean(params.data?.id && params.data.id === editingRowId),
            },
          }}
        />
      </div>

    </div>
  );
}

