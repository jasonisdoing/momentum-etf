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
import { useToast } from "../components/ToastProvider";
import { createAppGridTheme } from "../components/app-grid-theme";
import { reorderHoldings } from "@/lib/holdings-store";
import { fetchAlertBadges, normalizeBadgeTicker, type AlertBadges } from "@/lib/alert-badges";

import { AccountHoldingsDetailPanel } from "./AccountHoldingsDetailPanel";
import {
  AccountSummary,
  AssetsHeaderSummary,
  CASH_ROW_TICKER,
  HoldingsResponse,
  HoldingsRow,
  ParentGridRow,
  assetsGridTheme,
  buildDirtyCellKey,
  formatHiddenAmount,
  formatKrw,
  formatNumber,
  getSignedClass,
  isDetailRow,
  isTotalRow,
  parseRawPrice,
} from "./assets-helpers";

export function AssetsManager({ onHeaderSummaryChange }: { onHeaderSummaryChange?: (summary: AssetsHeaderSummary) => void }) {
  const toast = useToast();
  const [allRows, setAllRows] = useState<HoldingsRow[]>([]);
  const [summaries, setSummaries] = useState<AccountSummary[]>([]);
  const [dashTotals, setDashTotals] = useState<{
    daily_profit?: number;
    daily_return_pct?: number;
    weekly_profit?: number;
    weekly_return_pct?: number;
  } | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [showAmounts, setShowAmounts] = useState(true);
  const [loading, setLoading] = useState(true);
  const [parentDirtyCellKeys, setParentDirtyCellKeys] = useState<string[]>([]);
  const [editingParentId, setEditingParentId] = useState<string | null>(null);
  const summariesRef = useRef<AccountSummary[]>([]);
  const parentSaveTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const parentSavingAccountIdsRef = useRef<Set<string>>(new Set());
  const parentQueuedAccountIdsRef = useRef<Set<string>>(new Set());
  const childSortStatesRef = useRef<Record<string, ColumnState[]>>({});

  // silent: 로딩 화면으로 덮지 않고 뒤에서 값만 갱신한다. 현금 저장처럼 화면을 이미
  // 로컬로 맞춰 둔 뒤 정확한 값(금일 손익 등)만 따라오게 할 때 쓴다.
  const load = useCallback(async (options?: { silent?: boolean }) => {
    const silent = Boolean(options?.silent);
    if (!silent) {
      setLoading(true);
    }
    try {
      const [response, dashResponse] = await Promise.all([
        fetch("/api/assets", { cache: "no-store" }),
        fetch("/api/dashboard", { cache: "no-store" }).catch(() => null),
      ]);
      const payload = (await response.json()) as HoldingsResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "자산 정보를 불러오지 못했습니다.");
      }
      const dashData = dashResponse?.ok ? await dashResponse.json() : null;
      const dashAccounts: Record<string, { cash_ratio: number; net_profit: number; net_profit_pct: number; daily_profit: number; daily_return_pct: number; benchmark_name: string | null; benchmark_pct: number | null; index_result: "win" | "lose" | "draw" | null; weekly_profit: number; weekly_return_pct: number }> = {};
      if (dashData?.accounts) {
        for (const a of dashData.accounts) {
          dashAccounts[a.account_id] = {
            cash_ratio: a.cash_ratio ?? 0,
            net_profit: a.net_profit ?? 0,
            net_profit_pct: a.net_profit_pct ?? 0,
            daily_profit: a.daily_profit ?? 0,
            daily_return_pct: a.daily_return_pct ?? 0,
            benchmark_name: a.benchmark_name ?? null,
            benchmark_pct: a.benchmark_pct ?? null,
            index_result: a.index_result ?? null,
            weekly_profit: a.weekly_profit ?? 0,
            weekly_return_pct: a.weekly_return_pct ?? 0,
          };
        }
      }
      const dashTotals = dashData?.totals ?? null;
      setDashTotals(dashTotals);
      const defaultDash = { cash_ratio: 0, net_profit: 0, net_profit_pct: 0, daily_profit: 0, daily_return_pct: 0, benchmark_name: null, benchmark_pct: null, index_result: null, weekly_profit: 0, weekly_return_pct: 0 };
      const mergedSummaries = (payload.account_summaries ?? []).map((s) => ({
        ...s,
        ...(dashAccounts[s.account_id] ?? defaultDash),
      }));
      setAllRows(payload.rows ?? []);
      setSummaries(mergedSummaries);
      if (!silent) {
        setParentDirtyCellKeys([]);
        setEditingParentId(null);
      }
    } catch (error) {
      if (!silent) {
        toast.error(error instanceof Error ? error.message : "자산 정보를 불러오지 못했습니다.");
      }
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }, [toast]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    summariesRef.current = summaries;
  }, [summaries]);

  useEffect(() => {
    return () => {
      parentSaveTimersRef.current.forEach((timer) => clearTimeout(timer));
      parentSaveTimersRef.current.clear();
    };
  }, []);

  const groupedRows = useMemo(() => {
    const grouped = new Map<string, HoldingsRow[]>();
    for (const row of allRows) {
      const key = String(row.account_id || "").trim();
      if (!grouped.has(key)) {
        grouped.set(key, []);
      }
      grouped.get(key)?.push(row);
    }
    return grouped;
  }, [allRows]);

  const totalAssets = useMemo(
    () => summaries.reduce((sum, summary) => sum + Number(summary.total_assets_krw ?? 0), 0),
    [summaries],
  );
  const totalValuation = useMemo(
    () => summaries.reduce((sum, summary) => sum + Number(summary.valuation_krw ?? 0), 0),
    [summaries],
  );
  const totalCash = useMemo(
    () => summaries.reduce((sum, summary) => sum + Number(summary.cash_balance_krw ?? 0), 0),
    [summaries],
  );
  const totalPrincipal = useMemo(
    () => summaries.reduce((sum, summary) => sum + Number(summary.total_principal ?? 0), 0),
    [summaries],
  );
  const totalHoldingsCount = useMemo(
    () => summaries.reduce((sum, summary) => sum + Number(summary.holdings_count ?? 0), 0),
    [summaries],
  );

  const parentRows = useMemo<ParentGridRow[]>(() => {
    // 합계 지수(%) = 각 계좌 지수(%)를 자산 비중(total_assets_krw)으로 가중평균.
    // 승부 = 합계 금일(%) − 가중 지수(%). (지수 없는 계좌는 가중에서 제외 후 재정규화)
    let benchWeightedSum = 0;
    let benchAssetSum = 0;
    for (const s of summaries) {
      const b = s.benchmark_pct;
      const a = Number(s.total_assets_krw ?? 0);
      if (b !== null && b !== undefined && a > 0) {
        benchWeightedSum += a * b;
        benchAssetSum += a;
      }
    }
    const totalBenchmarkPct = benchAssetSum > 0 ? benchWeightedSum / benchAssetSum : null;
    const totalDailyPct = dashTotals?.daily_return_pct ?? 0;
    const totalIndexResult: "win" | "lose" | "draw" | null =
      totalBenchmarkPct === null
        ? null
        : totalDailyPct - totalBenchmarkPct > 0
          ? "win"
          : totalDailyPct - totalBenchmarkPct < 0
            ? "lose"
            : "draw";

    const totalRow: ParentGridRow = {
      id: "__total__",
      rowType: "total",
      name: "합계",
      total_assets_krw: totalAssets,
      valuation_krw: totalValuation,
      total_principal: totalPrincipal,
      cash_krw: totalCash,
      target_ratio_total: null,
      holdings_count: totalHoldingsCount,
      cash_ratio: totalAssets > 0 ? (totalCash / totalAssets) * 100 : 0,
      net_profit: totalAssets - totalPrincipal,
      net_profit_pct: totalPrincipal > 0 ? ((totalAssets - totalPrincipal) / totalPrincipal) * 100 : 0,
      daily_profit: dashTotals?.daily_profit ?? summaries.reduce((sum, s) => sum + (s.daily_profit ?? 0), 0),
      daily_return_pct: dashTotals?.daily_return_pct ?? 0,
      weekly_profit: dashTotals?.weekly_profit ?? summaries.reduce((sum, s) => sum + (s.weekly_profit ?? 0), 0),
      weekly_return_pct: dashTotals?.weekly_return_pct ?? 0,
      benchmark_pct: totalBenchmarkPct,
      index_result: totalIndexResult,
    };

    const detailRows = summaries.flatMap((summary): ParentGridRow[] => {
      const mainRow: ParentGridRow = {
        ...summary,
        id: summary.account_id,
        rowType: "main",
        // 다른 금액 열(총자산·평가액·원금)과 같은 원화 기준. 예전에는 계좌 통화의 native
        // 금액을 섞어 넣고 통화 기호로 찍어서, 미국 계좌의 원화 현금이 "$60,927,623.99"
        // 처럼 달러로 보였다. 통화별 잔액은 계좌를 펼쳐 상단 박스에서 본다.
        cash_krw: Number(summary.cash_balance_krw ?? 0),
      };

      if (expandedId !== summary.account_id) {
        return [mainRow];
      }

      const detailRow: ParentGridRow = {
        id: `${summary.account_id}__detail`,
        rowType: "detail",
        parentId: summary.account_id,
        summary,
        rows: groupedRows.get(summary.account_id) ?? [],
      };

      return [
        mainRow,
        detailRow,
      ];
    });
    return [totalRow, ...detailRows];
  }, [expandedId, groupedRows, summaries, totalAssets, totalCash, totalHoldingsCount, totalPrincipal, totalValuation]);

  useEffect(() => {
    onHeaderSummaryChange?.({
      totalAssets,
      totalValuation,
      totalCash,
      accountCount: summaries.length,
    });
  }, [onHeaderSummaryChange, summaries.length, totalAssets, totalCash, totalValuation]);

  const isDirtyParentCell = useCallback(
    (rowId: string | undefined, field: string) => Boolean(rowId && parentDirtyCellKeys.includes(buildDirtyCellKey(rowId, field))),
    [parentDirtyCellKeys],
  );

  const clearDirtyParentState = useCallback((accountId: string) => {
    setParentDirtyCellKeys((previous) => previous.filter((key) => !key.startsWith(`${accountId}::`)));
  }, []);

  const silentlySaveParent = useCallback(async (accountId: string) => {
    if (parentSavingAccountIdsRef.current.has(accountId)) {
      parentQueuedAccountIdsRef.current.add(accountId);
      return;
    }

    const summary = summariesRef.current.find((item) => item.account_id === accountId);
    if (!summary) {
      clearDirtyParentState(accountId);
      return;
    }

    parentSavingAccountIdsRef.current.add(accountId);
    try {
      const isAud = String(summary.currency || "KRW").toUpperCase() === "AUD";
      const response = await fetch("/api/assets", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          account_id: summary.account_id,
          total_principal: summary.total_principal,
          cash_balance_krw: isAud ? 0 : summary.cash_balance_krw,
          cash_balance_native: isAud ? summary.cash_balance_native : summary.cash_balance_krw,
          cash_currency: summary.cash_currency,
          intl_shares_value: summary.account_id === "aus_account" ? summary.intl_shares_value : null,
          intl_shares_change: summary.account_id === "aus_account" ? summary.intl_shares_change : null,
          cash_target_ratio: summary.cash_target_ratio,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "계좌 저장에 실패했습니다.");
      }
      clearDirtyParentState(accountId);
    } catch (error) {
      await load();
      toast.error(error instanceof Error ? error.message : "계좌 저장에 실패했습니다.");
    } finally {
      parentSavingAccountIdsRef.current.delete(accountId);
      if (parentQueuedAccountIdsRef.current.has(accountId)) {
        parentQueuedAccountIdsRef.current.delete(accountId);
        const nextTimer = setTimeout(() => {
          parentSaveTimersRef.current.delete(accountId);
          void silentlySaveParent(accountId);
        }, 400);
        parentSaveTimersRef.current.set(accountId, nextTimer);
      }
    }
  }, [clearDirtyParentState, load, toast]);

  const scheduleSilentParentSave = useCallback((accountId: string) => {
    const currentTimer = parentSaveTimersRef.current.get(accountId);
    if (currentTimer) {
      clearTimeout(currentTimer);
    }
    const nextTimer = setTimeout(() => {
      parentSaveTimersRef.current.delete(accountId);
      void silentlySaveParent(accountId);
    }, 700);
    parentSaveTimersRef.current.set(accountId, nextTimer);
  }, [silentlySaveParent]);

  const handleParentCellValueChanged = useCallback((row: ParentGridRow | undefined, field: string | undefined) => {
    if (!row || isDetailRow(row) || isTotalRow(row) || !field) {
      return;
    }

    setSummaries((previous) =>
      previous.map((summary) => {
        if (summary.account_id !== row.account_id) {
          return summary;
        }

        // 이 그리드에서 편집 가능한 값은 '총 원금' 뿐이다 — 현금은 계좌를 펼쳐 통화별로
        // 저장하므로 여기서 건드리지 않는다(건드리면 원화·native 가 서로 어긋난다).
        return {
          ...summary,
          total_principal: Number(row.total_principal ?? summary.total_principal),
        };
      }),
    );
    const dirtyCellKey = buildDirtyCellKey(row.account_id, field);
    setParentDirtyCellKeys((previous) => (previous.includes(dirtyCellKey) ? previous : [...previous, dirtyCellKey]));
    scheduleSilentParentSave(row.account_id);
  }, [scheduleSilentParentSave]);

  const handleChildRowsSync = useCallback((accountId: string, nextRows: HoldingsRow[]) => {
    setAllRows((previous) => [
      ...previous.filter((row) => row.account_id !== accountId),
      ...nextRows,
    ]);

    const nextValuation = nextRows.reduce((sum, row) => sum + Number(row.valuation_krw ?? 0), 0);
    // 자식 행의 목표 비중 변경이 부모 '목표비중합'에 즉시 반영되도록 합산 (현금/IS 행 제외).
    const nextTargetRatioTotal = nextRows.reduce((sum, row) => {
      const ticker = String(row.ticker || "").trim().toUpperCase();
      if (ticker === CASH_ROW_TICKER || ticker === "IS") {
        return sum;
      }
      return sum + Number(row.target_ratio ?? 0);
    }, 0);
    setSummaries((previous) =>
      previous.map((summary) => {
        if (summary.account_id !== accountId) {
          return summary;
        }
        return {
          ...summary,
          valuation_krw: nextValuation,
          total_assets_krw: nextValuation + Number(summary.cash_balance_krw ?? 0),
          holdings_count: nextRows.length,
          target_ratio_total: nextTargetRatioTotal,
        };
      }),
    );
  }, []);

  const handleChildSortStateChange = useCallback((accountId: string, state: ColumnState[]) => {
    childSortStatesRef.current = {
      ...childSortStatesRef.current,
      [accountId]: state,
    };
  }, []);

  const DetailRenderer = useCallback(
    (params: { data?: ParentGridRow }) => {
      const data = params.data;
      if (!data || !isDetailRow(data)) {
        return null;
      }
      return (
        <AccountHoldingsDetailPanel
          summary={data.summary}
          initialRows={data.rows}
          onRowsSync={handleChildRowsSync}
          onCashSync={(accountId, cashBalanceKrw, cashTargetRatio, saved) => {
            setSummaries((previous) =>
              previous.map((summary) => {
                if (summary.account_id !== accountId) {
                  return summary;
                }
                const currentCashKrw = Number(summary.cash_balance_krw ?? 0);
                const currentCashNative = Number(summary.cash_balance_native ?? 0);
                const nextCashNative =
                  saved?.cash_balance_native != null
                    ? saved.cash_balance_native
                    : String(summary.currency || "KRW").toUpperCase() === "AUD" && currentCashKrw > 0 && currentCashNative > 0
                      ? (cashBalanceKrw / currentCashKrw) * currentCashNative
                      : summary.cash_balance_native;
                // 현금에서 곧바로 나오는 값들은 여기서 다시 계산한다 — 대시보드가 내려주는
                // 값을 그대로 두면 리로드 전까지 현금 비중·평가손익이 옛 현금 기준으로 남는다.
                const nextTotalAssets = Number(summary.valuation_krw ?? 0) + cashBalanceKrw;
                const nextPrincipal = Number(saved?.total_principal ?? summary.total_principal ?? 0);
                const nextNetProfit = nextTotalAssets - nextPrincipal;
                return {
                  ...summary,
                  ...(saved?.cash ? { cash: saved.cash } : {}),
                  ...(saved?.updated_at !== undefined ? { updated_at: saved.updated_at } : {}),
                  ...(saved?.updated_by !== undefined ? { updated_by: saved.updated_by } : {}),
                  cash_balance_krw: cashBalanceKrw,
                  cash_balance_native: nextCashNative,
                  cash_target_ratio: cashTargetRatio,
                  total_principal: nextPrincipal,
                  total_assets_krw: nextTotalAssets,
                  cash_ratio: nextTotalAssets > 0 ? (cashBalanceKrw / nextTotalAssets) * 100 : 0,
                  net_profit: nextNetProfit,
                  net_profit_pct: nextPrincipal > 0 ? (nextNetProfit / nextPrincipal) * 100 : 0,
                };
              }),
            );
          }}
          onSortStateChange={handleChildSortStateChange}
          onReload={load}
          showAmounts={showAmounts}
        />
      );
    },
    [handleChildRowsSync, handleChildSortStateChange, load, showAmounts],
  );

  const parentColumns = useMemo<ColDef<ParentGridRow>[]>(() => [
    {
      field: "name",
      headerName: "계좌",
      minWidth: 220,
      flex: 1.2,
      cellRenderer: (params: { data?: ParentGridRow; value?: string }) => {
        const data = params.data;
        if (!data || isDetailRow(data)) {
          return "";
        }
        if (isTotalRow(data)) {
          return <span className="fw-bold">{data.name}</span>;
        }
        const label = (
          <>
            {data.icon} {params.value}
          </>
        );
        return (
          <div className="snapshotsExpandCell">
            <span className="snapshotsExpandIcon" aria-hidden="true">
              {data.account_id === expandedId ? "▾" : "▸"}
            </span>
            <span>{label}</span>
          </div>
        );
      },
    },
    {
      colId: "asset_weight",
      headerName: "비중",
      minWidth: 80,
      flex: 0.6,
      type: "rightAligned",
      valueGetter: (params) => {
        if (!params.data || isDetailRow(params.data) || isTotalRow(params.data)) {
          return null;
        }
        // 합계 행의 총 자산을 분모로 사용
        let totalSum = 0;
        params.api.forEachNode((node) => {
          const d = node.data as ParentGridRow | undefined;
          if (d && isTotalRow(d)) {
            totalSum = Number((d as Record<string, unknown>).total_assets_krw ?? 0);
          }
        });
        if (totalSum <= 0) return null;
        const own = Number((params.data as Record<string, unknown>).total_assets_krw ?? 0);
        return (own / totalSum) * 100;
      },
      cellRenderer: (params: { value?: number | null }) =>
        params.value !== null && params.value !== undefined
          ? `${params.value.toFixed(2)}%`
          : "",
    },
    {
      field: "cash_ratio",
      headerName: "현금 비중",
      minWidth: 90,
      flex: 0.7,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) =>
        params.data && !isDetailRow(params.data) ? `${(params.value ?? 0).toFixed(2)}%` : "",
    },
    {
      colId: "daily_profit_pct",
      headerName: "금일(%)",
      minWidth: 88,
      flex: 0.7,
      type: "rightAligned",
      valueGetter: (params) => {
        if (!params.data || isDetailRow(params.data)) {
          return null;
        }
        // 백엔드에서 계산된 daily_return_pct 사용 (입출금 영향 제거).
        return Number(params.data.daily_return_pct ?? 0);
      },
      cellRenderer: (params: { data?: ParentGridRow; value?: number | null }) =>
        params.data && !isDetailRow(params.data) && params.value !== null && params.value !== undefined
          ? <span className={getSignedClass(params.value)}>{`${params.value.toFixed(2)}%`}</span>
          : "",
    },
    {
      colId: "benchmark_pct",
      headerName: "지수(%)",
      minWidth: 88,
      flex: 0.7,
      type: "rightAligned",
      valueGetter: (params) => {
        const data = params.data;
        if (!data || isDetailRow(data)) return null;
        const v = (data as { benchmark_pct?: number | null }).benchmark_pct;
        return v === null || v === undefined ? null : Number(v);
      },
      headerTooltip:
        "각 계좌 벤치마크(계좌 설정) 의 금일 등락률 — 원화 기준 (외화 상장 ETF 는 환율 변동을 반영). 합계는 비중 가중평균",
      cellRenderer: (params: { data?: ParentGridRow; value?: number | null }) => {
        const data = params.data;
        if (!data || isDetailRow(data)) return "";
        const v = params.value;
        if (v === null || v === undefined) return <span style={{ color: "var(--text-muted)" }}>-</span>;
        const name = isTotalRow(data) ? "비중 가중 지수" : (data as AccountSummary).benchmark_name ?? "";
        return (
          <span className={getSignedClass(v)} title={name}>{`${v.toFixed(2)}%`}</span>
        );
      },
    },
    {
      colId: "index_result",
      headerName: "승부",
      minWidth: 130,
      flex: 0.85,
      cellStyle: { display: "flex", alignItems: "center", justifyContent: "center", textAlign: "center" },
      cellRenderer: (params: { data?: ParentGridRow }) => {
        const data = params.data;
        if (!data || isDetailRow(data)) return "";
        const summary = data as {
          index_result?: "win" | "lose" | "draw" | null;
          daily_return_pct?: number | null;
          benchmark_pct?: number | null;
        };
        const r = summary.index_result;
        if (!r) return <span style={{ color: "var(--text-muted)" }}>-</span>;
        // 격차 = 금일% − 지수 금일% (앞선/뒤쳐진 폭, 퍼센트포인트). 합계는 비중 가중 기준.
        const acc = summary.daily_return_pct;
        const bench = summary.benchmark_pct;
        const diff = acc !== null && acc !== undefined && bench !== null && bench !== undefined ? acc - bench : null;
        const diffText = diff !== null ? ` ${diff >= 0 ? "+" : ""}${diff.toFixed(2)}%p` : "";
        if (r === "win") return <span style={{ color: "#dc2626", fontWeight: 700 }}>{`🏆 승${diffText}`}</span>;
        if (r === "lose") return <span style={{ color: "#1971c2", fontWeight: 700 }}>{`😢 패${diffText}`}</span>;
        return <span style={{ color: "var(--text-muted)", fontWeight: 700 }}>🤝 무</span>;
      },
    },
    {
      field: "total_assets_krw",
      headerName: "총 자산",
      minWidth: 132,
      flex: 1,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) =>
        params.data && !isDetailRow(params.data) ? formatHiddenAmount(showAmounts, formatKrw(params.value ?? 0)) : "",
    },
    {
      field: "account_url",
      headerName: "링크",
      minWidth: 48,
      maxWidth: 52,
      editable: false,
      sortable: false,
      filter: false,
      cellRenderer: (params: { data?: ParentGridRow }) => {
        const data = params.data;
        if (!data || isDetailRow(data) || isTotalRow(data)) {
          return "";
        }
        if (!data.account_url) {
          return <span>-</span>;
        }
        return (
          <a
            href={data.account_url}
            target="_blank"
            rel="noreferrer"
            className="assetsInlineLinkButton assetsMoveLinkButton"
            onClick={(event) => {
              event.stopPropagation();
            }}
          >
            이동
          </a>
        );
      },
    },
    {
      field: "total_principal",
      headerName: "총 원금",
      minWidth: 124,
      flex: 1,
      type: "rightAligned",
      editable: (params) => Boolean(params.data && !isDetailRow(params.data) && !isTotalRow(params.data)),
      cellClass: (params) => {
        if (!params.data || isDetailRow(params.data) || isTotalRow(params.data)) {
          return undefined;
        }
        return isDirtyParentCell(params.data.account_id, "total_principal")
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
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) =>
        params.data && !isDetailRow(params.data) ? formatHiddenAmount(showAmounts, formatKrw(params.value ?? 0)) : "",
    },
    {
      field: "valuation_krw",
      headerName: "평가액",
      minWidth: 132,
      flex: 1,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) =>
        params.data && !isDetailRow(params.data) ? formatHiddenAmount(showAmounts, formatKrw(params.value ?? 0)) : "",
    },
    {
      field: "cash_krw",
      headerName: "현금",
      minWidth: 124,
      flex: 1,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) =>
        params.data && !isDetailRow(params.data)
          ? formatHiddenAmount(showAmounts, formatKrw(params.value ?? 0))
          : "",
    },
    {
      field: "daily_profit",
      headerName: "금일 손익",
      minWidth: 110,
      flex: 0.9,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) =>
        params.data && !isDetailRow(params.data)
          ? <span className={getSignedClass(params.value ?? 0)}>{formatHiddenAmount(showAmounts, formatKrw(params.value ?? 0))}</span>
          : "",
    },
    {
      field: "net_profit",
      headerName: "계좌 손익",
      minWidth: 120,
      flex: 1,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) =>
        params.data && !isDetailRow(params.data)
          ? <span className={getSignedClass(params.value ?? 0)}>{formatHiddenAmount(showAmounts, formatKrw(params.value ?? 0))}</span>
          : "",
    },
    {
      field: "net_profit_pct",
      headerName: "수익률",
      minWidth: 90,
      flex: 0.7,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) =>
        params.data && !isDetailRow(params.data)
          ? <span className={getSignedClass(params.value ?? 0)}>{`${(params.value ?? 0).toFixed(2)}%`}</span>
          : "",
    },
    {
      field: "weekly_profit",
      headerName: "금주 손익",
      minWidth: 110,
      flex: 0.9,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) =>
        params.data && !isDetailRow(params.data)
          ? <span className={getSignedClass(params.value ?? 0)}>{formatHiddenAmount(showAmounts, formatKrw(params.value ?? 0))}</span>
          : "",
    },
    {
      colId: "weekly_profit_pct",
      headerName: "금주(%)",
      minWidth: 88,
      flex: 0.7,
      type: "rightAligned",
      valueGetter: (params) => {
        if (!params.data || isDetailRow(params.data)) {
          return null;
        }
        // 백엔드에서 계산된 weekly_return_pct 사용 (입출금 영향 제거).
        return Number(params.data.weekly_return_pct ?? 0);
      },
      cellRenderer: (params: { data?: ParentGridRow; value?: number | null }) =>
        params.data && !isDetailRow(params.data) && params.value !== null && params.value !== undefined
          ? <span className={getSignedClass(params.value)}>{`${params.value.toFixed(2)}%`}</span>
          : "",
    },

    {
      field: "holdings_count",
      headerName: "종목수",
      minWidth: 76,
      flex: 0.6,
      type: "rightAligned",
      cellRenderer: (params: { data?: ParentGridRow; value?: number }) =>
        params.data && !isDetailRow(params.data) ? formatNumber(params.value) : "",
    },
  ], [expandedId, isDirtyParentCell, showAmounts]);

  const gridOptions = useMemo<GridOptions<ParentGridRow>>(
    () => ({
      suppressMovableColumns: true,
      ensureDomOrder: true,
      stopEditingWhenCellsLoseFocus: true,
      isFullWidthRow: (params) => isDetailRow(params.rowNode.data),
      fullWidthCellRenderer: DetailRenderer,
      getRowHeight: (params) => {
        if (!isDetailRow(params.data)) {
          return 38;
        }
        // 기본적으로 현금(1) + 추가 가능 공간(1)을 고려하여 최소 +2행 공간을 확보합니다.
        const rowCount = (params.data.rows?.length ?? 0) + 2;
        // 툴바(50) + 그리드 헤더(36) + 행(실제 rowHeight 34) + 가로 스크롤바/테두리 안전 마진(5 + 30).
        return 50 + 36 + rowCount * 34 + 5 + 30;
      },
      onCellClicked: (params) => {
        if (!params.data || isDetailRow(params.data) || isTotalRow(params.data)) {
          return;
        }
        if (params.colDef.field !== "name") {
          return;
        }
        const accountId = params.data.account_id;
        setExpandedId((current) => (current === accountId ? null : accountId));
      },
      onCellEditingStarted: (params) => {
        if (params.data && !isDetailRow(params.data) && !isTotalRow(params.data)) {
          setEditingParentId(params.data.account_id);
        }
      },
      onCellEditingStopped: () => {
        setEditingParentId(null);
      },
      onCellValueChanged: (params) => {
        if (params.newValue === params.oldValue) {
          return;
        }
        handleParentCellValueChanged(params.data, params.colDef.field);
      },
      rowClassRules: {
        assetsEditingRow: (params) =>
          Boolean(params.data && !isDetailRow(params.data) && !isTotalRow(params.data) && params.data.account_id === editingParentId),
        snapshotsExpandedMainRow: (params) =>
          Boolean(params.data && !isDetailRow(params.data) && !isTotalRow(params.data) && params.data.account_id === expandedId),
      },
    }),
    [DetailRenderer, editingParentId, expandedId, handleParentCellValueChanged],
  );

  if (loading && !summaries.length) {
    return (
      <div className="appPageStack">
        <div className="appPageLoading">
          <AppLoadingState label="자산 정보를 불러오는 중..." />
        </div>
      </div>
    );
  }


  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection appSectionFill">
        <div className="card appCard shadow-sm appTableCardFill">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft" />
              <div className="appMainHeaderRight">
                <button
                  type="button"
                  className={`btn btn-sm shadow-sm ${showAmounts ? "btn-outline-secondary" : "btn-dark"}`}
                  onClick={() => setShowAmounts((previous) => !previous)}
                >
                  {showAmounts ? "금액 가리기" : "금액 보기"}
                </button>
              </div>
            </div>
          </div>
          <div className="card-body p-2 appTableCardBodyFill">
            <AppAgGrid
              rowData={parentRows}
              columnDefs={parentColumns}
              loading={loading}
              minHeight="100%"
              className="assetsAgGrid assetsParentAgGrid"
              theme={assetsGridTheme}
              getRowClass={(params: RowClassParams<ParentGridRow>) => {
                if (isDetailRow(params.data)) {
                  return "assetsDetailFullRow";
                }
                return "";
              }}
              gridOptions={gridOptions}
            />
          </div>
        </div>
      </section>
    </div>
  );
}
