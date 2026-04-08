"use client";

import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { iconSetQuartzBold, themeQuartz } from "ag-grid-community";
import type { ColDef, RowClassParams } from "ag-grid-community";

import {
  readRememberedTickerType,
  writeRememberedTickerType,
} from "../components/account-selection";
import { AppAgGrid } from "../components/AppAgGrid";
import { useToast } from "../components/ToastProvider";

type RankTickerType = {
  ticker_type: string;
  order: number;
  name: string;
  icon: string;
  country_code: string;
};

type RankMaRule = {
  order: number;
  ma_type: string;
  ma_months: number;
  ma_days: number;
  score_column: string;
};

type RankRow = {
  [key: string]: string | number | null;
  순번: string;
  순위: number | null;
  이전순위: number | null;
  추천: string | null;
  추천요약: string | null;
  버킷: string;
  bucket: number;
  티커: string;
  종목명: string;
  상장일: string;
  점수: number | null;
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
  배당률: number | null;
  보수: number | null;
  순자산총액: number | null;
};

type RankResponse = {
  ticker_types?: RankTickerType[];
  ticker_type?: string;
  ma_rules?: RankMaRule[];
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
  previous_trading_day?: string | null;
  missing_tickers?: string[];
  missing_ticker_labels?: string[];
  stale_tickers?: string[];
  error?: string;
};

type RankGridRow = RankRow & {
  id: string;
};

const rankGridTheme = themeQuartz
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

type RankToolbarCache = {
  ticker_types: RankTickerType[];
  ticker_type: string;
  ma_rules: RankMaRule[];
  ma_type_options: string[];
  ma_months_max: number;
};

type RankHeaderSummary = {
  upCount: number;
  upPct: number;
  totalCount: number;
  ruleSummary: string;
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

function renderSignedPercentCell(value: number | null) {
  return <span className={getSignedClass(value ?? null)}>{formatPercent(value ?? null)}</span>;
}

function formatCurrencyValue(value: number | null, countryCode?: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  const digits = countryCode === "au" ? 2 : 0;
  return formatNumber(value, digits);
}

function formatAssetInEok(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${formatNumber(value / 100_000_000, 0)}억`;
}

export function RankManager({ onHeaderSummaryChange }: { onHeaderSummaryChange?: (summary: RankHeaderSummary) => void }) {
  const toast = useToast();
  const lastBlockedToastRef = useRef<string | null>(null);
  const [ticker_types, setAccounts] = useState<RankTickerType[]>(rankToolbarCache?.ticker_types ?? []);
  const [selectedTickerType, setSelectedAccountId] = useState(
    rankToolbarCache?.ticker_type ?? readRememberedTickerType() ?? "",
  );
  const [maRules, setMaRules] = useState<RankMaRule[]>(rankToolbarCache?.ma_rules ?? []);
  const [maTypeOptions, setMaTypeOptions] = useState<string[]>(rankToolbarCache?.ma_type_options ?? []);
  const [maMonthsMax, setMaMonthsMax] = useState(rankToolbarCache?.ma_months_max ?? 12);
  const [metricMode, setMetricMode] = useState<"cumulative" | "monthly" | "info">("cumulative");
  const [dedupeEnabled, setDedupeEnabled] = useState(false);
  const [monthlyReturnLabels, setMonthlyReturnLabels] = useState<string[]>([]);
  const [selectedAsOfDate, setSelectedAsOfDate] = useState<string>(getTodayDateInputValue());
  const [nameKeyword, setNameKeyword] = useState("");
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

  async function load(next?: { ticker_type?: string; ma_rule_overrides?: RankMaRule[]; as_of_date?: string }) {
    setLoading(true);
    setError(null);

    try {
      const search = new URLSearchParams();
      if (next?.ticker_type) {
        search.set("ticker_type", next.ticker_type);
      }
      if (next?.as_of_date) {
        search.set("as_of_date", next.as_of_date);
      }
      for (const rule of next?.ma_rule_overrides ?? []) {
        search.set(`rule${rule.order}_ma_type`, rule.ma_type);
        search.set(`rule${rule.order}_ma_months`, String(rule.ma_months));
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
      setMaRules(payload.ma_rules ?? []);
      setMaTypeOptions(payload.ma_type_options ?? []);
      setMaMonthsMax(payload.ma_months_max ?? 12);
      setSelectedAsOfDate(toDateInputValue(payload.as_of_date));
      setMonthlyReturnLabels(payload.monthly_return_labels ?? []);
      rankToolbarCache = {
        ticker_types: payload.ticker_types ?? [],
        ticker_type: nextAccountId,
        ma_rules: payload.ma_rules ?? [],
        ma_type_options: payload.ma_type_options ?? [],
        ma_months_max: payload.ma_months_max ?? 12,
      };
      setRows(payload.rows ?? []);
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

  const gridRows = useMemo<RankGridRow[]>(
    () =>
      rows.map((row, index) => ({
        ...row,
        id: `${row.티커}-${row.순번 || "none"}-${index}`,
      })),
    [rows],
  );

  const filteredGridRows = useMemo(() => {
    const keyword = deferredNameKeyword.trim().toLowerCase();
    return gridRows.filter((row) => {
      if (dedupeEnabled && String(row.추천 ?? "").trim()) {
        return false;
      }
      if (!keyword) {
        return true;
      }
      return String(row.종목명 ?? "").toLowerCase().includes(keyword);
    });
  }, [gridRows, deferredNameKeyword, dedupeEnabled]);

  const maRuleSummary = useMemo(
    () => maRules.map((rule) => `추세${rule.order}: ${rule.ma_type} ${rule.ma_months}개월`),
    [maRules],
  );

  const columns = useMemo<ColDef<RankGridRow>[]>(() => {
    const leadingColumns: ColDef<RankGridRow>[] = [
      {
        field: "순위",
        headerName: "순위",
        minWidth: 72,
        width: 72,
        cellStyle: { textAlign: "center" },
        cellRenderer: (params: { value: number | null | undefined }) => {
          return (
            <span style={{ fontWeight: 700 }}>{params.value == null ? "-" : formatNumber(params.value, 0)}</span>
          );
        },
      },
      {
        field: "이전순위",
        headerName: "이전순위",
        minWidth: 98,
        width: 98,
        cellStyle: { textAlign: "center" },
        sortable: false,
        cellRenderer: (params: { data?: RankGridRow; value: number | null | undefined }) => {
          const currentRank = params.data?.순위 ?? null;
          const previousRank = params.value ?? null;
          if (currentRank === null || currentRank === undefined || previousRank === null || previousRank === undefined) {
            return <span style={{ fontWeight: 600 }}>-</span>;
          }

          if (currentRank === previousRank) {
            return (
              <span style={{ fontWeight: 600 }}>
                {currentRank}(-)
              </span>
            );
          }

          const isRise = currentRank < previousRank;
          const delta = Math.abs(currentRank - previousRank);
          return (
            <span style={{ color: isRise ? "#d63939" : "#206bc4", fontWeight: 700 }}>
              {currentRank}({isRise ? `+${delta}` : `-${delta}`} {isRise ? "▲" : "▼"})
            </span>
          );
        },
      },
      {
        field: "추천요약",
        headerName: "중복",
        minWidth: 92,
        width: 92,
        sortable: false,
        cellStyle: { textAlign: "center" },
        cellRenderer: (params: { value: string | null | undefined }) => {
          const value = String(params.value ?? "").trim();
          if (!value) {
            return "";
          }
          return (
            <span style={{ color: "#182433", fontWeight: 400 }} title={value}>
              {value}
            </span>
          );
        },
      },
      {
        field: "버킷",
        headerName: "버킷",
        minWidth: 108,
        width: 108,
        cellClass: (params) => getBucketCellClass(String(params.value ?? "")),
        cellRenderer: (params: { value: string | null | undefined }) => <span>{String(params.value ?? "-")}</span>,
      },
      {
        field: "티커",
        headerName: "티커",
        minWidth: 92,
        width: 92,
        cellRenderer: (params: { value: string | null | undefined; data?: RankGridRow }) => {
          const value = String(params.value ?? "-");
          const href = `/ticker?ticker=${encodeURIComponent(value)}`;
          return (
            <a href={href} className="appCodeText" style={{ color: "#206bc4", textDecoration: "none" }}>
              {value}
            </a>
          );
        },
      },
      {
        field: "종목명",
        headerName: "종목명",
        minWidth: 260,
        flex: 1.2,
        cellRenderer: (params: { value: string | null | undefined; data?: RankGridRow }) => {
          const value = String(params.value ?? "-");
          const ticker = params.data?.티커 ?? "";
          const href = `/ticker?ticker=${encodeURIComponent(ticker)}`;
          return (
            <a href={href} className="rankNameCellText" style={{ color: "inherit", textDecoration: "none" }} title={value}>
              {value}
            </a>
          );
        },
      },
      {
        field: "현재가",
        headerName: "현재가",
        minWidth: 110,
        width: 110,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) =>
          formatNumber(params.value ?? null, selectedTickerTypeItem?.country_code === "au" ? 2 : 0),
      },
      {
        field: "일간(%)",
        headerName: "일간(%)",
        minWidth: 96,
        width: 96,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => renderSignedPercentCell(params.value ?? null),
      },
    ];

    const cumulativeColumns: ColDef<RankGridRow>[] = [
      {
        field: "점수",
        headerName: "점수",
        minWidth: 72,
        width: 72,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatNumber(params.value ?? null, 1),
      },
      ...maRules.map(
        (rule) =>
          ({
            field: rule.score_column,
            headerName: `추세${rule.order}`,
            minWidth: 112,
            width: 112,
            type: "rightAligned",
            cellRenderer: (params: { value: number | null | undefined }) => formatNumber(params.value ?? null, 1),
          }) as ColDef<RankGridRow>,
      ),
      ...(selectedTickerTypeItem?.country_code !== "au"
        ? [
            {
              field: "괴리율",
              headerName: "괴리율",
              minWidth: 88,
              width: 88,
              type: "rightAligned",
              cellRenderer: (params: { value: number | null | undefined }) => {
                const val = params.value ?? 0;
                const isExtreme = val > 2.0 || val < -2.0;
                return (
                  <span style={{ color: isExtreme ? "#d63939" : "inherit", fontWeight: isExtreme ? 700 : 400 }}>
                    {formatPercent(params.value ?? null)}
                  </span>
                );
              },
            } as ColDef<RankGridRow>,
          ]
        : []),
      {
        field: "고점",
        headerName: "고점",
        minWidth: 92,
        width: 92,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => {
          const value = params.value ?? null;
          const isHighlighted = value !== null && value >= -5;
          return (
            <span style={{ color: isHighlighted ? "#7952b3" : "inherit", fontWeight: isHighlighted ? 700 : 400 }}>
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
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => renderSignedPercentCell(params.value ?? null),
      },
      {
        field: "2주(%)",
        headerName: "2주(%)",
        minWidth: 88,
        width: 88,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => renderSignedPercentCell(params.value ?? null),
      },
      {
        field: "3주(%)",
        headerName: "3주(%)",
        minWidth: 88,
        width: 88,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => renderSignedPercentCell(params.value ?? null),
      },
      {
        field: "4주(%)",
        headerName: "4주(%)",
        minWidth: 88,
        width: 88,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => renderSignedPercentCell(params.value ?? null),
      },
      {
        field: "RSI",
        headerName: "RSI",
        minWidth: 74,
        width: 74,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatNumber(params.value ?? null, 1),
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
            type: "rightAligned",
            cellRenderer: (params: { value: number | null | undefined }) => renderSignedPercentCell(params.value ?? null),
          }) as ColDef<RankGridRow>,
      ),
    ];

    const duplicateColumn: ColDef<RankGridRow> = {
      field: "추천",
      headerName: "중복상세",
      minWidth: 340,
      width: 340,
      sortable: false,
      cellRenderer: (params: { value: string | null | undefined }) => {
        const value = String(params.value ?? "").trim();
        if (!value) {
          return "";
        }
        return (
          <span className="rankNameCellText" style={{ color: "#d63939", fontWeight: 700 }} title={value}>
            {value}
          </span>
        );
      },
    };

    const monthlyColumns: ColDef<RankGridRow>[] = monthlyReturnLabels.map(
      (label) =>
        ({
          field: label,
          headerName: label,
          minWidth: 108,
          width: 108,
          type: "rightAligned",
          cellRenderer: (params: { value: number | null | undefined }) => renderSignedPercentCell(params.value ?? null),
        }) as ColDef<RankGridRow>,
    );

    const monthlyLeadingColumns: ColDef<RankGridRow>[] = [
      {
        field: "점수",
        headerName: "점수",
        minWidth: 72,
        width: 72,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatNumber(params.value ?? null, 1),
      },
      ...maRules.map(
        (rule) =>
          ({
            field: rule.score_column,
            headerName: `추세${rule.order}`,
            minWidth: 112,
            width: 112,
            type: "rightAligned",
            cellRenderer: (params: { value: number | null | undefined }) => formatNumber(params.value ?? null, 1),
          }) as ColDef<RankGridRow>,
      ),
    ];

    const infoColumns: ColDef<RankGridRow>[] = [
      {
        field: "배당률",
        headerName: "배당률",
        minWidth: 92,
        width: 92,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatPercent(params.value ?? null),
      },
      {
        field: "보수",
        headerName: "보수",
        minWidth: 92,
        width: 92,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatPercent(params.value ?? null),
      },
      {
        field: "순자산총액",
        headerName: "순자산총액",
        minWidth: 132,
        width: 132,
        type: "rightAligned",
        cellRenderer: (params: { value: number | null | undefined }) => formatAssetInEok(params.value ?? null),
      },
      {
        field: "상장일",
        headerName: "상장일",
        minWidth: 110,
        width: 110,
        cellRenderer: (params: { value: string | null | undefined }) => String(params.value ?? "-"),
      },
    ];

    return [
      ...leadingColumns,
      ...(metricMode === "cumulative"
        ? [...cumulativeColumns, duplicateColumn]
        : metricMode === "monthly"
          ? [...monthlyLeadingColumns, ...monthlyColumns, duplicateColumn]
          : [...monthlyLeadingColumns, ...infoColumns, duplicateColumn]),
    ];
  }, [maRules, metricMode, monthlyReturnLabels, selectedTickerType, selectedTickerTypeItem?.country_code]);

  function handleTickerTypeChange(accountId: string) {
    setSelectedAccountId(accountId);
    writeRememberedTickerType(accountId);
    void load({ ticker_type: accountId, as_of_date: selectedAsOfDate });
  }

  function handleMaRuleTypeChange(order: number, nextMaType: string) {
    const nextRules = maRules.map((rule) => (rule.order === order ? { ...rule, ma_type: nextMaType } : rule));
    setMaRules(nextRules);
    void load({ ticker_type: selectedTickerType, ma_rule_overrides: nextRules, as_of_date: selectedAsOfDate });
  }

  function handleMaRuleMonthsChange(order: number, nextMaMonths: number) {
    const nextRules = maRules.map((rule) => (rule.order === order ? { ...rule, ma_months: nextMaMonths } : rule));
    setMaRules(nextRules);
    void load({ ticker_type: selectedTickerType, ma_rule_overrides: nextRules, as_of_date: selectedAsOfDate });
  }

  function handleAsOfDateChange(nextAsOfDate: string) {
    setSelectedAsOfDate(nextAsOfDate);
    void load({ ticker_type: selectedTickerType, ma_rule_overrides: maRules, as_of_date: nextAsOfDate });
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

  const headerSummary = useMemo<RankHeaderSummary>(() => {
    const totalCount = filteredGridRows.length;
    const upCount = filteredGridRows.filter((r) => (r["점수"] ?? 0) > 0).length;
    const upPct = totalCount > 0 ? Math.round((upCount / totalCount) * 100) : 0;
    return {
      upCount,
      upPct,
      totalCount,
      ruleSummary: maRuleSummary.join(" / ") || "-",
    };
  }, [filteredGridRows, maRuleSummary]);

  useEffect(() => {
    onHeaderSummaryChange?.(headerSummary);
  }, [headerSummary, onHeaderSummaryChange]);

  return (
    <div className="appPageStack appPageStackFill">
      {error ? (
        <div className="appBannerStack">
          <div className="bannerError alert alert-danger mb-0">{error}</div>
        </div>
      ) : null}

      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft rankMainHeaderLeft">
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">기준일</span>
                  <input
                    className="form-control"
                    type="date"
                    value={selectedAsOfDate}
                    max={getTodayDateInputValue()}
                    onChange={(event) => handleAsOfDateChange(event.target.value)}
                  />
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">종목 타입</span>
                  <select
                    className="form-select"
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
                </label>

                {maRules.map((rule) => (
                  <label key={rule.order} className="appLabeledField">
                    <span className="appLabeledFieldLabel">{`추세${rule.order}`}</span>
                    <div className="rankRuleFieldRow">
                      <select
                        className="form-select"
                        value={rule.ma_type}
                        onChange={(event) => handleMaRuleTypeChange(rule.order, event.target.value)}
                        disabled={maTypeOptions.length === 0}
                      >
                        {maTypeOptions.map((option) => (
                          <option key={`${rule.order}-${option}`} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                      <select
                        className="form-select"
                        value={String(rule.ma_months)}
                        onChange={(event) => handleMaRuleMonthsChange(rule.order, Number(event.target.value))}
                        disabled={maTypeOptions.length === 0}
                      >
                        {Array.from({ length: maMonthsMax }, (_, index) => index + 1).map((month) => (
                          <option key={`${rule.order}-${month}`} value={month}>
                            {month}개월
                          </option>
                        ))}
                      </select>
                    </div>
                  </label>
                ))}
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">수익률 보기</span>
                  <div className="appSegmentedToggle appSegmentedToggleCompact" role="group" aria-label="수익률 보기 방식">
                    <button
                      type="button"
                      className={metricMode === "cumulative" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                      onClick={() => setMetricMode("cumulative")}
                    >
                      누적
                    </button>
                    <button
                      type="button"
                      className={metricMode === "monthly" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                      onClick={() => setMetricMode("monthly")}
                    >
                      월별
                    </button>
                    <button
                      type="button"
                      className={metricMode === "info" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
                      onClick={() => setMetricMode("info")}
                    >
                      정보
                    </button>
                  </div>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">중복제거</span>
                  <span className="rankSwitchField">
                    <label className="form-check form-switch mb-0 rankSwitchFieldInner">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        checked={dedupeEnabled}
                        onChange={(event) => setDedupeEnabled(event.target.checked)}
                      />
                    </label>
                  </span>
                </label>
                <label className="appLabeledField">
                  <span className="appLabeledFieldLabel">종목명 검색</span>
                  <input
                    className="form-control"
                    type="text"
                    value={nameKeyword}
                    placeholder="종목명을 입력"
                    onChange={(event) => setNameKeyword(event.target.value)}
                  />
                </label>
              </div>

            </div>
          </div>

          <div className="card-body appCardBodyTight appTableCardBodyFill">
            <div className="appGridFillWrap">
              <AppAgGrid
                className="rankAgGrid"
                rowData={filteredGridRows}
                columnDefs={columns}
                loading={loading}
                theme={rankGridTheme}
                getRowClass={(params: RowClassParams<RankGridRow>) => {
                  const classes: string[] = [];
                  if ((params.data?.점수 ?? 0) < 0) {
                    classes.push("rankNegativeTrendRow");
                  }
                  if (String(params.data?.보유 || "").trim() !== "") {
                    classes.push("rankHeldRow");
                  }
                  return classes.join(" ");
                }}
                minHeight="100%"
                gridOptions={{
                  suppressMovableColumns: true,
                }}
              />
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
