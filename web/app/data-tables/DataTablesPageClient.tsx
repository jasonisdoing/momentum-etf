"use client";

import type { ColDef } from "ag-grid-community";
import { useCallback, useEffect, useMemo, useState } from "react";

import { AppAgGrid } from "../components/AppAgGrid";
import { createAppGridTheme } from "../components/app-grid-theme";
import { PageFrame } from "../components/PageFrame";

type TableRow = {
  name: string;
  category: string;
  category_label: string;
  policy: string;
  purpose: string;
  owner_field: string;
  owner_note: string;
  /** 이름이 소유자에서 파생되는 컬렉션(`cache_<풀>_stocks`)만 붙는다. */
  owner?: string;
  /** 카탈로그에는 있는데 DB 에 아직 없는 경우. */
  missing?: boolean;
  count: number;
  size: number;
  index_size: number;
};

/** 주인 없는 데이터 한 자리 — 컬렉션·문서 안 키·참조 필드를 같은 형태로 담는다. */
type OrphanItem = {
  location: string;
  owner_kind: string;
  detail: string;
  owners: string[];
  count: number;
  size?: number;
};

type Payload = {
  rows: TableRow[];
  unclassified: TableRow[];
  orphans: { items: OrphanItem[]; total: number };
  category_order: string[];
  category_labels: Record<string, string>;
  totals: { collections: number; count: number; size: number; index_size: number };
  pools: string[];
  accounts: string[];
};

/** 분류 배지 색 — 삭제 정책의 성격을 색으로 구분한다.
 *  소유자별(파랑 계열) · 보존(초록) · 설정(보라) · 재생성 가능(회색) · 미분류(빨강). */
const CATEGORY_TONE: Record<string, { bg: string; fg: string }> = {
  pool: { bg: "#dbeafe", fg: "#1e40af" },
  account: { bg: "#e0e7ff", fg: "#3730a3" },
  aggregate: { bg: "#dcfce7", fg: "#166534" },
  config: { bg: "#f3e8ff", fg: "#6b21a8" },
  reference: { bg: "#f1f5f9", fg: "#334155" },
  runtime: { bg: "#fef3c7", fg: "#92400e" },
  personal: { bg: "#f1f5f9", fg: "#334155" },
  unclassified: { bg: "#fee2e2", fg: "#991b1b" },
};

const gridTheme = createAppGridTheme();

function formatBytes(value: number): string {
  if (!value) return "-";
  if (value >= 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`;
  return `${Math.round(value / 1024).toLocaleString("ko-KR")} KB`;
}

export function DataTablesPageClient() {
  const [data, setData] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch("/api/data-tables", { cache: "no-store" });
      const payload = (await resp.json()) as Payload & { error?: string };
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "테이블 목록을 불러오지 못했습니다.");
      setData(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "테이블 목록을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // 미분류를 **맨 위**에 둔다 — 등록 누락이 목적이라 눈에 먼저 들어와야 한다.
  const gridRows = useMemo<TableRow[]>(() => {
    if (!data) return [];
    const order = new Map(data.category_order.map((key, index) => [key, index]));
    const rest = [...data.rows].sort((a, b) => {
      const byCategory = (order.get(a.category) ?? 99) - (order.get(b.category) ?? 99);
      if (byCategory !== 0) return byCategory;
      return b.size - a.size;
    });
    return [...data.unclassified, ...rest];
  }, [data]);

  // 컬렉션 이름 → 고아 건수. 표에 컬럼 하나로 붙인다.
  const orphanByLocation = useMemo(() => {
    const map = new Map<string, OrphanItem>();
    for (const item of data?.orphans.items ?? []) map.set(item.location, item);
    return map;
  }, [data]);

  const columnDefs = useMemo<ColDef<TableRow>[]>(
    () => [
      {
        field: "category_label",
        headerName: "분류",
        width: 104,
        cellRenderer: (params: { data?: TableRow }) => {
          const row = params.data;
          if (!row) return null;
          const tone = CATEGORY_TONE[row.category] ?? CATEGORY_TONE.reference;
          return (
            <span
              style={{
                display: "inline-block",
                padding: "0.1rem 0.45rem",
                borderRadius: "0.35rem",
                background: tone.bg,
                color: tone.fg,
                fontSize: "var(--fs-sm)",
                fontWeight: 700,
                whiteSpace: "nowrap",
              }}
            >
              {row.category_label}
            </span>
          );
        },
      },
      {
        field: "name",
        headerName: "컬렉션",
        width: 260,
        cellStyle: { fontWeight: 600 },
        cellRenderer: (params: { data?: TableRow; value?: string }) => (
          <span>
            {params.value}
            {params.data?.missing ? <span style={{ color: "var(--text-muted)" }}> (DB 에 없음)</span> : null}
          </span>
        ),
      },
      {
        field: "owner",
        headerName: "소유자",
        width: 116,
        valueGetter: (params) => params.data?.owner ?? "",
        cellRenderer: (params: { value?: string }) =>
          params.value ? <span>{params.value}</span> : <span style={{ color: "var(--text-muted)" }}>-</span>,
        headerTooltip: "이름이 소유자에서 파생되는 컬렉션(cache_<소유자>_stocks)만 값이 있습니다.",
      },
      {
        field: "owner_field",
        headerName: "소유자 필드",
        width: 118,
        cellRenderer: (params: { data?: TableRow; value?: string }) => {
          if (!params.value) return <span style={{ color: "var(--text-muted)" }}>-</span>;
          // 필드명과 실제 내용이 다른 경우(옛 명명 잔재)는 주의 표시를 붙인다.
          return (
            <span title={params.data?.owner_note || undefined}>
              <code>{params.value}</code>
              {params.data?.owner_note ? " ⚠️" : ""}
            </span>
          );
        },
      },
      {
        field: "count",
        headerName: "문서",
        width: 96,
        type: "numericColumn",
        valueFormatter: (params) => (params.value ? Number(params.value).toLocaleString("ko-KR") : "-"),
      },
      {
        field: "size",
        headerName: "데이터",
        width: 100,
        type: "numericColumn",
        valueFormatter: (params) => formatBytes(Number(params.value ?? 0)),
      },
      {
        field: "index_size",
        headerName: "인덱스",
        width: 96,
        type: "numericColumn",
        valueFormatter: (params) => formatBytes(Number(params.value ?? 0)),
      },
      {
        colId: "orphan",
        headerName: "고아",
        width: 92,
        type: "numericColumn",
        sortable: true,
        valueGetter: (params) => orphanByLocation.get(params.data?.name ?? "")?.count ?? null,
        headerTooltip: "살아있는 종목풀·계좌 어디에도 속하지 않는 데이터. 소유자를 지울 때 함께 정리되지 않은 것입니다.",
        cellRenderer: (params: { data?: TableRow; value?: number | null }) => {
          const item = orphanByLocation.get(params.data?.name ?? "");
          if (!item) return <span style={{ color: "var(--text-muted)" }}>-</span>;
          return (
            <span style={{ color: "#991b1b", fontWeight: 700 }} title={item.detail}>
              {item.count.toLocaleString("ko-KR")}
            </span>
          );
        },
      },
      { field: "policy", headerName: "삭제 정책", width: 190 },
      { field: "purpose", headerName: "설명", flex: 1, minWidth: 320, tooltipField: "purpose" },
    ],
    [orphanByLocation],
  );

  const titleRight = data ? (
    <div className="appHeaderMetrics rankToolbarMeta">
      <div className="appHeaderMetric">
        <span>컬렉션:</span>
        <span className="appHeaderMetricValue is-primary">{data.totals.collections}개</span>
      </div>
      <div className="appHeaderMetric">
        <span>문서:</span>
        <span className="appHeaderMetricValue">{data.totals.count.toLocaleString("ko-KR")}</span>
      </div>
      <div className="appHeaderMetric">
        <span>데이터:</span>
        <span className="appHeaderMetricValue">{formatBytes(data.totals.size)}</span>
      </div>
      <div className="appHeaderMetric">
        <span>인덱스:</span>
        <span className="appHeaderMetricValue">{formatBytes(data.totals.index_size)}</span>
      </div>
      <div className="appHeaderMetric">
        <span>고아:</span>
        <span
          className="appHeaderMetricValue"
          style={data.orphans.total > 0 ? { color: "#991b1b" } : undefined}
        >
          {data.orphans.total.toLocaleString("ko-KR")}건
        </span>
      </div>
    </div>
  ) : null;

  return (
    <PageFrame title="테이블" fullHeight fullWidth titleRight={titleRight}>
      <div className="appPageStack appPageStackFill">
        {error ? (
          <div className="appBannerStack">
            <div className="bannerError alert alert-danger mb-0">{error}</div>
          </div>
        ) : null}

        {data && data.orphans.items.length > 0 ? (
          <div className="appBannerStack">
            <div className="alert alert-danger mb-0">
              주인 없는 데이터 {data.orphans.total.toLocaleString("ko-KR")}건 — 종목풀·계좌를 지울 때 함께
              정리되지 않은 것입니다.
              <div style={{ fontSize: "var(--fs-sm)", marginTop: 6, display: "flex", flexDirection: "column", gap: 2 }}>
                {data.orphans.items.map((item) => (
                  <div key={item.location}>
                    <code>{item.location}</code> — {item.count.toLocaleString("ko-KR")}건 · {item.detail}
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : null}

        {data && data.unclassified.length > 0 ? (
          <div className="appBannerStack">
            <div className="alert alert-warning mb-0">
              카탈로그에 없는 컬렉션 {data.unclassified.length}개:{" "}
              <b>{data.unclassified.map((row) => row.name).join(", ")}</b>
              <div style={{ fontSize: "var(--fs-sm)", marginTop: 4 }}>
                새로 만든 컬렉션이면 <code>utils/data_table_catalog.py</code> 에 등록하고, 쓰지 않는 것이면
                지웁니다. 등록하지 않으면 종목풀·계좌를 지울 때 함께 정리되지 않습니다.
              </div>
            </div>
          </div>
        ) : null}

        <section className="appSection appSectionFill">
          <div className="card appCard appTableCardFill">
            <div className="card-body appCardBodyTight appTableCardBodyFill">
              <div className="appGridFillWrap">
                <AppAgGrid<TableRow>
                  rowData={gridRows}
                  columnDefs={columnDefs}
                  theme={gridTheme}
                  loading={loading}
                  minHeight="100%"
                  getRowId={(params) => params.data.name}
                  gridOptions={{
                    suppressMovableColumns: true,
                    suppressCellFocus: false,
                    defaultColDef: {
                      sortable: true,
                      resizable: true,
                      // 폭에 맞춰 말줄임하므로, 잘린 값은 마우스를 올려 전체를 본다.
                      tooltipValueGetter: (params) => params.valueFormatted ?? params.value,
                    },
                  }}
                />
              </div>
            </div>
          </div>
        </section>
      </div>
    </PageFrame>
  );
}
