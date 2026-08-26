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
  /** 이 자리가 언제 지워지는지 — 이 화면의 본래 질문. */
  deleted_with: string;
  deleted_with_label: string;
  /** 컬렉션 통째가 아니라 문서 안쪽 자리(키·배열 항목·참조)인 경우. */
  is_place?: boolean;
  /** 「건수」의 단위 — 컬렉션은 문서, 문서 안쪽 자리는 키·항목. */
  count_unit: string;
  /** 소유자별로 무엇으로 나뉘는지 — 소유자 값 또는 구분 필드명. */
  split_by: string;
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
  totals: { collections: number; places: number; count: number; size: number; index_size: number };
  pools: string[];
  accounts: string[];
};

/** 분류 배지 색 — 삭제 정책의 성격을 색으로 구분한다.
 *  소유자별(파랑 계열) · 보존(초록) · 설정(보라) · 재생성 가능(회색) · 미분류(빨강). */
const DELETED_WITH_TONE: Record<string, { bg: string; fg: string }> = {
  pool: { bg: "#dbeafe", fg: "#1e40af" },
  account: { bg: "#e0e7ff", fg: "#3730a3" },
  keep: { bg: "#f1f5f9", fg: "#334155" },
};

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

  // 정렬 축은 **삭제 시점** — 이 화면의 질문 그대로 읽히게 한다.
  // (미분류는 등록 누락을 드러내는 게 목적이라 맨 위에 둔다.)
  const gridRows = useMemo<TableRow[]>(() => {
    if (!data) return [];
    const deleteOrder: Record<string, number> = { pool: 0, account: 1, keep: 2 };
    const categoryOrder = new Map(data.category_order.map((key, index) => [key, index]));
    const rest = [...data.rows].sort((a, b) => {
      const byDelete = (deleteOrder[a.deleted_with] ?? 9) - (deleteOrder[b.deleted_with] ?? 9);
      if (byDelete !== 0) return byDelete;
      const byCategory = (categoryOrder.get(a.category) ?? 99) - (categoryOrder.get(b.category) ?? 99);
      if (byCategory !== 0) return byCategory;
      // 컬렉션을 먼저, 그 안쪽 자리를 뒤에. 같은 종류면 큰 것부터.
      if (Boolean(a.is_place) !== Boolean(b.is_place)) return a.is_place ? 1 : -1;
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
        field: "deleted_with_label",
        headerName: "삭제 트리거",
        width: 116,
        headerTooltip: "무엇을 지울 때 이 자리가 함께 지워지는지입니다.",
        cellRenderer: (params: { data?: TableRow }) => {
          const row = params.data;
          if (!row) return null;
          const tone = DELETED_WITH_TONE[row.deleted_with] ?? DELETED_WITH_TONE.keep;
          return (
            <span
              title={row.policy}
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
              {row.deleted_with_label}
            </span>
          );
        },
      },
      {
        field: "category_label",
        headerName: "성격",
        width: 96,
        headerTooltip: "무슨 성격의 데이터인지. 삭제 트리거가 '없음' 인 행에서 특히 중요합니다.",
        cellRenderer: (params: { data?: TableRow }) => {
          const row = params.data;
          if (!row) return null;
          const tone = CATEGORY_TONE[row.category] ?? CATEGORY_TONE.reference;
          // 집계·이력은 '절대 지우면 안 되는 것' 이라 유일하게 배지로 강조한다.
          if (row.category !== "aggregate" && row.category !== "unclassified") {
            return <span style={{ color: "var(--text-muted)" }}>{row.category_label}</span>;
          }
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
        headerName: "위치",
        width: 420,
        headerTooltip: "컬렉션 이름. `›` 가 있으면 컬렉션이 아니라 그 문서 안쪽 자리입니다.",
        cellStyle: { fontWeight: 600 },
        cellRenderer: (params: { data?: TableRow; value?: string }) => (
          <span>
            {params.value}
            {params.data?.missing ? <span style={{ color: "var(--text-muted)" }}> (DB 에 없음)</span> : null}
          </span>
        ),
      },
      {
        field: "split_by",
        headerName: "구분",
        width: 132,
        headerTooltip:
          "소유자별로 무엇으로 나뉘는지. 컬렉션이 통째로 나뉘면 소유자 이름, 한 컬렉션 안에서 나뉘면 그 필드명입니다.",
        cellRenderer: (params: { value?: string }) =>
          params.value ? <code>{params.value}</code> : <span style={{ color: "var(--text-muted)" }}>-</span>,
      },
      {
        field: "count",
        headerName: "건수",
        width: 112,
        type: "numericColumn",
        headerTooltip: "단위가 행마다 다릅니다 — 컬렉션은 문서 수, 문서 안쪽 자리는 키·항목 수입니다.",
        cellRenderer: (params: { data?: TableRow; value?: number }) => {
          const row = params.data;
          if (!row) return null;
          if (!row.count) return <span style={{ color: "var(--text-muted)" }}>-</span>;
          return (
            <span>
              {row.count.toLocaleString("ko-KR")}
              <span style={{ color: "var(--text-muted)" }}>{` ${row.count_unit}`}</span>
            </span>
          );
        },
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
        cellRenderer: (params: { data?: TableRow }) => {
          const row = params.data;
          if (!row) return null;
          const item = orphanByLocation.get(row.name);
          if (item) {
            return (
              <span style={{ color: "#991b1b", fontWeight: 700 }} title={item.detail}>
                {item.count.toLocaleString("ko-KR")}
              </span>
            );
          }
          // 소유자 개념이 있는 자리는 '0'(깨끗함), 없는 자리는 '-'(해당 없음)로 구분한다.
          const hasOwner = row.deleted_with !== "keep";
          return <span style={{ color: "var(--text-muted)" }}>{hasOwner ? "0" : "-"}</span>;
        },
      },
      {
        field: "purpose",
        headerName: "설명",
        flex: 1,
        minWidth: 360,
        // 소유자 필드에 덧붙일 참고사항(`owner_note`)도 여기 이어 쓴다 — 아이콘 뒤에 숨기면
        // 있는 줄도 모른다. 잘리면 마우스를 올려 전체를 본다.
        valueGetter: (params) =>
          [params.data?.purpose, params.data?.owner_note].filter(Boolean).join("  ·  "),
        cellRenderer: (params: { data?: TableRow }) => {
          const row = params.data;
          if (!row) return null;
          return (
            <span>
              {row.purpose}
              {row.owner_note ? (
                <span style={{ color: "var(--text-muted)" }}>{`  ·  ${row.owner_note}`}</span>
              ) : null}
            </span>
          );
        },
      },
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
        <span>안쪽 자리:</span>
        <span className="appHeaderMetricValue">{data.totals.places}곳</span>
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
