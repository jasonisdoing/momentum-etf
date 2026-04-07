"use client";

import { useEffect, useState, useTransition } from "react";
import { iconSetQuartzBold, themeQuartz } from "ag-grid-community";
import type { ColDef } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";
import { useToast } from "../components/ToastProvider";

type SystemSummaryRow = {
  category: string;
  count: number;
  target: string;
};

type SystemScheduleRow = {
  job: string;
  target: string;
  cadence: string;
  command: string;
};

type SystemResponse = {
  summary_rows?: SystemSummaryRow[];
  schedule_rows?: SystemScheduleRow[];
  schedule_note?: string;
  error?: string;
};

type SystemSummaryGridRow = SystemSummaryRow & { id: string };
type SystemScheduleGridRow = SystemScheduleRow & { id: string };

function formatCount(value: number): string {
  return new Intl.NumberFormat("ko-KR").format(value);
}

const appGridTheme = themeQuartz
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

const summaryColumns: ColDef<SystemSummaryGridRow>[] = [
  {
    field: "count",
    headerName: "순서",
    minWidth: 72,
    width: 72,
    type: "rightAligned",
    cellRenderer: (params: { value: number }) => formatCount(params.value),
  },
  { field: "target", headerName: "계좌 ID", minWidth: 200, flex: 1 },
  { field: "category", headerName: "계좌", minWidth: 200, flex: 1.2 },
];

const scheduleColumns: ColDef<SystemScheduleGridRow>[] = [
  { field: "job", headerName: "작업", minWidth: 180, flex: 1 },
  { field: "target", headerName: "대상", minWidth: 180, flex: 1 },
  { field: "cadence", headerName: "자동 주기", minWidth: 140, width: 140 },
  {
    field: "command",
    headerName: "실행 명령",
    minWidth: 320,
    flex: 1.6,
    cellRenderer: (params: { value: string }) => (
      <span className="appCodeText">{params.value}</span>
    ),
  },
];

export function SystemManager() {
  const [summaryRows, setSummaryRows] = useState<SystemSummaryRow[]>([]);
  const [scheduleRows, setScheduleRows] = useState<SystemScheduleRow[]>([]);
  const [scheduleNote, setScheduleNote] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runningAction, setRunningAction] = useState<"asset_summary" | null>(null);
  const [isPending, startTransition] = useTransition();
  const toast = useToast();
  const summaryGridRows: SystemSummaryGridRow[] = summaryRows.map((row) => ({ ...row, id: row.category }));
  const scheduleGridRows: SystemScheduleGridRow[] = scheduleRows.map((row) => ({ ...row, id: row.job }));

  useEffect(() => {
    let alive = true;

    async function load() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch("/api/system", { cache: "no-store" });
        const payload = (await response.json()) as SystemResponse;
        if (!response.ok) {
          throw new Error(payload.error ?? "시스템정보 데이터를 불러오지 못했습니다.");
        }

        if (!alive) {
          return;
        }

        setSummaryRows(payload.summary_rows ?? []);
        setScheduleRows(payload.schedule_rows ?? []);
        setScheduleNote(payload.schedule_note ?? "");
      } catch (loadError) {
        if (alive) {
          setError(loadError instanceof Error ? loadError.message : "시스템정보 데이터를 불러오지 못했습니다.");
        }
      } finally {
        if (alive) {
          setLoading(false);
        }
      }
    }

    load();
    return () => {
      alive = false;
    };
  }, []);

  function handleAction(action: "asset_summary") {
    startTransition(async () => {
      try {
        setError(null);
        setRunningAction(action);
        const response = await fetch("/api/system", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action }),
        });
        const payload = (await response.json()) as { message?: string; error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "시스템 작업 실행에 실패했습니다.");
        }
        toast.success(String(payload.message ?? "[시스템-정보] 작업 시작"));
      } catch (actionError) {
        setError(actionError instanceof Error ? actionError.message : "시스템 작업 실행에 실패했습니다.");
      } finally {
        setRunningAction(null);
      }
    });
  }

  return (
    <div className="appPageStack">
      {error ? (
        <div className="appBannerStack">
          <div className="bannerError">{error}</div>
        </div>
      ) : null}

      <section className="appSection">
        <div className="card appCard">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <span className="appHeaderMetricValue">수동 실행</span>
              </div>
              <div className="appMainHeaderRight">
                <span className="appHeaderSubtle">메타데이터/가격 캐시는 종목별로 관리됩니다. (종목 편집 모달에서 새로고침 가능)</span>
              </div>
            </div>
          </div>
          <div className="card-header appActionHeader bg-light-subtle border-top">
            <div className="appActionHeaderInner">
                <button
                  className="btn btn-primary btn-sm px-3 fw-bold"
                  type="button"
                  onClick={() => handleAction("asset_summary")}
                  disabled={isPending}
                >
                  {runningAction === "asset_summary" ? "실행 중..." : "전체 자산 요약 알림 전송"}
                </button>
            </div>
          </div>
          <div className="card-body appCardBody" />
        </div>
      </section>

      <section className="appSection">
        <div className="card appCard">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <span className="appHeaderMetricValue">계좌 요약</span>
              </div>
            </div>
          </div>
          <div className="card-body appCardBodyTight">
            <AppAgGrid
              rowData={summaryGridRows}
              columnDefs={summaryColumns}
              loading={loading}
              minHeight="18rem"
              theme={appGridTheme}
              gridOptions={{ suppressMovableColumns: true, domLayout: "autoHeight" }}
            />
          </div>
        </div>
      </section>

      <section className="appSection">
        <div className="card appCard">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <span className="appHeaderMetricValue">자동 작업</span>
              </div>
            </div>
          </div>
          <div className="card-body appCardBodyTight">
            <AppAgGrid
              rowData={scheduleGridRows}
              columnDefs={scheduleColumns}
              loading={loading}
              minHeight="18rem"
              theme={appGridTheme}
              gridOptions={{ suppressMovableColumns: true, domLayout: "autoHeight" }}
            />
            {scheduleNote ? <div className="tableFooterMeta">{scheduleNote}</div> : null}
          </div>
        </div>
      </section>
    </div>
  );
}
