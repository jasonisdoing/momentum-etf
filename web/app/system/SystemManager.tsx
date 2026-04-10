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
  key: string;
  job: string;
  target: string;
  cadence: string;
  command: string;
};

type SystemJobKey =
  | "cache_refresh"
  | "market_hours_analysis"
  | "metadata_updater"
  | "asset_summary";

type SystemResponse = {
  summary_rows?: SystemSummaryRow[];
  schedule_rows?: SystemScheduleRow[];
  schedule_note?: string;
  error?: string;
};

type SystemSummaryGridRow = SystemSummaryRow & { id: string };
type SystemScheduleGridRow = SystemScheduleRow & { id: string; running: boolean };

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
  { field: "job", headerName: "작업", minWidth: 140, width: 180 },
  { field: "target", headerName: "대상", minWidth: 120, width: 140 },
  { field: "cadence", headerName: "자동 주기", minWidth: 140, width: 180 },
  {
    field: "command",
    headerName: "실행 명령 (클릭하여 백그라운드 실행)",
    minWidth: 320,
    flex: 1.6,
    cellStyle: { cursor: "pointer" },
    tooltipValueGetter: (params) =>
      params.data?.running
        ? "실행 중..."
        : `클릭 시 "${params.data?.job ?? ""}" 배치를 백그라운드로 실행합니다.`,
    cellRenderer: (params: { value: string; data?: SystemScheduleGridRow }) => (
      <span className="appCodeText">
        {params.data?.running ? "▶ 실행 중... " : ""}
        {params.value}
      </span>
    ),
  },
];

export function SystemManager({
  onHeaderSummaryChange,
}: {
  onHeaderSummaryChange?: (summary: { accountCount: number; scheduleCount: number }) => void;
}) {
  const [summaryRows, setSummaryRows] = useState<SystemSummaryRow[]>([]);
  const [scheduleRows, setScheduleRows] = useState<SystemScheduleRow[]>([]);
  const [scheduleNote, setScheduleNote] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runningJobs, setRunningJobs] = useState<Set<string>>(new Set());
  const [, startTransition] = useTransition();
  const toast = useToast();
  const summaryGridRows: SystemSummaryGridRow[] = summaryRows.map((row) => ({ ...row, id: row.category }));
  const scheduleGridRows: SystemScheduleGridRow[] = scheduleRows.map((row) => ({
    ...row,
    id: row.key,
    running: runningJobs.has(row.key),
  }));

  useEffect(() => {
    onHeaderSummaryChange?.({
      accountCount: summaryRows.length,
      scheduleCount: scheduleRows.length,
    });
  }, [onHeaderSummaryChange, scheduleRows.length, summaryRows.length]);

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

  function handleTriggerJob(action: SystemJobKey, label: string) {
    if (runningJobs.has(action)) {
      return;
    }
    startTransition(async () => {
      setError(null);
      setRunningJobs((prev) => {
        const next = new Set(prev);
        next.add(action);
        return next;
      });
      try {
        const response = await fetch("/api/system", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action }),
        });
        const payload = (await response.json()) as { message?: string; error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "배치 실행에 실패했습니다.");
        }
        toast.success(String(payload.message ?? `[배치] ${label} 실행 시작`));
      } catch (actionError) {
        const msg = actionError instanceof Error ? actionError.message : "배치 실행에 실패했습니다.";
        setError(msg);
        toast.error(msg);
      } finally {
        // 백그라운드 실행이므로 트리거 성공 후 짧은 시간만 "실행 중" 표시 유지
        setTimeout(() => {
          setRunningJobs((prev) => {
            const next = new Set(prev);
            next.delete(action);
            return next;
          });
        }, 3000);
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
                <span className="appHeaderMetricValue">배치</span>
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
              gridOptions={{
                suppressMovableColumns: true,
                domLayout: "autoHeight",
                onCellClicked: (event) => {
                  if (event.colDef.field !== "command") return;
                  const row = event.data as SystemScheduleGridRow | undefined;
                  if (!row?.key) return;
                  handleTriggerJob(row.key as SystemJobKey, row.job);
                },
              }}
            />
            {scheduleNote ? <div className="tableFooterMeta">{scheduleNote}</div> : null}
          </div>
        </div>
      </section>
    </div>
  );
}
