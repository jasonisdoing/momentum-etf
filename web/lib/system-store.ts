import { fetchFastApiJson } from "./internal-api";

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

type SystemLastRunInfo = {
  status?: string | null;
  display?: string | null;
};

type SystemData = {
  summary_rows: SystemSummaryRow[];
  schedule_rows: SystemScheduleRow[];
  schedule_note: string;
  running_jobs: string[];
  last_run_by_job?: Record<string, SystemLastRunInfo>;
};

type SystemAction =
  | "data_aggregate"
  | "cache_refresh"
  | "market_hours_analysis"
  | "metadata_updater"
  | "asset_summary"
  | "us_index_constituents";

export async function loadSystemData(): Promise<SystemData> {
  return fetchFastApiJson<SystemData>("/internal/system");
}

export async function triggerSystemAction(action: SystemAction): Promise<string> {
  const payload = await fetchFastApiJson<{ message: string }>("/internal/system", {
    method: "POST",
    body: JSON.stringify({ action }),
  });
  return payload.message;
}

export type { SystemAction, SystemData, SystemLastRunInfo, SystemScheduleRow, SystemSummaryRow };
