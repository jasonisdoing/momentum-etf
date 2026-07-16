import { fetchFastApiJson } from "./internal-api";

type SystemPoolRow = {
  id: string;
  order: number;
  pool: string;
  ticker_type: string;
  country_code: string;
  stock_count: number;
  rising_count: number;
  rising_ratio: number;
  etf_count: number;
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

type SystemRunningJobDetail = {
  started_at?: string | null;
  estimated_seconds?: number | null;
  elapsed_seconds?: number | null;
  remaining_seconds?: number | null;
  estimated_display?: string | null;
  remaining_display?: string | null;
};

type SystemData = {
  pool_rows: SystemPoolRow[];
  schedule_rows: SystemScheduleRow[];
  schedule_note: string;
  running_jobs: string[];
  last_run_by_job?: Record<string, SystemLastRunInfo>;
  running_job_details?: Record<string, SystemRunningJobDetail>;
};

type SystemAction =
  | "data_aggregate"
  | "cache_refresh"
  | "market_hours_analysis"
  | "metadata_updater"
  | "asset_summary"
  | "us_market_stocks"
  | "live_24h_slack"
  | "leverage_switch"
  | "leverage_tune";

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

export type {
  SystemAction,
  SystemData,
  SystemLastRunInfo,
  SystemRunningJobDetail,
  SystemScheduleRow,
  SystemPoolRow,
};
