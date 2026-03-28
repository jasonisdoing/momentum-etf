import { fetchFastApiJson } from "./internal-api";

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

type SystemData = {
  summary_rows: SystemSummaryRow[];
  schedule_rows: SystemScheduleRow[];
  schedule_note: string;
};

type SystemAction = "meta_all" | "cache_all" | "asset_summary";

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

export type { SystemAction, SystemData, SystemScheduleRow, SystemSummaryRow };
