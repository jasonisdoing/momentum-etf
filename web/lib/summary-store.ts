import { fetchFastApiJson } from "./internal-api";

type SummaryData = {
  accounts: {
    account_id: string;
    order: number;
    name: string;
    icon: string;
  }[];
  account_id: string;
  content: string;
  updated_at: string | null;
};

type SummaryGeneratePayload = {
  account_id: string;
  text: string;
  warnings: string[];
  memo_content: string;
};

export async function loadSummaryPageData(requestedAccountId?: string): Promise<SummaryData> {
  const search = requestedAccountId ? `?account_id=${encodeURIComponent(requestedAccountId)}` : "";
  return fetchFastApiJson<SummaryData>(`/internal/summary${search}`);
}

export async function generateAiSummary(accountIdRaw: string): Promise<SummaryGeneratePayload> {
  const accountId = String(accountIdRaw || "").trim().toLowerCase();
  if (!accountId) {
    throw new Error("account_id가 필요합니다.");
  }
  return fetchFastApiJson<SummaryGeneratePayload>("/internal/summary", {
    method: "POST",
    body: JSON.stringify({ account_id: accountId }),
  });
}
