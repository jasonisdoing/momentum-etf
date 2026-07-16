import { NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

type HoldingsRow = {
  account_id?: string;
  account_name: string;
  currency: string;
  bucket: string;
  bucket_id: number;
  ticker: string;
  name: string;
  quantity: number;
  average_buy_price: string;
  current_price: string;
  pnl_krw: number;
  return_pct: number;
  buy_amount_krw: number;
  valuation_krw: number;
  target_ratio?: number | null;
  memo?: string | null;
  target_quantity?: number | null;
  target_weight_pct?: number | null;
  target_amount?: number | null;
  sort_order: number;
  ticker_type?: string;
  country_code?: string;
  is_etf?: boolean;
};

type TopPickWeightRow = {
  ticker?: string;
  target_quantity?: number | null;
  target_weight_pct?: number | null;
};

function normalizeTicker(value: string | undefined): string {
  return String(value ?? "").trim().toUpperCase().replace(/^ASX:/, "").replace(/^KR:/, "");
}

function normalizeAccountId(value: string | undefined): string {
  return String(value ?? "").trim().toLowerCase();
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const account = searchParams.get("account");

  try {
    const search = account ? `?account=${encodeURIComponent(account)}` : "";
    const payload = await fetchFastApiJson<{
      accounts?: any[];
      account_id?: string;
      rows: HoldingsRow[];
      account_summaries?: Array<{ account_id?: string; top_pick_cash_target_weight_pct?: number | null }>;
    }>(`/internal/holdings${search}`);
    const topPickAccounts = await fetchFastApiJson<{ accounts?: string[] }>("/internal/top-pick/accounts");
    const requestedAccountIds = new Set(
      (payload.account_summaries ?? []).map((item) => normalizeAccountId(item.account_id)).filter(Boolean),
    );
    const targetAccountIds = (topPickAccounts.accounts ?? []).filter((accountId) =>
      requestedAccountIds.has(normalizeAccountId(accountId)),
    );
    const weightPayloadResults = await Promise.allSettled(
      targetAccountIds.map(async (accountId) => ({
        accountId,
        payload: await fetchFastApiJson<{ rows?: TopPickWeightRow[] }>(
          `/internal/top-pick/weights?account_id=${encodeURIComponent(accountId)}`,
        ),
      })),
    );
    const weightPayloads: Array<{ accountId: string; payload: { rows?: TopPickWeightRow[] } }> = [];
    const topPickTargetErrors: Array<{ account_id: string; error: string }> = [];
    weightPayloadResults.forEach((result, index) => {
      const accountId = targetAccountIds[index];
      if (result.status === "fulfilled") {
        weightPayloads.push(result.value);
        return;
      }
      topPickTargetErrors.push({
        account_id: accountId,
        error: result.reason instanceof Error ? result.reason.message : "탑픽 목표비중을 불러오지 못했습니다.",
      });
    });
    const targetMap = new Map<string, { quantity: number | null; weightPct: number | null }>();
    for (const { accountId, payload: weightPayload } of weightPayloads) {
      for (const row of weightPayload.rows ?? []) {
        targetMap.set(`${normalizeAccountId(accountId)}::${normalizeTicker(row.ticker)}`, {
          quantity: row.target_quantity ?? null,
          weightPct: row.target_weight_pct ?? null,
        });
      }
    }
    return jsonNoStore({
      ...payload,
      rows: payload.rows.map((row) => ({
        ...row,
        target_quantity: targetMap.get(`${normalizeAccountId(row.account_id)}::${normalizeTicker(row.ticker)}`)?.quantity ?? null,
        target_weight_pct: targetMap.get(`${normalizeAccountId(row.account_id)}::${normalizeTicker(row.ticker)}`)?.weightPct ?? null,
      })),
      account_summaries: (payload.account_summaries ?? []).map((summary) => ({
        ...summary,
        top_pick_cash_target_weight_pct:
          targetMap.get(`${normalizeAccountId(summary.account_id)}::__CASH__`)?.weightPct ?? null,
      })),
      top_pick_target_errors: topPickTargetErrors,
    });
  } catch (error) {
    return jsonNoStore(
      { error: error instanceof Error ? error.message : "보유 종목을 불러오지 못했습니다." },
      { status: 400 },
    );
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const account = searchParams.get("account") ?? "";
  const ticker = searchParams.get("ticker") ?? "";

  try {
    const payload = await fetchFastApiJson<{ deleted?: string; error?: string }>(
      `/internal/holdings?account=${encodeURIComponent(account)}&ticker=${encodeURIComponent(ticker)}`,
      { method: "DELETE" },
    );
    return jsonNoStore(payload);
  } catch (error) {
    return jsonNoStore(
      { error: error instanceof Error ? error.message : "종목 삭제에 실패했습니다." },
      { status: 400 },
    );
  }
}

export async function PUT(request: Request) {
  try {
    const body = await request.json();
    const payload = await fetchFastApiJson<{ message?: string; error?: string }>(
      "/internal/cash",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ accounts: [body] }),
      },
    );
    return jsonNoStore(payload);
  } catch (error) {
    return jsonNoStore(
      { error: error instanceof Error ? error.message : "자산 정보 저장에 실패했습니다." },
      { status: 400 },
    );
  }
}

export async function PATCH(request: Request) {
  try {
    const body = await request.json();
    const endpoint = body.action === "reorder" ? "/internal/holdings/order" : "/internal/holdings";
    const payload = await fetchFastApiJson<{ updated?: string; reordered?: number; error?: string }>(
      endpoint,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    );
    return jsonNoStore(payload);
  } catch (error) {
    return jsonNoStore(
      { error: error instanceof Error ? error.message : "종목 수정에 실패했습니다." },
      { status: 400 },
    );
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const action = body.action ?? "create";

    if (action === "validate") {
      const payload = await fetchFastApiJson<{ ticker?: string; name?: string; bucket_id?: number; error?: string }>(
        "/internal/holdings/validate",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account_id: body.account_id,
            ticker: body.ticker,
          }),
        },
      );
      return jsonNoStore(payload);
    }

    const payload = await fetchFastApiJson<{ added?: string; error?: string }>(
      "/internal/holdings",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          account_id: body.account_id,
          ticker: body.ticker,
          quantity: body.quantity,
          average_buy_price: body.average_buy_price,
          target_ratio: body.target_ratio,
        }),
      },
    );
    return jsonNoStore(payload);
  } catch (error) {
    return jsonNoStore(
      { error: error instanceof Error ? error.message : "요청 처리 중 오류가 발생했습니다." },
      { status: 400 },
    );
  }
}

export const dynamic = "force-dynamic";
