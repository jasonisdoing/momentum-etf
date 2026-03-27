import { NextRequest, NextResponse } from "next/server";

import { saveCashAccounts, type CashAccountUpdate } from "@/lib/cash-store";

export const dynamic = "force-dynamic";

function toNumber(value: unknown, field: string): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    throw new Error(`${field} 값이 올바른 숫자가 아닙니다.`);
  }
  return numeric;
}

function toNullableNumber(value: unknown, field: string): number | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  return toNumber(value, field);
}

function validatePayload(value: unknown): CashAccountUpdate[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error("저장할 계좌 데이터가 없습니다.");
  }

  return value.map((item, index) => {
    if (!item || typeof item !== "object") {
      throw new Error(`${index + 1}번째 계좌 데이터 형식이 올바르지 않습니다.`);
    }

    const row = item as Record<string, unknown>;
    const accountId = String(row.account_id ?? "").trim();
    const cashCurrency = String(row.cash_currency ?? "").trim().toUpperCase();

    if (!accountId) {
      throw new Error(`${index + 1}번째 계좌에 account_id가 없습니다.`);
    }
    if (!cashCurrency) {
      throw new Error(`${accountId} 계좌의 cash_currency가 비어 있습니다.`);
    }

    return {
      account_id: accountId,
      total_principal: toNumber(row.total_principal, `${accountId}.total_principal`),
      cash_balance_krw: toNumber(row.cash_balance_krw, `${accountId}.cash_balance_krw`),
      cash_balance_native: toNullableNumber(row.cash_balance_native, `${accountId}.cash_balance_native`),
      cash_currency: cashCurrency,
      intl_shares_value: toNullableNumber(row.intl_shares_value, `${accountId}.intl_shares_value`),
      intl_shares_change: toNullableNumber(row.intl_shares_change, `${accountId}.intl_shares_change`),
    };
  });
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const payload = validatePayload(body.accounts);
    await saveCashAccounts(payload);
    return NextResponse.json({ ok: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : "자산관리 저장에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
