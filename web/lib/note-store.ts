import { loadAccountConfigs } from "./accounts";
import { getMongoDb } from "./mongo";

type AccountOption = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
};

type AccountNoteData = {
  accounts: AccountOption[];
  account_id: string;
  content: string;
  updated_at: string | null;
};

function normalizeAccountId(value: string): string {
  const accountId = String(value || "").trim().toLowerCase();
  if (!accountId) {
    throw new Error("account_id가 필요합니다.");
  }
  return accountId;
}

async function loadAccounts() {
  const accounts = await loadAccountConfigs();
  return accounts.map((account) => ({
    account_id: account.account_id,
    order: account.order,
    name: account.name,
    icon: account.icon,
  }));
}

export async function loadAccountNoteData(requestedAccountId?: string): Promise<AccountNoteData> {
  const accounts = await loadAccounts();
  if (accounts.length === 0) {
    throw new Error("선택 가능한 계좌가 없습니다.");
  }

  const accountId = normalizeAccountId(requestedAccountId || accounts[0].account_id);
  const targetAccount = accounts.find((account) => account.account_id === accountId);
  if (!targetAccount) {
    throw new Error(`계좌 '${accountId}'을(를) 찾을 수 없습니다.`);
  }

  const db = await getMongoDb();
  const doc = await db.collection("account_notes").findOne(
    { account_id: accountId },
    { projection: { _id: 0, content: 1, updated_at: 1 } },
  );

  return {
    accounts,
    account_id: accountId,
    content: String(doc?.content ?? ""),
    updated_at: doc?.updated_at instanceof Date ? doc.updated_at.toISOString() : doc?.updated_at ? String(doc.updated_at) : null,
  };
}

export async function saveAccountNoteData(accountIdRaw: string, contentRaw: string) {
  const accountId = normalizeAccountId(accountIdRaw);
  const content = String(contentRaw ?? "");
  const db = await getMongoDb();
  const updatedAt = new Date();

  const result = await db.collection("account_notes").updateOne(
    { account_id: accountId },
    {
      $set: {
        account_id: accountId,
        content,
        updated_at: updatedAt,
      },
    },
    { upsert: true },
  );

  if (!result.acknowledged) {
    throw new Error("메모 저장이 확인되지 않았습니다.");
  }

  return { updated_at: updatedAt.toISOString() };
}

export type { AccountNoteData };
