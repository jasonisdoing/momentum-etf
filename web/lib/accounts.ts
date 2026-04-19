import fs from "node:fs/promises";
import path from "node:path";

type AccountConfig = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
  country_code: string;
  currency: string;
};

async function getAccountsConfigPath(): Promise<string> {
  const candidates = [
    path.join(process.cwd(), "accounts.json"),
    path.join(process.cwd(), "..", "accounts.json"),
  ];

  for (const candidate of candidates) {
    try {
      // 개발(`web/`)과 Docker(`/app`)를 모두 지원하기 위해 실제 존재 경로를 우선 사용한다.
      await fs.access(candidate);
      return candidate;
    } catch {
      continue;
    }
  }

  throw new Error("accounts.json 파일을 찾을 수 없습니다.");
}

export async function loadAccountConfigs(): Promise<AccountConfig[]> {
  const configPath = await getAccountsConfigPath();
  const raw = await fs.readFile(configPath, "utf-8");
  const parsed = JSON.parse(raw) as { accounts?: Array<Record<string, unknown>> };
  if (!Array.isArray(parsed.accounts)) {
    throw new Error("accounts.json의 accounts 필드는 배열이어야 합니다.");
  }

  const configs: AccountConfig[] = parsed.accounts.map((entry) => ({
    account_id: String(entry.account_id ?? "").trim().toLowerCase(),
    order: Number(entry.order ?? 0),
    name: String(entry.name ?? "").trim(),
    icon: String(entry.icon ?? ""),
    country_code: String(entry.country_code ?? "").trim().toLowerCase(),
    currency: String(entry.currency ?? "").trim().toUpperCase(),
  }));

  configs.sort((left, right) => left.order - right.order);
  return configs;
}

export type { AccountConfig };
