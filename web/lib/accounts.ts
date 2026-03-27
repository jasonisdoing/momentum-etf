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

const ACCOUNT_DIR_PATTERN = /^(?<order>\d+)_(?<account>[a-z0-9_]+)$/;

function getAccountsRoot(): string {
  return path.join(process.cwd(), "zaccounts");
}

export async function loadAccountConfigs(): Promise<AccountConfig[]> {
  const entries = await fs.readdir(getAccountsRoot(), { withFileTypes: true });
  const configs: AccountConfig[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith(".") || entry.name.startsWith("_")) {
      continue;
    }

    const match = ACCOUNT_DIR_PATTERN.exec(entry.name);
    if (!match?.groups) {
      continue;
    }

    const configPath = path.join(getAccountsRoot(), entry.name, "config.json");
    const raw = await fs.readFile(configPath, "utf-8");
    const parsed = JSON.parse(raw) as Record<string, unknown>;

    configs.push({
      account_id: match.groups.account,
      order: Number(match.groups.order),
      name: String(parsed.name ?? match.groups.account),
      icon: String(parsed.icon ?? ""),
      country_code: String(parsed.country_code ?? "").trim().toLowerCase(),
      currency: String(parsed.currency ?? "").trim().toUpperCase(),
    });
  }

  configs.sort((left, right) => left.order - right.order);
  return configs;
}

export type { AccountConfig };
