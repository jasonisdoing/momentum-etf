import fs from "node:fs";
import path from "node:path";
import { MongoClient } from "mongodb";

declare global {
  // eslint-disable-next-line no-var
  var __momentumMongoClientPromise: Promise<MongoClient> | undefined;
}

function readRootEnvValue(key: string): string | null {
  const envPath = path.join(process.cwd(), "..", ".env");
  if (!fs.existsSync(envPath)) {
    return null;
  }

  const raw = fs.readFileSync(envPath, "utf-8");
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }
    const separatorIndex = trimmed.indexOf("=");
    if (separatorIndex <= 0) {
      continue;
    }
    const envKey = trimmed.slice(0, separatorIndex).trim();
    if (envKey !== key) {
      continue;
    }
    const envValue = trimmed.slice(separatorIndex + 1).trim();
    return envValue.replace(/^['"]|['"]$/g, "");
  }

  return null;
}

function resolveEnv(key: string): string {
  const fromProcess = process.env[key]?.trim();
  if (fromProcess) {
    return fromProcess;
  }
  const fromFile = readRootEnvValue(key)?.trim() ?? "";
  return fromFile;
}

function getMongoUri(): string {
  const uri = resolveEnv("MONGO_DB_CONNECTION_STRING");
  if (!uri) {
    throw new Error(
      "MongoDB 연결 정보가 없습니다. MONGO_DB_CONNECTION_STRING 환경변수를 설정하세요.",
    );
  }
  return uri;
}

function getMongoDbName(): string {
  return resolveEnv("MONGO_DB_NAME") || "momentum_etf_db";
}

export async function getMongoClient(): Promise<MongoClient> {
  if (!global.__momentumMongoClientPromise) {
    const client = new MongoClient(getMongoUri());
    global.__momentumMongoClientPromise = client.connect();
  }
  return global.__momentumMongoClientPromise;
}

export async function getMongoDb() {
  const client = await getMongoClient();
  return client.db(getMongoDbName());
}
