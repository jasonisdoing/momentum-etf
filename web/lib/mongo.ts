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

function getMongoUri(): string {
  // 로컬 web 개발 서버에서는 저장소 루트 .env를 직접 읽어 Mongo 연결 문자열을 맞춘다.
  const value = process.env.MONGO_DB_CONNECTION_STRING?.trim() || readRootEnvValue("MONGO_DB_CONNECTION_STRING");
  if (!value) {
    throw new Error("MONGO_DB_CONNECTION_STRING 환경변수가 필요합니다.");
  }
  return value;
}

function getMongoDbName(): string {
  return "momentum_etf_db";
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
