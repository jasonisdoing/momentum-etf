import { MongoClient } from "mongodb";

declare global {
  // eslint-disable-next-line no-var
  var __momentumMongoClientPromise: Promise<MongoClient> | undefined;
}

function getMongoUri(): string {
  const value = process.env.MONGO_DB_CONNECTION_STRING?.trim();
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
