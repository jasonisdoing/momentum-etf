import fs from "node:fs";
import path from "node:path";
import { execFileSync, spawn } from "node:child_process";

function parseEnvLine(line) {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith("#")) {
    return null;
  }

  const separatorIndex = trimmed.indexOf("=");
  if (separatorIndex <= 0) {
    return null;
  }

  const key = trimmed.slice(0, separatorIndex).trim();
  let value = trimmed.slice(separatorIndex + 1).trim();

  if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
    value = value.slice(1, -1);
  }

  return [key, value];
}

function loadEnvFile(filePath) {
  if (!fs.existsSync(filePath)) {
    return;
  }

  const raw = fs.readFileSync(filePath, "utf-8");
  for (const line of raw.split(/\r?\n/)) {
    const parsed = parseEnvLine(line);
    if (!parsed) {
      continue;
    }
    const [key, value] = parsed;
    process.env[key] = value;
  }
}

function loadAppEnv() {
  const candidates = [
    path.join(process.cwd(), ".env"),
    path.join(process.cwd(), ".env.local"),
    path.join(process.cwd(), "..", ".env"),
    path.join(process.cwd(), "..", ".env.local"),
  ];

  for (const candidate of candidates) {
    loadEnvFile(candidate);
  }
}

function requireEnv(name) {
  const value = process.env[name];
  if (!value || !String(value).trim()) {
    throw new Error(`${name} 환경변수가 필요합니다.`);
  }
}

function validateStartEnv() {
  requireEnv("AUTH_SESSION_SECRET");
  requireEnv("GOOGLE_CLIENT_ID");
  requireEnv("GOOGLE_CLIENT_SECRET");
  requireEnv("AUTH_ALLOWED_EMAILS");
  requireEnv("APP_BASE_URL");
}

function listeningPids(port) {
  try {
    const out = execFileSync("lsof", ["-ti", `tcp:${port}`, "-sTCP:LISTEN"], {
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "ignore"],
    });
    return [...new Set(out.split(/\s+/).map(Number).filter((pid) => Number.isInteger(pid) && pid > 0 && pid !== process.pid))];
  } catch {
    return [];
  }
}

async function freePort(port) {
  const pids = listeningPids(port);
  if (pids.length === 0) {
    return;
  }

  console.log(`${port} 포트를 사용 중인 기존 프로세스를 종료합니다: PID ${pids.join(", ")}`);
  for (const pid of pids) {
    try {
      process.kill(pid, "SIGTERM");
    } catch {
      // 이미 종료된 프로세스는 무시
    }
  }

  for (let i = 0; i < 15; i += 1) {
    if (listeningPids(port).length === 0) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  for (const pid of listeningPids(port)) {
    try {
      process.kill(pid, "SIGKILL");
    } catch {
      // 이미 종료된 프로세스는 무시
    }
  }
}

async function main() {
  const nextCommand = process.argv[2];
  const nextArgs = process.argv.slice(3);

  if (!nextCommand) {
    throw new Error("next 실행 명령이 필요합니다.");
  }

  loadAppEnv();

  if (nextCommand === "start") {
    validateStartEnv();
  }

  const finalArgs = [...nextArgs];
  if (nextCommand === "dev") {
    await freePort(3000);
    if (!finalArgs.includes("-p") && !finalArgs.includes("--port")) {
      finalArgs.push("-p", "3000");
    }
  }

  const child = spawn("next", [nextCommand, ...finalArgs], {
    stdio: "inherit",
    shell: true,
    env: process.env,
  });

  child.on("exit", (code) => {
    process.exit(code ?? 0);
  });
}

void main();
