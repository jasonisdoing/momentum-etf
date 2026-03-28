import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";

import { loadAccountNoteData } from "./note-store";
import { getWorkspaceRoot, resolvePythonBinary } from "./python-runtime";

const execFileAsync = promisify(execFile);
type SummaryData = Awaited<ReturnType<typeof loadAccountNoteData>>;

export async function loadSummaryPageData(requestedAccountId?: string): Promise<SummaryData> {
  return loadAccountNoteData(requestedAccountId);
}

export async function generateAiSummary(accountIdRaw: string) {
  const accountId = String(accountIdRaw || "").trim().toLowerCase();
  if (!accountId) {
    throw new Error("account_id가 필요합니다.");
  }

  const root = getWorkspaceRoot();
  const pythonBinary = await resolvePythonBinary();
  const scriptPath = path.join(root, "scripts", "generate_ai_summary.py");
  const { stdout, stderr } = await execFileAsync(pythonBinary, [scriptPath, "--account", accountId], {
    cwd: root,
    env: process.env,
    maxBuffer: 20 * 1024 * 1024,
  });

  const raw = String(stdout || "").trim();
  if (!raw) {
    throw new Error(stderr?.trim() || "AI용 요약 결과가 비어 있습니다.");
  }

  const payload = JSON.parse(raw) as {
    error?: string;
    text?: string;
    warnings?: string[];
    memo_content?: string;
    account_id?: string;
  };

  if (payload.error) {
    throw new Error(payload.error);
  }

  return {
    account_id: payload.account_id ?? accountId,
    text: String(payload.text ?? ""),
    warnings: Array.isArray(payload.warnings) ? payload.warnings.map((item) => String(item)) : [],
    memo_content: String(payload.memo_content ?? ""),
  };
}
