"use client";

import { useEffect, useMemo, useState, useTransition } from "react";

import { useToast } from "../components/ToastProvider";

type AccountOption = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
};

type SummaryPageResponse = {
  accounts?: AccountOption[];
  account_id?: string;
  content?: string;
  updated_at?: string | null;
  error?: string;
};

type SummaryGenerateResponse = {
  account_id?: string;
  text?: string;
  warnings?: string[];
  memo_content?: string;
  error?: string;
};

export function SummaryManager() {
  const [accounts, setAccounts] = useState<AccountOption[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState("");
  const [memoContent, setMemoContent] = useState("");
  const [summaryText, setSummaryText] = useState("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const toast = useToast();

  async function load(accountId?: string) {
    try {
      setLoading(true);
      setError(null);
      const search = accountId ? `?account=${encodeURIComponent(accountId)}` : "";
      const response = await fetch(`/api/summary${search}`, { cache: "no-store" });
      const payload = (await response.json()) as SummaryPageResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "AI용 요약 데이터를 불러오지 못했습니다.");
      }

      const nextAccounts = payload.accounts ?? [];
      setAccounts(nextAccounts);
      setSelectedAccountId(payload.account_id ?? nextAccounts[0]?.account_id ?? "");
      setMemoContent(String(payload.content ?? ""));
      setSummaryText("");
      setWarnings([]);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "AI용 요약 데이터를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  const selectedAccount = useMemo(
    () => accounts.find((account) => account.account_id === selectedAccountId) ?? null,
    [accounts, selectedAccountId],
  );

  function handleAccountChange(nextAccountId: string) {
    void load(nextAccountId);
  }

  function handleGenerate() {
    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/summary", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ account_id: selectedAccountId }),
        });
        const payload = (await response.json()) as SummaryGenerateResponse;
        if (!response.ok) {
          throw new Error(payload.error ?? "AI용 요약 생성에 실패했습니다.");
        }

        setSummaryText(String(payload.text ?? ""));
        setWarnings(Array.isArray(payload.warnings) ? payload.warnings.map((item) => String(item)) : []);
        setMemoContent(String(payload.memo_content ?? memoContent));
        toast.success("[Momentum ETF-AI용 요약] 생성 완료");
      } catch (generateError) {
        setError(generateError instanceof Error ? generateError.message : "AI용 요약 생성에 실패했습니다.");
      }
    });
  }

  if (loading) {
    return (
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBody">
            <p>AI용 요약 데이터를 불러오는 중...</p>
          </div>
        </div>
      </section>
    );
  }

  return (
    <div className="appPageStack">
      {error || warnings.length > 0 ? (
        <div className="appBannerStack">
          {error ? <div className="bannerError">{error}</div> : null}
          {warnings.length > 0 ? <div className="bannerWarn">{warnings.join("\n")}</div> : null}
        </div>
      ) : null}

      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBody">
            <div className="tableToolbar">
              <div className="toolbarActions">
                <select
                  className="field compactField"
                  value={selectedAccountId}
                  onChange={(event) => handleAccountChange(event.target.value)}
                >
                  {accounts.map((account) => (
                    <option key={account.account_id} value={account.account_id}>
                      {account.order}. {account.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="tableMeta">
                {selectedAccount ? (
                  <span>
                    {selectedAccount.icon} {selectedAccount.name}
                  </span>
                ) : null}
              </div>
            </div>

            <div className="summaryReadonlyBlock">
              <div className="summaryReadonlyLabel">계좌 메모 (AI 분석 시 상단 포함)</div>
              <textarea className="bulkTextarea" value={memoContent} readOnly />
            </div>

            <div className="tableToolbar">
              <div className="tableMeta">
                <span>선택 계좌 기준으로 TSV 형식의 AI용 요약 텍스트를 생성한다.</span>
              </div>
              <div className="toolbarActions">
                <button className="primaryButton" type="button" onClick={handleGenerate} disabled={isPending}>
                  {isPending ? "생성 중..." : "AI용 요약 생성"}
                </button>
              </div>
            </div>

            <div className="summaryReadonlyBlock">
              <div className="summaryReadonlyLabel">AI용 요약 결과 (TSV)</div>
              <textarea className="bulkTextarea" value={summaryText} readOnly placeholder="생성 버튼을 눌러 결과를 확인하세요." />
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
