"use client";

import { useEffect, useMemo, useState, useTransition } from "react";

import { useToast } from "../components/ToastProvider";

type AccountOption = {
  account_id: string;
  order: number;
  name: string;
  icon: string;
};

type NoteResponse = {
  accounts?: AccountOption[];
  account_id?: string;
  content?: string;
  updated_at?: string | null;
  error?: string;
};

function formatUpdatedAt(value: string | null): string {
  if (!value) {
    return "아직 저장된 메모가 없습니다.";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("ko-KR", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function NoteManager() {
  const [accounts, setAccounts] = useState<AccountOption[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState("");
  const [content, setContent] = useState("");
  const [savedContent, setSavedContent] = useState("");
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const toast = useToast();

  const isDirty = content !== savedContent;

  useEffect(() => {
    function handleBeforeUnload(event: BeforeUnloadEvent) {
      if (!isDirty) {
        return;
      }
      event.preventDefault();
      event.returnValue = "";
    }

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [isDirty]);

  async function load(accountId?: string) {
    try {
      setLoading(true);
      setError(null);
      const search = accountId ? `?account=${encodeURIComponent(accountId)}` : "";
      const response = await fetch(`/api/note${search}`, { cache: "no-store" });
      const payload = (await response.json()) as NoteResponse;
      if (!response.ok) {
        throw new Error(payload.error ?? "메모 데이터를 불러오지 못했습니다.");
      }

      setAccounts(payload.accounts ?? []);
      setSelectedAccountId(payload.account_id ?? "");
      const nextContent = String(payload.content ?? "");
      setContent(nextContent);
      setSavedContent(nextContent);
      setUpdatedAt(payload.updated_at ?? null);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "메모 데이터를 불러오지 못했습니다.");
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
    if (isDirty) {
      const confirmed = window.confirm("저장되지 않은 변경이 있습니다. 이동하면 변경 내용이 사라집니다.");
      if (!confirmed) {
        return;
      }
    }
    void load(nextAccountId);
  }

  function handleSave() {
    startTransition(async () => {
      try {
        setError(null);
        const response = await fetch("/api/note", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ account_id: selectedAccountId, content }),
        });
        const payload = (await response.json()) as { updated_at?: string; error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "메모 저장에 실패했습니다.");
        }

        setSavedContent(content);
        setUpdatedAt(payload.updated_at ?? null);
        toast.success("[Momentum ETF-계좌 메모] 메모 저장 완료");
      } catch (saveError) {
        setError(saveError instanceof Error ? saveError.message : "메모 저장에 실패했습니다.");
      }
    });
  }

  if (loading) {
    return (
      <section className="appSection">
        <div className="card appCard">
          <div className="card-body appCardBody">
            <p>메모 데이터를 불러오는 중...</p>
          </div>
        </div>
      </section>
    );
  }

  return (
    <div className="appPageStack">
      {error ? (
        <div className="appBannerStack">
          {error ? <div className="bannerError">{error}</div> : null}
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

            <textarea
              className="bulkTextarea"
              value={content}
              onChange={(event) => setContent(event.target.value)}
              placeholder="이 계좌에 대한 투자 전략이나 주의사항을 메모하세요. AI가 요약할 때 함께 참고합니다."
            />

            <div className="tableToolbar">
              <div className="tableMeta">
                {isDirty ? <span>저장되지 않은 변경이 있습니다.</span> : <span>마지막 저장: {formatUpdatedAt(updatedAt)}</span>}
              </div>
              <div className="toolbarActions">
                <button className="primaryButton" type="button" onClick={handleSave} disabled={isPending}>
                  {isPending ? "저장 중..." : "메모 저장"}
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
