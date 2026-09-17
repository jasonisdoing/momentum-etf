"use client";

import { useEffect, useState } from "react";

import { GridToolbarButton } from "../components/GridToolbarButton";
import { useToast } from "../components/ToastProvider";

/** 계좌 메모(/api/note) — 자식 테이블 아래 접이 섹션.

    부모 요약이 갱신되면 패널이 통째로 리마운트되므로(정렬 보존과 같은 이유),
    펼침 상태와 작성 중 초안은 모듈 맵에 계좌별로 보존한다(새로고침하면 사라짐). */
const OPEN_BY_ACCOUNT = new Map<string, boolean>();
const DRAFT_BY_ACCOUNT = new Map<string, string>();

function formatNoteUpdatedAt(value: string | null): string {
  if (!value) return "아직 저장된 메모가 없습니다.";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

export function AccountMemoSection({
  accountId,
  variant = "collapsible",
}: {
  accountId: string;
  /** "side" 는 모달 오른쪽 세로 패널 — 항상 펼쳐져 남은 높이를 채운다. */
  variant?: "collapsible" | "side";
}) {
  const toast = useToast();
  const [open, setOpen] = useState(() => OPEN_BY_ACCOUNT.get(accountId) ?? false);
  const [memo, setMemo] = useState(() => DRAFT_BY_ACCOUNT.get(accountId) ?? "");
  const [savedMemo, setSavedMemo] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    void (async () => {
      try {
        const resp = await fetch(`/api/note?account=${encodeURIComponent(accountId)}`, { cache: "no-store" });
        const data = (await resp.json()) as { content?: string; updated_at?: string; error?: string };
        if (!alive || !resp.ok || data.error) return;
        const content = data.content ?? "";
        setSavedMemo(content);
        setUpdatedAt(data.updated_at ?? null);
        // 리마운트로 남은 초안이 있으면 그것을 유지한다 — 없을 때만 저장본을 채운다.
        if (!DRAFT_BY_ACCOUNT.has(accountId)) setMemo(content);
      } finally {
        if (alive) setLoaded(true);
      }
    })();
    return () => {
      alive = false;
    };
  }, [accountId]);

  const toggle = () => {
    const next = !open;
    OPEN_BY_ACCOUNT.set(accountId, next);
    setOpen(next);
  };

  const changeMemo = (value: string) => {
    DRAFT_BY_ACCOUNT.set(accountId, value);
    setMemo(value);
  };

  const save = async () => {
    try {
      setSaving(true);
      const resp = await fetch("/api/note", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account_id: accountId, content: memo }),
      });
      const data = (await resp.json()) as { updated_at?: string; error?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? "메모 저장에 실패했습니다.");
      setSavedMemo(memo);
      setUpdatedAt(data.updated_at ?? null);
      DRAFT_BY_ACCOUNT.delete(accountId);
      toast.success("메모 저장 완료");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "메모 저장에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  };

  const dirty = memo !== savedMemo;
  if (variant === "side") {
    return (
      <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 8,
            padding: "3px 8px",
            marginBottom: 6,
            background: "rgba(240, 180, 41, 0.14)",
            borderRadius: 6,
            fontSize: "var(--fs-sm)",
          }}
        >
          <span style={{ fontWeight: 700 }}>
            메모
            <span style={{ fontWeight: 400, marginLeft: 6, color: "var(--text-muted)" }}>
              {!loaded ? "" : savedMemo ? `저장 ${formatNoteUpdatedAt(updatedAt)}` : "없음"}
              {dirty ? " · 저장 안 됨" : ""}
            </span>
          </span>
          <GridToolbarButton variant="save" disabled={saving || !dirty} onClick={() => void save()}>
            {saving ? "저장 중..." : "저장"}
          </GridToolbarButton>
        </div>
        <textarea
          className="form-control"
          style={{ fontSize: "var(--fs-base)", flex: "1 1 auto", minHeight: 0, resize: "none" }}
          placeholder="이 계좌에 대한 투자 전략이나 주의사항을 메모하세요."
          value={memo}
          onChange={(e) => changeMemo(e.target.value)}
        />
      </div>
    );
  }
  return (
    // 패널(flex column, overflow hidden) 안에서 메모가 줄어들거나 잘리지 않게 고정한다 —
    // 높이 부족분은 위의 그리드 래퍼(flex 1, min-height 0)가 흡수한다.
    <div style={{ flexShrink: 0, borderTop: "1px solid rgba(148,163,184,0.3)", paddingTop: 3 }}>
      {/* 배경은 종목 메모 칸(.appMemoCell)과 같은 노랑 — 메모 자리임이 한눈에 보이게. */}
      <button
        type="button"
        onClick={toggle}
        style={{
          border: "none",
          background: "rgba(240, 180, 41, 0.14)",
          borderRadius: 6,
          padding: "3px 8px",
          width: "100%",
          cursor: "pointer",
          color: "var(--text-normal, #1f2937)",
          fontSize: "var(--fs-sm)",
          fontWeight: 700,
          display: "flex",
          alignItems: "center",
          gap: 6,
        }}
      >
        <span>{open ? "▾" : "▸"} 메모</span>
        <span style={{ fontWeight: 400 }}>
          {!loaded ? "" : savedMemo ? `· 저장 ${formatNoteUpdatedAt(updatedAt)}` : "· 없음"}
          {dirty ? " · 저장 안 됨" : ""}
        </span>
      </button>
      {open ? (
        <div style={{ marginTop: 4 }}>
          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 4, height: 30 }}>
            <GridToolbarButton variant="save" disabled={saving || !dirty} onClick={() => void save()}>
              {saving ? "저장 중..." : "메모 저장"}
            </GridToolbarButton>
          </div>
          <textarea
            className="form-control"
            style={{ fontSize: "var(--fs-base)", minHeight: 224 }}
            placeholder="이 계좌에 대한 투자 전략이나 주의사항을 메모하세요."
            value={memo}
            onChange={(e) => changeMemo(e.target.value)}
          />
        </div>
      ) : null}
    </div>
  );
}
