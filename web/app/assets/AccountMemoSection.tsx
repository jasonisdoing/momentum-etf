"use client";

import { useEffect, useState } from "react";

import { GridToolbarButton } from "../components/GridToolbarButton";
import { useToast } from "../components/ToastProvider";

/** 계좌 메모(/api/note) — 자식 테이블 아래 접이 섹션.

    부모 요약이 갱신되면 패널이 통째로 리마운트되므로(정렬 보존과 같은 이유),
    펼침 상태와 작성 중 초안은 모듈 맵에 계좌별로 보존한다(새로고침하면 사라짐). */
const OPEN_BY_ACCOUNT = new Map<string, boolean>();
const DRAFT_BY_ACCOUNT = new Map<string, string>();

// 부모(자산 관리) 그리드는 자식 패널 행 높이를 공식으로 계산한다 — 메모 펼침이 높이를
// 바꾸므로, 부모가 상태를 읽고(toggle 시점에) 행 높이를 재계산할 수 있게 내보낸다.
const TOGGLE_LISTENERS = new Set<() => void>();

export function isAccountMemoOpen(accountId: string): boolean {
  return OPEN_BY_ACCOUNT.get(accountId) ?? false;
}

/** 접힌 제목 줄/펼친 편집기의 높이(px) — 부모 행 높이 공식에 더한다.
    접힘은 제목 줄만큼만 더한다(모자라면 그리드의 여유 공간이 흡수) — 표 아래 공백을 만들지 않는다.
    펼침 추가분은 편집기 실제 높이(버튼줄 34 + 입력창 224 + 여백)와 **정확히** 맞춘다 —
    크면 남는 높이가 그리드(flex 1)로 흘러 표 아래 공백이 같이 늘어난다. */
export const MEMO_COLLAPSED_HEIGHT = 34;
export const MEMO_OPEN_EXTRA_HEIGHT = 264;

export function subscribeAccountMemoToggle(listener: () => void): () => void {
  TOGGLE_LISTENERS.add(listener);
  return () => {
    TOGGLE_LISTENERS.delete(listener);
  };
}

function formatNoteUpdatedAt(value: string | null): string {
  if (!value) return "아직 저장된 메모가 없습니다.";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

export function AccountMemoSection({ accountId }: { accountId: string }) {
  const toast = useToast();
  const [open, setOpen] = useState(() => OPEN_BY_ACCOUNT.get(accountId) ?? false);
  const [memo, setMemo] = useState(() => DRAFT_BY_ACCOUNT.get(accountId) ?? "");
  const [savedMemo, setSavedMemo] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);

  // 패널이 (다시) 만들어질 때 부모 그리드가 메모 포함 높이로 재계산하게 알린다 —
  // 행 높이는 행 생성 시점 값이 캐시되므로, 마운트마다 한 번 건드린다.
  useEffect(() => {
    TOGGLE_LISTENERS.forEach((listener) => listener());
  }, [accountId]);

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
    TOGGLE_LISTENERS.forEach((listener) => listener());
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
          {/* 높이를 고정한다(리사이즈 금지) — 부모 행 높이 공식(MEMO_OPEN_EXTRA_HEIGHT)과 맞아야
              남는 높이가 표 아래 공백으로 흐르지 않는다. */}
          <textarea
            className="form-control"
            style={{ fontSize: "var(--fs-base)", height: 224, resize: "none" }}
            placeholder="이 계좌에 대한 투자 전략이나 주의사항을 메모하세요."
            value={memo}
            onChange={(e) => changeMemo(e.target.value)}
          />
        </div>
      ) : null}
    </div>
  );
}
