"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { IconGripVertical } from "@tabler/icons-react";

import { GridToolbarButton } from "../components/GridToolbarButton";
import { PageFrame } from "../components/PageFrame";
import { useToast } from "../components/ToastProvider";

/** 리스트형 메모의 한 줄. `done` 은 할일 리스트에서만 쓰지만, 타입을 오갈 때
 *  체크 상태가 사라지지 않도록 두 타입 모두 그대로 보관한다. */
type MemoItem = { text: string; done: boolean };

type MemoType = "text" | "list" | "todo";

type Memo = {
  id: string;
  type: MemoType;
  title: string;
  /** 텍스트형 본문 */
  content: string;
  /** 리스트형 항목 */
  items: MemoItem[];
  updated_at: string | null;
};

// 메모 타입 토글 — 다른 화면의 세그먼트 토글과 같은 마크업을 쓴다.
const MEMO_TYPES = [
  { key: "text", label: "텍스트" },
  { key: "list", label: "리스트" },
  { key: "todo", label: "할일 리스트" },
] as const;

/** 항목 목록으로 편집하는 타입 (텍스트만 본문 편집기를 쓴다). */
function isListType(type: MemoType): boolean {
  return type === "list" || type === "todo";
}

/** 새 메모의 임시 id — 저장 전까지만 쓰이며 서버로 보내지 않는다. */
const DRAFT_ID = "__draft__";

const hintStyle: React.CSSProperties = { color: "var(--text-muted)", fontSize: "var(--fs-sm)" };

function formatUpdatedAt(value: string | null): string {
  if (!value) return "저장 전";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function MemosClient() {
  const toast = useToast();
  const [memos, setMemos] = useState<Memo[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // 편집 중인 값 — 목록의 원본과 비교해 변경 여부를 판단한다.
  const [draftTitle, setDraftTitle] = useState("");
  const [draftContent, setDraftContent] = useState("");
  const [draftType, setDraftType] = useState<MemoType>("text");
  const [draftItems, setDraftItems] = useState<MemoItem[]>([]);
  // 항목 드래그 재정렬 — 손잡이를 누른 동안에만 draggable 을 켠다.
  // (행 전체를 항상 draggable 로 두면 텍스트 입력칸에서 글자 선택이 안 된다)
  const [handleHeld, setHandleHeld] = useState(false);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);

  const moveItem = useCallback((from: number, to: number) => {
    if (from === to) return;
    setDraftItems((prev) => {
      const next = [...prev];
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      return next;
    });
  }, []);

  const endDrag = useCallback(() => {
    setHandleHeld(false);
    setDragIndex(null);
    setDragOverIndex(null);
  }, []);

  const load = useCallback(
    async (keepId?: string) => {
      setLoading(true);
      try {
        const resp = await fetch("/api/memos", { cache: "no-store" });
        const payload = await resp.json();
        if (!resp.ok) throw new Error(payload?.error ?? "메모를 불러오지 못했습니다.");
        const rows: Memo[] = payload.memos ?? [];
        setLoadError(null);
        setMemos(rows);
        const next = keepId && rows.some((m) => m.id === keepId) ? keepId : (rows[0]?.id ?? null);
        setSelectedId(next);
        const selected = rows.find((m) => m.id === next);
        setDraftTitle(selected?.title ?? "");
        setDraftContent(selected?.content ?? "");
        setDraftType(selected?.type ?? "text");
        setDraftItems(selected?.items ?? []);
      } catch (error) {
        const message = error instanceof Error ? error.message : "메모를 불러오지 못했습니다.";
        setLoadError(message);
        setMemos([]);
        setSelectedId(null);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    void load();
  }, [load]);

  const selected = useMemo(
    () => memos.find((memo) => memo.id === selectedId) ?? null,
    [memos, selectedId],
  );
  const isDraft = selectedId === DRAFT_ID;
  const itemsChanged = (a: MemoItem[], b: MemoItem[]) =>
    a.length !== b.length || a.some((item, i) => item.text !== b[i].text || item.done !== b[i].done);
  const isDirty = isDraft
    ? draftTitle.trim().length > 0 || draftContent.length > 0 || draftItems.length > 0
    : Boolean(selected) &&
      (draftTitle !== selected!.title ||
        draftType !== selected!.type ||
        draftContent !== selected!.content ||
        itemsChanged(draftItems, selected!.items));

  const selectMemo = useCallback(
    (memo: Memo) => {
      if (isDirty && !window.confirm("저장하지 않은 변경이 있습니다. 이동할까요?")) return;
      setSelectedId(memo.id);
      setDraftTitle(memo.title);
      setDraftContent(memo.content);
      setDraftType(memo.type);
      setDraftItems(memo.items);
    },
    [isDirty],
  );

  const addMemo = useCallback(() => {
    if (isDirty && !window.confirm("저장하지 않은 변경이 있습니다. 새 메모를 시작할까요?")) return;
    setMemos((prev) => [
      { id: DRAFT_ID, type: "text", title: "", content: "", items: [], updated_at: null },
      ...prev.filter((m) => m.id !== DRAFT_ID),
    ]);
    setSelectedId(DRAFT_ID);
    setDraftTitle("");
    setDraftContent("");
    setDraftType("text");
    setDraftItems([]);
  }, [isDirty]);

  const save = useCallback(async () => {
    const title = draftTitle.trim();
    if (!title) {
      toast.error("제목을 입력하세요.");
      return;
    }
    setSaving(true);
    try {
      const isNew = selectedId === DRAFT_ID || selectedId === null;
      const resp = await fetch(isNew ? "/api/memos" : `/api/memos/${encodeURIComponent(selectedId!)}`, {
        method: isNew ? "POST" : "PUT",
        headers: { "Content-Type": "application/json" },
        // 두 형식의 값을 모두 보낸다 — 토글로 오가도 반대편 내용이 지워지지 않는다.
        body: JSON.stringify({ type: draftType, title, content: draftContent, items: draftItems }),
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "메모를 저장하지 못했습니다.");
      toast.success("메모를 저장했습니다.");
      await load(payload?.memo?.id ?? selectedId ?? undefined);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "메모를 저장하지 못했습니다.");
    } finally {
      setSaving(false);
    }
  }, [draftContent, draftItems, draftTitle, draftType, load, selectedId, toast]);

  const remove = useCallback(async () => {
    if (!selectedId || selectedId === DRAFT_ID) {
      setMemos((prev) => prev.filter((m) => m.id !== DRAFT_ID));
      setSelectedId(memos.find((m) => m.id !== DRAFT_ID)?.id ?? null);
      return;
    }
    if (!window.confirm(`"${selected?.title ?? ""}" 메모를 삭제할까요?`)) return;
    setDeleting(true);
    try {
      const resp = await fetch(`/api/memos/${encodeURIComponent(selectedId)}`, { method: "DELETE" });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload?.error ?? "메모를 삭제하지 못했습니다.");
      toast.success("메모를 삭제했습니다.");
      await load();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "메모를 삭제하지 못했습니다.");
    } finally {
      setDeleting(false);
    }
  }, [load, memos, selected, selectedId, toast]);

  return (
    <PageFrame title="메모" fullWidth>
      <div className="appPageStack">
        <div className="card appCard">
          <div className="card-body">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft" />
              <div className="appMainHeaderRight">
                {isDirty ? (
                  <span style={{ ...hintStyle, color: "var(--up-color, #d64545)", fontWeight: 700 }}>
                    저장하지 않은 변경
                  </span>
                ) : null}
                <GridToolbarButton variant="add" onClick={addMemo} disabled={saving || deleting} />
                <GridToolbarButton
                  variant="save"
                  onClick={() => void save()}
                  disabled={saving || deleting || !isDirty || selectedId === null}
                />
                <GridToolbarButton
                  variant="delete"
                  onClick={() => void remove()}
                  disabled={saving || deleting || selectedId === null}
                />
              </div>
            </div>
          </div>
        </div>

        {loadError ? (
          <div className="card appCard">
            <div className="card-body" style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
              <span style={{ fontWeight: 700 }}>메모를 불러오지 못했습니다.</span>
              <span style={hintStyle}>{loadError}</span>
              <button type="button" className="btn btn-sm btn-primary" onClick={() => void load()} disabled={loading}>
                {loading ? "다시 시도 중…" : "다시 시도"}
              </button>
            </div>
          </div>
        ) : (
          <div className="memoLayout">
            {/* 왼쪽 — 메모 목록 */}
            <div className="card appCard">
              <div className="card-body">
                <div style={{ ...hintStyle, marginBottom: 8 }}>메모 {memos.length}개</div>
                {memos.length === 0 ? (
                  <div style={{ ...hintStyle, padding: "12px 0" }}>
                    {loading ? "불러오는 중…" : "메모가 없습니다. 신규 메모를 눌러 추가하세요."}
                  </div>
                ) : (
                  <ul className="memoList">
                    {memos.map((memo) => (
                      <li key={memo.id}>
                        <button
                          type="button"
                          className={memo.id === selectedId ? "memoListItem is-active" : "memoListItem"}
                          onClick={() => selectMemo(memo)}
                        >
                          <span className="memoListTitle">
                            {memo.id === DRAFT_ID ? draftTitle || "(제목 없음)" : memo.title || "(제목 없음)"}
                          </span>
                          <span className="memoListMeta">{formatUpdatedAt(memo.updated_at)}</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

            {/* 오른쪽 — 편집 */}
            <div className="card appCard">
              <div className="card-body">
                {selectedId === null ? (
                  <div style={{ ...hintStyle, padding: "12px 0" }}>
                    왼쪽에서 메모를 고르거나 신규 메모를 눌러 시작하세요.
                  </div>
                ) : (
                  <>
                    <div className="appMainHeader">
                      <div className="appMainHeaderLeft">
                        <label className="appLabeledField" style={{ flex: 1, minWidth: 0 }}>
                          <span className="appLabeledFieldLabel">제목</span>
                          <input
                            className="form-control"
                            value={draftTitle}
                            onChange={(e) => setDraftTitle(e.target.value)}
                            placeholder="메모 제목"
                          />
                        </label>
                      </div>
                      <div className="appMainHeaderRight">
                        <label className="appLabeledField">
                          <span className="appLabeledFieldLabel">타입</span>
                          <div
                            className="appSegmentedToggle appSegmentedToggleCompact"
                            role="group"
                            aria-label="메모 타입"
                          >
                            {MEMO_TYPES.map((option) => (
                              <button
                                key={option.key}
                                type="button"
                                className={
                                  draftType === option.key
                                    ? "btn appSegmentedToggleButton is-active"
                                    : "btn appSegmentedToggleButton"
                                }
                                onClick={() => setDraftType(option.key)}
                              >
                                {option.label}
                              </button>
                            ))}
                          </div>
                        </label>
                      </div>
                    </div>
                    {!isListType(draftType) ? (
                      <textarea
                        className="form-control"
                        style={{ minHeight: "24rem", marginTop: 10 }}
                        rows={18}
                        placeholder="내용을 입력하세요. 서식 없는 일반 텍스트로 저장됩니다."
                        value={draftContent}
                        onChange={(e) => setDraftContent(e.target.value)}
                      />
                    ) : (
                      <div className="memoChecklist">
                        {draftItems.length === 0 ? (
                          <div style={{ ...hintStyle, padding: "8px 0" }}>
                            항목이 없습니다. 아래에서 추가하세요.
                          </div>
                        ) : (
                          draftItems.map((item, index) => (
                            <div
                              key={index}
                              className={
                                dragOverIndex === index && dragIndex !== index
                                  ? "memoChecklistRow is-dragover"
                                  : "memoChecklistRow"
                              }
                              draggable={handleHeld}
                              onDragStart={(e) => {
                                setDragIndex(index);
                                e.dataTransfer.effectAllowed = "move";
                              }}
                              onDragOver={(e) => {
                                if (dragIndex === null) return;
                                e.preventDefault();
                                setDragOverIndex(index);
                              }}
                              onDrop={(e) => {
                                e.preventDefault();
                                if (dragIndex !== null) moveItem(dragIndex, index);
                                endDrag();
                              }}
                              onDragEnd={endDrag}
                            >
                              <span
                                className="memoChecklistHandle"
                                aria-label="순서 이동"
                                onMouseDown={() => setHandleHeld(true)}
                                onMouseUp={() => setHandleHeld(false)}
                              >
                                <IconGripVertical size={16} />
                              </span>
                              {draftType === "todo" ? (
                                <input
                                  type="checkbox"
                                  className="form-check-input"
                                  checked={item.done}
                                  aria-label={item.text || `항목 ${index + 1}`}
                                  onChange={(e) =>
                                    setDraftItems((prev) =>
                                      prev.map((row, i) =>
                                        i === index ? { ...row, done: e.target.checked } : row,
                                      ),
                                    )
                                  }
                                />
                              ) : null}
                              <input
                                className={
                                  draftType === "todo" && item.done
                                    ? "form-control memoChecklistText is-done"
                                    : "form-control memoChecklistText"
                                }
                                value={item.text}
                                placeholder="할 일"
                                onChange={(e) =>
                                  setDraftItems((prev) =>
                                    prev.map((row, i) =>
                                      i === index ? { ...row, text: e.target.value } : row,
                                    ),
                                  )
                                }
                              />
                              <button
                                type="button"
                                className="memoChecklistRemove"
                                aria-label="항목 삭제"
                                onClick={() => setDraftItems((prev) => prev.filter((_, i) => i !== index))}
                              >
                                ×
                              </button>
                            </div>
                          ))
                        )}
                        <button
                          type="button"
                          className="btn btn-sm btn-outline-primary"
                          style={{ alignSelf: "flex-start", marginTop: 6 }}
                          onClick={() => setDraftItems((prev) => [...prev, { text: "", done: false }])}
                        >
                          + 항목 추가
                        </button>
                      </div>
                    )}
                    <div style={{ ...hintStyle, marginTop: 6 }}>
                      마지막 저장 {formatUpdatedAt(selected?.updated_at ?? null)}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </PageFrame>
  );
}
