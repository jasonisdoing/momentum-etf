"use client";

import { AppModal } from "./AppModal";

export type HoldingsGroupDraft = { id: string | null; name: string };

export function HoldingsGroupButton({ disabled, onAdd }: { disabled?: boolean; onAdd: () => void }) {
  return (
    <button
      type="button"
      className="btn btn-outline-secondary btn-sm"
      title="표시용 그룹 구분선 추가 — 드래그로 위치를 옮기고, 이름을 클릭해 수정합니다."
      disabled={disabled}
      onMouseDown={(event) => event.stopPropagation()}
      onClick={(event) => { event.stopPropagation(); onAdd(); }}
    >
      그룹
    </button>
  );
}

export function HoldingsGroupName({ name, disabled, onEdit, onDelete }: {
  name: string; disabled?: boolean; onEdit: () => void; onDelete: () => void;
}) {
  return (
    <span className="assetsGroupName">
      <button
        type="button"
        className="btn btn-sm btn-link p-0 text-reset fw-bold text-decoration-none"
        title="이름 수정"
        disabled={disabled}
        onClick={(event) => { event.stopPropagation(); onEdit(); }}
      >
        {name}
      </button>
      <button
        type="button"
        className="btn btn-sm btn-link p-0 assetsGroupDelete"
        title="그룹 삭제"
        aria-label={`${name} 그룹 삭제`}
        disabled={disabled}
        onMouseDown={(event) => event.stopPropagation()}
        onClick={(event) => { event.stopPropagation(); onDelete(); }}
      >
        ✕
      </button>
    </span>
  );
}

export function HoldingsGroupDialog({ draft, onChange, onSave }: {
  draft: HoldingsGroupDraft | null;
  onChange: (draft: HoldingsGroupDraft | null) => void;
  onSave: (draft: HoldingsGroupDraft) => void;
}) {
  const close = () => onChange(null);
  const save = () => {
    const name = draft?.name.trim();
    if (!draft || !name) return;
    onSave({ id: draft.id, name });
    close();
  };
  return (
    <AppModal
      open={draft !== null}
      title={draft?.id ? "그룹 이름 수정" : "그룹 추가"}
      subtitle="표시용 구분선입니다."
      onClose={close}
      footer={(
        <>
          <button type="button" className="btn btn-outline-secondary" onClick={close}>취소</button>
          <button type="button" className="btn btn-primary" disabled={!draft?.name.trim()} onClick={save}>저장</button>
        </>
      )}
    >
      <input
        className="form-control"
        autoFocus
        placeholder="그룹 이름"
        aria-label="그룹 이름"
        value={draft?.name ?? ""}
        onChange={(event) => { if (draft) onChange({ ...draft, name: event.target.value }); }}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.nativeEvent.isComposing) {
            event.preventDefault();
            save();
          }
        }}
      />
    </AppModal>
  );
}
