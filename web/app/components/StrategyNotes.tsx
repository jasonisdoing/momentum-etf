"use client";

import { useState } from "react";

/** 네 전략 화면이 공유하는 접이식 설명 — 제목·본문을 세로로 배치하며 기본은 접힘. */
export type StrategyNoteItem = {
  title: string;
  body: string;
};

export function StrategyNotes({ label = "전략 설명", items }: { label?: string; items: StrategyNoteItem[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        style={{
          color: "var(--text-muted)",
          fontSize: "var(--fs-sm)",
          fontWeight: 700,
          margin: "2px 0 6px",
          padding: 0,
          background: "none",
          border: "none",
          cursor: "pointer",
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
        }}
      >
        <span>{open ? "▾" : "▸"}</span>
        {label}
      </button>
      {open ? (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            marginBottom: 10,
          }}
        >
          {items.map((item, index) => (
            <div
              key={item.title}
              style={{
                fontSize: "var(--fs-sm)",
                lineHeight: 1.7,
                padding: "12px 0",
                borderTop: index > 0 ? "1px solid var(--bs-border-color, #dee2e6)" : undefined,
              }}
            >
              <strong>{item.title}</strong>
              <div style={{ color: "var(--text-muted)", marginTop: 4 }}>{item.body}</div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
