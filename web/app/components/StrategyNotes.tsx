"use client";

import { useState } from "react";

/** 섹션 상단의 접이식 전략 설명 — 기본은 접힘.
 *
 *  신고가 화면의 '진입 후보' 범례와 같은 모양(토글 줄 + 회색 그리드 패널)을 공용화했다.
 *  전략 3개 화면의 현재 상태·백테스트 섹션이 규칙·로직·주의점을 여기에 담는다. */
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
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
            gap: "10px 20px",
            padding: "12px 14px",
            marginBottom: 10,
            borderRadius: 8,
            background: "var(--bs-secondary-bg, #f1f5f9)",
          }}
        >
          {items.map((item) => (
            <div key={item.title} style={{ fontSize: "var(--fs-sm)", lineHeight: 1.5 }}>
              <strong>{item.title}</strong>
              <div style={{ color: "var(--text-muted)" }}>{item.body}</div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
