"use client";

import type { ButtonHTMLAttributes, ReactNode } from "react";
import { IconCheck, IconPlus, IconTrash } from "@tabler/icons-react";

/**
 * 그리드 상단 툴바의 추가/삭제 버튼 표준.
 *
 * 크기·모양(패딩/폰트/높이/테두리)을 이 컴포넌트 안 자체 스타일에 직접 못 박아, 화면별 전역/부모
 * CSS(`.btn` 등)의 영향을 받지 않는다 — 어느 화면에 놓아도 항상 동일하다. 색상만 앱 테마 변수를
 * 따른다. 화면마다 클래스 문자열을 복붙하지 말고 이 컴포넌트를 재사용한다(AGENTS.md 3-1).
 */
type Variant = "add" | "delete" | "save";

const VARIANT_ICON: Record<Variant, typeof IconPlus> = {
  add: IconPlus,
  delete: IconTrash,
  save: IconCheck,
};

const VARIANT_LABEL: Record<Variant, string> = {
  add: "추가",
  delete: "삭제",
  save: "저장",
};

export function GridToolbarButton({
  variant,
  children,
  className,
  ...rest
}: { variant: Variant; children?: ReactNode } & ButtonHTMLAttributes<HTMLButtonElement>) {
  const Icon = VARIANT_ICON[variant];
  const mergedClassName = className
    ? `gridToolbarBtn gridToolbarBtn--${variant} ${className}`
    : `gridToolbarBtn gridToolbarBtn--${variant}`;
  return (
    <button type="button" className={mergedClassName} {...rest}>
      <Icon size={16} /> {children ?? VARIANT_LABEL[variant]}
      <style jsx>{`
        .gridToolbarBtn {
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
          height: 34px;
          padding: 0 1rem;
          font-size: var(--fs-sm);
          font-weight: 700;
          line-height: 1;
          white-space: nowrap;
          border: 1px solid transparent;
          border-radius: 6px;
          cursor: pointer;
          transition: background-color 0.12s ease, color 0.12s ease, opacity 0.12s ease;
        }
        .gridToolbarBtn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .gridToolbarBtn--add {
          background: var(--tblr-primary, #206bc4);
          color: #fff;
          border-color: var(--tblr-primary, #206bc4);
        }
        .gridToolbarBtn--add:not(:disabled):hover {
          filter: brightness(0.93);
        }
        .gridToolbarBtn--delete {
          background: transparent;
          color: var(--tblr-danger, #d63939);
          border-color: var(--tblr-danger, #d63939);
        }
        .gridToolbarBtn--delete:not(:disabled):hover {
          background: var(--tblr-danger, #d63939);
          color: #fff;
        }
        .gridToolbarBtn--save {
          background: var(--tblr-success, #2fb344);
          color: #fff;
          border-color: var(--tblr-success, #2fb344);
        }
        .gridToolbarBtn--save:not(:disabled):hover {
          filter: brightness(0.93);
        }
      `}</style>
    </button>
  );
}
