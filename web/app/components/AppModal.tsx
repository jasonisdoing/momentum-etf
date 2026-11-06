import type { ReactNode } from "react";

type AppModalProps = {
  open: boolean;
  title: string;
  subtitle?: string;
  /** "full" 은 화면을 거의 채우는 최대창 — 자식 표처럼 내용 높이가 유동적인 화면용. */
  size?: "md" | "xl" | "full";
  onClose: () => void;
  footer?: ReactNode;
  /** 테두리·배지 색 — 실제 돈이 나가는 창(danger)과 시험 창(success)을 한눈에 가른다. */
  tone?: "danger" | "success";
  /** 제목 옆 배지 글자 — 색만으로 구분하지 않게 `tone` 과 함께 쓴다. */
  badge?: string;
  children: ReactNode;
};

export function AppModal({
  open,
  title,
  subtitle,
  size = "md",
  onClose,
  footer,
  tone,
  badge,
  children,
}: AppModalProps) {
  if (!open) {
    return null;
  }

  const dialogClassName =
    size === "full"
      ? "modal-dialog modal-dialog-centered appModalFull"
      : size === "xl"
        ? "modal-dialog modal-xl modal-dialog-centered"
        : "modal-dialog modal-dialog-centered";

  return (
    <>
      <div className="modal modal-blur fade show d-block" role="dialog" aria-modal="true">
        <div className={dialogClassName}>
          <div className={tone ? `modal-content appModalTone appModalTone-${tone}` : "modal-content"}>
            <div className="modal-header">
              <div>
                <h5 className="modal-title">
                  {title}
                  {badge ? <span className={`appModalBadge appModalBadge-${tone}`}>{badge}</span> : null}
                </h5>
                {subtitle ? <div className="tableMuted">{subtitle}</div> : null}
              </div>
              <button type="button" className="btn-close" aria-label="닫기" onClick={onClose} />
            </div>
            <div className="modal-body">{children}</div>
            {footer ? <div className="modal-footer">{footer}</div> : null}
          </div>
        </div>
      </div>
      <div className="modal-backdrop fade show" onClick={onClose} />
    </>
  );
}
