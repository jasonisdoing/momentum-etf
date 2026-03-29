import type { ReactNode } from "react";

type PageFrameProps = {
  title: string;
  children: ReactNode;
  fullHeight?: boolean;
  fullWidth?: boolean;
};

export function PageFrame({ title, children, fullHeight = false, fullWidth = false }: PageFrameProps) {
  const frameClassName = [
    "pageFrame",
    fullHeight ? "pageFrameFullHeight" : "",
    fullWidth ? "pageFrameFullWidth" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <main className={frameClassName}>
      <div className="pageFrameTitle">
        <h1>{title}</h1>
      </div>
      <div className={fullHeight ? "pageFrameBody pageFrameBodyFullHeight" : "pageFrameBody"}>{children}</div>
    </main>
  );
}
