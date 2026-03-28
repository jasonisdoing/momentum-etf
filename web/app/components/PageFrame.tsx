import type { ReactNode } from "react";

type PageFrameProps = {
  title: string;
  children: ReactNode;
  fullHeight?: boolean;
};

export function PageFrame({ title, children, fullHeight = false }: PageFrameProps) {
  return (
    <main className={fullHeight ? "pageFrame pageFrameFullHeight" : "pageFrame"}>
      <div className="pageFrameTitle">
        <h1>{title}</h1>
      </div>
      <div className={fullHeight ? "pageFrameBody pageFrameBodyFullHeight" : "pageFrameBody"}>{children}</div>
    </main>
  );
}
