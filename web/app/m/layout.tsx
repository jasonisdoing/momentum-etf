import type { ReactNode } from "react";

import { MobileMaskProvider } from "./MobileFrame";

/** `/m` 하위 공통 — 금액 가림 상태를 화면 이동 사이에 유지한다(앱을 다시 열면 초기화). */
export default function MobileLayout({ children }: { children: ReactNode }) {
  return <MobileMaskProvider>{children}</MobileMaskProvider>;
}
