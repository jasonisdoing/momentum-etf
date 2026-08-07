import { PageFrame } from "../components/PageFrame";
import { DashboardManager } from "./DashboardManager";

export const dynamic = "force-dynamic";

// 자산 요약 전체 화면 — 홈 허브의 요약 스트립에서 "전체 화면 →" 로 진입한다.
// (과거에는 / 로 redirect 했으나, 홈이 허브(요약)로 바뀌면서 상세 대시보드가 이 라우트를 가진다.)
export default function DashboardPage() {
  return (
    <PageFrame title="자산 요약">
      <DashboardManager />
    </PageFrame>
  );
}
