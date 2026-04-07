import { PageFrame } from "../components/PageFrame";
import { DashboardManager } from "./DashboardManager";

export const dynamic = "force-dynamic";

export default function DashboardPage() {
  return (
    <PageFrame title="대시보드">
      <DashboardManager />
    </PageFrame>
  );
}
