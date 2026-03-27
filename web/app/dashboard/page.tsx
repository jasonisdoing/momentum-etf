import { DashboardManager } from "./DashboardManager";

export const dynamic = "force-dynamic";

export default function DashboardPage() {
  return (
    <main className="shell">
      <DashboardManager />
    </main>
  );
}
