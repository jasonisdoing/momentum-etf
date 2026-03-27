import { CashManager } from "./CashManager";

export const dynamic = "force-dynamic";

export default function CashPage() {
  return (
    <main className="shell">
      <section className="pageHeaderCompact">
        <div><h1>자산관리</h1></div>
      </section>
      <CashManager />
    </main>
  );
}
