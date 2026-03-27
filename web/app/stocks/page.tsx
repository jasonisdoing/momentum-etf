import { StocksManager } from "./StocksManager";

export const dynamic = "force-dynamic";

export default function StocksPage() {
  return (
    <main className="shell">
      <section className="pageHeaderCompact">
        <div><h1>종목 관리</h1></div>
      </section>
      <StocksManager />
    </main>
  );
}
