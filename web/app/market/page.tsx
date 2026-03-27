import { MarketManager } from "./MarketManager";

export const dynamic = "force-dynamic";

export default function MarketPage() {
  return (
    <main className="shell">
      <section className="pageHeaderCompact">
        <div><h1>ETF 마켓</h1></div>
      </section>
      <MarketManager />
    </main>
  );
}
