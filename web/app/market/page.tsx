import { PageFrame } from "../components/PageFrame";
import { MarketManager } from "./MarketManager";

export const dynamic = "force-dynamic";

export default function MarketPage() {
  return (
    <PageFrame title="ETF 마켓">
      <MarketManager />
    </PageFrame>
  );
}
