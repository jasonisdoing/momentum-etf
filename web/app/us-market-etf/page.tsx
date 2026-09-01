import { MarketPageClient } from "../market/MarketPageClient";

export const dynamic = "force-dynamic";

export default function UsMarketEtfPage() {
  return <MarketPageClient title="🇺🇸 미국 ETF" market="us" />;
}
