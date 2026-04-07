import { PageFrame } from "../components/PageFrame";
import { TickerDetailManager } from "./TickerDetailManager";

export const dynamic = "force-dynamic";

export default function TickerPage() {
  return (
    <PageFrame title="개별종목" fullHeight fullWidth>
      <TickerDetailManager />
    </PageFrame>
  );
}
