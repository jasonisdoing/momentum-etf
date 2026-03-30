import { PageFrame } from "../components/PageFrame";
import { HoldingsManager } from "./HoldingsManager";

export const dynamic = "force-dynamic";

export default function HoldingsPage() {
  return (
    <PageFrame title="계좌 상세">
      <HoldingsManager />
    </PageFrame>
  );
}
