import { PageFrame } from "../components/PageFrame";
import { CashManager } from "./CashManager";

export const dynamic = "force-dynamic";

export default function CashPage() {
  return (
    <PageFrame title="자산관리">
      <CashManager />
    </PageFrame>
  );
}
