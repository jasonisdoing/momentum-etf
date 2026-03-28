import { PageFrame } from "../components/PageFrame";
import { StocksManager } from "./StocksManager";

export const dynamic = "force-dynamic";

export default function StocksPage() {
  return (
    <PageFrame title="종목 관리" fullHeight fullWidth>
      <StocksManager />
    </PageFrame>
  );
}
