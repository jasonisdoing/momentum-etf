import { PageFrame } from "../components/PageFrame";
import { BacktestPlaceholder } from "./BacktestPlaceholder";

export const dynamic = "force-dynamic";

export default function BacktestPage() {
  return (
    <PageFrame title="백테스트">
      <BacktestPlaceholder />
    </PageFrame>
  );
}
