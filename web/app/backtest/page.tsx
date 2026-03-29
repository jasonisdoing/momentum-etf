import { PageFrame } from "../components/PageFrame";
import { BacktestBuilder } from "./BacktestBuilder";

export const dynamic = "force-dynamic";

export default function BacktestPage() {
  return (
    <PageFrame title="백테스트">
      <BacktestBuilder />
    </PageFrame>
  );
}
