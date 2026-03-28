import { PageFrame } from "../components/PageFrame";
import { SummaryManager } from "./SummaryManager";

export const dynamic = "force-dynamic";

export default function SummaryPage() {
  return (
    <PageFrame title="AI용 요약">
      <SummaryManager />
    </PageFrame>
  );
}
