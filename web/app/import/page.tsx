import { PageFrame } from "../components/PageFrame";
import { BulkImportManager } from "./BulkImportManager";

export const dynamic = "force-dynamic";

export default function ImportPage() {
  return (
    <PageFrame title="벌크 입력">
      <BulkImportManager />
    </PageFrame>
  );
}
