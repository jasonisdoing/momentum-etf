import { PageFrame } from "../components/PageFrame";
import { SnapshotsManager } from "./SnapshotsManager";

export const dynamic = "force-dynamic";

export default function SnapshotsPage() {
  return (
    <PageFrame title="스냅샷">
      <SnapshotsManager />
    </PageFrame>
  );
}
