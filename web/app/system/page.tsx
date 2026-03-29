import { PageFrame } from "../components/PageFrame";
import { SystemManager } from "./SystemManager";

export const dynamic = "force-dynamic";

export default function SystemPage() {
  return (
    <PageFrame title="정보">
      <SystemManager />
    </PageFrame>
  );
}
