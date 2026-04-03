import { PageFrame } from "../components/PageFrame";
import { AssetsManager } from "./AssetsManager";

export const dynamic = "force-dynamic";

export default function AssetsPage() {
  return (
    <PageFrame title="자산 관리" fullHeight fullWidth>
      <AssetsManager />
    </PageFrame>
  );
}
