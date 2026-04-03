import { PageFrame } from "../components/PageFrame";
import { RankManager } from "./RankManager";

export const dynamic = "force-dynamic";

export default function RankPage() {
  return (
    <PageFrame title="순위" fullHeight fullWidth>
      <RankManager />
    </PageFrame>
  );
}
