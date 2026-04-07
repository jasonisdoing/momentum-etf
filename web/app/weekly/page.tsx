import { PageFrame } from "../components/PageFrame";
import { WeeklyManager } from "./WeeklyManager";

export const dynamic = "force-dynamic";

export default function WeeklyPage() {
  return (
    <PageFrame title="주별" fullHeight fullWidth>
      <WeeklyManager />
    </PageFrame>
  );
}
