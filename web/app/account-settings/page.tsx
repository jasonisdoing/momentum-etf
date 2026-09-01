import { PageFrame } from "../components/PageFrame";
import { AccountSettingsManager } from "./AccountSettingsManager";

export const dynamic = "force-dynamic";

export default function AccountSettingsPage() {
  // 항목이 많아 가로를 최대한 쓴다 — 그리드가 16개 컬럼을 한 화면에 놓는다.
  return (
    <PageFrame title="계좌 설정" fullHeight fullWidth>
      <AccountSettingsManager />
    </PageFrame>
  );
}
