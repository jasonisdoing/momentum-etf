import { PageFrame } from "../components/PageFrame";
import { NoteManager } from "./NoteManager";

export const dynamic = "force-dynamic";

export default function NotePage() {
  return (
    <PageFrame title="계좌 메모">
      <NoteManager />
    </PageFrame>
  );
}
