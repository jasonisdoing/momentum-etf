import { PageFrame } from "../components/PageFrame";
import { DeletedStocksManager } from "./DeletedStocksManager";

export default function DeletedStocksPage() {
  return (
    <PageFrame title="삭제된 종목">
      <DeletedStocksManager />
    </PageFrame>
  );
}
