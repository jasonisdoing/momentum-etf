import { BulkImportManager } from "./BulkImportManager";

export const dynamic = "force-dynamic";

export default function ImportPage() {
  return (
    <main className="shell">
      <section className="pageHeaderCompact">
        <div><h1>벌크 입력</h1></div>
      </section>
      <BulkImportManager />
    </main>
  );
}
