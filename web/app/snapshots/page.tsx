import { SnapshotsManager } from "./SnapshotsManager";

export const dynamic = "force-dynamic";

export default function SnapshotsPage() {
  return (
    <main className="shell">
      <section className="pageHeaderCompact">
        <div><h1>스냅샷</h1></div>
      </section>
      <SnapshotsManager />
    </main>
  );
}
