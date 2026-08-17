import { MobileAccountClient } from "./MobileAccountClient";

export const dynamic = "force-dynamic";

export default async function MobileAccountPage({ params }: { params: Promise<{ accountId: string }> }) {
  const { accountId } = await params;
  return <MobileAccountClient accountId={accountId} />;
}
