import { MarketCalendarClient } from "./MarketCalendarClient";

export const dynamic = "force-dynamic";

export default function MarketCalendarPage() {
  const today = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());

  return <MarketCalendarClient today={today} />;
}
