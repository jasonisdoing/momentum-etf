"use client";

import { PageFrame } from "../components/PageFrame";
import { SettingsManager } from "./SettingsManager";

export function SettingsPageClient() {
  return (
    <PageFrame title="설정">
      <SettingsManager />
    </PageFrame>
  );
}
