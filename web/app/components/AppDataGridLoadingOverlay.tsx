"use client";

import { AppLoadingState } from "./AppLoadingState";

export function AppDataGridLoadingOverlay() {
  return (
    <div className="appDataGridOverlay">
      <AppLoadingState />
    </div>
  );
}
