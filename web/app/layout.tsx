import type { Metadata } from "next";
import type { ReactNode } from "react";

import "@tabler/core/dist/css/tabler.min.css";

import { AppShell } from "./AppShell";
import "./globals.css";

export const metadata: Metadata = {
  title: "Momentum ETF",
  description: "Momentum ETF 운영 UI",
};

type RootLayoutProps = {
  children: ReactNode;
};

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="ko">
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
