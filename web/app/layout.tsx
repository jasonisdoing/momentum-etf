import type { Metadata } from "next";
import type { ReactNode } from "react";

import "@tabler/core/dist/css/tabler.min.css";
import { AppShell } from "./AppShell";
import { buildBucketCssVariables } from "../lib/bucket-theme";
import { ToastProvider } from "./components/ToastProvider";
import { HideMoneyProvider } from "@/lib/hide-money-context";
import "./globals.css";

export const metadata: Metadata = {
  title: "Jason Momentum",
  description: "투자는 원칙에 맞춰서 합시다",
};

type RootLayoutProps = {
  children: ReactNode;
};

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="ko" data-scroll-behavior="smooth">
      <body>
        <style>{buildBucketCssVariables()}</style>
        <ToastProvider>
          <HideMoneyProvider>
            <AppShell>{children}</AppShell>
          </HideMoneyProvider>
        </ToastProvider>
      </body>
    </html>
  );
}
