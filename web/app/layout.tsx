import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";

// Pretendard 가변 폰트. dynamic-subset 은 unicode-range 로 잘게 쪼갠 woff2 라
// 브라우저가 실제 쓰는 글자 구간만 받는다(전체 2MB 통짜 대신).
// 이 import 가 없으면 globals.css 의 font-family 는 이름만 적힌 셈이 되어
// 맥은 Apple SD Gothic Neo, 윈도우는 맑은 고딕으로 제각각 렌더링된다.
import "pretendard/dist/web/variable/pretendardvariable-dynamic-subset.css";
import "@tabler/core/dist/css/tabler.min.css";
import { AppShell } from "./AppShell";
import { buildBucketCssVariables } from "../lib/bucket-theme";
import { ToastProvider } from "./components/ToastProvider";
import { HideMoneyProvider } from "@/lib/hide-money-context";
import "./globals.css";

export const metadata: Metadata = {
  title: "Jason Momentum",
  description: "투자는 원칙에 맞춰서 합시다",
  // 홈 화면에 추가했을 때 사파리 UI 없이 앱처럼 뜨게 한다(아이콘은 public/apple-touch-icon.png).
  appleWebApp: { capable: true, title: "Invest", statusBarStyle: "default" },
  icons: { apple: "/apple-touch-icon.png" },
};

// 폰에서 확대·축소로 레이아웃이 흔들리지 않게 하고, 노치 영역까지 배경을 채운다.
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#206bc4",
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
