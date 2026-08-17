import type { MetadataRoute } from "next";

/** PWA 매니페스트 — 홈 화면에 추가했을 때 모바일 화면(`/m`)으로 바로 열린다.
 *  데스크톱 화면은 브라우저에서 그대로 쓰고, 앱으로 쓰는 건 모바일 쪽이다. */
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Jason Invest",
    short_name: "Invest",
    description: "자산 현황과 전략 운영 상태",
    start_url: "/m",
    scope: "/",
    display: "standalone",
    orientation: "portrait",
    background_color: "#ffffff",
    theme_color: "#206bc4",
    lang: "ko",
    icons: [
      { src: "/icon-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png", purpose: "any" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
