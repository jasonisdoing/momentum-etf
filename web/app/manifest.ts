import type { MetadataRoute } from "next";

/** PWA 매니페스트 — 데스크톱·모바일이 같은 앱을 쓴다.
 *  시작 주소는 `/` 하나이고, 좁은 화면이면 홈에서 `/m` 으로 보낸다
 *  (`MobileEntryRedirect`) — manifest 는 사이트당 하나라 기기별로 나눌 수 없다. */
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Jason Invest",
    short_name: "Invest",
    description: "자산 현황과 전략 운영 상태",
    start_url: "/",
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
