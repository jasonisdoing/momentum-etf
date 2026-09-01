// PWA 설치 조건용 서비스워커.
//
// 크롬은 "오프라인에서 응답할 수 있는가"를 설치 조건으로 보므로 fetch 핸들러가 필요하다.
// 자산·시세는 항상 최신이어야 하니 **캐시를 쓰지 않는다** — 네트워크로 그대로 보내고,
// 완전히 끊겼을 때만 안내 문구를 돌려준다. 오래된 잔고를 보여주는 것보다 낫다.

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  event.respondWith(
    fetch(event.request).catch(
      () =>
        new Response("<!doctype html><meta charset='utf-8'><p>네트워크에 연결되어 있지 않습니다.</p>", {
          status: 503,
          headers: { "Content-Type": "text/html; charset=utf-8" },
        }),
    ),
  );
});
