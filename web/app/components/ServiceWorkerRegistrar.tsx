"use client";

import { useEffect } from "react";

/** 서비스워커 등록 — 앱 설치(크롬 설치 아이콘)의 조건이다.
 *  캐시는 두지 않는다(`public/sw.js` 참고) — 잔고·시세는 항상 서버 값을 쓴다. */
export function ServiceWorkerRegistrar() {
  useEffect(() => {
    if (!("serviceWorker" in navigator)) return;
    navigator.serviceWorker.register("/sw.js").catch(() => {
      // 등록에 실패해도 화면 동작에는 영향이 없다(설치 아이콘만 안 뜬다).
    });
  }, []);

  return null;
}
