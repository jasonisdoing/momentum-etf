"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

/** 좁은 화면으로 홈에 들어오면 모바일 화면(`/m`)으로 보낸다.
 *
 * PWA manifest 는 사이트당 하나라 `start_url` 을 기기별로 나눌 수 없다. 그래서 시작 주소는
 * `/` 하나로 두고 여기서 폭을 보고 가른다. 폰에서 데스크톱 화면을 보려면 `/?desktop=1` 로
 * 들어오면 되고, 그 선택은 탭이 살아 있는 동안 유지된다.
 */
const MOBILE_MAX_WIDTH = 768;
const SKIP_KEY = "momentum-etf:skip-mobile-redirect";

export function MobileEntryRedirect() {
  const router = useRouter();

  useEffect(() => {
    let skip = false;
    try {
      if (new URLSearchParams(window.location.search).has("desktop")) {
        window.sessionStorage.setItem(SKIP_KEY, "1");
      }
      skip = window.sessionStorage.getItem(SKIP_KEY) === "1";
    } catch {
      // 저장소를 못 쓰면 폭 기준만 본다.
    }
    if (skip) return;
    if (window.innerWidth <= MOBILE_MAX_WIDTH) {
      router.replace("/m");
    }
  }, [router]);

  return null;
}
