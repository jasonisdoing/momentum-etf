"use client";

import { useLayoutEffect, type RefObject } from "react";

// 컨테이너 폭에 "한 줄로 들어가는 것만" 보여주고, 넘치는 자식은 숨긴다(오른쪽부터).
// - 숨김 대상은 `data-fit-hideable` 속성이 있는 자식만(예: 검색창처럼 항상 유지할 요소는 표시 유지).
// - 컨테이너는 `overflow: hidden; flex-wrap: nowrap` 이어야 한다.
// - deps 가 바뀌거나(내용 변경) 컨테이너 폭이 바뀌면(ResizeObserver) 다시 계산한다.
export function useFitOneLine(ref: RefObject<HTMLElement | null>, deps: readonly unknown[] = []): void {
  useLayoutEffect(() => {
    const container = ref.current;
    if (!container) return;

    const compute = () => {
      const hideable = Array.from(container.querySelectorAll<HTMLElement>("[data-fit-hideable]"));
      // 측정을 위해 먼저 모두 보이게 한 뒤, 오른쪽 끝이 컨테이너를 넘는 첫 항목부터 숨긴다.
      // offsetLeft 는 offsetParent 기준이라 부정확하므로, getBoundingClientRect 로 실제 위치를 비교한다.
      hideable.forEach((el) => {
        el.style.display = "";
      });
      const containerRight = container.getBoundingClientRect().right;
      let fits = true;
      for (const el of hideable) {
        if (fits && el.getBoundingClientRect().right <= containerRight + 0.5) {
          el.style.display = "";
        } else {
          fits = false;
          el.style.display = "none";
        }
      }
    };

    compute();
    const observer = new ResizeObserver(compute);
    observer.observe(container);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ref, ...deps]);
}

// 우선순위 기반 한 줄 맞춤 — 숫자가 큰 `data-fit-priority` 부터 숨긴다(1=끝까지 유지).
// useFitOneLine 과 달리 컨테이너 어디에 있든(중첩 포함) 전체 폭 기준으로 접는다.
// 전제: 컨테이너는 `flex-wrap: nowrap; overflow: hidden`, 중간 래퍼는 줄어들지 않아야
// (flex-shrink: 0) 넘침이 scrollWidth 로 드러난다. 줄바꿈 레이아웃(모바일)에선 아무것도 안 한다.
export function useFitByPriority(ref: RefObject<HTMLElement | null>, deps: readonly unknown[] = []): void {
  useLayoutEffect(() => {
    const container = ref.current;
    if (!container) return;

    const compute = () => {
      const items = Array.from(container.querySelectorAll<HTMLElement>("[data-fit-priority]")).sort(
        (a, b) => Number(b.dataset.fitPriority ?? 0) - Number(a.dataset.fitPriority ?? 0),
      );
      items.forEach((el) => {
        el.style.display = "";
      });
      if (getComputedStyle(container).flexWrap === "wrap") return;
      for (const el of items) {
        if (container.scrollWidth <= container.clientWidth + 1) break;
        el.style.display = "none";
      }
    };

    compute();
    const observer = new ResizeObserver(compute);
    observer.observe(container);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ref, ...deps]);
}
