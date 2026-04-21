"use client";

import { useEffect, useState, type ReactNode } from "react";

const MOBILE_LANDSCAPE_BREAKPOINT_PX = 1200;

type ResponsiveFiltersSectionProps = {
  children: ReactNode;
};

export function ResponsiveFiltersSection({ children }: ResponsiveFiltersSectionProps) {
  const [isNarrow, setIsNarrow] = useState(false);
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    const mediaQuery = window.matchMedia(`(max-width: ${MOBILE_LANDSCAPE_BREAKPOINT_PX}px)`);

    const applyMatches = (matches: boolean) => {
      setIsNarrow(matches);
      setIsExpanded(!matches);
    };

    applyMatches(mediaQuery.matches);

    const handleChange = (event: MediaQueryListEvent) => {
      applyMatches(event.matches);
    };

    mediaQuery.addEventListener("change", handleChange);
    return () => {
      mediaQuery.removeEventListener("change", handleChange);
    };
  }, []);

  return (
    <>
      {isNarrow ? (
        <div className="appResponsiveFiltersToggleWrap">
          <button
            type="button"
            className="btn btn-outline-secondary btn-sm appResponsiveFiltersToggle"
            onClick={() => setIsExpanded((prev) => !prev)}
          >
            {isExpanded ? "필터 접기" : "필터 펼치기"}
          </button>
        </div>
      ) : null}
      {!isNarrow || isExpanded ? children : null}
    </>
  );
}
