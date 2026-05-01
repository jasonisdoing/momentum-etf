"use client";

import { Children, cloneElement, isValidElement, useEffect, useState, type ReactElement, type ReactNode } from "react";

const MOBILE_LANDSCAPE_BREAKPOINT_PX = 1200;

type ResponsiveFiltersSectionProps = {
  children: ReactNode;
};

function buildToggleField(isExpanded: boolean, onToggle: () => void): ReactElement {
  return (
    <div className="appResponsiveFiltersField" key="responsive-filters-toggle">
      <span className="appLabeledFieldLabel">필터</span>
      <button
        type="button"
        className="btn btn-outline-secondary btn-sm appResponsiveFiltersToggle"
        onClick={onToggle}
      >
        {isExpanded ? "필터 접기" : "필터 펼치기"}
      </button>
    </div>
  );
}

function injectToggleFieldIntoHeader(children: ReactNode, toggleField: ReactElement): ReactNode {
  if (!isValidElement(children)) {
    return children;
  }

  const rootElement = children as ReactElement<{ children?: ReactNode }>;
  const rootChildren = Children.toArray(rootElement.props.children);
  const nextRootChildren = rootChildren.map((child) => {
    if (!isValidElement(child)) {
      return child;
    }
    const leftElement = child as ReactElement<{ className?: string; children?: ReactNode }>;
    const className = typeof leftElement.props.className === "string" ? leftElement.props.className : "";
    const classNames = className.split(/\s+/).filter(Boolean);
    if (!classNames.includes("appMainHeaderLeft")) {
      return child;
    }
    const leftChildren = Children.toArray(leftElement.props.children);
    return cloneElement(leftElement, {
      children: [toggleField, ...leftChildren],
    });
  });

  return cloneElement(rootElement, {
    children: nextRootChildren,
  });
}

export function ResponsiveFiltersSection({ children }: ResponsiveFiltersSectionProps) {
  const [mounted, setMounted] = useState(false);
  const [isNarrow, setIsNarrow] = useState(false);
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    setMounted(true);
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

  if (!mounted || !isNarrow) {
    return <>{children}</>;
  }

  const toggleField = buildToggleField(isExpanded, () => setIsExpanded((prev) => !prev));

  if (!isExpanded) {
    return (
      <div className="appMainHeader">
        <div className="appMainHeaderLeft">{toggleField}</div>
      </div>
    );
  }

  return <>{injectToggleFieldIntoHeader(children, toggleField)}</>;
}
