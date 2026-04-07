"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { usePathname, useRouter } from "next/navigation";

type TickerItem = {
  ticker: string;
  name: string;
  ticker_type: string;
  country_code: string;
};

export function GlobalTickerSearch() {
  const router = useRouter();
  const pathname = usePathname();
  const [allTickers, setAllTickers] = useState<TickerItem[]>([]);
  const [searchInput, setSearchInput] = useState("");
  const [showDropdown, setShowDropdown] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(-1);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let alive = true;

    async function fetchAllTickers() {
      try {
        const response = await fetch("/api/ticker-tickers", { cache: "no-store" });
        if (!response.ok) {
          return;
        }
        const payload = (await response.json()) as TickerItem[];
        if (alive && Array.isArray(payload)) {
          setAllTickers(payload);
        }
      } catch {
        // 전역 검색은 실패 시 조용히 비활성화합니다.
      }
    }

    void fetchAllTickers();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    setSearchInput("");
    setShowDropdown(false);
    setHighlightIndex(-1);
  }, [pathname]);

  const filteredTickers = useMemo(() => {
    const query = searchInput.trim().toLowerCase();
    if (!query) {
      return [];
    }

    return allTickers
      .filter((item) => item.ticker.toLowerCase().includes(query) || item.name.toLowerCase().includes(query))
      .slice(0, 20);
  }, [allTickers, searchInput]);

  function navigateToTicker(item: TickerItem) {
    setSearchInput("");
    setShowDropdown(false);
    setHighlightIndex(-1);
    router.push(`/ticker?ticker=${encodeURIComponent(item.ticker)}`);
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setHighlightIndex((prev) => Math.min(prev + 1, filteredTickers.length - 1));
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      setHighlightIndex((prev) => Math.max(prev - 1, 0));
      return;
    }

    if (event.key === "Escape") {
      setShowDropdown(false);
      return;
    }

    if (event.key !== "Enter") {
      return;
    }

    event.preventDefault();
    if (highlightIndex >= 0 && highlightIndex < filteredTickers.length) {
      navigateToTicker(filteredTickers[highlightIndex]);
      return;
    }

    if (filteredTickers.length === 1) {
      navigateToTicker(filteredTickers[0]);
    }
  }

  return (
    <div ref={wrapRef} className="globalTickerSearch">
      <input
        className="form-control form-control-sm globalTickerSearchInput"
        type="text"
        value={searchInput}
        placeholder="티커/종목명"
        onChange={(event) => {
          setSearchInput(event.target.value);
          setShowDropdown(true);
          setHighlightIndex(-1);
        }}
        onFocus={() => {
          if (searchInput.trim()) {
            setShowDropdown(true);
          }
        }}
        onKeyDown={handleKeyDown}
      />
      {showDropdown && filteredTickers.length > 0 ? (
        <div className="globalTickerSearchDropdown">
          {filteredTickers.map((item, index) => (
            <button
              key={`${item.ticker_type}-${item.ticker}`}
              type="button"
              className={index === highlightIndex ? "globalTickerSearchOption is-active" : "globalTickerSearchOption"}
              onMouseEnter={() => setHighlightIndex(index)}
              onMouseDown={(event) => {
                event.preventDefault();
                navigateToTicker(item);
              }}
            >
              <span className="appCodeText globalTickerSearchTicker">{item.ticker}</span>
              <span className="globalTickerSearchName">{item.name}</span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
