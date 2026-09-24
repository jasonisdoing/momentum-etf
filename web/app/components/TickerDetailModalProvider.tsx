"use client";

import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

import { TickerDetailManager } from "../ticker/TickerDetailManager";
import { AppModal } from "./AppModal";

const TickerDetailModalContext = createContext<((ticker: string) => void) | null>(null);

export function useTickerDetailModal() {
  const openTicker = useContext(TickerDetailModalContext);
  if (!openTicker) throw new Error("TickerDetailModalProvider가 필요합니다.");
  return openTicker;
}

export function TickerDetailModalProvider({ children }: { children: ReactNode }) {
  const [ticker, setTicker] = useState<string | null>(null);

  useEffect(() => {
    if (!ticker) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setTicker(null);
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [ticker]);

  return (
    <TickerDetailModalContext.Provider value={setTicker}>
      {children}
      <AppModal open={ticker !== null} title={ticker ? `${ticker} 상세` : "종목 상세"} size="full" onClose={() => setTicker(null)}>
        {ticker ? <div className="tickerDetailModalContent"><TickerDetailManager tickerOverride={ticker} /></div> : null}
      </AppModal>
    </TickerDetailModalContext.Provider>
  );
}
