"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

const STORAGE_KEY = "jason-invest-hide-money";

type HideMoneyContextValue = {
  hideMoney: boolean;
  setHideMoney: (value: boolean) => void;
  toggleHideMoney: () => void;
};

const HideMoneyContext = createContext<HideMoneyContextValue | null>(null);

export function HideMoneyProvider({ children }: { children: ReactNode }) {
  const [hideMoney, setHideMoneyState] = useState(false);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored === "1") {
        setHideMoneyState(true);
      }
    } catch {
      // 무시
    }
  }, []);

  const setHideMoney = useCallback((value: boolean) => {
    setHideMoneyState(value);
    try {
      window.localStorage.setItem(STORAGE_KEY, value ? "1" : "0");
    } catch {
      // 무시
    }
  }, []);

  const toggleHideMoney = useCallback(() => {
    setHideMoneyState((current) => {
      const next = !current;
      try {
        window.localStorage.setItem(STORAGE_KEY, next ? "1" : "0");
      } catch {
        // 무시
      }
      return next;
    });
  }, []);

  return (
    <HideMoneyContext.Provider value={{ hideMoney, setHideMoney, toggleHideMoney }}>
      {children}
    </HideMoneyContext.Provider>
  );
}

export function useHideMoney(): HideMoneyContextValue {
  const ctx = useContext(HideMoneyContext);
  if (!ctx) {
    // Provider 외부에서 사용해도 깨지지 않게 기본값 제공
    return {
      hideMoney: false,
      setHideMoney: () => {},
      toggleHideMoney: () => {},
    };
  }
  return ctx;
}
