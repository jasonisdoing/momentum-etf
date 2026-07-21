"use client";

import { useCallback, useRef, useState } from "react";

/**
 * 그리드 상단 "신규 티커 추가 행"의 상태머신.
 *
 * `/assets`(AssetsManager) 의 addingRow(ticker 입력 → 확인 조회 → 확정) 흐름을 공용화한 것.
 * 조회 엔드포인트·중복검사·커밋 방식은 화면마다 다르므로 `resolve`/`onValidated` 로 주입한다.
 * 화면마다 같은 상태 로직을 복붙하지 말고 이 훅을 재사용한다(AGENTS.md 3-1).
 */
export type ResolvedTicker = { ticker: string; name: string };

export type AddingTickerRow = {
  ticker: string;
  name: string;
  isValidated: boolean;
  isValidating: boolean;
};

export function useAddingTickerRow<T extends ResolvedTicker>(options: {
  /** 티커 문자열을 조회해 확정 정보를 반환한다(실패 시 throw). 중복검사도 여기서 처리. */
  resolve: (ticker: string) => Promise<T>;
  /** 조회 성공 시 호출 — 리스트 추가 등 확정 처리. */
  onValidated: (resolved: T) => void;
  onError?: (message: string) => void;
  /** 입력 티커 정규화(대문자/접두사 등). 없으면 trim 만. */
  normalize?: (raw: string) => string;
  /** 조회 성공 즉시 추가 행을 비운다(수량 등 후속 입력이 없는 화면용). 기본 false. */
  resetOnValidated?: boolean;
}) {
  const { resolve, onValidated, onError, normalize, resetOnValidated = false } = options;
  const [addingRow, setAddingRow] = useState<AddingTickerRow | null>(null);
  // validate 가 최신 ticker 를 읽도록 ref 로 미러링(낡은 클로저 방지).
  const rowRef = useRef<AddingTickerRow | null>(null);
  rowRef.current = addingRow;

  const start = useCallback(() => {
    setAddingRow({ ticker: "", name: "", isValidated: false, isValidating: false });
  }, []);

  const cancel = useCallback(() => setAddingRow(null), []);

  const setTicker = useCallback((value: string) => {
    setAddingRow((prev) => (prev ? { ...prev, ticker: value, name: "", isValidated: false } : null));
  }, []);

  const validate = useCallback(
    async (tickerToUse?: string) => {
      const current = rowRef.current;
      if (!current || current.isValidating) return;
      const source = tickerToUse ?? current.ticker;
      const raw = (normalize ? normalize(source) : source).trim();
      if (!raw) {
        onError?.("티커를 입력해주세요.");
        return;
      }
      setAddingRow((prev) => (prev ? { ...prev, ticker: raw, name: "", isValidated: false, isValidating: true } : null));
      try {
        const resolved = await resolve(raw);
        setAddingRow((prev) =>
          resetOnValidated
            ? null
            : prev
              ? { ...prev, ticker: resolved.ticker, name: resolved.name, isValidated: true, isValidating: false }
              : null,
        );
        onValidated(resolved);
      } catch (err) {
        setAddingRow((prev) => (prev ? { ...prev, isValidating: false } : null));
        onError?.(err instanceof Error ? err.message : "티커 조회에 실패했습니다.");
      }
    },
    [resolve, onValidated, onError, normalize, resetOnValidated],
  );

  return { addingRow, start, cancel, setTicker, validate };
}
