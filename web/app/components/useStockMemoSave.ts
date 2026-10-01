"use client";

import { useCallback } from "react";

import { updateStockMemo } from "@/lib/stocks-store";
import { useToast } from "./ToastProvider";

/** 행 목록에서 그 티커의 메모만 바꾼 새 목록 — 상태를 갱신할 때 쓴다. */
export function withTickerMemo<T extends { ticker: string }>(rows: T[], ticker: string, memo: string): T[] {
  return rows.map((row) => (row.ticker === ticker ? { ...row, memo } : row));
}

/**
 * 종목 메모 저장 — 모멘텀·신고가·포트폴리오·순위 화면 공용.
 *
 * 메모는 계좌가 아니라 종목에 붙는 값이라 서버에는 한 곳(`stock_meta.memo`)에 저장한다.
 * 화면은 저장이 성공하면 **자기 상태도 같이 바꿔야** 한다(`applyMemo`). 표가 상태에서 다시
 * 만들어지는 화면에서 서버에만 저장하면, 토스트나 시세 갱신으로 리렌더될 때 표가 옛 메모로
 * 되돌아간다(2026-10 모멘텀 화면에서 실제로 그랬다).
 */
export function useStockMemoSave(applyMemo: (ticker: string, memo: string) => void, errorPrefix = "") {
  const toast = useToast();
  return useCallback(
    async (ticker: string, memo: string) => {
      if (!ticker) return;
      try {
        await updateStockMemo(ticker, memo);
        applyMemo(ticker, memo);
        toast.success("메모 저장 완료");
      } catch (error) {
        toast.error(`${errorPrefix}${error instanceof Error ? error.message : "메모 저장에 실패했습니다."}`);
      }
    },
    [applyMemo, errorPrefix, toast],
  );
}
