"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import type { HoldingsRow } from "../../assets/assets-helpers";
import styles from "../mobile.module.css";
import {
  formatCompactKrw,
  formatKrw,
  formatPct,
  loadMobileSnapshot,
  signColorOf,
  type MobileAccount,
} from "../mobile-data";

/** 계좌 상세 — 그 계좌가 담고 있는 종목을 평가금액 순으로 본다. */
export function MobileAccountClient({ accountId }: { accountId: string }) {
  const [account, setAccount] = useState<MobileAccount | null>(null);
  const [rows, setRows] = useState<HoldingsRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const snapshot = await loadMobileSnapshot();
      const found = snapshot.accounts.find((row) => row.account_id === accountId) ?? null;
      if (!found) throw new Error(`계좌를 찾을 수 없습니다: ${accountId}`);
      setAccount(found);
      setRows(snapshot.rows.filter((row) => row.account_id === accountId));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [accountId]);

  useEffect(() => {
    void load();
  }, [load]);

  // 평가금액 큰 순 — 폰에서는 정렬 컨트롤 대신 한 가지 기준으로 고정한다.
  const holdings = useMemo(
    () => [...rows].sort((a, b) => (b.valuation_krw ?? 0) - (a.valuation_krw ?? 0)),
    [rows],
  );
  const totalAssets = account?.total_assets_krw ?? 0;

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <Link href="/m" className={styles.backLink}>
          ← 자산
        </Link>
        <span className={styles.headerTitle}>
          {account ? `${account.icon ? `${account.icon} ` : ""}${account.name}` : accountId}
        </span>
      </div>

      {error ? (
        <div className={styles.state}>{error}</div>
      ) : loading || !account ? (
        <div className={styles.state}>불러오는 중…</div>
      ) : (
        <>
          <div className={styles.summaryCard}>
            <span className={styles.summaryLabel}>총자산</span>
            <span className={styles.summaryValue}>{formatKrw(account.total_assets_krw)}</span>
            <div className={styles.summaryMetrics}>
              <span>
                <span className={styles.metricLabel}>금일</span>
                <span className={styles.metricValue} style={{ color: signColorOf(account.daily_return_pct) }}>
                  {formatPct(account.daily_return_pct)}
                </span>
              </span>
              <span>
                <span className={styles.metricLabel}>누적</span>
                <span className={styles.metricValue} style={{ color: signColorOf(account.net_profit_pct) }}>
                  {formatPct(account.net_profit_pct)}
                </span>
              </span>
              <span>
                <span className={styles.metricLabel}>현금</span>
                <span className={styles.metricValue}>{formatCompactKrw(account.cash_balance_krw)}</span>
              </span>
            </div>
          </div>

          <div className={styles.list}>
            {holdings.length === 0 ? (
              <div className={styles.state}>보유 종목이 없습니다.</div>
            ) : (
              holdings.map((row) => {
                const weight = totalAssets > 0 ? (row.valuation_krw / totalAssets) * 100 : 0;
                return (
                  <div key={`${row.ticker}`} className={styles.row}>
                    <span className={styles.rowMain}>
                      <span className={styles.rowName}>{row.name}</span>
                      <span className={styles.rowSub}>
                        {row.ticker} · {weight.toFixed(1)}%
                      </span>
                    </span>
                    <span className={styles.rowSide}>
                      <span className={styles.rowAmount}>{formatKrw(row.valuation_krw)}</span>
                      <span className={styles.rowPct} style={{ color: signColorOf(row.return_pct) }}>
                        {formatPct(row.return_pct)}
                      </span>
                    </span>
                  </div>
                );
              })
            )}
          </div>
        </>
      )}
    </div>
  );
}
