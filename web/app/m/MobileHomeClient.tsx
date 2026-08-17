"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import styles from "./mobile.module.css";
import {
  formatCompactKrw,
  formatKrw,
  formatPct,
  loadMobileSnapshot,
  signColorOf,
  type MobileAccount,
  type MobileTotals,
} from "./mobile-data";

/** 모바일 홈 — 전체 요약 + 계좌 목록. 계좌를 누르면 그 계좌의 종목으로 들어간다. */
export function MobileHomeClient() {
  const [accounts, setAccounts] = useState<MobileAccount[]>([]);
  const [totals, setTotals] = useState<MobileTotals | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const snapshot = await loadMobileSnapshot();
      setAccounts(snapshot.accounts);
      setTotals(snapshot.totals);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <span className={styles.headerTitle}>자산</span>
      </div>

      {error ? (
        <div className={styles.state}>{error}</div>
      ) : loading || !totals ? (
        <div className={styles.state}>불러오는 중…</div>
      ) : (
        <>
          <div className={styles.summaryCard}>
            <span className={styles.summaryLabel}>총자산</span>
            <span className={styles.summaryValue}>{formatKrw(totals.total_assets)}</span>
            <div className={styles.summaryMetrics}>
              <span>
                <span className={styles.metricLabel}>금일</span>
                <span className={styles.metricValue} style={{ color: signColorOf(totals.daily_return_pct) }}>
                  {formatPct(totals.daily_return_pct)}
                </span>
              </span>
              <span>
                <span className={styles.metricLabel}>누적</span>
                <span className={styles.metricValue} style={{ color: signColorOf(totals.net_profit_pct) }}>
                  {formatPct(totals.net_profit_pct)} ({formatCompactKrw(totals.net_profit)})
                </span>
              </span>
            </div>
          </div>

          <div className={styles.list}>
            {accounts.map((account) => (
              <Link key={account.account_id} href={`/m/${account.account_id}`} className={styles.row}>
                <span className={styles.rowMain}>
                  <span className={styles.rowName}>
                    {account.icon ? `${account.icon} ` : ""}
                    {account.name}
                  </span>
                  <span className={styles.rowSub}>현금 {formatCompactKrw(account.cash_balance_krw)}</span>
                </span>
                <span className={styles.rowSide}>
                  <span className={styles.rowAmount}>{formatKrw(account.total_assets_krw)}</span>
                  <span className={styles.rowPct} style={{ color: signColorOf(account.daily_return_pct) }}>
                    {formatPct(account.daily_return_pct)}
                  </span>
                </span>
              </Link>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
