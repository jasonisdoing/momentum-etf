"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import type { HoldingsRow } from "../../assets/assets-helpers";
import { MobileFrame, useMaskedAmount } from "../MobileFrame";
import styles from "../mobile.module.css";
import {
  accountLabel,
  formatCompactKrw,
  formatKrw,
  formatPct,
  loadMobileSnapshot,
  signColorOf,
  type MobileAccount,
  type MobileTotals,
} from "../mobile-data";

/** 자산 — 전체 요약 + 계좌 목록. 계좌를 누르면 그 자리에서 보유 종목이 펼쳐진다. */
export function MobileAssetsClient() {
  const [accounts, setAccounts] = useState<MobileAccount[]>([]);
  const [totals, setTotals] = useState<MobileTotals | null>(null);
  const [rows, setRows] = useState<HoldingsRow[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [loadedAt, setLoadedAt] = useState<Date | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mask = useMaskedAmount();

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const snapshot = await loadMobileSnapshot();
      setAccounts(snapshot.accounts);
      setTotals(snapshot.totals);
      setRows(snapshot.rows);
      setLoadedAt(new Date());
    } catch (loadError) {
      setError(
        loadError instanceof Error ? loadError.message : "불러오지 못했습니다.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // 계좌별 보유 종목 — 평가금액 큰 순. 폰에서는 정렬 컨트롤 없이 한 기준으로 고정한다.
  const holdingsByAccount = useMemo(() => {
    const grouped = new Map<string, HoldingsRow[]>();
    for (const row of rows) {
      const key = row.account_id;
      if (!key) continue;
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key)!.push(row);
    }
    for (const list of grouped.values()) {
      list.sort((a, b) => (b.valuation_krw ?? 0) - (a.valuation_krw ?? 0));
    }
    return grouped;
  }, [rows]);

  return (
    <MobileFrame
      title="자산"
      backHref="/m"
      loadedAt={loadedAt}
      onRefresh={() => void load()}
      refreshing={loading}
    >
      <div className={styles.page}>
        {error ? (
          <div className={styles.state}>{error}</div>
        ) : loading || !totals ? (
          <div className={styles.state}>불러오는 중…</div>
        ) : (
          <>
            <div className={styles.summaryCard}>
              <span className={styles.summaryLabel}>총자산</span>
              <span className={styles.summaryValue}>
                {mask(formatKrw(totals.total_assets))}
              </span>
              <div className={styles.summaryMetrics}>
                <span>
                  <span className={styles.metricLabel}>금일</span>
                  <span
                    className={styles.metricValue}
                    style={{ color: signColorOf(totals.daily_return_pct) }}
                  >
                    {formatPct(totals.daily_return_pct)}
                  </span>
                </span>
                <span>
                  <span className={styles.metricLabel}>누적</span>
                  <span
                    className={styles.metricValue}
                    style={{ color: signColorOf(totals.net_profit_pct) }}
                  >
                    {formatPct(totals.net_profit_pct)} (
                    {mask(formatCompactKrw(totals.net_profit))})
                  </span>
                </span>
              </div>
            </div>

            <div className={styles.list}>
              {accounts.map((account) => {
                const expanded = expandedId === account.account_id;
                const holdings =
                  holdingsByAccount.get(account.account_id) ?? [];
                return (
                  <div key={account.account_id} className={styles.group}>
                    <button
                      type="button"
                      className={styles.row}
                      aria-expanded={expanded}
                      onClick={() =>
                        setExpandedId(expanded ? null : account.account_id)
                      }
                    >
                      <span className={styles.rowMain}>
                        <span className={styles.rowName}>
                          {expanded ? "▾" : "▸"} {accountLabel(account)}
                        </span>
                        <span className={styles.rowSub}>
                          {holdings.length}종목 · 현금{" "}
                          {mask(formatCompactKrw(account.cash_balance_krw))}
                        </span>
                      </span>
                      <span className={styles.rowSide}>
                        <span className={styles.rowAmount}>
                          {mask(formatKrw(account.total_assets_krw))}
                        </span>
                        <span
                          className={styles.rowPct}
                          style={{
                            color: signColorOf(account.daily_return_pct),
                          }}
                        >
                          {formatPct(account.daily_return_pct)}
                        </span>
                      </span>
                    </button>

                    {expanded ? (
                      <div className={styles.children}>
                        {holdings.length === 0 ? (
                          <div className={styles.state}>
                            보유 종목이 없습니다.
                          </div>
                        ) : (
                          holdings.map((row) => {
                            const weight =
                              account.total_assets_krw > 0
                                ? (row.valuation_krw /
                                    account.total_assets_krw) *
                                  100
                                : 0;
                            return (
                              <div
                                key={`${account.account_id}:${row.ticker}`}
                                className={styles.childRow}
                              >
                                <span className={styles.rowMain}>
                                  <span className={styles.rowName}>
                                    {row.name}
                                  </span>
                                  <span className={styles.rowSub}>
                                    {row.ticker} · {weight.toFixed(1)}%
                                  </span>
                                </span>
                                <span className={styles.rowSide}>
                                  <span className={styles.rowAmount}>
                                    {mask(formatKrw(row.valuation_krw))}
                                  </span>
                                  <span
                                    className={styles.rowPct}
                                    style={{
                                      color: signColorOf(row.return_pct),
                                    }}
                                  >
                                    {formatPct(row.return_pct)}
                                  </span>
                                </span>
                              </div>
                            );
                          })
                        )}
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </MobileFrame>
  );
}
