"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Cell, Pie, PieChart } from "recharts";

import { MobileFrame, useMaskedAmount } from "./MobileFrame";
import styles from "./mobile.module.css";
import {
  ACCOUNT_COLORS,
  accountLabel,
  formatKoreanMoney,
  formatKrw,
  formatPct,
  loadMobileSnapshot,
  signColorOf,
  type MobileAccount,
  type MobilePeriods,
  type MobileTotals,
} from "./mobile-data";

const PERIOD_LABELS: { key: keyof MobilePeriods; label: string }[] = [
  { key: "daily", label: "금일" },
  { key: "weekly", label: "금주" },
  { key: "monthly", label: "금월" },
  { key: "yearly", label: "금년" },
];

/** 모바일 홈 — 계좌별 비중(파이) · 기간 수익률 · 화면 버튼. */
export function MobileHomeClient() {
  const [accounts, setAccounts] = useState<MobileAccount[]>([]);
  const [totals, setTotals] = useState<MobileTotals | null>(null);
  const [periods, setPeriods] = useState<MobilePeriods>({});
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
      setPeriods(snapshot.periods);
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

  // 계좌별 비중 — 총자산 기준. 비중이 큰 순으로 돌려 색이 섞이지 않게 한다.
  const pieData = useMemo(() => {
    const total = accounts.reduce(
      (sum, account) => sum + account.total_assets_krw,
      0,
    );
    if (total <= 0) return [];
    return [...accounts]
      .sort((a, b) => b.total_assets_krw - a.total_assets_krw)
      .map((account) => ({
        name: accountLabel(account),
        amount: account.total_assets_krw,
        weight_pct: (account.total_assets_krw / total) * 100,
      }));
  }, [accounts]);

  return (
    <MobileFrame
      title="Jason Invest"
      loadedAt={loadedAt}
      onRefresh={() => void load()}
      refreshing={loading}
    >
      <div className={`${styles.page} ${styles.homePage}`}>
        {error ? (
          <div className={styles.state}>{error}</div>
        ) : (
          <>
            <div className={styles.summaryCard}>
              <span className={styles.summaryLabel}>총자산</span>
              <span className={styles.summaryValue}>
                {mask(formatKrw(totals?.total_assets ?? null))}
              </span>
            </div>

            <section className={styles.section}>
              <h2 className={styles.sectionTitle}>계좌 비중</h2>
              <div className={styles.pieBlock}>
                <PieChart
                  width={150}
                  height={150}
                  margin={{ top: 0, right: 0, bottom: 0, left: 0 }}
                >
                  <Pie
                    data={pieData}
                    dataKey="weight_pct"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={38}
                    outerRadius={68}
                    paddingAngle={2}
                    strokeWidth={0}
                    isAnimationActive={false}
                  >
                    {pieData.map((_, index) => (
                      <Cell
                        key={index}
                        fill={ACCOUNT_COLORS[index % ACCOUNT_COLORS.length]}
                      />
                    ))}
                  </Pie>
                </PieChart>
                <div className={styles.pieLegend}>
                  {pieData.map((item, index) => (
                    <span key={item.name} className={styles.legendRow}>
                      <span
                        className={styles.legendDot}
                        style={{
                          background:
                            ACCOUNT_COLORS[index % ACCOUNT_COLORS.length],
                        }}
                      />
                      <span className={styles.legendName}>{item.name}</span>
                      <span className={styles.legendAmount}>
                        {mask(formatKoreanMoney(item.amount))}
                      </span>
                      <span className={styles.legendValue}>
                        {item.weight_pct.toFixed(1)}%
                      </span>
                    </span>
                  ))}
                </div>
              </div>
            </section>

            <section className={styles.section}>
              <h2 className={styles.sectionTitle}>수익률</h2>
              <div className={styles.periodCard}>
                {PERIOD_LABELS.map(({ key, label }) => {
                  const item = periods[key];
                  return (
                    <div key={key} className={styles.periodRow}>
                      <span className={styles.metricLabel}>{label}</span>
                      <span
                        className={styles.periodValue}
                        style={{ color: signColorOf(item?.return_pct) }}
                      >
                        {formatPct(item?.return_pct)}
                        <span className={styles.periodProfit}>
                          {mask(formatKoreanMoney(item?.profit))}
                        </span>
                      </span>
                    </div>
                  );
                })}
              </div>
            </section>

            <div className={styles.homeMenu}>
              <Link href="/m/assets" className={styles.homeButton}>
                자산
              </Link>
            </div>
          </>
        )}
      </div>
    </MobileFrame>
  );
}
