"use client";

import { useCallback, useEffect, useState } from "react";

import { formatKstDateTime } from "@/lib/datetime";
import { formatPoolLabel } from "@/lib/pool-label";
import { ensureAsxPrefix } from "../components/TickerDetailLink";
import { useToast } from "../components/ToastProvider";

type Benchmark = { ticker?: string; name?: string };
type MarketIndexOption = { ticker: string; name: string };

type AccountEntry = {
  account_id: string;
  name?: string;
  icon?: string;
  order?: number;
  country_code?: string;
  currency?: string;
  cash_currencies?: string[];
  benchmark?: Benchmark;
  market_regime_index?: MarketIndexOption | null;
  /** 합성 전략에서 이 계좌로 운용할 종목풀 — 없으면 `/strategy-mix` 목록에 오르지 않는다. */
  mix_pool?: string | null;
  /** 증권사 API 연동 — 잔고 수동 불러오기·배치 동기화가 이 값으로 동작한다. */
  broker_api?: { provider: string; account_no: string } | null;
  URL?: string;
  updated_at?: string | null;
  save_method?: string | null;
};

type PoolOption = { ticker_type: string; name?: string | null; icon?: string | null; order?: number | null };

type BrokerProvider = { id: string; name: string; env_ok: boolean };
type BrokerAccountRow = {
  account_no: string;
  masked: string;
  acct_type: string;
  ok: boolean;
  cash?: number;
  holdings_count?: number;
  error?: string;
};

type BrokerHolding = { ticker: string; name: string; quantity: number; average_buy_price: number };
type BrokerBalanceDiff = {
  fetched: { cash: number; cash_d0: number; holdings: BrokerHolding[] };
  current: { cash: number; holdings: BrokerHolding[] };
};

type ApiResponse = {
  accounts?: AccountEntry[];
  market_indices?: MarketIndexOption[];
  pool_options?: PoolOption[];
  error?: string;
};

// 시장 레짐 지수 기본값 — S&P 500 (yf_ticker ^GSPC). 필수값이라 미설정 계좌는 이 값으로 시작한다.
const DEFAULT_REGIME_TICKER = "^GSPC";

// 현금 잔액(보유 현금 통화) 선택 옵션. 주 통화는 항상 포함되며 해제 불가.
const CASH_CURRENCY_OPTIONS = ["KRW", "USD", "AUD"] as const;

const inputStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "4px 8px",
  fontSize: "var(--fs-base)",
};
const labelStyle: React.CSSProperties = { color: "var(--text-muted)", fontWeight: 600, fontSize: "var(--fs-sm)", flexShrink: 0 };

/** 계좌 1개 편집 행 (자체 저장). account_id 는 원장 FK 라 수정 불가. */
function AccountRow({
  account,
  marketIndices,
  poolOptions,
  onSaved,
  onDeleted,
}: {
  account: AccountEntry;
  marketIndices: MarketIndexOption[];
  poolOptions: PoolOption[];
  onSaved: () => void;
  onDeleted: () => void;
}) {
  const toast = useToast();
  const [deleting, setDeleting] = useState(false);

  const remove = async () => {
    if (!window.confirm(`'${account.name ?? account.account_id}' 계좌를 삭제할까요?\n(보유종목이 있으면 삭제되지 않습니다)`)) return;
    setDeleting(true);
    try {
      const resp = await fetch("/api/account-settings", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account_id: account.account_id }),
      });
      const data = (await resp.json()) as { error?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? "계좌 삭제에 실패했습니다.");
      toast.success(`[계좌] ${account.account_id} 삭제 완료`);
      onDeleted();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "계좌 삭제에 실패했습니다.");
    } finally {
      setDeleting(false);
    }
  };
  const [name, setName] = useState(account.name ?? "");
  const [icon, setIcon] = useState(account.icon ?? "");
  const [order, setOrder] = useState(String(account.order ?? 0));
  const [countryCode, setCountryCode] = useState(account.country_code ?? "kor");
  const [currency, setCurrency] = useState(account.currency ?? "KRW");
  const [cashCurrencies, setCashCurrencies] = useState<string[]>(
    account.cash_currencies && account.cash_currencies.length > 0
      ? account.cash_currencies.map((c) => c.toUpperCase())
      : [(account.currency ?? "KRW").toUpperCase()],
  );
  const [url, setUrl] = useState(account.URL ?? "");
  // 시장 레짐 지수(필수) — 미설정 계좌는 S&P 500 기본값으로 시작.
  const [regimeTicker, setRegimeTicker] = useState(account.market_regime_index?.ticker || DEFAULT_REGIME_TICKER);
  const [mixPool, setMixPool] = useState(account.mix_pool ?? "");
  // 증권사 API 연동 — provider 를 고르고 '확인' 으로 계좌를 나열한 뒤 하나를 고른다.
  const [brokerProvider, setBrokerProvider] = useState(account.broker_api?.provider ?? "");
  const [brokerAccountNo, setBrokerAccountNo] = useState(account.broker_api?.account_no ?? "");
  const [brokerProviders, setBrokerProviders] = useState<BrokerProvider[]>([]);
  const [brokerAccounts, setBrokerAccounts] = useState<BrokerAccountRow[] | null>(null);
  const [brokerChecking, setBrokerChecking] = useState(false);
  // 잔고 불러오기 — 저장된 연동으로 API 잔고를 받아 현재 저장값과의 차이를 보여준다.
  const [balanceDiff, setBalanceDiff] = useState<BrokerBalanceDiff | null>(null);
  const [balanceLoading, setBalanceLoading] = useState(false);
  const [applying, setApplying] = useState(false);

  const fetchBalance = async () => {
    try {
      setBalanceLoading(true);
      setBalanceDiff(null);
      const resp = await fetch(`/api/broker-api/balance?account_id=${encodeURIComponent(account.account_id)}`, {
        cache: "no-store",
      });
      const data = (await resp.json()) as (BrokerBalanceDiff & { error?: string }) | { error?: string };
      if (!resp.ok || "error" in data && data.error) {
        toast.error(("error" in data && data.error) || "잔고를 불러오지 못했습니다.");
        return;
      }
      setBalanceDiff(data as BrokerBalanceDiff);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "잔고 조회 중 오류가 발생했습니다.");
    } finally {
      setBalanceLoading(false);
    }
  };

  const applyBalance = async () => {
    try {
      setApplying(true);
      const resp = await fetch("/api/broker-api/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account_id: account.account_id }),
      });
      const data = (await resp.json()) as { ok?: boolean; holdings_count?: number; error?: string };
      if (!resp.ok || data.error || !data.ok) {
        toast.error(data.error ?? "잔고 반영에 실패했습니다.");
        return;
      }
      toast.success(`반영 완료 — 보유 ${data.holdings_count}종목 · 현금 갱신`);
      setBalanceDiff(null);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "잔고 반영 중 오류가 발생했습니다.");
    } finally {
      setApplying(false);
    }
  };

  useEffect(() => {
    // 커넥터 목록 — 등록된 것만 셀렉트에 올려 오타를 원천 차단한다.
    fetch("/api/broker-api/providers", { cache: "no-store" })
      .then((resp) => resp.json())
      .then((data: { providers?: BrokerProvider[] }) => setBrokerProviders(data.providers ?? []))
      .catch(() => setBrokerProviders([]));
  }, []);

  const checkBroker = async () => {
    if (!brokerProvider) return;
    try {
      setBrokerChecking(true);
      setBrokerAccounts(null);
      const resp = await fetch(`/api/broker-api/accounts?provider=${encodeURIComponent(brokerProvider)}`, {
        cache: "no-store",
      });
      const data = (await resp.json()) as { accounts?: BrokerAccountRow[]; error?: string };
      if (!resp.ok || data.error) {
        toast.error(data.error ?? "증권사 계좌 조회에 실패했습니다.");
        return;
      }
      const rows = data.accounts ?? [];
      setBrokerAccounts(rows);
      const usable = rows.filter((row) => row.ok);
      toast.success(`연결 확인 — 계좌 ${rows.length}개 (조회 가능 ${usable.length}개)`);
      // 저장된 계좌가 목록에 없으면 선택을 비운다(계좌가 사라진 경우 그대로 두면 저장이 헛값이 된다).
      if (brokerAccountNo && !rows.some((row) => row.account_no === brokerAccountNo)) {
        setBrokerAccountNo("");
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "증권사 계좌 조회 중 오류가 발생했습니다.");
    } finally {
      setBrokerChecking(false);
    }
  };
  const [benchTicker, setBenchTicker] = useState(account.benchmark?.ticker ?? "");
  const [benchName, setBenchName] = useState(account.benchmark?.name ?? "");
  const [benchEditing, setBenchEditing] = useState(!(account.benchmark?.ticker && account.benchmark?.name));
  const [updatedAt, setUpdatedAt] = useState<string | null | undefined>(account.updated_at);
  const [saving, setSaving] = useState(false);
  const [resolving, setResolving] = useState(false);

  // 벤치마크 티커 → 종목명 조회 (공통: 종목풀 stock_meta). 이름은 조회로만 채움.
  const resolveBench = async () => {
    // 호주 계좌는 `ASX:` 를 붙여 조회·저장한다 — 종목풀·가격 캐시가 그 형태로 보관하고,
    // 미국에도 같은 티커가 있어(예: IVV) 접두사가 없으면 구분되지 않는다.
    const t = countryCode === "au" ? ensureAsxPrefix(benchTicker) : benchTicker.trim().toUpperCase();
    if (!t) {
      toast.error("티커를 입력해주세요.");
      return;
    }
    try {
      setResolving(true);
      const resp = await fetch(`/api/leverage-config/resolve?ticker=${encodeURIComponent(t)}`);
      const data = (await resp.json()) as { name?: string; error?: string };
      if (!resp.ok || data.error || !data.name) {
        toast.error(data.error ?? "종목명을 찾을 수 없습니다.");
        return;
      }
      // 정규화된 티커를 입력칸에도 되돌려 저장값과 화면이 어긋나지 않게 한다.
      setBenchTicker(t);
      setBenchName(data.name);
      setBenchEditing(false);
      toast.success(`${data.name}(${t}) 확인 완료`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "티커 조회 중 오류가 발생했습니다.");
    } finally {
      setResolving(false);
    }
  };

  const baseCurrency = currency.trim().toUpperCase();

  // 현금 잔액 토글. 주 통화는 항상 포함(해제 불가).
  const toggleCashCurrency = (code: string) => {
    setCashCurrencies((cur) => {
      if (cur.includes(code)) {
        if (code === baseCurrency) return cur; // 주 통화는 해제 불가
        return cur.filter((c) => c !== code);
      }
      return [...cur, code];
    });
  };

  const save = async () => {
    try {
      setSaving(true);
      // 주 통화는 반드시 현금 잔액에 포함(백엔드 검증과 동일). 순서 보존 + 중복 제거.
      const finalCashCurrencies = cashCurrencies.includes(baseCurrency)
        ? cashCurrencies
        : [baseCurrency, ...cashCurrencies];
      const values: Record<string, unknown> = {
        name: name.trim(),
        icon: icon.trim(),
        order: Math.trunc(Number(order)),
        country_code: countryCode,
        currency: baseCurrency,
        cash_currencies: finalCashCurrencies,
        market_regime_index: {
          ticker: regimeTicker,
          name: marketIndices.find((item) => item.ticker === regimeTicker)?.name ?? "",
        },
        // 합성 전략 종목풀 — 없음이면 null 로 저장한다(그 계좌는 합성 화면에 안 뜬다).
        mix_pool: mixPool || null,
        // 증권사 API 연동 — provider·계좌 둘 다 있어야 저장, 아니면 해제(null).
        broker_api:
          brokerProvider && brokerAccountNo
            ? { provider: brokerProvider, account_no: brokerAccountNo }
            : null,
        URL: url.trim(),
      };
      // 벤치마크는 선택 — 둘 다 채워졌을 때만 저장(빈 값이면 백엔드 검증에 걸리므로 생략).
      if (benchTicker.trim() && benchName.trim()) {
        values.benchmark = { ticker: benchTicker.trim(), name: benchName.trim() };
      }
      const resp = await fetch("/api/account-settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account_id: account.account_id, values }),
      });
      const data = (await resp.json()) as { updated_at?: string | null; error?: string; detail?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? data.detail ?? "저장에 실패했습니다.");
      setUpdatedAt(data.updated_at);
      toast.success(`[계좌] ${account.account_id} 저장 완료`);
      onSaved();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "저장에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  };

  const rowStyle: React.CSSProperties = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 6 };

  return (
    <div style={{ border: "1px solid rgba(148,163,184,0.25)", borderRadius: 8, padding: "10px 12px", marginBottom: 10 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
        <span style={{ fontWeight: 800 }}>
          {icon} {name} <span style={{ color: "var(--text-muted)", fontWeight: 500 }}>({account.account_id})</span>
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
            마지막 저장: {updatedAt ? formatKstDateTime(updatedAt) : "저장 이력 없음"}
          </span>
          <button
            type="button"
            className="btn btn-sm btn-dark"
            disabled={saving || !name.trim()}
            onClick={() => void save()}
          >
            {saving ? "저장 중…" : "저장"}
          </button>
          <button
            type="button"
            className="btn btn-sm btn-outline-danger"
            disabled={deleting}
            onClick={() => void remove()}
          >
            {deleting ? "삭제 중…" : "삭제"}
          </button>
        </span>
      </div>

      <div style={rowStyle}>
        <span style={{ ...labelStyle, width: 60 }}>이름</span>
        <input style={{ ...inputStyle, width: 160 }} value={name} onChange={(e) => setName(e.target.value)} />
        <span style={{ ...labelStyle, marginLeft: 8 }}>아이콘</span>
        <input style={{ ...inputStyle, width: 60 }} value={icon} onChange={(e) => setIcon(e.target.value)} />
        <span style={{ ...labelStyle, marginLeft: 8 }}>순서</span>
        <input style={{ ...inputStyle, width: 56 }} type="number" value={order} onChange={(e) => setOrder(e.target.value)} />
        <span style={{ ...labelStyle, marginLeft: 8 }}>국가</span>
        <select className="form-select form-select-sm" style={{ width: 90 }} value={countryCode} onChange={(e) => setCountryCode(e.target.value)}>
          <option value="kor">kor</option>
          <option value="au">au</option>
          <option value="us">us</option>
        </select>
        <span style={{ ...labelStyle, marginLeft: 8 }}>통화</span>
        <select
          className="form-select form-select-sm"
          style={{ width: 96 }}
          value={currency}
          onChange={(e) => {
            const next = e.target.value.toUpperCase();
            setCurrency(next);
            // 새 주 통화는 현금 잔액에 자동 포함(주 통화 현금은 항상 가능).
            setCashCurrencies((cur) => (cur.includes(next) ? cur : [...cur, next]));
          }}
        >
          <option value="KRW">KRW</option>
          <option value="AUD">AUD</option>
          <option value="USD">USD</option>
        </select>
      </div>

      <div style={rowStyle}>
        <span style={{ ...labelStyle, width: 60 }}>현금 잔액</span>
        {CASH_CURRENCY_OPTIONS.map((code) => {
          const checked = cashCurrencies.includes(code);
          const isBase = code === baseCurrency;
          return (
            <label
              key={code}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 4,
                fontSize: "var(--fs-sm)",
                color: isBase ? "var(--text-muted)" : undefined,
                cursor: isBase ? "not-allowed" : "pointer",
              }}
              title={isBase ? "주 통화는 항상 포함됩니다" : undefined}
            >
              <input
                type="checkbox"
                checked={checked}
                disabled={isBase}
                onChange={() => toggleCashCurrency(code)}
              />
              {code}
              {isBase ? " (주)" : ""}
            </label>
          );
        })}
      </div>

      <div style={rowStyle}>
        <span style={{ ...labelStyle, width: 60 }}>벤치마크</span>
        {benchEditing ? (
          <>
            <input
              style={{ ...inputStyle, width: 110 }}
              placeholder="티커"
              value={benchTicker}
              onChange={(e) => {
                setBenchTicker(e.target.value);
                setBenchName("");
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void resolveBench();
                }
              }}
            />
            <button type="button" className="btn btn-sm btn-outline-secondary" disabled={resolving} onClick={() => void resolveBench()}>
              {resolving ? "조회 중…" : "조회"}
            </button>
            <input
              style={{ ...inputStyle, flex: 1, minWidth: 140, background: "#f8fafc", color: "var(--text-muted)" }}
              placeholder="이름 (티커 입력 후 조회)"
              value={benchName}
              readOnly
            />
          </>
        ) : (
          <>
            <span style={{ fontSize: "var(--fs-base)", fontWeight: 600 }}>
              {benchName} <span style={{ color: "var(--text-muted)", fontWeight: 500 }}>({benchTicker})</span>
            </span>
            <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => setBenchEditing(true)}>
              변경
            </button>
          </>
        )}
      </div>


      <div style={rowStyle}>
        <span style={{ ...labelStyle, width: 60 }}>시장 레짐</span>
        <select
          className="form-select form-select-sm"
          style={{ width: 200 }}
          value={regimeTicker}
          onChange={(e) => setRegimeTicker(e.target.value)}
        >
          {marketIndices.map((item) => (
            <option key={item.ticker} value={item.ticker}>
              {item.name}
            </option>
          ))}
        </select>
      </div>

      <div style={rowStyle}>
        <span style={{ ...labelStyle, width: 60 }}>합성 전략</span>
        <select
          className="form-select form-select-sm"
          style={{ width: 240 }}
          value={mixPool}
          onChange={(e) => setMixPool(e.target.value)}
          title="이 계좌로 합성 전략을 운용할 종목풀. 지정한 계좌만 /strategy-mix 목록에 오른다."
        >
          <option value="">없음</option>
          {poolOptions.map((pool) => (
            <option key={pool.ticker_type} value={pool.ticker_type}>
              {formatPoolLabel(pool)}
            </option>
          ))}
        </select>
      </div>

      <div style={rowStyle}>
        <span style={{ ...labelStyle, width: 60 }}>API</span>
        <select
          className="form-select form-select-sm"
          style={{ width: 200 }}
          value={brokerProvider}
          onChange={(e) => {
            setBrokerProvider(e.target.value);
            setBrokerAccounts(null);
            setBrokerAccountNo("");
          }}
          title="증권사 API 연동 — 잔고 불러오기·배치 동기화가 이 연동으로 동작한다."
        >
          <option value="">없음</option>
          {brokerProviders.map((provider) => (
            <option key={provider.id} value={provider.id} disabled={!provider.env_ok}>
              {provider.name}
              {provider.env_ok ? "" : " (.env 키 없음)"}
            </option>
          ))}
        </select>
        {brokerProvider ? (
          <button
            type="button"
            className="btn btn-sm btn-outline-secondary"
            disabled={brokerChecking}
            onClick={() => void checkBroker()}
          >
            {brokerChecking ? "확인 중…" : "확인"}
          </button>
        ) : null}
        {/* 확인이 나열한 계좌 중 선택 — 조회 불가 계좌(종합/CMA 등)는 비활성. */}
        {brokerAccounts ? (
          <select
            className="form-select form-select-sm"
            style={{ width: 300 }}
            value={brokerAccountNo}
            onChange={(e) => setBrokerAccountNo(e.target.value)}
          >
            <option value="">계좌 선택</option>
            {brokerAccounts.map((row) => (
              <option key={row.account_no} value={row.account_no} disabled={!row.ok}>
                {row.masked}
                {row.ok
                  ? ` — 예수금 ${(row.cash ?? 0).toLocaleString("ko-KR")} · ${row.holdings_count}종목`
                  : " — 조회 불가"}
              </option>
            ))}
          </select>
        ) : brokerAccountNo ? (
          <span style={{ fontSize: "var(--fs-sm)", color: "var(--text-muted)" }}>
            저장된 계좌: {brokerAccountNo.slice(0, 3)}***{brokerAccountNo.slice(-2)}
          </span>
        ) : null}
        {/* 저장된 연동이 있어야 불러올 수 있다 — 셀렉트만 바꾼 상태에서는 먼저 저장. */}
        {account.broker_api?.account_no ? (
          <button
            type="button"
            className="btn btn-sm btn-outline-primary"
            disabled={balanceLoading}
            onClick={() => void fetchBalance()}
          >
            {balanceLoading ? "동기화 중…" : "잔고 동기화"}
          </button>
        ) : null}
      </div>

      {balanceDiff ? (
        <div
          style={{
            margin: "4px 0 10px",
            padding: "10px 12px",
            borderRadius: 8,
            background: "var(--bs-secondary-bg, #f1f5f9)",
            fontSize: "var(--fs-sm)",
          }}
        >
          {(() => {
            const fetchedBy = new Map(balanceDiff.fetched.holdings.map((h) => [h.ticker, h]));
            const currentBy = new Map(balanceDiff.current.holdings.map((h) => [h.ticker, h]));
            const tickers = [...new Set([...fetchedBy.keys(), ...currentBy.keys()])].sort();
            const stateOf = (tk: string): string => {
              const f = fetchedBy.get(tk);
              const c = currentBy.get(tk);
              if (!c) return "추가";
              if (!f) return "삭제";
              if (f.quantity !== c.quantity) return "수량 차이";
              if (Math.round(f.average_buy_price) !== Math.round(c.average_buy_price)) return "평단 차이";
              return "일치";
            };
            const diffCount = tickers.filter((tk) => stateOf(tk) !== "일치").length;
            const cashChanged = Math.round(balanceDiff.fetched.cash) !== Math.round(balanceDiff.current.cash);
            const number = (v: number | undefined) => (v == null ? "-" : v.toLocaleString("ko-KR"));
            const cellStyle: React.CSSProperties = { padding: "2px 10px 2px 0", textAlign: "right" };
            const diffColor = "var(--bs-danger, #dc3545)";
            return (
              <>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>
                  시스템 vs 증권사 API{" "}
                  <span style={{ fontWeight: 500, color: "var(--text-muted)" }}>
                    (현금은 D+2 예수금 기준 · 종목 차이 {diffCount}건{cashChanged ? " · 현금 차이" : ""})
                  </span>
                </div>
                <div style={{ marginBottom: 6, color: cashChanged ? diffColor : "var(--text-muted)" }}>
                  현금 — 시스템 {number(balanceDiff.current.cash)} · API {number(balanceDiff.fetched.cash)}
                </div>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ borderCollapse: "collapse", whiteSpace: "nowrap" }}>
                    <thead>
                      <tr style={{ color: "var(--text-muted)" }}>
                        <th style={{ ...cellStyle, textAlign: "left" }}>종목</th>
                        <th style={cellStyle}>시스템 수량</th>
                        <th style={cellStyle}>API 수량</th>
                        <th style={cellStyle}>시스템 평단</th>
                        <th style={cellStyle}>API 평단</th>
                        <th style={{ ...cellStyle, textAlign: "left" }}>상태</th>
                      </tr>
                    </thead>
                    <tbody>
                      {tickers.map((tk) => {
                        const f = fetchedBy.get(tk);
                        const c = currentBy.get(tk);
                        const state = stateOf(tk);
                        const changed = state !== "일치";
                        const qtyDiff = f && c && f.quantity !== c.quantity;
                        const avgDiff = f && c && Math.round(f.average_buy_price) !== Math.round(c.average_buy_price);
                        return (
                          <tr key={tk} style={changed ? { fontWeight: 600 } : undefined}>
                            <td style={{ ...cellStyle, textAlign: "left" }}>
                              {(f?.name || c?.name || tk)}({tk})
                            </td>
                            <td style={cellStyle}>{number(c?.quantity)}</td>
                            <td style={{ ...cellStyle, color: !c || qtyDiff ? diffColor : undefined }}>
                              {number(f?.quantity)}
                            </td>
                            <td style={cellStyle}>{number(c?.average_buy_price)}</td>
                            <td style={{ ...cellStyle, color: !c || avgDiff ? diffColor : undefined }}>
                              {number(f?.average_buy_price)}
                            </td>
                            <td style={{ ...cellStyle, textAlign: "left", color: changed ? diffColor : "var(--text-muted)" }}>
                              {state}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div style={{ marginTop: 8, display: "flex", gap: 8, alignItems: "center" }}>
                  <button
                    type="button"
                    className="btn btn-sm btn-primary"
                    disabled={applying || (diffCount === 0 && !cashChanged)}
                    onClick={() => void applyBalance()}
                  >
                    {applying ? "반영 중…" : "증권사 값으로 반영"}
                  </button>
                  <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => setBalanceDiff(null)}>
                    닫기
                  </button>
                  {diffCount === 0 && !cashChanged ? (
                    <span style={{ color: "var(--text-muted)" }}>차이가 없습니다.</span>
                  ) : null}
                </div>
              </>
            );
          })()}
        </div>
      ) : null}

      <div style={{ ...rowStyle, marginBottom: 0 }}>
        <span style={{ ...labelStyle, width: 60 }}>URL</span>
        <input style={{ ...inputStyle, flex: 1, minWidth: 220 }} placeholder="증권사 접속 URL (선택)" value={url} onChange={(e) => setUrl(e.target.value)} />
      </div>
    </div>
  );
}

function AddAccountModal({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const toast = useToast();
  const [accountId, setAccountId] = useState("");
  const [name, setName] = useState("");
  const [icon, setIcon] = useState("");
  const [order, setOrder] = useState("0");
  const [countryCode, setCountryCode] = useState("kor");
  const [currency, setCurrency] = useState("KRW");
  const [saving, setSaving] = useState(false);

  const create = async () => {
    setSaving(true);
    try {
      const resp = await fetch("/api/account-settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          account_id: accountId.trim().toLowerCase(),
          name: name.trim(),
          icon: icon.trim(),
          order: Math.trunc(Number(order) || 0),
          country_code: countryCode,
          currency: currency.trim().toUpperCase(),
        }),
      });
      const data = (await resp.json()) as { error?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? "계좌 추가에 실패했습니다.");
      toast.success(`[계좌] ${accountId.trim().toLowerCase()} 추가 완료`);
      onCreated();
      onClose();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "계좌 추가에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      style={{ position: "fixed", inset: 0, background: "rgba(15,23,42,0.45)", display: "grid", placeItems: "center", zIndex: 1000 }}
      onClick={onClose}
    >
      <div
        className="card appCard"
        style={{ width: 420, maxWidth: "92vw" }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <h3 style={{ fontSize: "var(--fs-base)", fontWeight: 800, margin: 0 }}>계좌 추가</h3>
          <p style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", margin: 0 }}>
            계좌 ID는 원장 FK라 생성 후 변경할 수 없습니다(영문 소문자·숫자·-·_). 벤치마크·전략 상세는 추가 후 편집하세요.
          </p>
          {[
            { label: "계좌 ID", el: <input style={{ ...inputStyle, flex: 1 }} value={accountId} placeholder="예: kiwoom-main" onChange={(e) => setAccountId(e.target.value)} /> },
            { label: "이름", el: <input style={{ ...inputStyle, flex: 1 }} value={name} onChange={(e) => setName(e.target.value)} /> },
            { label: "아이콘", el: <input style={{ ...inputStyle, flex: 1 }} value={icon} placeholder="선택(이모지)" onChange={(e) => setIcon(e.target.value)} /> },
            { label: "순서", el: <input style={{ ...inputStyle, flex: 1 }} type="number" value={order} onChange={(e) => setOrder(e.target.value)} /> },
          ].map((row) => (
            <label key={row.label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ ...labelStyle, width: 64 }}>{row.label}</span>
              {row.el}
            </label>
          ))}
          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ ...labelStyle, width: 64 }}>국가</span>
            <select className="form-select form-select-sm" style={{ flex: 1 }} value={countryCode} onChange={(e) => setCountryCode(e.target.value)}>
              <option value="kor">kor</option>
              <option value="au">au</option>
              <option value="us">us</option>
            </select>
            <span style={{ ...labelStyle, width: 40 }}>통화</span>
            <select className="form-select form-select-sm" style={{ width: 90 }} value={currency} onChange={(e) => setCurrency(e.target.value)}>
              <option value="KRW">KRW</option>
              <option value="USD">USD</option>
              <option value="AUD">AUD</option>
            </select>
          </label>
          <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 4 }}>
            <button type="button" className="btn btn-sm btn-light" onClick={onClose} disabled={saving}>취소</button>
            <button
              type="button"
              className="btn btn-sm btn-dark"
              disabled={saving || !accountId.trim() || !name.trim()}
              onClick={() => void create()}
            >
              {saving ? "추가 중…" : "추가"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export function AccountSettingsManager() {
  const toast = useToast();
  const [accounts, setAccounts] = useState<AccountEntry[]>([]);
  const [marketIndices, setMarketIndices] = useState<MarketIndexOption[]>([]);
  const [poolOptions, setPoolOptions] = useState<PoolOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [addOpen, setAddOpen] = useState(false);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const accResp = await fetch("/api/account-settings", { cache: "no-store" });
      const accData = (await accResp.json()) as ApiResponse;
      if (!accResp.ok || accData.error) throw new Error(accData.error ?? "계좌 설정을 불러오지 못했습니다.");
      setAccounts(accData.accounts ?? []);
      setMarketIndices(accData.market_indices ?? []);
      setPoolOptions(accData.pool_options ?? []);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "계좌 설정을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="card appCard">
      <div className="card-body">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 4 }}>
          <h2 style={{ fontSize: "var(--fs-lg)", fontWeight: 800, margin: 0 }}>계좌 설정</h2>
          <button type="button" className="btn btn-sm btn-primary" onClick={() => setAddOpen(true)}>+ 계좌 추가</button>
        </div>
        <p style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", marginBottom: 12 }}>
          계좌 메타(이름/순서/벤치마크 등)는 DB(account_settings)가 단일 소스입니다. 삭제는 보유종목이 없는 계좌만 가능합니다.
        </p>
        {loading ? (
          <div style={{ color: "var(--text-muted)", padding: 12 }}>불러오는 중…</div>
        ) : accounts.length === 0 ? (
          <div style={{ color: "var(--text-muted)", padding: 12 }}>등록된 계좌가 없습니다.</div>
        ) : (
          accounts.map((a) => (
            <AccountRow
              key={a.account_id}
              account={a}
              marketIndices={marketIndices}
              poolOptions={poolOptions}
              onSaved={() => {}}
              onDeleted={() => void load()}
            />
          ))
        )}
      </div>
      {addOpen ? <AddAccountModal onClose={() => setAddOpen(false)} onCreated={() => void load()} /> : null}
    </div>
  );
}
