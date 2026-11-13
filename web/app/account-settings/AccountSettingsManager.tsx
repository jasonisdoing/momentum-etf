"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { ColDef, ICellRendererParams } from "ag-grid-community";

import { formatKstDateTime } from "@/lib/datetime";
import { AppAgGrid } from "../components/AppAgGrid";
import { createAppGridTheme } from "../components/app-grid-theme";
import { AppModal } from "../components/AppModal";
import { LastSavedCell } from "../components/LastSavedCell";
import { ensureAsxPrefix } from "../components/TickerDetailLink";
import { useToast } from "../components/ToastProvider";
import { UnsavedChangesBadge } from "../components/UnsavedChangesBadge";
import { ApiOrderModal } from "./ApiOrderModal";
import { maskedBrokerAccount } from "./broker-label";

type Benchmark = { ticker?: string; name?: string };
type MarketIndexOption = { ticker: string; name: string };

type AccountEntry = {
  account_id: string;
  name?: string;
  icon?: string;
  order?: number;
  /** 계좌 구분 — 만든 뒤에는 바꿀 수 없다. */
  env: string;
  country_code?: string;
  currency?: string;
  cash_currencies?: string[];
  benchmark?: Benchmark;
  market_regime_index?: MarketIndexOption | null;
  /** 합성 전략에서 이 계좌로 운용할 종목풀 — 없으면 `/strategy-mix` 목록에 오르지 않는다. */
  /** 합성 슬리브별 종목풀 — 모멘텀·신고가가 서로 다른 풀을 볼 수 있다(계좌와 같은 국가). */
  /** 합성 전략 사용 여부 — 실제 조합(전략·종목풀·배분)은 `/strategy-mix` 에서 정한다. */
  mix_enabled?: boolean;
  /** 전략 사고팔기 사용 여부 — 켠 계좌만 `/strategy-trade` 에서 고를 수 있다. */
  trade_enabled?: boolean;
  /** 자산 화면·계좌 선택·집계에서 숨김 — 현금·보유·원금이 모두 0인 계좌만 켤 수 있다. */
  hidden?: boolean;
  /** 증권사 API 연동 — 잔고 수동 불러오기·배치 동기화가 이 값으로 동작한다. */
  /** 계좌별 키 커넥터(메리츠)는 `account_no` 대신 사용자가 입력한 표시용 마스킹 번호(`account_mask`)를 저장한다. */
  broker_api?: { provider: string; account_no?: string; account_mask?: string } | null;
  URL?: string;
  /** 보유종목 알림 — 이동선 이탈은 On/Off 만(기준은 종목의 종목풀 이평선). */
  ma_alarm_enabled?: boolean;
  /** 보유종목 알림 — 손절도 On/Off 만(기준 %는 종목의 종목풀 손절 기준). */
  stoploss_alarm_enabled?: boolean;
  /** API 주문 안전 장치(운영 계좌) — 켜져 있어야 주문이 나가고, 하루 건수·금액 한도를 넘으면 막힌다. */
  api_order_enabled?: boolean;
  api_order_max_count?: number | null;
  api_order_max_amount?: number | null;
  updated_at?: string | null;
  save_method?: string | null;
};

/** 커넥터 — `per_account_keys` 면 키가 이 계좌에 묶여(`{계좌ID}_API_KEY`) 증권사 계좌를 고르지 않는다. */
type BrokerProvider = { id: string; name: string; env_ok: boolean; per_account_keys: boolean; env_key_name: string };
type BrokerAccountRow = {
  account_no: string;
  masked: string;
  acct_type: string;
  /** 모의투자(03) 계좌 — 모의 계좌에는 이것만, 운영 계좌에는 이것만 빼고 연동한다. */
  mock: boolean;
  ok: boolean;
  cash?: number;
  net_asset?: number;
  holdings_count?: number;
  error?: string;
};

type BrokerHolding = { ticker: string; name: string; quantity: number; average_buy_price: number };
type BrokerBalanceDiff = {
  /** 현금은 통화별(계좌 현금 통화) — 미국 계좌는 USD·KRW 를 함께 담을 수 있다. */
  fetched: { cash_map: Record<string, number>; holdings: BrokerHolding[] };
  current: { cash_map: Record<string, number>; holdings: BrokerHolding[] };
};

type ApiResponse = {
  accounts?: AccountEntry[];
  market_indices?: MarketIndexOption[];
  error?: string;
};

// 시장 레짐 지수 기본값 — S&P 500 (yf_ticker ^GSPC). 필수값이라 미설정 계좌는 이 값으로 시작한다.
const DEFAULT_REGIME_TICKER = "^GSPC";

// 현금 잔액(보유 현금 통화) 선택 옵션. 주 통화는 항상 포함되며 해제 불가.
const CASH_CURRENCY_OPTIONS = ["KRW", "USD", "AUD"] as const;
const COUNTRY_OPTIONS = ["kor", "au", "us"] as const;
/** 계좌 구분 — 백엔드 `ACCOUNT_ENVS` 와 같다. 모의 계좌는 API 주문 시험용이라 합성·알림 설정과 사고팔기를 쓰지 않는다. */
const ENV_LIVE = "live";
const ENV_MOCK = "mock";
const ENV_LABELS: Record<string, string> = { [ENV_LIVE]: "운영", [ENV_MOCK]: "모의" };
const CURRENCY_OPTIONS = ["KRW", "USD", "AUD"] as const;

const inputStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "4px 8px",
  fontSize: "var(--fs-base)",
};
const labelStyle: React.CSSProperties = { color: "var(--text-muted)", fontWeight: 600, fontSize: "var(--fs-sm)", flexShrink: 0 };
const mutedStyle: React.CSSProperties = { color: "var(--text-muted)", fontSize: "var(--fs-sm)" };

/** 편집 중인 계좌 한 건. 그리드 셀이 그대로 읽고 쓰는 평평한 형태로 든다. */
type AccountDraft = {
  account_id: string;
  name: string;
  icon: string;
  order: string;
  env: string;
  country_code: string;
  currency: string;
  cash_currencies: string[];
  benchmarkTicker: string;
  benchmarkName: string;
  regimeTicker: string;
  mix_enabled: boolean;
  trade_enabled: boolean;
  hidden: boolean;
  brokerProvider: string;
  brokerAccountNo: string;
  /** 메리츠 계좌번호(표시용 수기 입력) — 증권사 API 로 대조할 수 없는 값이다. */
  brokerAccountMask: string;
  URL: string;
  ma_alarm_enabled: boolean;
  stoploss_alarm_enabled: boolean;
  api_order_enabled: boolean;
  api_order_max_count: string;
  api_order_max_amount: string;
};

/** 그리드 행 — 편집 중인 초안 그대로에 표시용 필드를 얹는다. */
type AccountGridRow = AccountDraft & { __dirty: boolean; __brokerLinked: boolean; __updatedAt?: string | null; __saveMethod?: string | null };

// 셀렉트·체크박스 에디터가 들어가는 행이라 기본(34px)보다 조금 높인다.
const accountGridTheme = createAppGridTheme({ rowHeight: 38 });

function toDraft(account: AccountEntry): AccountDraft {
  const currency = (account.currency ?? "KRW").toUpperCase();
  return {
    account_id: account.account_id,
    name: account.name ?? "",
    icon: account.icon ?? "",
    order: String(account.order ?? 0),
    env: account.env,
    country_code: account.country_code ?? "kor",
    currency,
    cash_currencies:
      account.cash_currencies && account.cash_currencies.length > 0
        ? account.cash_currencies.map((code) => code.toUpperCase())
        : [currency],
    benchmarkTicker: account.benchmark?.ticker ?? "",
    benchmarkName: account.benchmark?.name ?? "",
    // 시장 레짐 지수(필수) — 미설정 계좌는 S&P 500 기본값으로 시작.
    regimeTicker: account.market_regime_index?.ticker || DEFAULT_REGIME_TICKER,
    mix_enabled: Boolean(account.mix_enabled),
    trade_enabled: Boolean(account.trade_enabled),
    hidden: Boolean(account.hidden),
    brokerProvider: account.broker_api?.provider ?? "",
    brokerAccountNo: account.broker_api?.account_no ?? "",
    brokerAccountMask: account.broker_api?.account_mask ?? "",
    URL: account.URL ?? "",
    ma_alarm_enabled: Boolean(account.ma_alarm_enabled),
    stoploss_alarm_enabled: Boolean(account.stoploss_alarm_enabled),
    api_order_enabled: Boolean(account.api_order_enabled),
    api_order_max_count: account.api_order_max_count == null ? "" : String(account.api_order_max_count),
    api_order_max_amount: account.api_order_max_amount == null ? "" : String(account.api_order_max_amount),
  };
}

function draftToValues(draft: AccountDraft, marketIndices: MarketIndexOption[]): Record<string, unknown> {
  const baseCurrency = draft.currency.trim().toUpperCase();
  // 주 통화는 반드시 현금 잔액에 포함(백엔드 검증과 동일). 순서 보존 + 중복 제거.
  const cashCurrencies = draft.cash_currencies.includes(baseCurrency)
    ? draft.cash_currencies
    : [baseCurrency, ...draft.cash_currencies];

  const values: Record<string, unknown> = {
    name: draft.name.trim(),
    icon: draft.icon.trim(),
    order: Math.trunc(Number(draft.order)),
    country_code: draft.country_code,
    currency: baseCurrency,
    cash_currencies: cashCurrencies,
    market_regime_index: {
      ticker: draft.regimeTicker,
      name: marketIndices.find((item) => item.ticker === draft.regimeTicker)?.name ?? "",
    },
    // 합성 전략 종목풀 — 없음이면 null 로 저장한다(그 계좌는 합성 화면에 안 뜬다).
    mix_enabled: draft.mix_enabled,
    trade_enabled: draft.trade_enabled,
    hidden: draft.hidden,
    // 증권사 API 연동 — 커넥터를 고르면 그대로 보낸다. 계좌번호가 필요한지(나무)·없어야 하는지(메리츠)는
    // 백엔드가 커넥터별로 검증해 저장을 막는다(빠진 값을 조용히 해제하지 않는다).
    broker_api: draft.brokerProvider
      ? { provider: draft.brokerProvider, account_no: draft.brokerAccountNo, account_mask: draft.brokerAccountMask.trim() }
      : null,
    URL: draft.URL.trim(),
    ma_alarm_enabled: draft.ma_alarm_enabled,
    stoploss_alarm_enabled: draft.stoploss_alarm_enabled,
    api_order_enabled: draft.api_order_enabled,
    api_order_max_count: draft.api_order_max_count === "" ? null : Number(draft.api_order_max_count),
    api_order_max_amount: draft.api_order_max_amount === "" ? null : Number(draft.api_order_max_amount),
  };
  // 벤치마크는 선택 — 둘 다 채워졌을 때만 저장(빈 값이면 백엔드 검증에 걸리므로 생략).
  if (draft.benchmarkTicker.trim() && draft.benchmarkName.trim()) {
    values.benchmark = { ticker: draft.benchmarkTicker.trim(), name: draft.benchmarkName.trim() };
  }
  return values;
}

function isDirty(draft: AccountDraft, account: AccountEntry): boolean {
  return JSON.stringify(draft) !== JSON.stringify(toDraft(account));
}

/** 계좌 하나의 상세 설정 — 셀 한 칸에 담기지 않는 것만 모았다.
 *
 * 현금 잔액(체크 여러 개), 벤치마크(티커 조회 2단계), 증권사 API(선택 → 확인 → 계좌 선택),
 * 잔고 동기화(비교표)는 모두 한 번의 입력으로 끝나지 않아 그리드 셀에 넣을 수 없다.
 */
function AccountDetailModal({
  draft,
  savedBrokerProvider,
  onChange,
  onClose,
}: {
  draft: AccountDraft;
  /** 저장된 연동이 있어야 잔고를 불러올 수 있다 — 셀렉트만 바꾼 상태에서는 먼저 저장. */
  savedBrokerProvider: string;
  onChange: (patch: Partial<AccountDraft>) => void;
  onClose: () => void;
}) {
  const toast = useToast();
  const [brokerProviders, setBrokerProviders] = useState<BrokerProvider[]>([]);
  const [brokerAccounts, setBrokerAccounts] = useState<BrokerAccountRow[] | null>(null);
  const [brokerChecking, setBrokerChecking] = useState(false);
  const [balanceDiff, setBalanceDiff] = useState<BrokerBalanceDiff | null>(null);
  const [balanceLoading, setBalanceLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [resolving, setResolving] = useState(false);
  const [benchEditing, setBenchEditing] = useState(!(draft.benchmarkTicker && draft.benchmarkName));

  const baseCurrency = draft.currency.trim().toUpperCase();

  useEffect(() => {
    // 커넥터 목록 — 등록된 것만 셀렉트에 올려 오타를 원천 차단한다.
    // 키 등록 여부는 계좌마다 다르다(메리츠는 `{계좌ID}_API_KEY`) — 이 계좌 기준으로 받는다.
    fetch(`/api/broker-api/providers?account_id=${encodeURIComponent(draft.account_id)}`, { cache: "no-store" })
      .then((resp) => resp.json())
      .then((data: { providers?: BrokerProvider[] }) => setBrokerProviders(data.providers ?? []))
      .catch(() => setBrokerProviders([]));
  }, [draft.account_id]);
  const selectedProvider = brokerProviders.find((provider) => provider.id === draft.brokerProvider) ?? null;

  // 현금 잔액 토글. 주 통화는 항상 포함(해제 불가).
  const toggleCashCurrency = (code: string) => {
    if (draft.cash_currencies.includes(code)) {
      if (code === baseCurrency) return; // 주 통화는 해제 불가
      onChange({ cash_currencies: draft.cash_currencies.filter((item) => item !== code) });
      return;
    }
    onChange({ cash_currencies: [...draft.cash_currencies, code] });
  };

  // 벤치마크 티커 → 종목명 조회 (공통: 종목풀 stock_meta). 이름은 조회로만 채운다.
  const resolveBench = async () => {
    // 호주 계좌는 `ASX:` 를 붙여 조회·저장한다 — 종목풀·가격 캐시가 그 형태로 보관하고,
    // 미국에도 같은 티커가 있어(예: IVV) 접두사가 없으면 구분되지 않는다.
    const target = draft.country_code === "au" ? ensureAsxPrefix(draft.benchmarkTicker) : draft.benchmarkTicker.trim().toUpperCase();
    if (!target) {
      toast.error("티커를 입력해주세요.");
      return;
    }
    try {
      setResolving(true);
      const resp = await fetch(`/api/leverage-config/resolve?ticker=${encodeURIComponent(target)}`);
      const data = (await resp.json()) as { name?: string; error?: string };
      if (!resp.ok || data.error || !data.name) {
        toast.error(data.error ?? "종목명을 찾을 수 없습니다.");
        return;
      }
      // 정규화된 티커를 입력칸에도 되돌려 저장값과 화면이 어긋나지 않게 한다.
      onChange({ benchmarkTicker: target, benchmarkName: data.name });
      setBenchEditing(false);
      toast.success(`${data.name}(${target}) 확인 완료`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "티커 조회 중 오류가 발생했습니다.");
    } finally {
      setResolving(false);
    }
  };

  const checkBroker = async () => {
    if (!draft.brokerProvider) return;
    try {
      setBrokerChecking(true);
      setBrokerAccounts(null);
      const resp = await fetch(`/api/broker-api/accounts?provider=${encodeURIComponent(draft.brokerProvider)}`, {
        cache: "no-store",
      });
      const data = (await resp.json()) as { accounts?: BrokerAccountRow[]; error?: string };
      if (!resp.ok || data.error) {
        toast.error(data.error ?? "증권사 계좌 조회에 실패했습니다.");
        return;
      }
      const rows = data.accounts ?? [];
      setBrokerAccounts(rows);
      toast.success(`연결 확인 — 계좌 ${rows.length}개 (조회 가능 ${rows.filter((row) => row.ok).length}개)`);
      // 저장된 계좌가 목록에 없으면 선택을 비운다(계좌가 사라진 경우 그대로 두면 저장이 헛값이 된다).
      if (draft.brokerAccountNo && !rows.some((row) => row.account_no === draft.brokerAccountNo)) {
        onChange({ brokerAccountNo: "" });
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "증권사 계좌 조회 중 오류가 발생했습니다.");
    } finally {
      setBrokerChecking(false);
    }
  };

  const fetchBalance = async () => {
    try {
      setBalanceLoading(true);
      setBalanceDiff(null);
      const resp = await fetch(`/api/broker-api/balance?account_id=${encodeURIComponent(draft.account_id)}`, {
        cache: "no-store",
      });
      const data = (await resp.json()) as (BrokerBalanceDiff & { error?: string }) | { error?: string };
      if (!resp.ok || ("error" in data && data.error)) {
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
        body: JSON.stringify({ account_id: draft.account_id }),
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

  const rowStyle: React.CSSProperties = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 10 };

  return (
    <AppModal
      open
      title={`${draft.icon} ${draft.name}`.trim() || draft.account_id}
      subtitle={`${draft.account_id} — 셀에 담기지 않는 설정입니다. 변경은 그리드 상단 저장으로 확정됩니다.`}
      onClose={onClose}
      size="xl"
      footer={
        <div style={{ display: "flex", justifyContent: "flex-end", width: "100%" }}>
          <button type="button" className="btn btn-primary" onClick={onClose}>
            닫기
          </button>
        </div>
      }
    >
      <div style={rowStyle}>
        <span style={{ ...labelStyle, width: 72 }}>현금 잔액</span>
        {CASH_CURRENCY_OPTIONS.map((code) => {
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
                checked={draft.cash_currencies.includes(code)}
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
        <span style={{ ...labelStyle, width: 72 }}>URL</span>
        <input
          style={{ ...inputStyle, flex: 1, minWidth: 260 }}
          placeholder="증권사 접속 URL (선택)"
          value={draft.URL}
          onChange={(e) => onChange({ URL: e.target.value })}
        />
      </div>

      <div style={rowStyle}>
        <span style={{ ...labelStyle, width: 72 }}>벤치마크</span>
        {benchEditing ? (
          <>
            <input
              style={{ ...inputStyle, width: 130 }}
              placeholder="티커"
              value={draft.benchmarkTicker}
              onChange={(e) => onChange({ benchmarkTicker: e.target.value, benchmarkName: "" })}
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
              style={{ ...inputStyle, flex: 1, minWidth: 160, background: "#f8fafc", color: "var(--text-muted)" }}
              placeholder="이름 (티커 입력 후 조회)"
              value={draft.benchmarkName}
              readOnly
            />
          </>
        ) : (
          <>
            <span style={{ fontSize: "var(--fs-base)", fontWeight: 600 }}>
              {draft.benchmarkName} <span style={{ color: "var(--text-muted)", fontWeight: 500 }}>({draft.benchmarkTicker})</span>
            </span>
            <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => setBenchEditing(true)}>
              변경
            </button>
          </>
        )}
      </div>

      <div style={rowStyle}>
        <span style={{ ...labelStyle, width: 72 }}>증권사 API</span>
        <select
          className="form-select form-select-sm"
          style={{ width: 200 }}
          value={draft.brokerProvider}
          onChange={(e) => {
            onChange({ brokerProvider: e.target.value, brokerAccountNo: "", brokerAccountMask: "" });
            setBrokerAccounts(null);
          }}
          title="증권사 API 연동 — 잔고 불러오기·배치 동기화가 이 연동으로 동작한다."
        >
          <option value="">없음</option>
          {brokerProviders.map((provider) => (
            <option key={provider.id} value={provider.id} disabled={!provider.env_ok}>
              {provider.name}
              {provider.env_ok ? "" : ` (.env ${provider.env_key_name} 없음)`}
            </option>
          ))}
        </select>
        {/* 계좌별 키 커넥터(메리츠) — 키가 이 계좌에 묶여 있어 증권사 계좌를 고르지 않는다. 키 이름만 보여 준다. */}
        {selectedProvider?.per_account_keys ? (
          <>
            <input
              className="form-control form-control-sm"
              style={{ width: 140 }}
              value={draft.brokerAccountMask}
              placeholder="예: 3082****01"
              onChange={(e) => onChange({ brokerAccountMask: e.target.value })}
              title="표시용 계좌번호 — 증권사 API 가 계좌번호를 주지 않아 대조할 수 없는 수기 값입니다."
            />
            <span style={{ fontSize: "var(--fs-sm)", color: "var(--text-muted)" }}>
              키 {selectedProvider.env_key_name} {selectedProvider.env_ok ? "등록됨" : "없음"}
            </span>
          </>
        ) : draft.brokerProvider ? (
          <button type="button" className="btn btn-sm btn-outline-secondary" disabled={brokerChecking} onClick={() => void checkBroker()}>
            {brokerChecking ? "확인 중…" : "확인"}
          </button>
        ) : null}
        {/* 확인이 나열한 계좌 중 선택 — 조회 불가 계좌(종합/CMA 등)는 비활성. */}
        {brokerAccounts ? (
          <select
            className="form-select form-select-sm"
            style={{ width: 320 }}
            value={draft.brokerAccountNo}
            onChange={(e) => onChange({ brokerAccountNo: e.target.value })}
          >
            <option value="">계좌 선택</option>
            {brokerAccounts
              .filter((row) => row.mock === (draft.env === ENV_MOCK))
              .map((row) => (
                <option key={row.account_no} value={row.account_no} disabled={!row.mock && !row.ok}>
                  {row.masked}
                  {row.mock
                    ? " — 모의투자"
                    : row.ok
                      ? ` — 순자산 ${(row.net_asset ?? 0).toLocaleString("ko-KR")} · 예수금 ${(row.cash ?? 0).toLocaleString("ko-KR")} · ${row.holdings_count}종목`
                      : " — 조회 불가"}
                </option>
              ))}
          </select>
        ) : draft.brokerAccountNo ? (
          <span style={{ fontSize: "var(--fs-sm)", color: "var(--text-muted)" }}>
            저장된 계좌: {maskedBrokerAccount(draft.brokerAccountNo, null)}
          </span>
        ) : null}
        {savedBrokerProvider && draft.env !== ENV_MOCK ? (
          <button type="button" className="btn btn-sm btn-outline-primary" disabled={balanceLoading} onClick={() => void fetchBalance()}>
            {balanceLoading ? "동기화 중…" : "잔고 동기화"}
          </button>
        ) : null}
      </div>

      {balanceDiff ? (
        <div style={{ padding: "10px 12px", borderRadius: 8, background: "var(--bs-secondary-bg, #f1f5f9)", fontSize: "var(--fs-sm)" }}>
          {(() => {
            const fetchedBy = new Map(balanceDiff.fetched.holdings.map((h) => [h.ticker, h]));
            const currentBy = new Map(balanceDiff.current.holdings.map((h) => [h.ticker, h]));
            const tickers = [...new Set([...fetchedBy.keys(), ...currentBy.keys()])].sort();
            // 평단 비교는 소수 4자리 — 동기화 배치의 슬랙 비교(`broker_balance_sync._AVG_DIGITS`)와 같은 기준.
            const avgDiffers = (f: BrokerHolding, c: BrokerHolding) =>
              f.average_buy_price.toFixed(4) !== c.average_buy_price.toFixed(4);
            const stateOf = (tk: string): string => {
              const f = fetchedBy.get(tk);
              const c = currentBy.get(tk);
              if (!c) return "추가";
              if (!f) return "삭제";
              if (f.quantity !== c.quantity) return "수량 차이";
              if (avgDiffers(f, c)) return "평단 차이";
              return "일치";
            };
            const diffCount = tickers.filter((tk) => stateOf(tk) !== "일치").length;
            // 통화별 현금 비교 — 원화는 원, 외화는 센트 단위(동기화 배치의 비교와 같은 기준).
            const cashDigits = (code: string) => (code === "KRW" ? 0 : 2);
            const cashCodes = Object.keys(balanceDiff.fetched.cash_map);
            const cashDiffers = (code: string) =>
              balanceDiff.fetched.cash_map[code].toFixed(cashDigits(code)) !==
              (balanceDiff.current.cash_map[code] ?? 0).toFixed(cashDigits(code));
            const cashChanged = cashCodes.some(cashDiffers);
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
                {cashCodes.map((code) => (
                  <div key={code} style={{ marginBottom: 6, color: cashDiffers(code) ? diffColor : "var(--text-muted)" }}>
                    현금 {code} — 시스템 {number(balanceDiff.current.cash_map[code])} · API {number(balanceDiff.fetched.cash_map[code])}
                  </div>
                ))}
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
                        const avgDiff = f && c && avgDiffers(f, c);
                        return (
                          <tr key={tk} style={changed ? { fontWeight: 600 } : undefined}>
                            <td style={{ ...cellStyle, textAlign: "left" }}>{(f?.name || c?.name || tk)}({tk})</td>
                            <td style={cellStyle}>{number(c?.quantity)}</td>
                            <td style={{ ...cellStyle, color: !c || qtyDiff ? diffColor : undefined }}>{number(f?.quantity)}</td>
                            <td style={cellStyle}>{number(c?.average_buy_price)}</td>
                            <td style={{ ...cellStyle, color: !c || avgDiff ? diffColor : undefined }}>{number(f?.average_buy_price)}</td>
                            <td style={{ ...cellStyle, textAlign: "left", color: changed ? diffColor : "var(--text-muted)" }}>{state}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div style={{ marginTop: 8, display: "flex", gap: 8, alignItems: "center" }}>
                  <button type="button" className="btn btn-sm btn-primary" disabled={applying} onClick={() => void applyBalance()}>
                    {applying ? "반영 중…" : "증권사 데이터 덮어쓰기"}
                  </button>
                  <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => setBalanceDiff(null)}>
                    닫기
                  </button>
                  {diffCount === 0 && !cashChanged ? <span style={{ color: "var(--text-muted)" }}>차이가 없습니다.</span> : null}
                </div>
              </>
            );
          })()}
        </div>
      ) : null}
    </AppModal>
  );
}

function AddAccountModal({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const toast = useToast();
  const [accountId, setAccountId] = useState("");
  const [name, setName] = useState("");
  const [icon, setIcon] = useState("");
  const [order, setOrder] = useState("0");
  const [env, setEnv] = useState("");
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
          env,
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
    <AppModal
      open
      title="계좌 추가"
      subtitle="계좌 ID는 원장 FK라 생성 후 변경할 수 없습니다(영문 소문자·숫자·-·_). 벤치마크·전략 상세는 추가 후 편집하세요."
      onClose={onClose}
      footer={
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, width: "100%" }}>
          <button type="button" className="btn btn-outline-secondary" onClick={onClose} disabled={saving}>
            취소
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={saving || !accountId.trim() || !name.trim() || !env}
            onClick={() => void create()}
          >
            {saving ? "추가 중…" : "추가"}
          </button>
        </div>
      }
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
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
          <span style={{ ...labelStyle, width: 64 }}>구분</span>
          <select className="form-select form-select-sm" style={{ flex: 1 }} value={env} onChange={(e) => setEnv(e.target.value)}>
            <option value="">선택</option>
            {Object.entries(ENV_LABELS).map(([code, label]) => (
              <option key={code} value={code}>{label}</option>
            ))}
          </select>
          <span style={mutedStyle}>만든 뒤에는 바꿀 수 없습니다</span>
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ ...labelStyle, width: 64 }}>국가</span>
          <select className="form-select form-select-sm" style={{ flex: 1 }} value={countryCode} onChange={(e) => setCountryCode(e.target.value)}>
            {COUNTRY_OPTIONS.map((code) => (
              <option key={code} value={code}>{code}</option>
            ))}
          </select>
          <span style={{ ...labelStyle, width: 40 }}>통화</span>
          <select className="form-select form-select-sm" style={{ width: 90 }} value={currency} onChange={(e) => setCurrency(e.target.value)}>
            {CURRENCY_OPTIONS.map((code) => (
              <option key={code} value={code}>{code}</option>
            ))}
          </select>
        </label>
      </div>
    </AppModal>
  );
}

export function AccountSettingsManager() {
  const toast = useToast();
  const [accounts, setAccounts] = useState<AccountEntry[]>([]);
  const [marketIndices, setMarketIndices] = useState<MarketIndexOption[]>([]);
  const [drafts, setDrafts] = useState<Record<string, AccountDraft>>({});
  const [loading, setLoading] = useState(true);
  const [addOpen, setAddOpen] = useState(false);
  const [mockOrderId, setMockOrderId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [sendingAlarm, setSendingAlarm] = useState(false);
  // 삭제는 체크박스로 고른 행을 상단 버튼으로 한 번에 처리한다.
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [detailId, setDetailId] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const resp = await fetch("/api/account-settings", { cache: "no-store" });
      const data = (await resp.json()) as ApiResponse;
      if (!resp.ok || data.error) throw new Error(data.error ?? "계좌 설정을 불러오지 못했습니다.");
      const list = data.accounts ?? [];
      setAccounts(list);
      setMarketIndices(data.market_indices ?? []);
      const nextDrafts: Record<string, AccountDraft> = {};
      list.forEach((account) => {
        nextDrafts[account.account_id] = toDraft(account);
      });
      setDrafts(nextDrafts);
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

  const rows = useMemo(() => [...accounts].sort((a, b) => (a.order ?? 0) - (b.order ?? 0)), [accounts]);

  // 그리드 행 = 초안 그대로 + 변경 여부. 초안을 그리므로 저장 전 값이 화면에 남는다.
  // AppAgGrid 가 rowData 가 바뀔 때마다 행을 다시 그리므로, 실제로 바뀔 때만 새 배열을 만든다.
  const gridRows = useMemo<AccountGridRow[]>(
    () =>
      rows.map((account) => {
        const draft = drafts[account.account_id] ?? toDraft(account);
        return {
          ...draft,
          __dirty: isDirty(draft, account),
          // 저장된 증권사 연동(나무·메리츠) — API 주문은 저장된 연동으로 연다.
          __brokerLinked: Boolean(account.broker_api?.provider),
          __updatedAt: account.updated_at, __saveMethod: account.save_method };
      }),
    [drafts, rows],
  );
  const dirtyCount = gridRows.filter((row) => row.__dirty).length;

  const updateDraft = useCallback((id: string, patch: Partial<AccountDraft>) => {
    setDrafts((prev) => ({ ...prev, [id]: { ...prev[id], ...patch } }));
  }, []);

  /** 변경된 행만 모아 한 번에 저장한다 — 상단 저장 버튼 1개가 전부를 처리한다. */
  const handleSaveAll = useCallback(async () => {
    const targets = rows.filter((account) => {
      const draft = drafts[account.account_id];
      return draft && isDirty(draft, account);
    });
    if (targets.length === 0) return;

    setSaving(true);
    const failed: string[] = [];
    try {
      for (const account of targets) {
        try {
          const resp = await fetch("/api/account-settings", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              account_id: account.account_id,
              values: draftToValues(drafts[account.account_id], marketIndices),
            }),
          });
          const data = (await resp.json()) as { error?: string; detail?: string };
          if (!resp.ok || data.error) throw new Error(data.error ?? data.detail ?? "저장에 실패했습니다.");
        } catch (err) {
          failed.push(`${account.account_id}: ${err instanceof Error ? err.message : "저장 실패"}`);
        }
      }
      const savedCount = targets.length - failed.length;
      if (savedCount > 0) toast.success(`계좌 ${savedCount}개를 저장했습니다.`);
      if (failed.length > 0) toast.error(`저장 실패 ${failed.length}건 — ${failed.join(" / ")}`);
      await load();
    } finally {
      setSaving(false);
    }
  }, [drafts, load, marketIndices, rows, toast]);

  /** 체크한 행을 한 번에 삭제한다. 보유종목이 있는 계좌는 서버가 막는다. */
  const handleDeleteSelected = useCallback(async () => {
    const targets = rows.filter((account) => selectedIds.includes(account.account_id));
    if (targets.length === 0) return;

    const message = [
      `계좌 ${targets.length}개를 삭제합니다.`,
      targets.map((account) => `  • ${account.name ?? account.account_id} (${account.account_id})`).join("\n"),
      "",
      "보유종목이 있으면 삭제되지 않습니다.",
      "계속할까요?",
    ].join("\n");
    if (!window.confirm(message)) return;

    setDeleting(true);
    const failed: string[] = [];
    try {
      for (const account of targets) {
        try {
          const resp = await fetch("/api/account-settings", {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ account_id: account.account_id }),
          });
          const data = (await resp.json()) as { error?: string };
          if (!resp.ok || data.error) throw new Error(data.error ?? "계좌 삭제에 실패했습니다.");
        } catch (err) {
          failed.push(`${account.account_id}: ${err instanceof Error ? err.message : "삭제 실패"}`);
        }
      }
      const okCount = targets.length - failed.length;
      if (okCount > 0) toast.success(`계좌 ${okCount}개를 삭제했습니다.`);
      if (failed.length > 0) toast.error(`삭제 실패 ${failed.length}건 — ${failed.join(" / ")}`);
      setSelectedIds([]);
      await load();
    } finally {
      setDeleting(false);
    }
  }, [load, rows, selectedIds, toast]);

  /** 지금 조건을 만족하는 종목이 있으면 슬랙으로 보낸다 — 저장된 설정 그대로 확인용. */
  const sendAlarmTest = async () => {
    setSendingAlarm(true);
    try {
      const resp = await fetch("/api/alarms/send", { method: "POST" });
      const payload = (await resp.json()) as { sent?: boolean; reason?: string; accounts?: number; error?: string };
      if (!resp.ok || payload.error) throw new Error(payload.error ?? "발송에 실패했습니다.");
      if (payload.sent) toast.success(`슬랙 발송 완료 (${payload.accounts ?? 0}개 계좌)`);
      else toast.error(payload.reason ?? "발송 대상이 없습니다.");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "발송에 실패했습니다.");
    } finally {
      setSendingAlarm(false);
    }
  };

  /** 셀렉트 편집 컬럼 — 목록 밖 저장값도 후보에 남겨 빈 셀렉트가 되지 않게 한다. */
  const selectCol = (
    field: keyof AccountDraft & ColDef<AccountGridRow>["field"],
    headerName: string,
    width: number,
    values: (row: AccountGridRow) => (string | number)[],
    extra?: Partial<ColDef<AccountGridRow>>,
  ): ColDef<AccountGridRow> => ({
    field,
    headerName,
    width,
    editable: true,
    cellEditor: "agSelectCellEditor",
    cellEditorParams: (params: { data: AccountGridRow }) => {
      const list = values(params.data).map(String);
      const current = String(params.data[field] ?? "");
      return { values: current && !list.includes(current) ? [current, ...list] : list };
    },
    ...extra,
  });

  /** 숫자 편집 컬럼 — 초안은 문자열로 들지만 그리드에는 숫자로 넘긴다.
   *
   * 문자열인 채로 두면 AG Grid 가 셀 타입을 `text` 로 추론해 `agNumberCellEditor` 가
   * 호환되지 않는 것으로 판정돼 **편집 자체가 되지 않는다**. valueGetter/valueSetter 로
   * 그리드 쪽만 숫자로 바꿔 준다. (`field` 대신 `colId` 를 쓰므로 onCellValueChanged 는
   * 이 컬럼을 건너뛴다 — 저장은 valueSetter 가 이미 했다.)
   */
  const numberCol = (
    field: "order" | "api_order_max_count" | "api_order_max_amount",
    headerName: string,
    width: number,
    liveOnly = false,
    headerTooltip?: string,
  ): ColDef<AccountGridRow> => ({
    colId: field,
    headerName,
    headerTooltip,
    width,
    // 운영 계좌 전용 컬럼은 모의 계좌에서 편집하지 못하고 "—" 로 보인다.
    editable: (params) => !liveOnly || params.data?.env === ENV_LIVE,
    cellDataType: "number",
    type: "numericColumn",
    valueFormatter: (params) =>
      liveOnly && params.data?.env === ENV_MOCK ? "—" : params.value == null ? "" : Number(params.value).toLocaleString("ko-KR"),
    valueGetter: (params) => {
      const raw = params.data?.[field];
      return raw === "" || raw === undefined ? null : Number(raw);
    },
    valueSetter: (params) => {
      if (!params.data) return false;
      const next = params.newValue === null || params.newValue === "" ? "" : String(Math.trunc(Number(params.newValue)));
      if (next === params.data[field]) return false;
      updateDraft(params.data.account_id, { [field]: next });
      return true;
    },
  });

  /** 체크박스 컬럼 — AG Grid 편집 사이클을 거치지 않고 **직접 그린다**.
   *
   *  `cellDataType: "boolean"` + `editable` 로 두면 클릭이 편집 시작과 값 토글을 겸하는데,
   *  `stopEditingWhenCellsLoseFocus` 와 맞물려 켤 때 편집 중 상태가 남아 체크 표시가 보이지
   *  않았다(값은 바뀌어 저장은 됐다). 초안 값을 그대로 그리고 onChange 로만 바꾸면
   *  화면과 값이 항상 같다. */
  const booleanCol = (
    field: "mix_enabled" | "trade_enabled" | "hidden" | "ma_alarm_enabled" | "stoploss_alarm_enabled" | "api_order_enabled",
    headerName: string,
    width: number,
    headerTooltip: string,
  ): ColDef<AccountGridRow> => ({
    field,
    headerName,
    width,
    headerTooltip,
    editable: false,
    cellStyle: { display: "flex", alignItems: "center", justifyContent: "center" },
    cellRenderer: (params: { data?: AccountGridRow }) => {
      const row = params.data;
      if (!row) return null;
      // 모의 계좌는 합성·알림 대상이 아니다 — 체크박스 대신 막힌 표시.
      if (row.env === ENV_MOCK) return <span style={mutedStyle}>—</span>;
      return (
        <input
          type="checkbox"
          className="form-check-input"
          style={{ margin: 0, cursor: "pointer" }}
          checked={Boolean(row[field])}
          onChange={(event) => updateDraft(row.account_id, { [field]: event.target.checked })}
        />
      );
    },
  });

  const columnDefs: ColDef<AccountGridRow>[] = [
    { field: "account_id", headerName: "ID", width: 130 },
    { field: "name", headerName: "이름", width: 150, editable: true },
    {
      headerName: "상세",
      width: 64,
      sortable: false,
      headerClass: "settingsCenterHeader",
      cellStyle: { display: "flex", alignItems: "center", justifyContent: "center" },
      // 셀 한 칸에 담기지 않는 설정(현금·벤치마크·증권사 API·잔고 동기화)을 여기서 연다.
      cellRenderer: (params: ICellRendererParams<AccountGridRow>) =>
        params.data ? (
          <button
            type="button"
            className="btn btn-sm btn-outline-secondary"
            style={{ padding: "0 8px", lineHeight: 1.4 }}
            onClick={() => setDetailId(params.data!.account_id)}
          >
            열기
          </button>
        ) : null,
    },
    {
      field: "brokerProvider",
      headerName: "증권사 API",
      width: 190,
      sortable: false,
      valueFormatter: (params) => {
        const row = params.data;
        if (!row?.brokerProvider) return "없음";
        const masked = maskedBrokerAccount(row.brokerAccountNo, row.brokerAccountMask);
        return masked ? `${row.brokerProvider} · ${masked}` : row.brokerProvider;
      },
    },
    {
      field: "env",
      headerName: "구분",
      width: 70,
      valueFormatter: (params) => ENV_LABELS[String(params.value)] ?? String(params.value),
    },
    { field: "icon", headerName: "아이콘", width: 62, editable: true },
    numberCol("order", "순서", 64),
    selectCol("country_code", "국가", 82, () => [...COUNTRY_OPTIONS]),
    selectCol("currency", "통화", 70, () => [...CURRENCY_OPTIONS]),
    {
      field: "cash_currencies",
      headerName: "현금",
      width: 94,
      sortable: false,
      valueFormatter: (params) => (Array.isArray(params.value) ? params.value.join(" · ") : ""),
    },
    {
      field: "benchmarkTicker",
      headerName: "벤치마크",
      width: 250,
      sortable: false,
      valueFormatter: (params) => {
        const row = params.data;
        return row?.benchmarkTicker ? `${row.benchmarkName} (${row.benchmarkTicker})` : "미설정";
      },
    },
    selectCol("regimeTicker", "시장 레짐", 120, () => marketIndices.map((item) => item.ticker), {
      valueFormatter: (params) =>
        marketIndices.find((item) => item.ticker === params.value)?.name ?? (params.value ? String(params.value) : "미설정"),
    }),
    // 켜면 `/strategy-mix` 목록에 오른다. 어떤 전략을 어떤 풀로 굴릴지는 그 화면에서 고른다
    // — 배분(%)과 함께 보며 맞추는 값이라 한 화면에 모아 둔다.
    booleanCol(
      "mix_enabled",
      "합성",
      76,
      "합성 전략을 운용할 계좌입니다. 전략·종목풀·배분은 /strategy-mix 에서 정합니다.",
    ),
    booleanCol(
      "trade_enabled",
      "사고팔기",
      88,
      "전략 사고팔기를 운용할 계좌입니다. 켠 계좌만 /strategy-trade 에서 고를 수 있습니다. 증권사 API 가 연동된 한국·미국 계좌만 켤 수 있습니다.",
    ),
    booleanCol(
      "hidden",
      "숨김",
      70,
      "자산 화면·계좌 선택·집계에서 이 계좌를 숨깁니다. 현금·보유 종목·원금이 모두 0일 때만 켤 수 있습니다.",
    ),
    booleanCol(
      "ma_alarm_enabled",
      "이평선🔔",
      96,
      "보유 종목의 종가가 그 종목이 속한 종목풀의 단기·장기 이평선 중 하나라도 아래면 알립니다.",
    ),
    booleanCol(
      "stoploss_alarm_enabled",
      "손절🔔",
      88,
      "보유 종목의 수익률이 그 종목이 속한 종목풀의 손절 기준 이하면 알립니다.",
    ),
    booleanCol(
      "api_order_enabled",
      "주문허용",
      88,
      "운영 계좌의 API 주문 스위치 — 꺼져 있으면 주문이 나가지 않습니다. 켜려면 아래 하루 한도를 먼저 입력하세요.",
    ),
    numberCol("api_order_max_count", "일 건수", 78, true, "하루 API 주문 건수 한도(운영 계좌). 비우면 주문이 막힙니다."),
    numberCol("api_order_max_amount", "일매수한도(원)", 120, true, "하루 API 주문 순매수 금액 한도(운영 계좌, 매수 − 매도) — 모든 계좌 원화로 입력하며 미장은 환율로 환산해 비교합니다. 비우면 주문이 막힙니다."),
    {
      headerName: "API 주문",
      width: 104,
      sortable: false,
      headerClass: "settingsCenterHeader",
      headerTooltip: "증권사 연동 계좌로 지정가 주문·취소·체결 조회를 합니다. 운영 계좌는 실제 주문이 나갑니다.",
      cellStyle: { display: "flex", alignItems: "center", justifyContent: "center" },
      cellRenderer: (params: ICellRendererParams<AccountGridRow>) => {
        const row = params.data;
        if (!row) return null;
        return (
          <button
            type="button"
            className={`btn btn-sm ${row.env === ENV_LIVE ? "btn-outline-danger" : "btn-outline-success"}`}
            style={{ padding: "0 8px", lineHeight: 1.4 }}
            disabled={!row.__brokerLinked}
            title={row.__brokerLinked ? undefined : "증권사 API 연동 계좌를 먼저 저장하세요."}
            onClick={() => setMockOrderId(row.account_id)}
          >
            열기
          </button>
        );
      },
    },
    {
      // 마지막 컬럼이 남는 가로를 채운다 — 오른쪽에 빈 공간이 남지 않게.
      field: "__updatedAt",
      headerName: "마지막 저장",
      flex: 1,
      minWidth: 320,
      valueFormatter: (params) => (params.value ? formatKstDateTime(String(params.value)) : "저장 이력 없음"),
      cellRenderer: (params: ICellRendererParams<AccountGridRow>) => (
        <LastSavedCell value={params.value} method={params.data?.__saveMethod} />
      ),
    },
  ];

  const detailDraft = detailId ? drafts[detailId] : undefined;
  const detailAccount = detailId ? accounts.find((account) => account.account_id === detailId) : undefined;

  return (
    <div className="appPageStack appPageStackFill">
      <section className="appSection appSectionFill">
        <div className="card appCard appTableCardFill">
          <div className="card-header">
            <div className="appMainHeader">
              <div className="appMainHeaderLeft">
                <div>
                  <h2 style={{ fontSize: "var(--fs-lg)", fontWeight: 800, margin: 0 }}>계좌 설정</h2>
                  <div className="tableFooterMeta" style={{ margin: 0, color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
                    계좌 메타는 DB(account_settings)가 단일 소스입니다. 셀을 클릭해 고친 뒤 저장하세요.
                  </div>
                </div>
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <UnsavedChangesBadge show={dirtyCount > 0} message={`저장하지 않은 변경 ${dirtyCount}개`} />
                <button
                  type="button"
                  className="btn btn-sm btn-dark"
                  disabled={sendingAlarm}
                  title="저장된 알림 설정 그대로 지금 판정해, 조건을 만족하는 종목이 있으면 슬랙으로 보냅니다."
                  onClick={() => void sendAlarmTest()}
                >
                  {sendingAlarm ? "발송 중…" : "슬랙 알람 테스트"}
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-outline-danger"
                  disabled={selectedIds.length === 0 || deleting}
                  onClick={() => void handleDeleteSelected()}
                >
                  {deleting ? "삭제 중…" : `삭제${selectedIds.length > 0 ? ` (${selectedIds.length})` : ""}`}
                </button>
                <button type="button" className="btn btn-sm btn-primary" onClick={() => setAddOpen(true)}>
                  계좌 추가
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-dark"
                  disabled={dirtyCount === 0 || saving}
                  onClick={() => void handleSaveAll()}
                >
                  {saving ? "저장 중…" : "저장"}
                </button>
              </div>
            </div>
          </div>

          <div className="card-body appCardBodyTight appTableCardBodyFill">
            <div className="appGridFillWrap">
              <AppAgGrid<AccountGridRow>
                className="settingsAgGrid"
                rowData={gridRows}
                columnDefs={columnDefs}
                loading={loading}
                theme={accountGridTheme}
                minHeight="100%"
                getRowId={(params) => params.data.account_id}
                getRowClass={(params) => (params.data?.__dirty ? "settingsDirtyRow" : "")}
                gridOptions={{
                  suppressMovableColumns: true,
                  // 폭에 맞춰 말줄임하므로, 잘린 값은 마우스를 올려 전체를 본다.
                  defaultColDef: {
                    sortable: true,
                    resizable: true,
                    tooltipValueGetter: (params) => params.valueFormatted ?? params.value,
                  },
                  // AppAgGrid 기본값은 셀 포커스를 막는다(읽기 전용 표 기준). 편집하려면 켜야 한다.
                  suppressCellFocus: false,
                  singleClickEdit: true,
                  stopEditingWhenCellsLoseFocus: true,
                  rowSelection: {
                    mode: "multiRow",
                    checkboxes: true,
                    headerCheckbox: true,
                    enableClickSelection: false,
                  },
                  selectionColumnDef: {
                    width: 40,
                    minWidth: 40,
                    maxWidth: 40,
                    pinned: "left",
                    sortable: false,
                    resizable: false,
                    suppressMovable: true,
                    headerName: "",
                  },
                  onSelectionChanged: (params) => {
                    setSelectedIds(params.api.getSelectedRows().map((row) => row.account_id));
                  },
                  onCellValueChanged: (params) => {
                    const key = params.colDef.field as keyof AccountDraft | undefined;
                    if (!key || !params.data || params.newValue === params.oldValue) return;
                    const id = params.data.account_id;
                    if (key === "ma_alarm_enabled" || key === "stoploss_alarm_enabled") {
                      updateDraft(id, { [key]: Boolean(params.newValue) } as Partial<AccountDraft>);
                      return;
                    }
                    updateDraft(id, { [key]: String(params.newValue ?? "") } as Partial<AccountDraft>);
                    // 주 통화를 바꾸면 현금 잔액에 반드시 포함되어야 한다(백엔드 검증과 동일).
                    if (key === "currency") {
                      const next = String(params.newValue ?? "").toUpperCase();
                      const current = params.data.cash_currencies;
                      if (next && !current.includes(next)) {
                        updateDraft(id, { cash_currencies: [...current, next] });
                      }
                    }
                  },
                }}
              />
            </div>
          </div>
        </div>
      </section>

      {detailId && detailDraft ? (
        <AccountDetailModal
          draft={detailDraft}
          savedBrokerProvider={detailAccount?.broker_api?.provider ?? ""}
          onChange={(patch) => updateDraft(detailId, patch)}
          onClose={() => setDetailId(null)}
        />
      ) : null}

      {mockOrderId ? (
        <ApiOrderModal
          isLive={accounts.find((account) => account.account_id === mockOrderId)?.env === ENV_LIVE}
          accountId={mockOrderId}
          accountName={accounts.find((account) => account.account_id === mockOrderId)?.name ?? mockOrderId}
          accountLabel={(() => {
            const broker = accounts.find((account) => account.account_id === mockOrderId)?.broker_api;
            return `${broker?.provider ?? ""} · ${maskedBrokerAccount(broker?.account_no, broker?.account_mask)}`;
          })()}
          onClose={() => setMockOrderId(null)}
        />
      ) : null}

      {addOpen ? <AddAccountModal onClose={() => setAddOpen(false)} onCreated={() => void load()} /> : null}
    </div>
  );
}
