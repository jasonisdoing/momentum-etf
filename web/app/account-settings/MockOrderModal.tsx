"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { formatPrice } from "@/lib/price-format";
import { formatKrw } from "../assets/assets-helpers";
import { AppModal } from "../components/AppModal";
import { useToast } from "../components/ToastProvider";
import type { StrategyView } from "../strategy-trade/StrategyTradeClient";

/** 증권사 모의투자 서버 주소 — 이 모달은 여기로만 주문한다(운영 계좌는 고를 수 없다). */
const MOCK_DOMAIN = "moapi.nhplug.com:8443";
/** 나무증권 계좌구분 — 03 이 모의투자 전용이다. */
const MOCK_ACCOUNT_TYPE = "03";
const PROVIDER = "NAMU_PLUG";

type Side = "buy" | "sell";
type MockAccount = { account_no: string; masked: string; acct_type: string };

type MockHolding = {
  ticker: string;
  name: string;
  quantity: number;
  average_buy_price: number;
  current_price: number | null;
  value: number | null;
};
/** 증권사 잔고 — 계좌 설정의 잔고 동기화와 같은 값(현금은 D+2 예수금). */
type MockBalance = { cash: number; total_asset: number; holdings: MockHolding[] };

/** 주문 상태 — 증권사 체결 조회의 체결·미체결·취소 수량으로 판정한다. */
type OrderStatus = "pending" | "partial" | "filled" | "canceled" | "rejected";
const STATUS_CHIPS: Record<OrderStatus, { label: string; className: string }> = {
  pending: { label: "대기중", className: "badge bg-yellow-lt" },
  partial: { label: "부분체결", className: "badge bg-blue-lt" },
  filled: { label: "완료", className: "badge bg-green-lt" },
  canceled: { label: "취소", className: "badge bg-secondary-lt" },
  rejected: { label: "거부", className: "badge bg-red-lt" },
};

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, { cache: "no-store", ...init });
  const data = (await resp.json()) as T & { error?: string };
  if (!resp.ok || data.error) throw new Error(data.error ?? "요청에 실패했습니다.");
  return data;
}

/** 우리가 보낸 주문 한 건 — 어느 전략의 주문인지와 현재 상태. */
type OrderRecord = {
  id: string;
  at: string;
  strategy_label: string;
  side: Side;
  ticker: string;
  quantity: number;
  price: number;
  filled_quantity: number;
  status: OrderStatus;
  /** 증권사 응답·정리 사유 — 비어 있으면 정상. */
  message: string;
};

/** 증권사 주문내역 한 줄 — 체결 조회 응답 그대로(우리 기록과 대조용). */
type BrokerOrder = {
  order_no: string;
  ticker: string;
  name: string;
  side_label: string;
  quantity: number;
  price: number;
  filled_quantity: number;
  unfilled_quantity: number;
  canceled_quantity: number;
  status: OrderStatus;
};

const sectionStyle: React.CSSProperties = {
  borderTop: "1px solid var(--border-color, rgba(148,163,184,0.3))",
  paddingTop: 12,
  marginTop: 16,
};
const fieldStyle: React.CSSProperties = { display: "flex", alignItems: "center", gap: 8, marginBottom: 10 };
const labelStyle: React.CSSProperties = {
  color: "var(--text-muted)",
  fontWeight: 600,
  fontSize: "var(--fs-sm)",
  width: 72,
  flexShrink: 0,
};
const mutedStyle: React.CSSProperties = { color: "var(--text-muted)", fontSize: "var(--fs-sm)" };
const rightCell: React.CSSProperties = { textAlign: "right" };

function StatusChip({ status }: { status: OrderStatus }) {
  const chip = STATUS_CHIPS[status];
  return <span className={chip.className}>{chip.label}</span>;
}

/** 모의투자 계좌로 사고팔기 전략 종목의 현재가 지정가 주문·취소·상태 확인을 시험하는 모달 — 슬랙 버튼 주문의 서버 쪽 주문 경로를 검증한다. */
export function MockOrderModal({ onClose }: { onClose: () => void }) {
  const toast = useToast();
  const [accounts, setAccounts] = useState<MockAccount[] | null>(null);
  const [accountNo, setAccountNo] = useState("");
  const [strategies, setStrategies] = useState<StrategyView[] | null>(null);
  const [pool, setPool] = useState("");
  const [strategyId, setStrategyId] = useState("");
  const [side, setSide] = useState<Side>("buy");
  const [balance, setBalance] = useState<MockBalance | null>(null);
  const [orders, setOrders] = useState<OrderRecord[]>([]);
  const [brokerOrders, setBrokerOrders] = useState<BrokerOrder[] | null>(null);
  const [loadingStrategies, setLoadingStrategies] = useState(false);
  // 진행 중인 요청 — 같은 버튼을 연달아 눌러 주문이 두 번 나가지 않게 한다.
  const [busy, setBusy] = useState<"balance" | "orders" | "order" | "broker" | string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const resp = await fetch(`/api/broker-api/accounts?provider=${PROVIDER}`, { cache: "no-store" });
        const data = (await resp.json()) as { accounts?: MockAccount[]; error?: string };
        if (!resp.ok || data.error) throw new Error(data.error ?? "증권사 계좌를 불러오지 못했습니다.");
        if (cancelled) return;
        const mock = (data.accounts ?? []).filter((row) => row.acct_type === MOCK_ACCOUNT_TYPE);
        setAccounts(mock);
        if (mock.length === 1) setAccountNo(mock[0].account_no);
      } catch (err) {
        if (cancelled) return;
        setAccounts([]);
        toast.error(err instanceof Error ? err.message : "증권사 계좌를 불러오지 못했습니다.");
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 주문할 전략 종목과 현재가 — /strategy-trade 화면과 같은 운용 현황이다(현재가는 화면을 열 때와 새로고침 때 읽는다).
  const loadStrategies = useCallback(async () => {
    setLoadingStrategies(true);
    try {
      const resp = await fetch("/api/strategy-trade", { cache: "no-store" });
      const data = (await resp.json()) as { strategies?: StrategyView[]; pool?: string | null; error?: string };
      if (!resp.ok || data.error) throw new Error(data.error ?? "사고팔기 전략을 불러오지 못했습니다.");
      const list = data.strategies ?? [];
      setStrategies(list);
      setPool(data.pool ?? "");
      setStrategyId((current) => (list.some((row) => row.strategy_id === current) ? current : (list[0]?.strategy_id ?? "")));
    } catch (err) {
      setStrategies((current) => current ?? []);
      toast.error(err instanceof Error ? err.message : "사고팔기 전략을 불러오지 못했습니다.");
    } finally {
      setLoadingStrategies(false);
    }
  }, [toast]);

  useEffect(() => {
    void loadStrategies();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const strategy = useMemo(() => (strategies ?? []).find((row) => row.strategy_id === strategyId), [strategies, strategyId]);
  const currentPrice = strategy?.index.close ?? null;
  const quantity = strategy?.config.round_quantity ?? null;

  const run = useCallback(
    async (key: string, work: () => Promise<void>) => {
      setBusy(key);
      try {
        await work();
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "요청에 실패했습니다.");
      } finally {
        setBusy(null);
      }
    },
    [toast],
  );

  const query = `account_no=${encodeURIComponent(accountNo)}`;
  const loadBalance = () =>
    run("balance", async () => {
      setBalance(await requestJson<MockBalance>(`/api/broker-api/mock/balance?${query}`));
    });
  const loadOrders = () =>
    run("orders", async () => {
      setOrders((await requestJson<{ orders: OrderRecord[] }>(`/api/broker-api/mock/orders?${query}`)).orders);
    });
  const loadBrokerOrders = () =>
    run("broker", async () => {
      setBrokerOrders((await requestJson<{ orders: BrokerOrder[] }>(`/api/broker-api/mock/broker-orders?${query}`)).orders);
    });

  // 계좌를 고르면 잔고와 주문 기록을 함께 읽는다.
  useEffect(() => {
    if (!accountNo) {
      setBalance(null);
      setOrders([]);
      setBrokerOrders(null);
      return;
    }
    // 모의투자 서버는 호출 간격을 두고 직렬로 받으므로 순서대로 읽는다.
    void (async () => {
      await loadBalance();
      await loadOrders();
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accountNo]);

  const placeOrder = () =>
    run("order", async () => {
      const data = await requestJson<{ order: OrderRecord }>("/api/broker-api/mock/orders", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account_no: accountNo, pool, strategy_id: strategyId, side }),
      });
      toast.success(`${data.order.strategy_label} ${side === "buy" ? "매수" : "매도"} 주문을 접수했습니다.`);
      setOrders((await requestJson<{ orders: OrderRecord[] }>(`/api/broker-api/mock/orders?${query}`)).orders);
      setBalance(await requestJson<MockBalance>(`/api/broker-api/mock/balance?${query}`));
    });

  // 같은 주문 내용을 슬랙 메시지(버튼·모달)로 보낸다 — 주문은 슬랙에서 확인을 눌러야 나간다.
  const sendSlackOrder = () =>
    run("slack", async () => {
      await requestJson("/api/broker-api/mock/slack-order", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account_no: accountNo, pool, strategy_id: strategyId, side }),
      });
      toast.success(`${strategy?.label ?? ""} ${side === "buy" ? "매수" : "매도"} 주문 메시지를 슬랙으로 보냈습니다.`);
    });

  const cancelOrder = (orderId: string) =>
    run(`cancel:${orderId}`, async () => {
      await requestJson("/api/broker-api/mock/orders/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ order_id: orderId }),
      });
      toast.success("취소 요청을 보냈습니다.");
      setOrders((await requestJson<{ orders: OrderRecord[] }>(`/api/broker-api/mock/orders?${query}`)).orders);
      setBalance(await requestJson<MockBalance>(`/api/broker-api/mock/balance?${query}`));
    });

  const canOrder = Boolean(accountNo && pool && strategy && currentPrice && quantity && busy === null && !loadingStrategies);

  return (
    <AppModal
      open
      title="모의 주문 테스트"
      subtitle={`모의투자 서버(${MOCK_DOMAIN})로만 보냅니다. 운영 계좌는 선택할 수 없습니다.`}
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
      <label style={fieldStyle}>
        <span style={labelStyle}>모의 계좌</span>
        <select
          className="form-select form-select-sm"
          style={{ width: 280 }}
          value={accountNo}
          disabled={accounts === null}
          onChange={(e) => setAccountNo(e.target.value)}
        >
          <option value="">{accounts === null ? "불러오는 중…" : accounts.length === 0 ? "모의투자 계좌 없음" : "계좌 선택"}</option>
          {(accounts ?? []).map((row) => (
            <option key={row.account_no} value={row.account_no}>
              {row.masked}
            </option>
          ))}
        </select>
      </label>

      <div style={{ ...sectionStyle, marginTop: 4 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
          <span style={{ fontWeight: 700 }}>잔고</span>
          <button type="button" className="btn btn-sm btn-outline-secondary" disabled={!accountNo || busy !== null} onClick={() => void loadBalance()}>
            {busy === "balance" ? "불러오는 중…" : "잔고 새로고침"}
          </button>
          <span style={{ marginLeft: "auto", ...mutedStyle }}>
            예수금(D+2) <strong style={{ color: "inherit" }}>{balance ? formatKrw(balance.cash) : "-"}</strong>
            {" · "}총자산 <strong style={{ color: "inherit" }}>{balance ? formatKrw(balance.total_asset) : "-"}</strong>
          </span>
        </div>
        {balance === null ? (
          <div style={mutedStyle}>잔고를 불러오지 않았습니다.</div>
        ) : balance.holdings.length === 0 ? (
          <div style={mutedStyle}>보유 종목이 없습니다.</div>
        ) : (
          <table className="table table-sm" style={{ fontSize: "var(--fs-sm)", marginBottom: 0 }}>
            <thead>
              <tr>
                <th style={{ width: 90 }}>종목코드</th>
                <th>종목명</th>
                <th style={{ width: 90, ...rightCell }}>수량</th>
                <th style={{ width: 110, ...rightCell }}>평단</th>
                <th style={{ width: 110, ...rightCell }}>현재가</th>
                <th style={{ width: 130, ...rightCell }}>평가금액</th>
              </tr>
            </thead>
            <tbody>
              {balance.holdings.map((row) => (
                <tr key={row.ticker}>
                  <td>{row.ticker}</td>
                  <td>{row.name}</td>
                  <td style={rightCell}>{row.quantity.toLocaleString("ko-KR")}</td>
                  <td style={rightCell}>{formatPrice(row.average_buy_price, "KRW")}</td>
                  <td style={rightCell}>{formatPrice(row.current_price, "KRW")}</td>
                  <td style={rightCell}>{formatPrice(row.value, "KRW")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div style={sectionStyle}>
        <div style={{ fontWeight: 700, marginBottom: 10 }}>지정가 주문</div>
        <label style={fieldStyle}>
          <span style={labelStyle}>전략</span>
          <select
            className="form-select form-select-sm"
            style={{ width: 280 }}
            value={strategyId}
            disabled={strategies === null}
            onChange={(e) => setStrategyId(e.target.value)}
          >
            {strategies === null ? <option value="">불러오는 중…</option> : null}
            {strategies !== null && strategies.length === 0 ? <option value="">사고팔기 전략 없음</option> : null}
            {(strategies ?? []).map((row) => (
              <option key={row.strategy_id} value={row.strategy_id}>
                {row.label} ({row.ticker})
              </option>
            ))}
          </select>
        </label>
        <div style={fieldStyle}>
          <span style={labelStyle}>구분</span>
          <div className="appSegmentedToggle" role="group" aria-label="매수·매도">
            <button
              type="button"
              className={side === "buy" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
              onClick={() => setSide("buy")}
            >
              매수
            </button>
            <button
              type="button"
              className={side === "sell" ? "btn appSegmentedToggleButton is-active" : "btn appSegmentedToggleButton"}
              onClick={() => setSide("sell")}
            >
              매도
            </button>
          </div>
        </div>
        <div style={fieldStyle}>
          <span style={labelStyle}>주문 내용</span>
          <span style={{ fontSize: "var(--fs-base)" }}>
            {strategy && currentPrice && quantity
              ? `${strategy.label} · ${quantity.toLocaleString("ko-KR")}주 × 현재가 ${formatPrice(currentPrice, "KRW")}원 지정가 ${side === "buy" ? "매수" : "매도"}`
              : "-"}
          </span>
          <button type="button" className="btn btn-sm btn-outline-secondary" disabled={loadingStrategies || busy !== null} onClick={() => void loadStrategies()}>
            {loadingStrategies ? "불러오는 중…" : "현재가 새로고침"}
          </button>
        </div>
        <div style={{ ...mutedStyle, marginBottom: 10 }}>
          수량은 전략의 회차당 수량이고, 가격은 주문 순간의 현재가를 호가 단위로 반올림해 서버가 정합니다(직접 입력 없음).
        </div>
        <button
          type="button"
          className={side === "buy" ? "btn btn-sm btn-primary" : "btn btn-sm btn-danger"}
          disabled={!canOrder}
          onClick={() => void placeOrder()}
        >
          {busy === "order" ? "주문 중…" : side === "buy" ? "매수 주문" : "매도 주문"}
        </button>
        <button
          type="button"
          className="btn btn-sm btn-outline-secondary"
          style={{ marginLeft: 8 }}
          disabled={!canOrder}
          onClick={() => void sendSlackOrder()}
        >
          {busy === "slack" ? "보내는 중…" : side === "buy" ? "매수 주문 메시지(슬랙)" : "매도 주문 메시지(슬랙)"}
        </button>
      </div>

      <div style={sectionStyle}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
          <span style={{ fontWeight: 700 }}>주문 기록</span>
          <button type="button" className="btn btn-sm btn-outline-secondary" disabled={!accountNo || busy !== null} onClick={() => void loadOrders()}>
            {busy === "orders" ? "불러오는 중…" : "상태 새로고침"}
          </button>
          <span style={{ marginLeft: "auto", display: "flex", gap: 6, alignItems: "center" }}>
            {(Object.keys(STATUS_CHIPS) as OrderStatus[]).map((status) => (
              <StatusChip key={status} status={status} />
            ))}
          </span>
        </div>
        {orders.length === 0 ? (
          <div style={mutedStyle}>주문 기록이 없습니다.</div>
        ) : (
          <table className="table table-sm" style={{ fontSize: "var(--fs-sm)", marginBottom: 0 }}>
            <thead>
              <tr>
                <th style={{ width: 90 }}>시각</th>
                <th>전략</th>
                <th style={{ width: 60 }}>구분</th>
                <th style={{ width: 90, ...rightCell }}>수량</th>
                <th style={{ width: 110, ...rightCell }}>지정가</th>
                <th style={{ width: 90, ...rightCell }}>체결</th>
                <th style={{ width: 90 }}>상태</th>
                <th style={{ width: 70 }} />
              </tr>
            </thead>
            <tbody>
              {orders.map((row) => (
                <tr key={row.id}>
                  <td>{row.at}</td>
                  <td>
                    {row.strategy_label} <span style={mutedStyle}>({row.ticker})</span>
                  </td>
                  <td>{row.side === "buy" ? "매수" : "매도"}</td>
                  <td style={rightCell}>{row.quantity.toLocaleString("ko-KR")}</td>
                  <td style={rightCell}>{formatPrice(row.price, "KRW")}</td>
                  <td style={rightCell}>{row.filled_quantity.toLocaleString("ko-KR")}</td>
                  <td>
                    <StatusChip status={row.status} />
                    {row.message ? (
                      <div style={{ ...mutedStyle, fontSize: "var(--fs-xs, 0.75rem)", whiteSpace: "normal" }}>{row.message}</div>
                    ) : null}
                  </td>
                  <td>
                    {row.status === "pending" || row.status === "partial" ? (
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-danger"
                        disabled={busy !== null}
                        onClick={() => void cancelOrder(row.id)}
                      >
                        {busy === `cancel:${row.id}` ? "취소 중…" : "취소"}
                      </button>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <details style={sectionStyle}>
        <summary style={{ fontWeight: 700, cursor: "pointer" }}>증권사 주문내역 (오늘)</summary>
        <div style={{ marginTop: 10 }}>
          <button type="button" className="btn btn-sm btn-outline-secondary" disabled={!accountNo || busy !== null} onClick={() => void loadBrokerOrders()}>
            {busy === "broker" ? "불러오는 중…" : "증권사 내역 불러오기"}
          </button>
          <span style={{ ...mutedStyle, marginLeft: 10 }}>우리 기록과 대조합니다 — 보내지 않은 주문이나 상태가 다른 주문을 찾는 용도입니다.</span>
        </div>
        <div style={{ marginTop: 10 }}>
          {brokerOrders === null ? (
            <div style={mutedStyle}>불러오지 않았습니다.</div>
          ) : brokerOrders.length === 0 ? (
            <div style={mutedStyle}>오늘 주문이 없습니다.</div>
          ) : (
            <table className="table table-sm" style={{ fontSize: "var(--fs-sm)", marginBottom: 0 }}>
              <thead>
                <tr>
                  <th style={{ width: 90 }}>주문번호</th>
                  <th>종목</th>
                  <th style={{ width: 90 }}>구분</th>
                  <th style={{ width: 90, ...rightCell }}>주문수량</th>
                  <th style={{ width: 110, ...rightCell }}>주문가</th>
                  <th style={{ width: 90, ...rightCell }}>체결</th>
                  <th style={{ width: 90, ...rightCell }}>미체결</th>
                  <th style={{ width: 90, ...rightCell }}>취소</th>
                  <th style={{ width: 90 }}>상태</th>
                </tr>
              </thead>
              <tbody>
                {brokerOrders.map((row) => (
                  <tr key={row.order_no}>
                    <td>{row.order_no}</td>
                    <td>
                      {row.name} <span style={mutedStyle}>({row.ticker})</span>
                    </td>
                    <td>{row.side_label}</td>
                    <td style={rightCell}>{row.quantity.toLocaleString("ko-KR")}</td>
                    <td style={rightCell}>{formatPrice(row.price, "KRW")}</td>
                    <td style={rightCell}>{row.filled_quantity.toLocaleString("ko-KR")}</td>
                    <td style={rightCell}>{row.unfilled_quantity.toLocaleString("ko-KR")}</td>
                    <td style={rightCell}>{row.canceled_quantity.toLocaleString("ko-KR")}</td>
                    <td>
                      <StatusChip status={row.status} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </details>
    </AppModal>
  );
}
