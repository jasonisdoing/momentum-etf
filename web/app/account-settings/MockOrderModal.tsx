"use client";

import { useEffect, useState } from "react";

import { AppModal } from "../components/AppModal";
import { useToast } from "../components/ToastProvider";

/** 증권사 모의투자 서버 주소 — 이 모달은 여기로만 주문한다(운영 계좌는 고를 수 없다). */
const MOCK_DOMAIN = "moapi.nhplug.com:8443";
/** 나무증권 계좌구분 — 03 이 모의투자 전용이다. */
const MOCK_ACCOUNT_TYPE = "03";
const PROVIDER = "NAMU_PLUG";

type Side = "buy" | "sell";
type MockAccount = { account_no: string; masked: string; acct_type: string };
type LogEntry = { at: string; action: string; code: string; message: string };

const fieldStyle: React.CSSProperties = { display: "flex", alignItems: "center", gap: 8, marginBottom: 10 };
const labelStyle: React.CSSProperties = {
  color: "var(--text-muted)",
  fontWeight: 600,
  fontSize: "var(--fs-sm)",
  width: 72,
  flexShrink: 0,
};

/** 모의투자 계좌로 지정가 주문·취소·체결 조회를 시험하는 모달 — 슬랙 버튼 주문의 서버 쪽 주문 경로를 검증한다. */
export function MockOrderModal({ onClose }: { onClose: () => void }) {
  const toast = useToast();
  const [accounts, setAccounts] = useState<MockAccount[] | null>(null);
  const [accountNo, setAccountNo] = useState("");
  const [side, setSide] = useState<Side>("buy");
  const [ticker, setTicker] = useState("");
  const [quantity, setQuantity] = useState("");
  const [price, setPrice] = useState("");
  const [orderNo, setOrderNo] = useState("");
  const [log] = useState<LogEntry[]>([]);

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

  // UI 확인 단계 — 서버 호출은 아직 연결하지 않았다.
  const notConnected = () => toast.warning("서버 연결 전입니다 (화면 확인 단계).");

  const canOrder = Boolean(accountNo && ticker.trim() && Number(quantity) > 0 && Number(price) > 0);

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

      <div style={{ borderTop: "1px solid var(--border-color, rgba(148,163,184,0.3))", paddingTop: 12, marginTop: 4 }}>
        <div style={{ fontWeight: 700, marginBottom: 10 }}>지정가 주문</div>
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
        <label style={fieldStyle}>
          <span style={labelStyle}>종목코드</span>
          <input
            className="form-control form-control-sm"
            style={{ width: 140 }}
            value={ticker}
            placeholder="예: 069500"
            onChange={(e) => setTicker(e.target.value)}
          />
        </label>
        <label style={fieldStyle}>
          <span style={labelStyle}>수량</span>
          <input
            className="form-control form-control-sm"
            style={{ width: 140 }}
            type="number"
            min={1}
            value={quantity}
            onChange={(e) => setQuantity(e.target.value)}
          />
          <span style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>주</span>
        </label>
        <label style={fieldStyle}>
          <span style={labelStyle}>지정가</span>
          <input
            className="form-control form-control-sm"
            style={{ width: 140 }}
            type="number"
            min={1}
            value={price}
            onChange={(e) => setPrice(e.target.value)}
          />
          <span style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>원 · 보통가(지정가)</span>
        </label>
        <button
          type="button"
          className={side === "buy" ? "btn btn-sm btn-primary" : "btn btn-sm btn-danger"}
          disabled={!canOrder}
          onClick={notConnected}
        >
          {side === "buy" ? "매수 주문" : "매도 주문"}
        </button>
      </div>

      <div style={{ borderTop: "1px solid var(--border-color, rgba(148,163,184,0.3))", paddingTop: 12, marginTop: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 10 }}>취소 · 체결 조회</div>
        <label style={fieldStyle}>
          <span style={labelStyle}>주문번호</span>
          <input
            className="form-control form-control-sm"
            style={{ width: 140 }}
            value={orderNo}
            placeholder="시장주문번호"
            onChange={(e) => setOrderNo(e.target.value)}
          />
          <button type="button" className="btn btn-sm btn-outline-danger" disabled={!accountNo || !orderNo.trim()} onClick={notConnected}>
            주문 취소
          </button>
          <button type="button" className="btn btn-sm btn-outline-secondary" disabled={!accountNo} onClick={notConnected}>
            체결 조회
          </button>
        </label>
      </div>

      <div style={{ borderTop: "1px solid var(--border-color, rgba(148,163,184,0.3))", paddingTop: 12, marginTop: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 10 }}>호출 결과</div>
        {log.length === 0 ? (
          <div style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>아직 호출 내역이 없습니다.</div>
        ) : (
          <table className="table table-sm" style={{ fontSize: "var(--fs-sm)", marginBottom: 0 }}>
            <thead>
              <tr>
                <th style={{ width: 90 }}>시각</th>
                <th style={{ width: 90 }}>동작</th>
                <th style={{ width: 90 }}>응답코드</th>
                <th>메시지</th>
              </tr>
            </thead>
            <tbody>
              {log.map((entry, index) => (
                <tr key={`${entry.at}-${index}`}>
                  <td>{entry.at}</td>
                  <td>{entry.action}</td>
                  <td>{entry.code}</td>
                  <td>{entry.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </AppModal>
  );
}
