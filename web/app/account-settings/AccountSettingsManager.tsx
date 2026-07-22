"use client";

import { useCallback, useEffect, useState } from "react";

import { formatKstDateTime } from "@/lib/datetime";
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
  benchmark?: Benchmark;
  market_regime_index?: MarketIndexOption | null;
  URL?: string;
  updated_at?: string | null;
  save_method?: string | null;
};

type ApiResponse = { accounts?: AccountEntry[]; market_indices?: MarketIndexOption[]; error?: string };

// 시장 레짐 지수 기본값 — S&P 500 (yf_ticker ^GSPC). 필수값이라 미설정 계좌는 이 값으로 시작한다.
const DEFAULT_REGIME_TICKER = "^GSPC";

const inputStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "4px 8px",
  fontSize: "0.88rem",
};
const labelStyle: React.CSSProperties = { color: "var(--text-muted)", fontWeight: 600, fontSize: "0.83rem", flexShrink: 0 };

/** 계좌 1개 편집 행 (자체 저장). account_id 는 원장 FK 라 수정 불가. */
function AccountRow({
  account,
  marketIndices,
  onSaved,
  onDeleted,
}: {
  account: AccountEntry;
  marketIndices: MarketIndexOption[];
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
  const [url, setUrl] = useState(account.URL ?? "");
  // 시장 레짐 지수(필수) — 미설정 계좌는 S&P 500 기본값으로 시작.
  const [regimeTicker, setRegimeTicker] = useState(account.market_regime_index?.ticker || DEFAULT_REGIME_TICKER);
  const [benchTicker, setBenchTicker] = useState(account.benchmark?.ticker ?? "");
  const [benchName, setBenchName] = useState(account.benchmark?.name ?? "");
  const [benchEditing, setBenchEditing] = useState(!(account.benchmark?.ticker && account.benchmark?.name));
  const [updatedAt, setUpdatedAt] = useState<string | null | undefined>(account.updated_at);
  const [saving, setSaving] = useState(false);
  const [resolving, setResolving] = useState(false);

  // 벤치마크 티커 → 종목명 조회 (공통: 종목풀 stock_meta). 이름은 조회로만 채움.
  const resolveBench = async () => {
    const t = benchTicker.trim();
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
      setBenchName(data.name);
      setBenchEditing(false);
      toast.success(`${data.name}(${t}) 확인 완료`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "티커 조회 중 오류가 발생했습니다.");
    } finally {
      setResolving(false);
    }
  };

  const save = async () => {
    try {
      setSaving(true);
      const resp = await fetch("/api/account-settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          account_id: account.account_id,
          values: {
            name: name.trim(),
            icon: icon.trim(),
            order: Math.trunc(Number(order)),
            country_code: countryCode,
            currency: currency.trim().toUpperCase(),
            benchmark: { ticker: benchTicker.trim(), name: benchName.trim() },
            market_regime_index: {
              ticker: regimeTicker,
              name: marketIndices.find((item) => item.ticker === regimeTicker)?.name ?? "",
            },
            URL: url.trim(),
          },
        }),
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
          <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
            마지막 저장: {updatedAt ? formatKstDateTime(updatedAt) : "저장 이력 없음"}
          </span>
          <button
            type="button"
            className="btn btn-sm btn-dark"
            disabled={saving || !name.trim() || !benchTicker.trim() || !benchName.trim()}
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
        <select className="form-select form-select-sm" style={{ width: 96 }} value={currency} onChange={(e) => setCurrency(e.target.value)}>
          <option value="KRW">KRW</option>
          <option value="AUD">AUD</option>
          <option value="USD">USD</option>
        </select>
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
            <span style={{ fontSize: "0.88rem", fontWeight: 600 }}>
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
  const [accountType, setAccountType] = useState<"fixed" | "trend" | "regime">("fixed");
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
          account_type: accountType,
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
          <h3 style={{ fontSize: "1rem", fontWeight: 800, margin: 0 }}>계좌 추가</h3>
          <p style={{ color: "var(--text-muted)", fontSize: "0.8rem", margin: 0 }}>
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
            <span style={{ ...labelStyle, width: 64 }}>타입</span>
            <select className="form-select form-select-sm" style={{ flex: 1 }} value={accountType} onChange={(e) => setAccountType(e.target.value as "fixed" | "trend" | "regime")}>
              <option value="fixed">fixed (고정 보유)</option>
              <option value="trend">trend (추세 순위)</option>
              <option value="regime">regime (레짐)</option>
            </select>
          </label>
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
          <h2 style={{ fontSize: "1.05rem", fontWeight: 800, margin: 0 }}>계좌 설정</h2>
          <button type="button" className="btn btn-sm btn-primary" onClick={() => setAddOpen(true)}>+ 계좌 추가</button>
        </div>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginBottom: 12 }}>
          계좌 메타(이름/순서/벤치마크 등)는 DB(account_settings)가 단일 소스입니다. 삭제는 보유종목이 없는 계좌만 가능합니다.
        </p>
        {loading ? (
          <div style={{ color: "var(--text-muted)", padding: 12 }}>불러오는 중…</div>
        ) : accounts.length === 0 ? (
          <div style={{ color: "var(--text-muted)", padding: 12 }}>등록된 계좌가 없습니다.</div>
        ) : (
          accounts.map((a) => (
            <AccountRow key={a.account_id} account={a} marketIndices={marketIndices} onSaved={() => {}} onDeleted={() => void load()} />
          ))
        )}
      </div>
      {addOpen ? <AddAccountModal onClose={() => setAddOpen(false)} onCreated={() => void load()} /> : null}
    </div>
  );
}
