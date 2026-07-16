"use client";

import { useCallback, useEffect, useState } from "react";

import { formatKstDateTime } from "@/lib/datetime";
import { useToast } from "../components/ToastProvider";

type Benchmark = { ticker?: string; name?: string };

type AccountEntry = {
  account_id: string;
  name?: string;
  icon?: string;
  order?: number;
  country_code?: string;
  currency?: string;
  benchmark?: Benchmark;
  ticker_types?: string[];
  memo?: string;
  top_pick_start_amount_manwon?: number | null;
  top_pick_start_date?: string | null;
  URL?: string;
  updated_at?: string | null;
  save_method?: string | null;
};

type PoolOption = { ticker_type: string; name?: string; icon?: string };

type ApiResponse = { accounts?: AccountEntry[]; error?: string };

const inputStyle: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,0.4)",
  borderRadius: 6,
  padding: "4px 8px",
  fontSize: "0.88rem",
};
const labelStyle: React.CSSProperties = { color: "var(--text-muted)", fontWeight: 600, fontSize: "0.83rem", flexShrink: 0 };

/** 계좌 1개 편집 행 (자체 저장). account_id 는 원장 FK 라 수정 불가. */
function AccountRow({ account, pools, onSaved }: { account: AccountEntry; pools: PoolOption[]; onSaved: () => void }) {
  const toast = useToast();
  const [name, setName] = useState(account.name ?? "");
  const [icon, setIcon] = useState(account.icon ?? "");
  const [order, setOrder] = useState(String(account.order ?? 0));
  const [countryCode, setCountryCode] = useState(account.country_code ?? "kor");
  const [currency, setCurrency] = useState(account.currency ?? "KRW");
  const [url, setUrl] = useState(account.URL ?? "");
  const [memo, setMemo] = useState(account.memo ?? "");
  const [topPickStartAmount, setTopPickStartAmount] = useState(
    account.top_pick_start_amount_manwon == null ? "" : String(account.top_pick_start_amount_manwon),
  );
  const [topPickStartDate, setTopPickStartDate] = useState(account.top_pick_start_date ?? "");
  const [tickerType, setTickerType] = useState((account.ticker_types ?? [])[0] ?? "");
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
            ticker_types: tickerType ? [tickerType] : [],
            memo: memo.trim(),
            top_pick_start_amount_manwon: topPickStartAmount === "" ? null : Number(topPickStartAmount),
            top_pick_start_date: topPickStartDate || null,
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
        <span style={{ ...labelStyle, marginLeft: 16 }}>종목풀</span>
        <select
          style={{ ...inputStyle, minWidth: 220 }}
          value={tickerType}
          onChange={(event) => setTickerType(event.target.value)}
          disabled={pools.length === 0}
        >
          <option value="">선택 안 함</option>
          {pools.map((pool) => (
            <option key={pool.ticker_type} value={pool.ticker_type}>
              {pool.icon ? `${pool.icon} ` : ""}
              {pool.name ?? pool.ticker_type} ({pool.ticker_type})
            </option>
          ))}
        </select>
      </div>

      <div style={rowStyle}>
        <span style={{ ...labelStyle, width: 60 }}>메모</span>
        <input
          style={{ ...inputStyle, flex: 1, minWidth: 220 }}
          placeholder="탑픽 설정 상단에 표시할 1줄 메모"
          value={memo}
          onChange={(e) => setMemo(e.target.value.replace(/\r?\n/g, " "))}
        />
      </div>

      <div style={rowStyle}>
        <span style={{ ...labelStyle, width: 60 }}>탑픽</span>
        <label style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 220 }}>
          <span style={labelStyle}>{currency === "KRW" ? "시작금액(만원)" : `시작금액(${currency})`}</span>
          <input
            type="number"
            style={{ ...inputStyle, width: 130 }}
            min={1}
            step={currency === "KRW" ? 1 : 0.01}
            placeholder="미설정"
            value={topPickStartAmount}
            onChange={(e) => setTopPickStartAmount(e.target.value)}
          />
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 220 }}>
          <span style={labelStyle}>시작일자</span>
          <input
            type="date"
            style={{ ...inputStyle, width: 150 }}
            value={topPickStartDate}
            onChange={(e) => setTopPickStartDate(e.target.value)}
          />
        </label>
      </div>

      <div style={{ ...rowStyle, marginBottom: 0 }}>
        <span style={{ ...labelStyle, width: 60 }}>URL</span>
        <input style={{ ...inputStyle, flex: 1, minWidth: 220 }} placeholder="증권사 접속 URL (선택)" value={url} onChange={(e) => setUrl(e.target.value)} />
      </div>
    </div>
  );
}

export function AccountSettingsManager() {
  const toast = useToast();
  const [accounts, setAccounts] = useState<AccountEntry[]>([]);
  const [pools, setPools] = useState<PoolOption[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const [accResp, poolResp] = await Promise.all([
        fetch("/api/account-settings", { cache: "no-store" }),
        fetch("/api/pool-settings", { cache: "no-store" }),
      ]);
      const accData = (await accResp.json()) as ApiResponse;
      if (!accResp.ok || accData.error) throw new Error(accData.error ?? "계좌 설정을 불러오지 못했습니다.");
      const poolData = (await poolResp.json()) as { pools?: PoolOption[]; error?: string };
      if (!poolResp.ok || poolData.error) throw new Error(poolData.error ?? "종목풀을 불러오지 못했습니다.");
      setAccounts(accData.accounts ?? []);
      setPools(poolData.pools ?? []);
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
        <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 4 }}>계좌 설정</h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginBottom: 12 }}>
          계좌 메타(이름/순서/벤치마크 등)는 DB(account_settings)가 단일 소스입니다. 계좌 추가/삭제는 화면에서 지원하지 않습니다.
        </p>
        {loading ? (
          <div style={{ color: "var(--text-muted)", padding: 12 }}>불러오는 중…</div>
        ) : accounts.length === 0 ? (
          <div style={{ color: "var(--text-muted)", padding: 12 }}>등록된 계좌가 없습니다.</div>
        ) : (
          accounts.map((a) => <AccountRow key={a.account_id} account={a} pools={pools} onSaved={() => {}} />)
        )}
      </div>
    </div>
  );
}
