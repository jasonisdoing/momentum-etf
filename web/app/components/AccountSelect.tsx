"use client";

import type { CSSProperties } from "react";

/** 계좌 셀렉터가 필요한 최소 필드. 화면별 추가 필드는 그대로 두고 넘기면 된다. */
export type AccountOptionBase = {
  account_id: string;
  name?: string | null;
  icon?: string | null;
  order?: number | null;
};

// 계좌 표기 — /pools-backtest 종목풀 셀렉터(formatPoolLabel)와 같은 조합 형식.
// `1. 💰 연금저축 계좌(pension_account)`. order/icon 이 없으면 그 부분만 빠진다.
export function formatAccountLabel(acc: AccountOptionBase): string {
  const name = String(acc.name ?? "").trim() || acc.account_id;
  const prefix = [
    acc.order === null || acc.order === undefined ? null : `${acc.order}.`,
    String(acc.icon ?? "").trim() || null,
  ]
    .filter(Boolean)
    .join(" ");
  const body = `${name}(${acc.account_id})`;
  return prefix ? `${prefix} ${body}` : body;
}

type Props = {
  accounts: AccountOptionBase[];
  value: string;
  onChange: (accountId: string) => void;
  /** 라벨 문구 — 화면에 따라 "계좌" / "적용 계좌". */
  label?: string;
  disabled?: boolean;
  /** 빈 값 선택지 문구. 주지 않으면 빈 값 선택지가 없다. */
  emptyLabel?: string;
  style?: CSSProperties;
  labelStyle?: CSSProperties;
};

/** 화면 공용 계좌 셀렉터 — 표기·마크업을 한 곳에서 맞춘다. */
export default function AccountSelect({
  accounts,
  value,
  onChange,
  label = "계좌",
  disabled = false,
  emptyLabel,
  style,
  labelStyle,
}: Props) {
  const hasAccounts = accounts.length > 0;
  return (
    <label className="appLabeledField" style={labelStyle}>
      <span className="appLabeledFieldLabel">{label}</span>
      <select
        className="form-select form-select-sm"
        style={style}
        value={value}
        disabled={disabled || !hasAccounts}
        onChange={(event) => onChange(event.target.value)}
      >
        {!hasAccounts ? <option value="">계좌 불러오는 중...</option> : null}
        {hasAccounts && emptyLabel !== undefined ? <option value="">{emptyLabel}</option> : null}
        {accounts.map((account) => (
          <option key={account.account_id} value={account.account_id}>
            {formatAccountLabel(account)}
          </option>
        ))}
      </select>
    </label>
  );
}
