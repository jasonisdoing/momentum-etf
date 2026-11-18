/** 증권사 연동 계좌의 표시용 번호 — 메리츠는 수기 입력한 마스킹 번호, 나무는 저장된 계좌번호의 앞 4·뒤 2자리. */
export function maskedBrokerAccount(accountNo: string | null | undefined, accountMask: string | null | undefined): string {
  if (accountMask) return accountMask;
  if (!accountNo) return "";
  return `${accountNo.slice(0, 4)}***${accountNo.slice(-2)}`;
}
