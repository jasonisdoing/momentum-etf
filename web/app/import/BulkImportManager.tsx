"use client";

import { useState, useTransition } from "react";

type ParsedImportRow = {
  account_name: string;
  account_id: string;
  currency: string;
  bucket_text: string;
  bucket: number;
  ticker: string;
  name: string;
  quantity: number;
  average_buy_price: number;
};

type PreviewResponse = {
  rows?: ParsedImportRow[];
  account_count?: number;
  row_count?: number;
  error?: string;
};

function formatCount(value: number | undefined): string {
  return new Intl.NumberFormat("ko-KR").format(value ?? 0);
}

function formatPrice(value: number): string {
  return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: 2 }).format(value);
}

export function BulkImportManager() {
  const [rawText, setRawText] = useState("");
  const [rows, setRows] = useState<ParsedImportRow[]>([]);
  const [accountCount, setAccountCount] = useState(0);
  const [rowCount, setRowCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [isParsing, startParsing] = useTransition();
  const [isSaving, startSaving] = useTransition();

  function handleParse() {
    startParsing(async () => {
      try {
        setError(null);
        setNotice(null);
        const response = await fetch("/api/import/preview", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: rawText }),
        });
        const payload = (await response.json()) as PreviewResponse;
        if (!response.ok) {
          throw new Error(payload.error ?? "벌크 입력 파싱에 실패했습니다.");
        }
        setRows(payload.rows ?? []);
        setAccountCount(payload.account_count ?? 0);
        setRowCount(payload.row_count ?? 0);
        setNotice("파싱 완료");
      } catch (parseError) {
        setRows([]);
        setAccountCount(0);
        setRowCount(0);
        setError(parseError instanceof Error ? parseError.message : "벌크 입력 파싱에 실패했습니다.");
      }
    });
  }

  function handleSave() {
    startSaving(async () => {
      try {
        setError(null);
        setNotice(null);
        const response = await fetch("/api/import/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rows }),
        });
        const payload = (await response.json()) as { updated_accounts?: number; error?: string };
        if (!response.ok) {
          throw new Error(payload.error ?? "벌크 입력 반영에 실패했습니다.");
        }
        setNotice(`총 ${formatCount(payload.updated_accounts)}개 계좌 업데이트 완료`);
      } catch (saveError) {
        setError(saveError instanceof Error ? saveError.message : "벌크 입력 반영에 실패했습니다.");
      }
    });
  }

  return (
    <div className="appPageStack">
      {error || notice ? (
        <div className="appBannerStack">
          {error ? <div className="bannerError prelineText">{error}</div> : null}
          {notice ? <div className="bannerSuccess">{notice}</div> : null}
        </div>
      ) : null}
      <section className="appSection">
        <div className="card appCard">
        <div className="card-body appCardBody">
          <div className="bulkGuide">
            <p>TSV 7컬럼 순서: 계좌, 환종, 버킷, 티커, 종목명, 수량, 평균 매입가</p>
            <p>입력에 포함된 계좌만 반영되고, 해당 계좌 내부 종목은 입력값 기준으로 완전 교체된다.</p>
            <p>동일 티커가 다시 들어오면 최초 매수일은 유지된다.</p>
          </div>

          <textarea
            className="bulkTextarea"
            value={rawText}
            onChange={(event) => setRawText(event.target.value)}
            placeholder={"1. 국내 계좌\tKRW\t1. 모멘텀\t005930\t삼성전자\t10\t59000"}
          />

          <div className="tableToolbar">
            <div className="tableMeta">
              <span>행 {formatCount(rowCount)}</span>
              <span>계좌 {formatCount(accountCount)}</span>
            </div>
            <div className="toolbarActions">
              <button className="secondaryButton" type="button" onClick={handleParse} disabled={isParsing}>
                {isParsing ? "파싱 중..." : "파싱 및 확인"}
              </button>
              <button className="primaryButton" type="button" onClick={handleSave} disabled={isSaving || rows.length === 0}>
                {isSaving ? "반영 중..." : "현재 잔고에 반영"}
              </button>
            </div>
          </div>

          <div className="tableWrap">
            <table className="erpTable">
              <thead>
                <tr>
                  <th>계좌</th>
                  <th>환종</th>
                  <th>버킷</th>
                  <th>티커</th>
                  <th>종목명</th>
                  <th className="tableAlignRight">수량</th>
                  <th className="tableAlignRight">평균 매입가</th>
                </tr>
              </thead>
              <tbody>
                {rows.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="emptyCell">
                      파싱한 데이터가 아직 없습니다.
                    </td>
                  </tr>
                ) : (
                  rows.map((row, index) => (
                    <tr key={`${row.account_id}-${row.ticker}-${index}`}>
                      <td>
                        <div className="tableAccountCell">
                          <strong>{row.account_name}</strong>
                          <span>{row.account_id}</span>
                        </div>
                      </td>
                      <td>{row.currency}</td>
                      <td>{row.bucket_text}</td>
                      <td>{row.ticker}</td>
                      <td>{row.name}</td>
                      <td className="tableAlignRight">{formatPrice(row.quantity)}</td>
                      <td className="tableAlignRight">{formatPrice(row.average_buy_price)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
        </div>
      </section>
    </div>
  );
}
