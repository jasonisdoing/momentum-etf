import type { ColDef } from "ag-grid-community";

import { CASH_ROW_TICKER, formatHiddenAmount, formatKrw, getSignedClass, type HoldingsRow } from "./assets-helpers";

type HoldingMetrics = Pick<HoldingsRow, "weight_pct" | "quantity" | "return_pct" | "pnl_krw" | "valuation_krw">;
type HoldingMetricRow = { ticker: string } & Partial<HoldingMetrics>;

const quantityFormat = new Intl.NumberFormat("ko-KR");

export function renderHoldingWeight(value: number) {
  return <span style={{ color: "#000000", fontWeight: 700 }}>{value.toFixed(2)}%</span>;
}

/** 계좌 보유 지표의 공용 열. 편집 화면은 필요한 열에만 편집 동작을 붙인다. */
export function holdingMetricColumns<T extends HoldingMetricRow>(showAmounts: boolean) {
  return {
    weight: {
      colId: "weight_pct", headerName: "비중", width: 82, type: "rightAligned",
      valueGetter: ({ data }) => data?.weight_pct,
      cellRenderer: ({ value }: { value?: number }) => value == null ? "-" : renderHoldingWeight(value),
    },
    quantity: {
      colId: "quantity", headerName: "수량", width: 72, type: "rightAligned",
      valueGetter: ({ data }) => data?.quantity,
      valueFormatter: ({ value, data }) => value == null || data?.ticker === CASH_ROW_TICKER ? "-" : quantityFormat.format(value),
    },
    returnPct: {
      colId: "return_pct", headerName: "수익률", width: 88, type: "rightAligned",
      valueGetter: ({ data }) => data?.return_pct,
      cellRenderer: ({ value, data }: { value?: number; data?: T }) =>
        value == null || data?.ticker === CASH_ROW_TICKER ? "-" : <span className={getSignedClass(value)}>{value.toFixed(2)}%</span>,
    },
    pnl: {
      colId: "pnl_krw", headerName: "평가손익", width: 120, type: "rightAligned",
      valueGetter: ({ data }) => data?.pnl_krw,
      cellRenderer: ({ value, data }: { value?: number; data?: T }) =>
        value == null || data?.ticker === CASH_ROW_TICKER ? "-" : (
          <span className={getSignedClass(value)}>{formatHiddenAmount(showAmounts, formatKrw(value))}</span>
        ),
    },
    valuation: {
      colId: "valuation_krw", headerName: "평가 금액", width: 124, type: "rightAligned",
      valueGetter: ({ data }) => data?.valuation_krw,
      cellRenderer: ({ value }: { value?: number }) => (
        <span className="appGridNumericValue">{value == null ? "-" : formatHiddenAmount(showAmounts, formatKrw(value))}</span>
      ),
    },
  } satisfies Record<string, ColDef<T, number>>;
}
