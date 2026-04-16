"use client";

import { iconSetQuartzBold, themeQuartz } from "ag-grid-community";
import type { ColDef } from "ag-grid-community";

import { AppAgGrid } from "../components/AppAgGrid";

type KorMarketStockRow = {
  순번: number;
  티커: string;
  종목명: string;
  현재가: number | null;
  "일간(%)": number | null;
  "전일 거래량(주)": number | null;
  "시가총액(억)": number | null;
};

const korMarketStockGridTheme = themeQuartz
  .withPart(iconSetQuartzBold)
  .withParams({
    accentColor: "#206bc4",
    backgroundColor: "#ffffff",
    foregroundColor: "#182433",
    headerBackgroundColor: "#f8fafc",
    headerTextColor: "#5b6778",
    spacing: 8,
    fontSize: 14,
    wrapperBorderRadius: 10,
    rowHeight: 38,
    headerHeight: 38,
    cellHorizontalPadding: 12,
    headerColumnBorder: true,
    headerColumnBorderHeight: "70%",
    columnBorder: true,
    oddRowBackgroundColor: "#fbfdff",
    headerCellHoverBackgroundColor: "#eef4fb",
    headerCellMovingBackgroundColor: "#e8f0fb",
    iconButtonHoverBackgroundColor: "#eef4fb",
    iconButtonHoverColor: "#206bc4",
    iconSize: 18,
  });

const columnDefs: ColDef<KorMarketStockRow>[] = [
  {
    headerName: "#",
    field: "순번",
    width: 72,
    minWidth: 60,
    maxWidth: 84,
    sortable: false,
    resizable: false,
  },
  {
    headerName: "티커",
    field: "티커",
    width: 120,
    minWidth: 96,
  },
  {
    headerName: "종목명",
    field: "종목명",
    flex: 1,
    minWidth: 260,
  },
  {
    headerName: "현재가",
    field: "현재가",
    width: 120,
    minWidth: 96,
  },
  {
    headerName: "일간(%)",
    field: "일간(%)",
    width: 110,
    minWidth: 96,
  },
  {
    headerName: "전일 거래량(주)",
    field: "전일 거래량(주)",
    width: 150,
    minWidth: 132,
  },
  {
    headerName: "시가총액(억)",
    field: "시가총액(억)",
    width: 140,
    minWidth: 124,
  },
];

export function KorMarketStockManager() {
  return (
    <section className="appSection appSectionFill">
      <div className="card appCard appTableCardFill">
        <div className="card-body appCardBodyTight appTableCardBodyFill">
          <div className="appGridFillWrap">
            <AppAgGrid<KorMarketStockRow>
              rowData={[]}
              columnDefs={columnDefs}
              loading={false}
              theme={korMarketStockGridTheme}
              minHeight="32rem"
              gridOptions={{
                overlayNoRowsTemplate: '<span style="color:#667382;">데이터 없음</span>',
              }}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
