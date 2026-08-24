"use client";

import { AgGridReact } from "ag-grid-react";
import { useEffect, useRef } from "react";
import { AllCommunityModule, ModuleRegistry } from "ag-grid-community";
import type {
  CellClassParams,
  ColDef,
  GridOptions,
  RowClassParams,
  Theme,
  GridApi,
} from "ag-grid-community";

ModuleRegistry.registerModules([AllCommunityModule]);

type AppAgGridProps<TData> = {
  rowData: TData[];
  columnDefs: ColDef<TData>[];
  loading?: boolean;
  minHeight?: string | number;
  height?: string | number;
  className?: string;
  getRowClass?: (params: RowClassParams<TData>) => string;
  getCellClass?: (params: CellClassParams<TData>) => string | string[] | undefined;
  getRowId?: (params: { data: TData }) => string;
  gridOptions?: GridOptions<TData>;
  theme?: Theme | "legacy";
};

export function AppAgGrid<TData>({
  rowData,
  columnDefs,
  loading = false,
  minHeight = "24rem",
  height = "100%",
  className,
  getRowClass,
  getRowId,
  gridOptions,
  theme = "legacy",
}: AppAgGridProps<TData>) {
  const themeClassName = theme === "legacy" ? "ag-theme-quartz appAgGridThemeLegacy" : "appAgGridTheme";
  // 행 클래스(getRowClass)는 AG Grid 가 행 생성 시점에만 평가한다 — getRowId 가 같은 행을
  // 데이터만 바꿔 갱신하면(이평선 변경 재계산 등) 셀 값은 새 값인데 행 배경은 옛 판정으로
  // 남는다. 데이터가 바뀌면 행을 다시 그려 클래스도 같은 데이터 기준으로 재평가한다.
  const apiRef = useRef<GridApi<TData> | null>(null);
  useEffect(() => {
    apiRef.current?.redrawRows();
  }, [rowData]);
  return (
    <div className={className ? `appAgGridWrap ${className}` : "appAgGridWrap"} style={{ minHeight, height }}>
      <div className={themeClassName}>
        <AgGridReact<TData>
          rowData={rowData}
          columnDefs={columnDefs}
          loading={loading}
          theme={theme}
          enableCellTextSelection
          ensureDomOrder
          suppressCellFocus
          rowSelection={{ mode: "singleRow", checkboxes: false, enableClickSelection: false }}
          animateRows={false}
          defaultColDef={{
            sortable: true,
            resizable: true,
          }}
          getRowClass={getRowClass}
          getRowId={getRowId}
          onGridReady={(event) => {
            apiRef.current = event.api;
          }}
          {...gridOptions}
        />
      </div>
    </div>
  );
}
