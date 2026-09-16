"use client";

import { AgGridReact } from "ag-grid-react";
import { useEffect, useRef } from "react";
import { AllCommunityModule, ModuleRegistry } from "ag-grid-community";
import type {
  CellClassParams,
  ColDef,
  ColGroupDef,
  GridOptions,
  RowClassParams,
  Theme,
  GridApi,
} from "ag-grid-community";

ModuleRegistry.registerModules([AllCommunityModule]);

type AppAgGridProps<TData> = {
  rowData: TData[];
  /** AG Grid 는 평범한 컬럼과 컬럼 그룹을 섞어 받는다(2단 헤더). */
  columnDefs: (ColDef<TData> | ColGroupDef<TData>)[];
  loading?: boolean;
  minHeight?: string | number;
  height?: string | number;
  className?: string;
  getRowClass?: (params: RowClassParams<TData>) => string;
  getCellClass?: (params: CellClassParams<TData>) => string | string[] | undefined;
  getRowId?: (params: { data: TData }) => string;
  /** getRowClass 가 rowData 밖에서 참조하는 값들 — 바뀌면 행을 다시 그려 클래스를 재평가한다. */
  rowClassDeps?: readonly unknown[];
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
  rowClassDeps,
  gridOptions,
  theme = "legacy",
}: AppAgGridProps<TData>) {
  const themeClassName = theme === "legacy" ? "ag-theme-quartz appAgGridThemeLegacy" : "appAgGridTheme";
  // 행 클래스(getRowClass)는 AG Grid 가 행 생성 시점에만 평가한다 — getRowId 가 같은 행을
  // 데이터만 바꿔 갱신하면(이평선 변경 재계산 등) 셀 값은 새 값인데 행 배경은 옛 판정으로
  // 남는다. 데이터가 바뀌면 행을 다시 그려 클래스도 같은 데이터 기준으로 재평가한다.
  const apiRef = useRef<GridApi<TData> | null>(null);
  // getRowClass 가 rowData 밖의 값(현재 선택 등)에 의존하면 그 값도 재그리기 트리거여야 한다 —
  // 안 그러면 선택을 바꿔도 이전 선택 행의 강조가 남는다(레버리지 튜닝 표 사례).
  const rowClassDepsKey = JSON.stringify(rowClassDeps ?? null);
  useEffect(() => {
    apiRef.current?.redrawRows();
  }, [rowData, rowClassDepsKey]);
  // 호출부가 gridOptions.onGridReady 를 넘겨도 내부 api 캡처가 덮이지 않게 둘을 합쳐 부른다.
  const { onGridReady: callerOnGridReady, ...restGridOptions } = gridOptions ?? {};
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
          {...restGridOptions}
          onGridReady={(event) => {
            apiRef.current = event.api;
            callerOnGridReady?.(event);
          }}
        />
      </div>
    </div>
  );
}
