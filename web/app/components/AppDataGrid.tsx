"use client";

import {
  DataGrid,
  type DataGridProps,
  type GridValidRowModel,
} from "@mui/x-data-grid";

import { AppDataGridLoadingOverlay } from "./AppDataGridLoadingOverlay";

type AppDataGridProps<R extends GridValidRowModel> = DataGridProps<R> & {
  minHeight?: string | number;
  maxHeight?: string | number;
  wrapClassName?: string;
  fitContentRows?: boolean;
  fitContentMaxRows?: number;
};

export function AppDataGrid<R extends GridValidRowModel>({
  className,
  wrapClassName,
  minHeight = "24rem",
  maxHeight,
  fitContentRows = false,
  fitContentMaxRows = 8,
  sx,
  rows,
  rowHeight = 40,
  columnHeaderHeight = 40,
  ...props
}: AppDataGridProps<R>) {
  const shouldFitContent =
    fitContentRows && Array.isArray(rows) && rows.length > 0 && rows.length <= fitContentMaxRows;
  const wrapStyle = shouldFitContent
    ? {
        height: columnHeaderHeight + rowHeight * rows.length + 2,
      }
    : { minHeight, ...(maxHeight ? { maxHeight } : {}) };

  return (
    <div
      className={wrapClassName ? `appDataGridWrap ${wrapClassName}` : "appDataGridWrap"}
      style={wrapStyle}
    >
      <DataGrid
        className={className ? `appDataGrid ${className}` : "appDataGrid"}
        disableColumnMenu
        disableRowSelectionOnClick
        hideFooter
        localeText={{
          noRowsLabel: "데이터가 없습니다.",
          noResultsOverlayLabel: "조건에 맞는 데이터가 없습니다.",
        }}
        slots={{
          loadingOverlay: AppDataGridLoadingOverlay,
        }}
        rows={rows}
        rowHeight={rowHeight}
        columnHeaderHeight={columnHeaderHeight}
        sx={{
          border: 0,
          minWidth: 0,
          "& .MuiDataGrid-cell:focus, & .MuiDataGrid-columnHeader:focus, & .MuiDataGrid-cell:focus-within, & .MuiDataGrid-columnHeader:focus-within":
            {
              outline: "none",
            },
          ...sx,
        }}
        {...props}
      />
    </div>
  );
}
