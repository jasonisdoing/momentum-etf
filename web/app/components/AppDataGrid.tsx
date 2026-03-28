"use client";

import {
  DataGrid,
  type DataGridProps,
  type GridValidRowModel,
} from "@mui/x-data-grid";

type AppDataGridProps<R extends GridValidRowModel> = DataGridProps<R> & {
  minHeight?: string | number;
  wrapClassName?: string;
};

export function AppDataGrid<R extends GridValidRowModel>({
  className,
  wrapClassName,
  minHeight = "24rem",
  sx,
  ...props
}: AppDataGridProps<R>) {
  return (
    <div
      className={wrapClassName ? `appDataGridWrap ${wrapClassName}` : "appDataGridWrap"}
      style={{ minHeight }}
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
        rowHeight={40}
        columnHeaderHeight={40}
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
