import { iconSetQuartzBold, themeQuartz } from "ag-grid-community";

type AppGridThemeOverrides = {
  fontSize?: number;
  rowHeight?: number;
  headerHeight?: number;
  wrapperBorderRadius?: number;
};

export function createAppGridTheme(overrides: AppGridThemeOverrides = {}) {
  return themeQuartz.withPart(iconSetQuartzBold).withParams({
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
    ...overrides,
  });
}
