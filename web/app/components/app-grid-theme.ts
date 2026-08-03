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
    // 표는 본문(--fs-base 16px)보다 한 단계 작은 --fs-sm(14px)을 쓴다.
    // 일부러 다르게 둔 것이다 — 한 화면에 행을 많이 담아야 하는 데이터 표라
    // 본문 크기를 그대로 쓰면 스크롤이 길어지고 훑어보기 나빠진다.
    // AG Grid 테마는 JS 객체라 CSS 토큰을 못 읽으므로 여기서 숫자로 적는다.
    // --fs-sm 을 바꾸면 이 줄도 같이 바꿔야 표만 옛 크기로 남지 않는다.
    fontSize: 14,
    wrapperBorderRadius: 10,
    rowHeight: 34,
    headerHeight: 36,
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
