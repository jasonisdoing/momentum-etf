/**
 * AG Grid 공용 셀 헬퍼 — 여러 화면(pools-rank, strategy-momentum 등)이 같은 표기를 쓰는
 * 셀의 단일 소스. 화면별 고유 스타일(클래스 기반 색 등)은 각 화면에 남긴다.
 */

import type { ColDef, ColDefField } from "ag-grid-community";
import type React from "react";

/** 부호를 붙인 퍼센트 표기: +1.23% / -4.56% / "-"(값 없음). */
export function formatSignedPct(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

/** 상승 빨강 · 하락 파랑 · 0/없음 상속 — 인라인 색 방식 화면용. */
export function signColor(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value) || value === 0) return "inherit";
  return value > 0 ? "var(--up-color, #d64545)" : "var(--down-color, #2f6fd0)";
}

/** 마켓(KOSPI/KOSDAQ) 배지 셀 스타일 — KOSPI 녹색 · KOSDAQ 파란색, 그 외 없음. */
export function marketBadgeCellStyle(value: unknown): React.CSSProperties | null {
  if (value === "KOSPI")
    return { textAlign: "center", backgroundColor: "#d1e7dd", color: "#0f5132", fontWeight: "bold" };
  if (value === "KOSDAQ")
    return { textAlign: "center", backgroundColor: "#cfe2ff", color: "#084298", fontWeight: "bold" };
  return { textAlign: "center" };
}

/** 종목명 컬럼 폭 — 종목명 컬럼이 있는 모든 화면이 같은 값을 쓴다.
 *  남는 폭은 종목명이 가져가되(flex), 좁아져도 이 폭 아래로는 줄지 않는다.
 *  긴 이름은 2줄까지 보이고 넘치면 말줄임 (renderStockNameCell). */
export const STOCK_NAME_COLUMN_MIN_WIDTH = 220;

/** 업종 컬럼 폭 — 종목풀 순위 화면 기준. 업종 컬럼이 있는 모든 화면이 같은 값을 쓴다.
 *  한글 최장은 `섬유,의류,신발,호화품`(12자). 영문은 25자 안팎이 흔하고
 *  40자짜리(`Drug Manufacturers - Specialty & Generic`)는 2줄까지 보이고 넘치면 말줄임. */
export const INDUSTRY_COLUMN_WIDTH = 200;
export const INDUSTRY_COLUMN_MIN_WIDTH = 150;

/** 업종 셀 — 종목풀 화면의 종목명과 같이 2줄까지 보이고 넘치면 말줄임(전체 값은 툴팁). */
export function renderIndustryCell(value: string | null | undefined) {
  const text = String(value ?? "").trim();
  if (!text) return <span>-</span>;
  return (
    <span className="appNameCellText" title={text}>
      {text}
    </span>
  );
}

/** 고점 대비(%) 셀 — 정확히 0 이면 ⭐신고점(빨강 볼드), 그 외 퍼센트 표기. */
export function renderHighDrawdownCell(value: number | null | undefined, digits = 1) {
  if (value === 0) return <span style={{ color: "#d93025", fontWeight: 700 }}>⭐신고점</span>;
  if (value == null || Number.isNaN(value)) return <span>-</span>;
  return <span>{`${value.toFixed(digits)}%`}</span>;
}

/** 거래대금 배수(20일 평균 대비) 셀 색 — 클수록 진해진다. 종목풀 순위·신고가 돌파 공용.
 *
 *  대부분 1배 근처에 몰려 있어(중앙값 0.94배) 평상시 값까지 물들이면 표가 시끄러워지고
 *  정작 봐야 할 종목이 묻힌다. 1.5배까지는 흐린 회색으로 눌러두고 그 위부터 같은 빨강의
 *  농도만 올린다 — 색상을 섞으면 등락률 색과 헷갈린다.
 *  단계는 실제 분포에 맞췄다(90%분위 1.6배 · 99%분위 6.3배).
 *
 *  `bold` 를 주면 농도와 별개로 굵기를 강제한다. 신고가 화면이 '진입 자격 통과' 를
 *  이걸로 표시한다 — 자격 하한은 설정값이라 농도 단계와 일치하지 않는다.
 */
export function tradeValueMultStyle(
  value: number | null | undefined,
  bold?: boolean,
): { color: string; fontWeight?: number } {
  const weight = bold ? { fontWeight: 700 } : {};
  if (value == null || Number.isNaN(value)) return { color: "var(--text-muted)", ...weight };
  if (value >= 5) return { color: "rgb(214, 40, 40)", fontWeight: 700 };
  if (value >= 3) return { color: "rgba(214, 40, 40, 0.85)", fontWeight: 700 };
  if (value >= 2) return { color: "rgba(214, 40, 40, 0.7)", ...weight };
  if (value >= 1.5) return { color: "rgba(214, 40, 40, 0.5)", ...weight };
  return { color: "var(--text-muted)", ...weight };
}

/** 시총 순위 컬럼 — 순위·모멘텀·신고가 화면 공용. 배치 B 가 메타 캐시에 적어 둔 국가별 시장 전체
 *  시총 순위(한국=KOSPI+KOSDAQ, 미국=S&P500∪NDX100, 호주=ASX200)다. 개별주 풀에서만 보이고
 *  (`hide`), 값이 없으면 "-". 티커 컬럼 바로 앞에 둔다. */
export function marketCapRankColumn<T>(field: ColDefField<T>, hide: boolean): ColDef<T> {
  return {
    headerName: "시총",
    field,
    width: 72,
    type: "numericColumn",
    hide,
    headerTooltip: "시장 전체 시가총액 순위 (배치 기준, 하루 1회 갱신)",
    valueFormatter: (p) => (p.value == null ? "-" : String(p.value)),
    cellStyle: () => ({ color: "var(--text-muted)" }),
  };
}

/** 종목 메모 컬럼 — 순위·모멘텀·합성·자산 관리가 **같은 값**을 보여주는 수기 칸.
 *  메모는 계좌가 아니라 **종목**에 붙는다(`utils/stock_memo_store`). 화면마다 컬럼을
 *  새로 짜면 라벨·폭·빈 값 표기·문자열 에디터 지정이 갈리므로 여기서만 정의한다.
 *
 *  저장 방식은 화면마다 다르다:
 *   · `onSave` 를 주면 셀을 벗어날 때 바로 저장한다(순위·모멘텀·합성).
 *   · 안 주면 그리드 값만 바뀐다 — 화면이 자기 저장 흐름을 태운다(자산 관리의 일괄 저장).
 *  `editable` 이 false 인 행은 편집도 안 되고 빈 값도 흐리게 표시하지 않는다(현금 행 등). */
export function stockMemoColumn<T>(options: {
  field: ColDefField<T>;
  width?: number;
  editable?: (row: T | undefined) => boolean;
  cellClass?: (row: T | undefined) => string | undefined;
  onSave?: (row: T, memo: string) => void;
}): ColDef<T> {
  const { field, width = 150, editable, cellClass, onSave } = options;
  const canEdit = (row: T | undefined) => (editable ? editable(row) : true);
  const column: ColDef<T> = {
    field,
    headerName: "메모",
    headerTooltip: "종목에 붙는 메모 — 자산 관리·순위·전략 화면이 같은 값을 본다",
    width,
    sortable: false,
    // 빈 문자열이 많아 자동 추론이 흔들린다 — 문자열 에디터를 명시한다.
    cellDataType: "text",
    editable: (params) => canEdit(params.data),
    cellClass: (params) =>
      cellClass ? cellClass(params.data) : canEdit(params.data) ? "appEditableCell" : undefined,
    valueParser: (params) => String(params.newValue ?? "").trim(),
    cellRenderer: (params: { data?: T; value?: string | null }) => {
      const text = String(params.value ?? "").trim();
      if (text) return <span>{text}</span>;
      if (!canEdit(params.data)) return <span>-</span>;
      return <span style={{ color: "var(--text-muted)" }}>-</span>;
    },
  };
  if (!onSave) return column;
  return {
    ...column,
    valueSetter: (params) => {
      const row = params.data;
      const next = String(params.newValue ?? "").trim();
      if (!row || !canEdit(row) || next === String((row as Record<string, unknown>)[field as string] ?? "")) {
        return false;
      }
      (row as Record<string, unknown>)[field as string] = next;
      onSave(row, next);
      return true;
    },
  };
}
