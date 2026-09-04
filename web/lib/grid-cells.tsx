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

/** 업종 컬럼 폭 — 업종 컬럼이 있는 모든 화면이 같은 값을 쓴다.
 *
 *  셀은 2줄까지 보이고 넘치면 말줄임(`renderIndustryCell`)이라, **2줄이 꽉 차는 폭**이
 *  기준이다. 실측: 미국·호주는 90%가 30자 안쪽(최장 35자)이고 한국은 최장 12자
 *  (`섬유,의류,신발,호화품`)다. 30자를 2줄에 담으려면 15자/줄, 13px 영문 평균 자폭
 *  6.8px 에 셀 좌우 패딩을 더해 약 124px 이다.
 *
 *  예전 200px 은 영문 기준으로도 남았고, 한국 화면에서는 절반이 빈 채로 있었다. */
export const INDUSTRY_COLUMN_WIDTH = 124;
export const INDUSTRY_COLUMN_MIN_WIDTH = 104;

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

/** 업종 컬럼 — 순위·전략·마켓 화면 공용.
 *
 *  폭·셀 표기만 공용이고 정의는 화면마다 복사돼 있어서, 툴팁이 제각각이거나 아예 없고
 *  빈 컬럼을 숨기는 기준도 갈라져 있었다. 여기 하나로 모은다.
 *
 *  `field` 는 화면의 데이터 키(순위 화면만 한글 `"업종"`), `hide` 는 업종이 아예 없는
 *  풀(ETF 모음 등)에서 빈 컬럼을 숨길 때 쓴다. */
export function industryColumn<T>(options?: {
  field?: string;
  hide?: boolean;
  cellClass?: string;
}): ColDef<T> {
  return {
    colId: "industry",
    field: (options?.field ?? "industry") as ColDefField<T>,
    headerName: "업종",
    hide: options?.hide,
    width: INDUSTRY_COLUMN_WIDTH,
    minWidth: INDUSTRY_COLUMN_MIN_WIDTH,
    headerTooltip: "한국은 네이버 분류, 미국·호주는 지수 구성종목의 yfinance 분류",
    cellClass: options?.cellClass,
    cellRenderer: (p: { value?: string }) => renderIndustryCell(p.value),
  };
}

/** 고점 대비(%) 셀 — 정확히 0 이면 ⭐신고점(빨강 볼드), 그 외 퍼센트 표기. */
export function renderHighDrawdownCell(value: number | null | undefined, digits = 1) {
  if (value === 0) return <span style={{ color: "#d93025", fontWeight: 700 }}>⭐신고점</span>;
  if (value == null || Number.isNaN(value)) return <span>-</span>;
  return <span>{`${value.toFixed(digits)}%`}</span>;
}

/** 이탈 이평선까지 남은 이격 컬럼 — 신고가·모멘텀 운용 현황 공용.
 *  0 이하면 이탈이고, 5% 안으로 가까워지면 빨강·굵게 표시한다. */
export function maExitGapColumn<T>(options: {
  field: ColDefField<T>;
  maDays: number | null | undefined;
  getMaValue?: (row: T | undefined) => number | null | undefined;
  formatMaValue?: (value: number, row: T | undefined) => string;
}): ColDef<T> {
  const label = `MA${options.maDays ?? ""}`;
  return {
    field: options.field,
    headerName: `${label} 이탈`,
    width: 96,
    type: "numericColumn",
    headerTooltip: `현재가가 이탈 이평선(${label})보다 몇 % 위인지. 0에 가까울수록 매도가 가깝다.`,
    tooltipValueGetter: (p) => {
      const value = options.getMaValue?.(p.data);
      if (value == null) return "";
      return `이탈선 ${options.formatMaValue?.(value, p.data) ?? String(value)}`;
    },
    valueFormatter: (p) => (p.value == null ? "-" : `${Number(p.value).toFixed(1)}%`),
    cellStyle: (p): { color: string; fontWeight: number } | null => {
      const value = p.value as number | null;
      if (value != null && value <= 5) return { color: "var(--up-color, #d64545)", fontWeight: 700 };
      return null;
    },
  };
}

/** 거래대금 배수(20일 평균 대비) 셀 색 — 클수록 진해진다. 종목풀 순위·신고가 공용.
 *
 *  대부분 1배 근처에 몰려 있어(중앙값 0.94배) 평상시 값까지 물들이면 표가 시끄러워지고
 *  정작 봐야 할 종목이 묻힌다. 1.5배까지는 흐린 회색으로 눌러두고 그 위부터 같은 빨강의
 *  농도만 올린다 — 색상을 섞으면 등락률 색과 헷갈린다.
 *  단계는 실제 분포에 맞췄다(90%분위 1.6배 · 99%분위 6.3배).
 *
 *  `bold` 를 주면 농도와 별개로 굵기를 강제한다. 신고가 화면이 '진입 자격 통과' 를
 *  이걸로 표시한다 — 자격 하한은 설정값이라 농도 단계와 일치하지 않는다.
 */
/** 거래대금 배수 표기 — 순위·신고가 화면 공용.
 *
 *  `value` 는 본값(장중=토스 실시간 누적, 마감 후=KRX 확정)이고, `live` 는 장중에만 오는
 *  **시간 환산 배수**(누적 ÷ 장 경과율 — 지금 페이스대로면 하루 기준 몇 배인지)다.
 *  둘이 다를 때 `3.0배 (19.5)` 처럼 나란히 보여준다(장중이 아니면 괄호가 안 붙는다).
 */
/** 거래대금(배수) 컬럼 폭 — 이 컬럼을 쓰는 모든 화면이 같은 값을 쓴다.
 *
 *  값은 대개 `3.9배`(실측 99%가 3.9 이하, 최대 14.7)라 짧고, 한국 장중에만 환산 배수가
 *  괄호로 붙어 `3.9배 (19.5)` 로 가장 길어진다. 13px 기준 약 90px 이고, 우측 정렬 헤더
 *  `거래대금`(4자 + 정렬 화살표)도 비슷해서 둘 다 100px 안에 들어간다. */
export const TRADE_VALUE_COLUMN_WIDTH = 100;
export const TRADE_VALUE_COLUMN_MIN_WIDTH = 88;

export function formatTradeValueMult(
  value: number | null | undefined,
  live?: number | null,
): string {
  const hasLive = live != null && !Number.isNaN(live);
  if (value == null || Number.isNaN(value)) return hasLive ? `- (${live.toFixed(1)})` : "-";
  const main = `${value.toFixed(1)}배`;
  // 0.05 미만 차이는 반올림 오차라 괄호를 붙이지 않는다.
  if (!hasLive || Math.abs(live - value) < 0.05) return main;
  return `${main} (${live.toFixed(1)})`;
}

export function tradeValueMultStyle(
  value: number | null | undefined,
  bold?: boolean,
): { color: string; fontWeight?: number } {
  // 단계는 **불투명 색**으로 만든다. 예전에는 알파(0.5~0.85)로 강도를 표현했는데,
  // 편집 칸처럼 배경색이 깔린 셀에서 옅은 단계가 배경에 묻혀 회색(--text-muted, #5b6675)
  // 보다 흐려졌다 — 신호가 있는 1.9배가 신호가 없는 0.6배보다 약해 보였다.
  // 아래로 갈수록 채도만 낮아지고 명도는 회색 이상을 유지한다.
  const weight = bold ? { fontWeight: 700 } : {};
  if (value == null || Number.isNaN(value)) return { color: "var(--text-muted)", ...weight };
  if (value >= 5) return { color: "#a11010", fontWeight: 700 };
  if (value >= 3) return { color: "#c62828", fontWeight: 700 };
  if (value >= 2) return { color: "#d13a3a", ...weight };
  if (value >= 1.5) return { color: "#b8514a", ...weight };
  return { color: "var(--text-muted)", ...weight };
}

/** 거래대금(20일 평균 대비 배수) 컬럼 — 순위·신고가·모멘텀·합성이 같은 정의를 쓴다.
 *  표준 배치는 「일간(%) → 현재가 → 거래대금」 — 이 컬럼을 현재가 바로 뒤에 둔다.
 *  필드명이 화면마다 다르면(fied·liveField) 지정하고, 진입 자격 굵기가 필요한 화면만 qualifies 를 준다. */
export function tradeValueMultColumn<T>(options?: {
  field?: string;
  liveField?: string;
  qualifies?: (row: T | undefined) => boolean | undefined;
  hide?: boolean;
  headerTooltip?: string;
}): ColDef<T> {
  const field = options?.field ?? "value_mult";
  const liveField = options?.liveField ?? "value_mult_live";
  return {
    colId: field,
    valueGetter: (p) => ((p.data as Record<string, unknown> | undefined)?.[field] as number | null | undefined) ?? null,
    headerName: "거래대금",
    width: TRADE_VALUE_COLUMN_WIDTH,
    minWidth: TRADE_VALUE_COLUMN_MIN_WIDTH,
    hide: options?.hide,
    type: "numericColumn",
    headerTooltip:
      options?.headerTooltip ??
      "20일 평균 거래대금 대비 배수 (순위·신고가 화면과 같은 값). " +
      "괄호는 장중 시간 비율로 환산한 배수 — 지금 페이스대로면 하루 기준 몇 배인지입니다.",
    valueFormatter: (p: { value?: unknown; data?: T }) =>
      formatTradeValueMult(
        p.value as number | null,
        ((p.data as Record<string, unknown> | undefined)?.[liveField] as number | null | undefined) ?? null,
      ),
    cellStyle: (p: { value?: unknown; data?: T }) =>
      tradeValueMultStyle(p.value as number | null, options?.qualifies?.(p.data)),
  };
}

/** 시장 ADR 컬럼 — 모멘텀·신고가 백테스트 표 공용. 일간=당일 값, 주간=판정일 값,
 *  월간·연간=기간 최저. 값이 하나도 없으면(레짐 시장 없는 풀) `hide` 로 숨긴다. */
export function adrColumn<T>(options: {
  headerName: string;
  headerTooltip: string;
  hide: boolean;
  getter: (row: T) => number | null | undefined;
}): ColDef<T> {
  return {
    headerName: options.headerName,
    colId: "adr",
    headerTooltip: options.headerTooltip,
    width: 96,
    hide: options.hide,
    type: "numericColumn",
    valueGetter: (p) => (p.data ? options.getter(p.data) ?? null : null),
    valueFormatter: (p) => (p.value == null ? "-" : Number(p.value).toFixed(1)),
  };
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
/** 메모 칸 배경(노랑) — 규칙은 globals.css 의 `.appMemoCell` 한 곳에 있다. */
const MEMO_CELL_CLASS = "appMemoCell";

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
    // 메모 칸 표시(노란 배경)는 화면이 `cellClass` 를 덮어써도 유지한다 — 같은 값을 여러
    // 화면이 함께 보는 칸이라 색이 화면마다 달라지면 같은 칸인지 알아보기 어렵다.
    cellClass: (params) => {
      const base = cellClass ? cellClass(params.data) : canEdit(params.data) ? "appEditableCell" : undefined;
      return [base, MEMO_CELL_CLASS].filter(Boolean).join(" ");
    },
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

/** 고점 대비(%) — 캐시 전 기간 최고가 대비 마지막 종가. 0 이면 ⭐신고점. */
export function highDrawdownColumn<T>(field: ColDefField<T>): ColDef<T> {
  return {
    field,
    headerName: "고점",
    headerTooltip: "최근 고점 대비 현재가(%) — pools-rank 고점과 같은 규칙, 0 = 신고점",
    width: 80,
    type: "rightAligned",
    cellRenderer: (p: { value?: number | null }) => renderHighDrawdownCell(p.value, 1),
  };
}

/**
 * 슬롯 전략의 **상태** 컬럼 — 신고가·모멘텀이 같은 폭·같은 문구를 쓴다.
 *
 * 두 표(보유·후보)가 나란히 있어 폭이 다르면 어긋나 보인다. 장중 판정은 오늘 종가로
 * 확정되기 전이라 `(예상)` 꼬리표를 붙인다.
 */
export const STATUS_COLUMN_WIDTH = 124;
export const STATUS_COLUMN_MIN_WIDTH = 110;

export type SlotPlan = "hold" | "sell" | "buy" | "exited" | "empty";

export function slotStatusColumn<T extends { plan: SlotPlan; days: number | null; is_new: boolean; exit_reason: string | null }>(options: {
  live: boolean;
}): ColDef<T> {
  const tag = options.live ? "(예상)" : "";
  return {
    headerName: "상태",
    width: STATUS_COLUMN_WIDTH,
    minWidth: STATUS_COLUMN_MIN_WIDTH,
    cellStyle: { display: "flex", alignItems: "center", justifyContent: "center" },
    valueGetter: (p) => p.data?.plan ?? "",
    cellRenderer: (p: { data?: T }) => {
      if (!p.data || p.data.plan === "empty") return null;
      const reason = p.data.exit_reason ? ` (${p.data.exit_reason})` : "";
      if (p.data.plan === "buy") return <strong style={{ color: "#d62828" }}>진입 예정{tag}</strong>;
      if (p.data.plan === "sell") {
        const label = `매도 예정${tag}${reason}`;
        return (
          <strong style={{ color: "#1971c2", whiteSpace: "nowrap" }} title={label}>
            {label}
          </strong>
        );
      }
      if (p.data.plan === "exited") {
        const label = `이탈${reason}`;
        return (
          <span style={{ color: "var(--text-muted)", whiteSpace: "nowrap" }} title={label}>
            {label}
          </span>
        );
      }
      return <span>{p.data.is_new ? "진입" : `${p.data.days}일`}</span>;
    },
  };
}

/** 편입일·매수가·청산가·수익률 — 보유 표의 공통 꼬리 컬럼(두 전략이 같다). */
export function slotTradeColumns<T>(options: { fillDay: string }): ColDef<T>[] {
  const price = (value: unknown) =>
    value == null || !Number.isFinite(Number(value))
      ? "-"
      : Number(value).toLocaleString("ko-KR", { maximumFractionDigits: 2 });
  return [
    {
      field: "entry_date" as ColDefField<T>,
      headerName: "편입일",
      width: 116,
      // 아직 안 산 종목은 편입일이 없다 — 다음 시가에 정해진다.
      valueFormatter: (p) => (p.value ? String(p.value) : "-"),
    },
    {
      field: "entry_price" as ColDefField<T>,
      headerName: "매수가",
      width: 110,
      type: "numericColumn",
      headerTooltip: `진입 예정 종목은 ${options.fillDay} 시가에 체결되므로 아직 값이 없다.`,
      valueFormatter: (p) => price(p.value),
    },
    {
      field: "exit_price" as ColDefField<T>,
      headerName: "청산가",
      width: 110,
      type: "numericColumn",
      headerTooltip: "오늘 이탈한 종목의 체결가. 아직 들고 있는 종목은 값이 없다.",
      valueFormatter: (p) => price(p.value),
    },
    {
      field: "return_pct" as ColDefField<T>,
      headerName: "수익률",
      width: 108,
      type: "numericColumn",
      headerTooltip: "아직 청산 전이라 매도 슬리피지는 빠져 있다.",
      valueFormatter: (p) => (p.value == null ? "-" : formatSignedPct(p.value as number, 2)),
      cellStyle: (p) => ({ color: signColor(p.value as number), fontWeight: 700 }),
    },
  ];
}
