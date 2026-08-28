/** 고정 자산(IS, International Shares) 표기 — 화면 공용.
 *
 *  호주 계좌에서 수동으로 관리하는 항목이라 실제 상장 종목이 아니다. 사고팔 수 없고
 *  평가액만 계좌에 반영된다(백엔드 `utils/strategy_mix_service.FIXED_ASSET_TICKER`).
 *
 *  세 화면(`/assets` · `/asset-helper` · `/strategy-mix`)이 각자 다른 티커·이름으로
 *  보여주던 것을 여기 하나로 모았다 — 같은 행이 화면마다 `ASX:VGS` · `ASX:IS` · `IS` 로
 *  달리 보이면 같은 자산인지 알 수 없다.
 */

export const FIXED_ASSET_TICKER = "IS";
export const FIXED_ASSET_NAME = "International Shares";

/** 티커·종목명 칸의 `cellClass` 로 준다 — 매매 대상이 아닌 행임을 배경색으로 구분한다.
 *  (셀 배경이라 `<span>` 이 아니라 컬럼 정의의 cellClass 에 붙여야 칠해진다.) */
export const FIXED_ASSET_CELL_CLASS = "appFixedAssetCell";

export function isFixedAssetTicker(ticker: unknown): boolean {
  return String(ticker ?? "").trim().toUpperCase() === FIXED_ASSET_TICKER;
}
