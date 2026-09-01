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

/** 그리드의 `getRowClass` 로 준다 — 사고팔 수 없는 항목이라 **줄 전체**를 노랗게 칠한다
 *  (추세 이탈 행이 회색 줄로 구분되는 것과 같은 방식). */
export const FIXED_ASSET_ROW_CLASS = "appFixedAssetRow";

/** IS 의 가격 프록시 — 실제 상장 종목이 아니라 시세가 없어, 호주 ETF 풀의 이 종목으로 대신 잰다.
 *  **접두사까지 정확히** 써야 한다: 종목풀에 없는 티커는 가격 캐시가 없어 조회 전체가 실패한다
 *  (예전에 `VGS` 로 적어 `/asset-helper` 의 지표 컬럼이 통째로 비었다). */
export const FIXED_ASSET_PRICE_PROXY = "ASX:VGS";

export function isFixedAssetTicker(ticker: unknown): boolean {
  return String(ticker ?? "").trim().toUpperCase() === FIXED_ASSET_TICKER;
}
