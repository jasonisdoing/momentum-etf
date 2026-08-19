/**
 * 이평선 "일수" 선택지 — 화면 공용 단일 소스.
 *
 * 백엔드 `utils/pool_settings_store.py` 의 `MA_DAY_OPTIONS` 와 같은 값이어야 한다.
 * 백엔드가 `constraints` 로 내려주는 화면(`/pools-settings`, `/pools-backtest`)에서는
 * 응답을 우선 쓰고, 이 목록은 응답을 못 받았을 때의 폴백이다.
 * 응답을 받지 않는 화면(`/pools-rank`)은 이 목록을 그대로 쓴다.
 *
 * 예전에는 세 화면이 각자 배열을 들고 있어서, 눈금을 바꿀 때 한 화면만 고치면
 * 나머지가 옛 목록을 계속 보여줬다.
 */

/** 단기·장기 이평선 일수. */
export const MA_DAY_OPTIONS: number[] = [5, 10, 20, 40, 60, 120, 160, 180, 240];
