/**
 * 이평선 "일수" 선택지 — 화면 공용 단일 소스.
 *
 * 백엔드 `utils/pool_settings_store.py` 의 `MA_DAY_OPTIONS` 와 같은 값이어야 한다.
 * 화면들은 백엔드가 응답으로 내려주는 목록을 우선 쓴다(`/pools-settings`,
 * `/pools-backtest` 는 `constraints`, `/pools-rank` 는 순위 응답). 이 배열은 그 응답을
 * 못 받았을 때만 쓰는 폴백이다.
 *
 * 값이 어긋나면 `web/scripts/check-ma-day-options.mjs` 가 빌드를 막는다 — 예전에
 * 백엔드에 80·100·140 이 추가됐는데 이 파일이 안 따라가서 한 화면에만 100 이
 * 안 보이는 일이 있었다.
 */

/** 단기·장기 이평선 일수 — 백엔드 목록이 아직 안 왔을 때만 쓰는 폴백. */
export const MA_DAY_OPTIONS: number[] = [5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 240];
