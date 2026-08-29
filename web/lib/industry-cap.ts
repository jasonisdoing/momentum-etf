/** 업종 상한(MAX_PER_INDUSTRY) 표기 — 화면 공용.
 *
 *  선택지의 단일 소스는 백엔드 `config.MAX_PER_INDUSTRY_OPTIONS` 다. 화면에 목록을
 *  베껴 두면 어긋난다 — 실제로 `/pools-rank` 는 4 가 빠져 있었고 `/pools-settings` 는
 *  아직 `["1","2","3",""]` 로 굳어 있었다.
 *
 *  백엔드의 '제한 없음'은 `None` 인데 JSON·쿼리스트링에 실으려면 숫자여야 해서
 *  API 경계에서 `-1` 로 바꿔 보낸다(`utils.pool_settings_store.max_per_industry_options`).
 */

/** API 가 '제한 없음'을 실어 보내는 값. */
export const INDUSTRY_CAP_NONE = -1;

export function isIndustryCapNone(value: number | null | undefined): boolean {
  return value == null || value < 0;
}

/** 셀렉트에 보일 문구 — `없음` · `3종목`. */
export function industryCapLabel(value: number | null | undefined): string {
  return isIndustryCapNone(value) ? "없음" : `${value}종목`;
}

/** 저장값이 선택지 밖이면 그 값도 목록에 남긴다 — 빈 셀렉트가 되어 값이 조용히 바뀌는 걸 막는다. */
export function withCurrentIndustryCap(options: number[], current: number): number[] {
  return options.includes(current) ? options : [...options, current];
}
