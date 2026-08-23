"use client";

/**
 * 기간(개월) 셀렉트 — 시스템 공용. 백테스트·튜닝 화면이 전부 이 컴포넌트를 쓴다.
 *
 * 예전에는 화면마다 `<select className="form-select form-select-sm">` 를 직접 써서, 담긴
 * 컨테이너의 전역 CSS(`appMainHeader` 는 30px, 카드 헤더는 부트스트랩 기본값)에 따라 크기와
 * 글자 굵기가 달라졌다. 같은 화면 안에서도 백테스트와 튜닝의 셀렉트가 달라 보였고,
 * 문구도 `12개월` 과 `최근 12개월` 두 가지로 갈렸다.
 *
 * 그래서 **겉모습과 표기를 이 컴포넌트 안에 못박는다** — 어느 컨테이너에 넣어도 같게 보인다.
 * 스타일은 이 파일에만 있고(전역 클래스에 기대지 않는다), 색은 지정하지 않아 테마를 따른다.
 * 화면은 값과 핸들러만 준다.
 */

export function MonthsSelect({
  value,
  options,
  onChange,
  disabled,
  title,
}: {
  value: number | null | undefined;
  options: number[] | undefined;
  onChange: (months: number) => void;
  disabled?: boolean;
  title?: string;
}) {
  const list = options ?? [];
  // 선택지가 바뀌어 저장값이 목록 밖이면 숨기지 않고 그대로 보여줘 사용자가 바꾸게 한다
  // (이평선 셀렉트와 같은 규칙).
  const outside = value != null && !list.includes(value);
  return (
    <>
      <select
        className="form-select appMonthsSelect"
        value={value == null ? "" : String(value)}
        disabled={disabled || list.length === 0}
        title={title}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        {list.map((months) => (
          <option key={months} value={months}>
            최근 {months}개월
          </option>
        ))}
        {outside ? <option value={String(value)}>최근 {value}개월 (선택지 밖)</option> : null}
      </select>
      <style jsx>{`
        .appMonthsSelect {
          width: auto;
          min-height: 30px;
          height: 30px;
          padding-top: 0;
          padding-bottom: 0;
          font-size: var(--fs-base);
          font-weight: 700;
        }
      `}</style>
    </>
  );
}
