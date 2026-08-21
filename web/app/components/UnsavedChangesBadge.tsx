"use client";

/**
 * 「저장하지 않은 변경」 배지 — 설정 화면 공용.
 *
 * 설정 화면의 표준 방식은 **초안 편집 + 명시적 저장**이다. 입력을 바꿔도 바로 저장되지
 * 않으므로, 저장 전이라는 사실을 이 배지로 알리고 저장 버튼은 변경이 있을 때만 켠다.
 * 세 전략 화면이 같은 문구·색을 쓰도록 여기서만 정의한다.
 */
export function UnsavedChangesBadge({ show, message = "저장하지 않은 변경" }: { show: boolean; message?: string }) {
  if (!show) {
    return null;
  }
  return (
    <span
      style={{
        color: "var(--up-color, #d64545)",
        fontSize: "var(--fs-sm)",
        fontWeight: 700,
      }}
    >
      {message}
    </span>
  );
}
