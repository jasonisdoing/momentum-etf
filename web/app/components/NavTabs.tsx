"use client";

/**
 * 화면 내 탭 — Tabler(@tabler/core)의 nav 컴포넌트를 감싼 공용 래퍼.
 *
 * 직접 만든 스타일을 쓰지 않는다. tabler.min.css 가 layout.tsx 에서 전역 로드돼 있어
 * 클래스만 붙이면 되고, 화면마다 클래스 문자열을 복붙하지 않도록 여기서만 조합한다.
 *
 * 층이 다른 탭은 variant 로 구분한다 — 같은 화면에서 최상위와 안쪽 탭이 똑같이 생기면
 * 지금 어디를 보고 있는지가 드러나지 않는다.
 *   - `card`      : 카드 헤더에 붙는 최상위 탭 (card-header-tabs)
 *   - `underline` : 패널 안쪽 탭
 *
 * 설정값을 고르는 선택지(예: 일별 보기 방식)는 탭이 아니라 토글(appSegmentedToggle)로 둔다 —
 * 화면 전환이 아니라 값 입력이기 때문이다.
 */

export type NavTabItem<K extends string> = { key: K; label: string };

const VARIANT_CLASS = {
  card: "nav nav-tabs card-header-tabs",
  underline: "nav nav-underline",
} as const;

export function NavTabs<K extends string>({
  items,
  value,
  onChange,
  variant = "underline",
  label,
  className,
  style,
}: {
  items: readonly NavTabItem<K>[];
  value: K;
  onChange: (key: K) => void;
  variant?: keyof typeof VARIANT_CLASS;
  /** 스크린리더용 이름 (예: "화면 전환"). */
  label: string;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <ul
      className={[VARIANT_CLASS[variant], className].filter(Boolean).join(" ")}
      role="tablist"
      aria-label={label}
      style={style}
    >
      {items.map((item) => (
        <li className="nav-item" key={item.key}>
          <button
            type="button"
            role="tab"
            aria-selected={value === item.key}
            className={value === item.key ? "nav-link active" : "nav-link"}
            onClick={() => onChange(item.key)}
          >
            {item.label}
          </button>
        </li>
      ))}
    </ul>
  );
}
