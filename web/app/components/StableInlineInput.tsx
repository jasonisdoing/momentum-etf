"use client";

import { useEffect, useState } from "react";
import type { CSSProperties } from "react";

/**
 * AG Grid 셀 안에서 포커스·입력 상태가 튀지 않는 인라인 입력창.
 * 내부에서 localValue 를 들고(uncontrolled 유사) initialValue 변경 시에만 동기화한다.
 * `/assets`(AssetsManager) 에서 쓰던 것을 공용화 — 그리드 셀 티커 입력은 이 컴포넌트를 재사용한다.
 */
export function StableInlineInput({
  initialValue,
  onSave,
  onCancel,
  onChange,
  className,
  style,
  placeholder,
  autoFocus = false,
  disabled = false,
}: {
  initialValue: string;
  onSave?: (val: string) => void;
  onCancel?: () => void;
  onChange?: (val: string) => void;
  className?: string;
  style?: CSSProperties;
  placeholder?: string;
  autoFocus?: boolean;
  disabled?: boolean;
}) {
  const [localValue, setLocalValue] = useState(initialValue);

  useEffect(() => {
    setLocalValue(initialValue);
  }, [initialValue]);

  return (
    <input
      type="text"
      className={className}
      style={style}
      placeholder={placeholder}
      value={localValue}
      autoFocus={autoFocus}
      disabled={disabled}
      onMouseDown={(event) => {
        event.stopPropagation();
      }}
      onClick={(event) => {
        event.stopPropagation();
      }}
      onDoubleClick={(event) => {
        event.stopPropagation();
      }}
      onChange={(event) => {
        event.stopPropagation();
        setLocalValue(event.target.value);
        onChange?.(event.target.value);
      }}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.nativeEvent.isComposing) return;
        if (event.key === "Enter") {
          onSave?.(localValue);
        } else if (event.key === "Escape") {
          setLocalValue(initialValue);
          onCancel?.();
        }
      }}
      onBlur={() => {
        if (localValue !== initialValue) {
          onSave?.(localValue);
        }
      }}
    />
  );
}
