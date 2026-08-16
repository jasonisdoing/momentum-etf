"use client";

/** 진행도 표시 공용 컴포넌트 — /compare 의 로딩바를 표준화한 것.
 *
 * 겉모습은 컴포넌트 자체 스타일로 고정해 화면별 CSS 의 영향을 받지 않게 한다.
 * 서버가 단일 요청으로 응답하는 긴 작업에는 호출부에서 타이머로 percent 를
 * 단계적으로 올리는 방식(compare 방식)을 쓴다.
 */

export type LoadingProgress = {
  percent: number;
  message: string;
};

/**
 * 진행률을 일정 간격으로 90%까지 올린다. 서버가 단일 요청으로 응답해 실제 단계를
 * 알 수 없으므로, 실제 소요 시간(수 초~수 분)에 맞춘 완만한 램프만 보여준다.
 * 반환값을 호출하면 타이머가 멈춘다 — 응답이 오면 반드시 호출한다.
 */
export function startProgressRamp(
  setProgress: (updater: (prev: LoadingProgress | null) => LoadingProgress | null) => void,
  stepPercent = 6,
  intervalMs = 400,
): () => void {
  const timer = window.setInterval(() => {
    setProgress((prev) => (prev ? { ...prev, percent: Math.min(90, prev.percent + stepPercent) } : prev));
  }, intervalMs);
  return () => window.clearInterval(timer);
}

type AppLoadingProgressProps = {
  title: string;
  progress: LoadingProgress | null;
  fallbackMessage?: string;
};

const boxStyle: React.CSSProperties = {
  display: "grid",
  gap: "0.55rem",
  padding: "1rem",
  border: "1px dashed #d8e0ec",
  borderRadius: "0.5rem",
  color: "#64748b",
  fontWeight: 700,
  textAlign: "left",
};

const textRowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: "1rem",
};

const barStyle: React.CSSProperties = {
  height: 8,
  overflow: "hidden",
  borderRadius: 999,
  background: "#e2e8f0",
};

export function AppLoadingProgress({ title, progress, fallbackMessage }: AppLoadingProgressProps) {
  const percent = progress?.percent ?? 0;
  return (
    <div style={boxStyle}>
      <div style={textRowStyle}>
        <span>{title}</span>
        <strong style={{ color: "var(--text-strong, #0f172a)", fontSize: "var(--fs-base)" }}>{percent}%</strong>
      </div>
      <div style={barStyle} aria-hidden="true">
        <div
          style={{
            height: "100%",
            borderRadius: "inherit",
            background: "#0b7bdc",
            transition: "width 0.25s ease",
            width: `${percent}%`,
          }}
        />
      </div>
      <small style={{ color: "#64748b", fontWeight: 600 }}>{progress?.message ?? fallbackMessage ?? ""}</small>
    </div>
  );
}
