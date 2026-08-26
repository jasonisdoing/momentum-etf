"use client";

import type { PoolAddProgress } from "@/lib/pool-add";

/**
 * 종목풀 추가 진행도 — 한국·미국·호주 시장 화면의 추가 모달 공용.
 *
 * 서버가 종목당 시세·메타 캐시까지 채워 한 건에 수 초씩 걸린다. 100개를 고르면 몇 분이
 * 되는데, 그동안 버튼만 "추가 중..." 이면 멈춘 건지 도는 건지 알 수 없다.
 *
 * 겉모습은 이 컴포넌트가 자체적으로 갖는다(전역 클래스에 의존하지 않는다). 색만 테마
 * 변수를 따른다.
 */
export function PoolAddProgressBar({ progress }: { progress: PoolAddProgress | null }) {
  if (!progress || progress.total <= 0) {
    return null;
  }
  const percent = Math.min(100, Math.round((progress.done / progress.total) * 100));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)" }}>
        <span style={{ fontWeight: 700 }}>
          {progress.done} / {progress.total}
        </span>
        {/* 지금 처리 중인 종목 — 어디서 멈췄는지 바로 보인다. 티커만으로는 알아보기 어렵다. */}
        <span style={{ color: "var(--text-muted)" }}>
          {progress.done < progress.total
            ? progress.name
              ? `${progress.name}(${progress.ticker})`
              : progress.ticker
            : "마무리 중…"}
        </span>
      </div>
      <div
        style={{ height: 8, borderRadius: 999, background: "rgba(148,163,184,0.25)", overflow: "hidden" }}
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          style={{
            width: `${percent}%`,
            height: "100%",
            background: "var(--tblr-success, #2f9e44)",
            transition: "width 120ms linear",
          }}
        />
      </div>
    </div>
  );
}
