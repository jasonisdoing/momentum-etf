#!/usr/bin/env node
/**
 * 글자 크기가 토큰(--fs-*)만 쓰는지 검사한다. 위반이 있으면 종료코드 1 로 빌드를 세운다.
 *
 * 왜 필요한가: 예전에는 크기를 그때그때 눈대중으로 붙여 45종(10~30.4px)이 흩어져 있었다.
 * 0.9rem 과 0.92rem 처럼 구분도 안 되는 값이 섞여 화면마다 크기가 제각각이었다.
 * 한 번 정리해도 검사가 없으면 곧 같은 상태로 돌아간다.
 *
 * 크기를 바꾸고 싶으면 app/globals.css 의 :root 에 있는 --fs-* 여섯 줄을 고친다.
 * 새 단계가 정말 필요하면 토큰을 추가하고 이 파일의 설명도 함께 갱신한다.
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SCAN_DIRS = ["app", "lib"];
const SKIP_DIRS = new Set(["node_modules", ".next", "dist", "build"]);

/**
 * 숫자만 받는 자리(차트·그리드 테마)를 면제한다.
 *
 * 면제는 `fontSize: 14` 같은 **따옴표 없는 숫자**에만 적용된다. 파일을 통째로 빼면
 * 그 파일의 CSS 까지 검사에서 빠져 구멍이 되므로, 아래 목록에 있는 파일이라도
 * `font-size:`(CSS)와 `fontSize: "..."`(문자열)은 그대로 검사한다.
 *
 * 이 자리들이 숫자여야 하는 이유: 차트는 SVG·캔버스에 직접 그려서 CSS 를 안 타고,
 * recharts 는 이 값으로 축 여백까지 계산한다. AG Grid 테마는 JS 객체라 토큰을 못 읽는다.
 */
const NUMERIC_ALLOWED = [
  { file: "app/components/app-grid-theme.ts", reason: "AG Grid JS 테마" },
  { file: "app/dashboard/DashboardManager.tsx", reason: "recharts 축 눈금" },
  { file: "app/asset-status/AssetChartsManager.tsx", reason: "recharts 축 눈금" },
  { file: "app/components/AssetHelperBacktestResult.tsx", reason: "recharts 축 눈금" },
  { file: "app/market-trend/MarketTrendChart.tsx", reason: "차트 라이브러리 설정" },
  { file: "app/ticker/TickerDetailManager.tsx", reason: "lightweight-charts layout" },
  { file: "app/compare/ComparePageClient.tsx", reason: "lightweight-charts layout" },
];

/** CSS `font-size: 13px` — 어느 파일에서도 허용하지 않는다(styled-jsx 포함). */
const CSS_RAW = /font-size:\s*(-?[0-9.]+(?:px|rem|em|pt|%))/g;
/** JSX `fontSize: "0.85rem"` — 문자열이면 토큰을 쓸 수 있으므로 허용하지 않는다. */
const JSX_STRING_RAW = /fontSize:\s*(["']-?[0-9.]+(?:px|rem|em|pt)?["'])/g;
/** JSX `fontSize: 14` — 숫자. NUMERIC_ALLOWED 파일에서만 허용한다. */
const JSX_NUMERIC_RAW = /fontSize:\s*(-?[0-9.]+)\s*[,}\n]/g;

function walk(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    if (SKIP_DIRS.has(entry)) continue;
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, out);
    else if (/\.(tsx?|css)$/.test(entry)) out.push(full);
  }
  return out;
}

const violations = [];
for (const dir of SCAN_DIRS) {
  for (const full of walk(join(WEB_ROOT, dir))) {
    const rel = relative(WEB_ROOT, full);
    const numericOk = NUMERIC_ALLOWED.some((a) => a.file === rel);
    const patterns = numericOk ? [CSS_RAW, JSX_STRING_RAW] : [CSS_RAW, JSX_STRING_RAW, JSX_NUMERIC_RAW];
    const lines = readFileSync(full, "utf8").split("\n");
    lines.forEach((line, i) => {
      for (const pattern of patterns) {
        pattern.lastIndex = 0;
        let m;
        while ((m = pattern.exec(line)) !== null) {
          violations.push({ file: rel, line: i + 1, value: m[1], text: line.trim().slice(0, 90) });
        }
      }
    });
  }
}

if (violations.length === 0) {
  console.log("[check-font-tokens] 통과 — 글자 크기가 모두 --fs-* 토큰입니다.");
  process.exit(0);
}

console.error(`[check-font-tokens] 토큰이 아닌 글자 크기 ${violations.length}건:\n`);
for (const v of violations) {
  console.error(`  ${v.file}:${v.line}  ${v.value}`);
  console.error(`    ${v.text}`);
}
console.error(`
고치는 법 — app/globals.css 의 --fs-* 중 가까운 단계를 쓰세요.
  ~14px → var(--fs-sm)      15~16px → var(--fs-base)
  ~18px → var(--fs-lg)      ~20px   → var(--fs-xl)      그 이상 → var(--fs-2xl)
차트처럼 숫자를 넣을 수밖에 없는 자리는 이 파일의 ALLOWED 에 이유와 함께 추가하세요.`);
process.exit(1);
