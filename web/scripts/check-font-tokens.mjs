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
  { file: "app/strategy-new-high/HoldingChart.tsx", reason: "lightweight-charts layout" },
];

/**
 * Tabler 가 자체 크기를 갖고 있는 클래스들. className 으로 크기가 정해지므로
 * `font-size` 문자열 검사에는 안 걸린다 — 여기서 따로 막는다.
 *
 * 이 중 우리가 쓰는 것은 globals.css 에서 토큰으로 덮어써 두었고(= PINNED),
 * 덮어쓰지 않은 클래스를 새로 쓰면 Tabler 기본값(예: .small 은 부모의 0.875em)이
 * 그대로 나와 표준 6단계 밖 크기가 생긴다. 그래서 빌드를 세운다.
 *
 * 새 클래스를 쓰고 싶으면 globals.css 에 토큰으로 크기를 지정하고 PINNED 에 추가한다.
 */
const TABLER_SIZE_CLASSES = new Set([
  "avatar", "avatar-2xl", "avatar-lg", "avatar-md", "avatar-sm", "avatar-upload-text",
  "avatar-xl", "avatar-xs", "avatar-xxs", "badge", "badge-lg", "badge-sm", "blockquote",
  "blockquote-footer", "btn", "btn-lg", "btn-sm", "btn-xl", "calendar", "card-title",
  "chart-sparkline-label", "col-form-label-lg", "col-form-label-sm", "datagrid-title",
  "display-1", "display-2", "display-3", "display-4", "display-5", "display-6",
  "dropdown-header", "dropdown-menu", "empty-header", "empty-title", "figure-caption",
  "form-check-description", "form-control", "form-control-lg", "form-control-sm",
  "form-help", "form-label", "form-select", "form-select-lg", "form-select-sm",
  "h1", "h2", "h3", "h4", "h5", "h6", "lead", "modal-title", "nav-link", "navbar",
  "page-title", "small", "subheader", "table", "text-h1", "text-h2", "text-h3",
]);

/** globals.css 에서 토큰으로 크기를 못박아 둔 클래스 — 이것만 써도 된다. */
const PINNED = new Set([
  "btn", "btn-sm", "form-control", "form-control-sm", "form-select", "form-select-sm",
  "small", "subheader", "modal-title", "h1", "navbar", "table",
]);

/** JSX `className="..."` 안의 클래스 이름들. */
const CLASS_ATTR = /className=\{?["'`]([^"'`]+)["'`]/g;

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
      if (!full.endsWith(".css")) {
        CLASS_ATTR.lastIndex = 0;
        let cm;
        while ((cm = CLASS_ATTR.exec(line)) !== null) {
          for (const name of cm[1].split(/\s+/)) {
            if (TABLER_SIZE_CLASSES.has(name) && !PINNED.has(name)) {
              violations.push({
                file: rel,
                line: i + 1,
                value: `.${name} (Tabler 기본 크기)`,
                text: line.trim().slice(0, 90),
              });
            }
          }
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
Tabler 클래스(.small, .badge 등)가 걸렸다면 globals.css 에서 그 클래스의 크기를
토큰으로 지정한 뒤 이 파일의 PINNED 에 이름을 추가하세요.
차트처럼 숫자를 넣을 수밖에 없는 자리는 NUMERIC_ALLOWED 에 이유와 함께 추가하세요.`);
process.exit(1);
