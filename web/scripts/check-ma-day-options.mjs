/**
 * 이평선 일수 선택지 동기화 검사 — 빌드 전에 돌린다.
 *
 * `web/lib/ma-day-options.ts` 의 `MA_DAY_OPTIONS` 는 백엔드
 * `utils/pool_settings_store.py` 의 같은 이름 상수를 복사한 **폴백**이다. 화면들은
 * 백엔드가 내려주는 목록을 우선 쓰지만, 응답을 못 받았을 때 이 값이 쓰이므로
 * 둘이 어긋나면 그때만 옛 목록이 나온다. 실제로 백엔드에 80·100·140 이 추가됐을 때
 * 이 파일이 안 따라가서 `/pools-rank` 에만 100 이 안 보이는 일이 있었다.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, "..", "..");

function parseList(text, label) {
  const match = text.match(/MA_DAY_OPTIONS[^=]*=\s*[[(]([^\])]*)[\])]/);
  if (!match) {
    throw new Error(`[check-ma-day-options] ${label} 에서 MA_DAY_OPTIONS 를 찾지 못했습니다.`);
  }
  return match[1]
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map(Number);
}

const backend = parseList(
  readFileSync(join(repoRoot, "utils", "pool_settings_store.py"), "utf8"),
  "utils/pool_settings_store.py",
);
const frontend = parseList(
  readFileSync(join(repoRoot, "web", "lib", "ma-day-options.ts"), "utf8"),
  "web/lib/ma-day-options.ts",
);

if (backend.join(",") !== frontend.join(",")) {
  console.error("[check-ma-day-options] 이평선 일수 선택지가 백엔드와 다릅니다.");
  console.error(`  백엔드 utils/pool_settings_store.py : ${backend.join(", ")}`);
  console.error(`  프론트 web/lib/ma-day-options.ts    : ${frontend.join(", ")}`);
  console.error("  web/lib/ma-day-options.ts 를 백엔드 값으로 맞추세요.");
  process.exit(1);
}

console.log(`[check-ma-day-options] 통과 — 선택지 ${backend.length}개가 백엔드와 같습니다.`);
