#!/bin/bash
# momentum-etf crontab 설치/재설치/제거 스크립트
#
# 같은 VM 에서 여러 앱의 cron 을 함께 운영하기 위해 마커 블록
# (# >>> momentum-etf >>> ~ # <<< momentum-etf <<<) 사이만 교체한다.
# 다른 앱(예: leverage-switching)의 cron 항목은 그대로 보존된다.
#
# 사용법:
#   bash infra/cron/install.sh             # 설치 / 재설치 (idempotent)
#   bash infra/cron/install.sh --uninstall # momentum-etf 블록만 제거

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_CRON="$SCRIPT_DIR/crontab"
MARKER_BEGIN="# >>> momentum-etf >>>"
MARKER_END="# <<< momentum-etf <<<"

current="$(crontab -l 2>/dev/null || true)"
filtered="$(printf '%s\n' "$current" | awk -v b="$MARKER_BEGIN" -v e="$MARKER_END" '
    $0 == b { skip=1; next }
    $0 == e { skip=0; next }
    skip != 1 { print }
')"

if [ "${1:-}" = "--uninstall" ]; then
    printf '%s\n' "$filtered" | crontab -
    echo "[install.sh] momentum-etf 블록 제거 완료"
    exit 0
fi

if [ ! -f "$APP_CRON" ]; then
    echo "[install.sh] crontab 파일을 찾을 수 없음: $APP_CRON" >&2
    exit 1
fi

{
    printf '%s\n' "$filtered" | sed '/./,$!d'
    echo
    echo "$MARKER_BEGIN"
    cat "$APP_CRON"
    echo "$MARKER_END"
} | crontab -

echo "[install.sh] momentum-etf crontab 설치 완료"
echo "--- 현재 crontab ---"
crontab -l
