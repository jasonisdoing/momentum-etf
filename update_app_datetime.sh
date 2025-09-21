#!/bin/bash
# settings.py 파일의 APP_DATE_TIME 값을 현재 시각으로 갱신

SETTINGS_FILE="settings.py"
CURRENT_TIME=$(date +"%Y-%m-%d-%H")

# APP_DATE_TIME 라인 치환
sed -i.bak -E "s/^APP_DATE_TIME = \".*\"/APP_DATE_TIME = \"${CURRENT_TIME}\"/" $SETTINGS_FILE

# 백업 파일(.bak) 제거
rm -f "${SETTINGS_FILE}.bak"

echo "🔄 APP_DATE_TIME updated to ${CURRENT_TIME}"