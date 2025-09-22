#!/bin/bash
# settings.py 파일의 APP_DATE_TIME 값을 현재 시각으로 갱신하고 자동으로 git add

SETTINGS_FILE="settings.py"
CURRENT_TIME=$(date +"%Y-%m-%d-%H")

# APP_DATE_TIME 치환 (macOS / Linux 둘 다 대응)
if grep -q '^APP_DATE_TIME = ' "$SETTINGS_FILE"; then
  if sed --version >/dev/null 2>&1; then
    # GNU sed (Linux)
    sed -i "s/^APP_DATE_TIME = \".*\"/APP_DATE_TIME = \"${CURRENT_TIME}\"/" "$SETTINGS_FILE"
  else
    # BSD sed (macOS)
    sed -i '' "s/^APP_DATE_TIME = \".*\"/APP_DATE_TIME = \"${CURRENT_TIME}\"/" "$SETTINGS_FILE"
  fi
  echo "🔄 APP_DATE_TIME updated to ${CURRENT_TIME}"
else
  echo "⚠️  APP_DATE_TIME not found in $SETTINGS_FILE"
fi

# 변경 사항을 staging 영역에 추가
git add "$SETTINGS_FILE"