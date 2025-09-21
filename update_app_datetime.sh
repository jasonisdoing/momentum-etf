#!/bin/bash
# settings.py 파일의 APP_DATE_TIME 값을 현재 시각으로 갱신

SETTINGS_FILE="settings.py"
CURRENT_TIME=$(date +"%Y-%m-%d-%H")

# 파일에 APP_DATE_TIME 가 있을 때만 치환
if grep -q '^APP_DATE_TIME = ' "$SETTINGS_FILE"; then
  # macOS와 Linux 모두 대응
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

# 자동으로 git add 실행 (변경사항 커밋에 포함시키기)
git add "$SETTINGS_FILE"