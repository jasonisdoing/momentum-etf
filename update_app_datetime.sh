#!/bin/bash
SETTINGS_FILE="utils/settings.py"
CURRENT_TIME=$(date +"%Y-%m-%d-%H")

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

# git add "$SETTINGS_FILE"  # ❌ 제거
