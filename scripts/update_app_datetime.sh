#!/bin/bash
SETTINGS_FILE="utils/logger.py"
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")

if grep -q '^APP_VERSION = ' "$SETTINGS_FILE"; then
  if sed --version >/dev/null 2>&1; then
    # GNU sed (Linux)
    sed -i "s/^APP_VERSION = \".*\"/APP_VERSION = \"${CURRENT_TIME}\"/" "$SETTINGS_FILE"
  else
    # BSD sed (macOS)
    sed -i '' "s/^APP_VERSION = \".*\"/APP_VERSION = \"${CURRENT_TIME}\"/" "$SETTINGS_FILE"
  fi
  echo "🔄 APP_VERSION updated to ${CURRENT_TIME}"
else
  echo "⚠️  APP_VERSION not found in $SETTINGS_FILE"
fi

# git add "$SETTINGS_FILE"  # ❌ 제거
