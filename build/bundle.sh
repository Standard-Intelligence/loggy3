#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ "$SCRIPT_DIR" == */build ]]; then
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
else
    PROJECT_DIR="$SCRIPT_DIR"
fi

FFMPEG_URL="https://publicr2.si.inc/ffmpeg_binaries/macos_arm/ffmpeg"
# FFMPEG_URL="https://publicr2.si.inc/ffmpeg_binaries/macos_x64/ffmpeg"

export CARGO_TARGET_DIR="${PROJECT_DIR}/target"

CARGO_TOML="${PROJECT_DIR}/Cargo.toml"
if [ ! -f "$CARGO_TOML" ]; then
    echo "Error: Cannot find Cargo.toml at $CARGO_TOML"
    exit 1
fi

APP_NAME=$(grep '^name =' "$CARGO_TOML" | head -1 | cut -d'"' -f2)
IDENTIFIER=$(grep 'identifier =' "$CARGO_TOML" | cut -d'"' -f2)
VERSION=$(grep '^version =' "$CARGO_TOML" | head -1 | cut -d'"' -f2)

(cd "$PROJECT_DIR" && cargo build --release)
# (cd "$PROJECT_DIR" && cargo build --target x86_64-apple-darwin --release)

APP_BUNDLE="${CARGO_TARGET_DIR}/release/${APP_NAME}.app"
mkdir -p "${APP_BUNDLE}/Contents/"{MacOS,Resources,Frameworks}
FRAMEWORKS_DIR="${APP_BUNDLE}/Contents/Frameworks"

# Download and install ffmpeg
echo "Downloading ffmpeg for $ARCH..."
FFMPEG_PATH="${FRAMEWORKS_DIR}/ffmpeg"
curl -L "$FFMPEG_URL" -o ./ffmpeg
cp -f ./ffmpeg "$FFMPEG_PATH"
rm ./ffmpeg
chmod +x "$FFMPEG_PATH"

if [ ! -f "$FFMPEG_PATH" ]; then
    echo "Error: Failed to download ffmpeg"
    exit 1
fi

cp "${CARGO_TARGET_DIR}/release/${APP_NAME}" "${APP_BUNDLE}/Contents/MacOS/"

cp "${PROJECT_DIR}/assets/icon.icns" "${APP_BUNDLE}/Contents/Resources/"

# Make the binary executable
chmod +x "${APP_BUNDLE}/Contents/MacOS/${APP_NAME}"

# Create a wrapper script to launch in Terminal
cat > "${APP_BUNDLE}/Contents/MacOS/${APP_NAME}_wrapper.sh" << EOF
#!/bin/bash
DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )"
BINARY="\${DIR}/${APP_NAME}_binary"

osascript <<END
tell application "Terminal"
    if not (exists window 1) then
        do script ""
    end if
    do script "\"\${BINARY}\"" in window 1
    activate
end tell
END
EOF

chmod +x "${APP_BUNDLE}/Contents/MacOS/${APP_NAME}_wrapper.sh"

# Move the original binary and update the wrapper to be the main executable
mv "${APP_BUNDLE}/Contents/MacOS/${APP_NAME}" "${APP_BUNDLE}/Contents/MacOS/${APP_NAME}_binary"
mv "${APP_BUNDLE}/Contents/MacOS/${APP_NAME}_wrapper.sh" "${APP_BUNDLE}/Contents/MacOS/${APP_NAME}"

cat > "${APP_BUNDLE}/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIconFile</key>
    <string>icon.icns</string>
    <key>CFBundleIdentifier</key>
    <string>${IDENTIFIER}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.utilities</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSScreenCapture</key>
    <true/>
    <key>NSScreenCaptureUsageDescription</key>
    <string>This app needs to capture your screen for recording.</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>This app needs to capture your screen for recording.</string>
    <key>com.apple.security.temporary-exception.apple-events</key>
    <true/>
    <key>com.apple.security.automation.apple-events</key>
    <true/>
    <key>com.apple.security.device.camera</key>
    <true/>
    <key>com.apple.security.device.microphone</key>
    <true/>
    <key>com.apple.security.personal-information.photos-library</key>
    <true/>
    <key>com.apple.security.files.user-selected.read-write</key>
    <true/>
    <key>NSCameraUsageDescription</key>
    <string>This app does not use your camera.</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>This app does not use your microphone.</string>
</dict>
</plist>
EOF

echo "App bundle created at: ${APP_BUNDLE}"