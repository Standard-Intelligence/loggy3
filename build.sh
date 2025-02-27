#!/bin/bash
set -e

# Create resources directory if it doesn't exist
mkdir -p resources

# Download FFmpeg binaries based on platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Downloading macOS FFmpeg binary..."
    curl -L https://publicr2.standardinternal.com/ffmpeg_binaries/macos/ffmpeg -o resources/ffmpeg
    chmod +x resources/ffmpeg
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Downloading Windows FFmpeg binary..."
    curl -L https://publicr2.standardinternal.com/ffmpeg_binaries/windows_x64/ffmpeg.exe -o resources/ffmpeg.exe
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

# Run the application
echo "Building and running the application..."
cargo build --release
