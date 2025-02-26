#!/bin/bash
set -e

INSTALL_DIR="$HOME/.local/bin"
REPO="Standard-Intelligence/loggy3"
BINARY_NAME="loggy3"

mkdir -p "$INSTALL_DIR"

rm -f "$INSTALL_DIR/$BINARY_NAME" 2>/dev/null

echo "📦 Downloading latest $BINARY_NAME..."

# Detect operating system and architecture
OS=$(uname -s)
ARCH=$(uname -m)

# Determine which binary to download based on OS and architecture
if [[ "$OS" == "Darwin" ]]; then
    if [[ "$ARCH" == "arm64" ]]; then
        BINARY_SUFFIX="macos-arm64"
    else
        echo "❌ Unsupported architecture: $ARCH"
        echo "This application currently only supports Apple Silicon (arm64) Macs."
        exit 1
    fi
elif [[ "$OS" =~ MINGW|MSYS|CYGWIN ]]; then
    BINARY_SUFFIX="windows.exe"
    BINARY_NAME="loggy3.exe"
else
    echo "❌ Unsupported operating system: $OS"
    echo "This application currently only supports macOS on Apple Silicon and Windows."
    exit 1
fi

echo "🔍 Detected platform: $OS on $ARCH, downloading $BINARY_SUFFIX binary..."

LATEST_RELEASE_URL=$(curl -s "https://api.github.com/repos/$REPO/releases/latest" | 
                    grep -o "https://github.com/$REPO/releases/download/[^\"]*/$BINARY_NAME-$BINARY_SUFFIX")

if [ -z "$LATEST_RELEASE_URL" ]; then
    echo "❌ Failed to find the download URL for the latest release."
    echo "Please check if the repository ($REPO) and binary ($BINARY_NAME-$BINARY_SUFFIX) are correct."
    exit 1
fi

curl -#L "$LATEST_RELEASE_URL" -o "$INSTALL_DIR/$BINARY_NAME" && 
chmod +x "$INSTALL_DIR/$BINARY_NAME" && 
echo "✅ Successfully installed $BINARY_NAME to $INSTALL_DIR/$BINARY_NAME"

if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
        echo "🔧 Added $INSTALL_DIR to your PATH in $SHELL_RC"
        echo "Please restart your terminal or run 'source $SHELL_RC'"
    elif [ -n "$BASH_VERSION" ] || [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bashrc"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
        echo "🔧 Added $INSTALL_DIR to your PATH in $SHELL_RC"
        echo "Please restart your terminal or run 'source $SHELL_RC'"
    else
        echo "⚠️ $INSTALL_DIR is not in your PATH. Please add it manually to your shell configuration."
    fi
fi

echo " "
echo "🚀 Running $BINARY_NAME, in the future you can run 'loggy3' directly"
echo "ℹ️  Auto-updates are enabled by default. You can disable them using 'loggy3 --disable-auto-update'"
"$INSTALL_DIR/$BINARY_NAME"