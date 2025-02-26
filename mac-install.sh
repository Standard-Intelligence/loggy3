#!/bin/bash
set -e

INSTALL_DIR="$HOME/.local/bin"
REPO="Standard-Intelligence/loggy3"
BINARY_NAME="loggy3"

mkdir -p "$INSTALL_DIR"

rm -f "$INSTALL_DIR/$BINARY_NAME" 2>/dev/null

echo "📦 Downloading latest $BINARY_NAME..."

BINARY_SUFFIX="macos-arm64"

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

export PATH="$INSTALL_DIR:$PATH"

echo "📝 Checking shell configuration files..."

ZSHRC="$HOME/.zshrc"
if [ -f "$ZSHRC" ]; then
    if ! grep -q "export PATH=\"$INSTALL_DIR:\$PATH\"" "$ZSHRC"; then
        echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$ZSHRC"
        echo "✅ Added PATH export to $ZSHRC"
    else
        echo "✅ PATH already configured in $ZSHRC"
    fi
else
    echo "export PATH=\"$INSTALL_DIR:\$PATH\"" > "$ZSHRC"
    echo "✅ Created $ZSHRC with PATH export"
fi

BASHRC="$HOME/.bashrc"
if [ -f "$BASHRC" ]; then
    if ! grep -q "export PATH=\"$INSTALL_DIR:\$PATH\"" "$BASHRC"; then
        echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$BASHRC"
        echo "✅ Added PATH export to $BASHRC"
    else
        echo "✅ PATH already configured in $BASHRC"
    fi
else
    echo "export PATH=\"$INSTALL_DIR:\$PATH\"" > "$BASHRC"
    echo "✅ Created $BASHRC with PATH export"
fi

echo " "
echo "🚀 Running $BINARY_NAME, in the future you can run 'loggy3' directly"
"$INSTALL_DIR/$BINARY_NAME"