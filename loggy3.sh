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

# Improved PATH handling
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    ADDED_TO_PROFILE=0
    
    # macOS zsh handling (the default shell since macOS Catalina)
    if [[ "$OS" == "Darwin" ]] && [ -n "$ZSH_VERSION" ] || [ "$SHELL" == *"zsh"* ]; then
        # Check common zsh config files for macOS
        for ZSHRC in "$HOME/.zshrc" "$HOME/.zprofile" "$HOME/.zshenv"; do
            if [ -f "$ZSHRC" ]; then
                if ! grep -q "export PATH=\"$INSTALL_DIR:\$PATH\"" "$ZSHRC"; then
                    echo "" >> "$ZSHRC"
                    echo "# Added by loggy3 installer" >> "$ZSHRC"
                    echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$ZSHRC"
                    echo "🔧 Added $INSTALL_DIR to your PATH in $ZSHRC"
                    ADDED_TO_PROFILE=1
                    SHELL_RC="$ZSHRC"
                    break
                else
                    echo "🔍 $INSTALL_DIR already in PATH configuration in $ZSHRC"
                    ADDED_TO_PROFILE=1
                    SHELL_RC="$ZSHRC"
                    break
                fi
            fi
        done
        
        # If no config file exists, create .zshrc
        if [ $ADDED_TO_PROFILE -eq 0 ]; then
            echo "# Added by loggy3 installer" > "$HOME/.zshrc"
            echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$HOME/.zshrc"
            echo "🔧 Created $HOME/.zshrc and added $INSTALL_DIR to your PATH"
            ADDED_TO_PROFILE=1
            SHELL_RC="$HOME/.zshrc"
        fi
    # bash handling
    elif [ -n "$BASH_VERSION" ] || [ "$SHELL" == *"bash"* ]; then
        # Check common bash config files for macOS (prefers .bash_profile on macOS)
        for BASHRC in "$HOME/.bash_profile" "$HOME/.bashrc" "$HOME/.profile"; do
            if [ -f "$BASHRC" ]; then
                if ! grep -q "export PATH=\"$INSTALL_DIR:\$PATH\"" "$BASHRC"; then
                    echo "" >> "$BASHRC"
                    echo "# Added by loggy3 installer" >> "$BASHRC"
                    echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$BASHRC"
                    echo "🔧 Added $INSTALL_DIR to your PATH in $BASHRC"
                    ADDED_TO_PROFILE=1
                    SHELL_RC="$BASHRC"
                    break
                else
                    echo "🔍 $INSTALL_DIR already in PATH configuration in $BASHRC"
                    ADDED_TO_PROFILE=1
                    SHELL_RC="$BASHRC"
                    break
                fi
            fi
        done
        
        # If no config file exists, create .bash_profile for macOS
        if [ $ADDED_TO_PROFILE -eq 0 ]; then
            if [[ "$OS" == "Darwin" ]]; then
                echo "# Added by loggy3 installer" > "$HOME/.bash_profile"
                echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$HOME/.bash_profile"
                echo "🔧 Created $HOME/.bash_profile and added $INSTALL_DIR to your PATH"
                ADDED_TO_PROFILE=1
                SHELL_RC="$HOME/.bash_profile"
            else
                echo "# Added by loggy3 installer" > "$HOME/.bashrc"
                echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$HOME/.bashrc"
                echo "🔧 Created $HOME/.bashrc and added $INSTALL_DIR to your PATH"
                ADDED_TO_PROFILE=1
                SHELL_RC="$HOME/.bashrc"
            fi
        fi
    else
        echo "⚠️ $INSTALL_DIR is not in your PATH. Please add it manually to your shell configuration."
    fi
    
    if [ $ADDED_TO_PROFILE -eq 1 ]; then
        echo "👉 To use loggy3 from any terminal, please restart your terminal or run: source $SHELL_RC"
    fi
fi

export PATH="$INSTALL_DIR:$PATH"

echo " "
echo "🚀 Running $BINARY_NAME, in the future you can run 'loggy3' directly"
"$INSTALL_DIR/$BINARY_NAME"