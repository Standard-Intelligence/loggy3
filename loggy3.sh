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
    echo "⚠️ Running in a Windows environment ($OS)"
    BINARY_SUFFIX="windows.exe"
    BINARY_NAME="loggy3.exe"
    
    # Windows doesn't use .bashrc or .zshrc for PATH
    echo "ℹ️ On Windows, you'll need to add $INSTALL_DIR to your PATH manually."
    echo "You can do this by editing your Environment Variables in System Properties."
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

# Only configure PATH on non-Windows systems
if [[ ! "$OS" =~ MINGW|MSYS|CYGWIN ]]; then
    # Check if already in PATH
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        ADDED_TO_PROFILE=0
        
        # Platform-specific shell configuration
        if [[ "$OS" == "Darwin" ]]; then
            # macOS shell detection - Check the actual current shell, not just what we're running now
            CURRENT_SHELL=$(dscl . -read /Users/$USER UserShell | sed 's/UserShell: //')
            
            # Check for zsh (default on macOS Catalina+)
            if [[ "$CURRENT_SHELL" == *"zsh"* ]]; then
                # Check common zsh config files
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
                            echo "🔍 $INSTALL_DIR already in PATH in $ZSHRC"
                            ADDED_TO_PROFILE=1
                            SHELL_RC="$ZSHRC"
                            break
                        fi
                    fi
                done
                
                # Create .zshrc if none exists
                if [ $ADDED_TO_PROFILE -eq 0 ]; then
                    echo "# Added by loggy3 installer" > "$HOME/.zshrc"
                    echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$HOME/.zshrc"
                    echo "🔧 Created $HOME/.zshrc and added $INSTALL_DIR to your PATH"
                    ADDED_TO_PROFILE=1
                    SHELL_RC="$HOME/.zshrc"
                fi
            # Check for bash
            elif [[ "$CURRENT_SHELL" == *"bash"* ]]; then
                # On macOS, prefer .bash_profile over .bashrc
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
                            echo "🔍 $INSTALL_DIR already in PATH in $BASHRC"
                            ADDED_TO_PROFILE=1
                            SHELL_RC="$BASHRC"
                            break
                        fi
                    fi
                done
                
                # Create .bash_profile if none exists (macOS standard)
                if [ $ADDED_TO_PROFILE -eq 0 ]; then
                    echo "# Added by loggy3 installer" > "$HOME/.bash_profile"
                    echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$HOME/.bash_profile"
                    echo "🔧 Created $HOME/.bash_profile and added $INSTALL_DIR to your PATH"
                    ADDED_TO_PROFILE=1
                    SHELL_RC="$HOME/.bash_profile"
                fi
            else
                echo "⚠️ Unknown shell: $CURRENT_SHELL"
                echo "⚠️ $INSTALL_DIR is not in your PATH. Please add it manually to your shell configuration."
            fi
        fi
        
        if [ $ADDED_TO_PROFILE -eq 1 ]; then
            echo "👉 To use loggy3 from any terminal, please restart your terminal or run: source $SHELL_RC"
        fi
    fi
fi

# Add to PATH temporarily for this session so we can run the binary immediately
export PATH="$INSTALL_DIR:$PATH"

echo " "
echo "🚀 Running $BINARY_NAME, in the future you can run 'loggy3' directly"
"$INSTALL_DIR/$BINARY_NAME"