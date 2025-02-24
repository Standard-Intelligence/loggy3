#!/bin/bash
mkdir -p ~/.local/bin

rm -f ~/.local/bin/loggy3 2>/dev/null

echo "📦 Downloading loggy3..."
curl -#L https://si.ml/loggy3 -o ~/.local/bin/loggy3 && \
chmod +x ~/.local/bin/loggy3 && \
echo "✅ Successfully installed loggy3 to ~/.local/bin/loggy3" && \
loggy3

if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
    echo "🔧 Added ~/.local/bin to your PATH. Please restart your terminal or run 'source ~/.zshrc'"
fi