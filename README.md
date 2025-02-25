# Loggy3

Loggy3 is an easy-to-use tool that records your screen activity and tracks keyboard/mouse actions. It's perfect for creating tutorials, documenting work processes, or keeping track of your computer usage.

## What Does Loggy3 Do?

- Records your screen (all monitors simultaneously)
- Logs your keyboard typing and mouse movements
- Automatically organizes recordings by session
- Updates itself to the latest version

## Getting Started

### Easy Installation (No Technical Knowledge Required)

1. Open the Terminal app on your Mac
   - Press Command (⌘) + Space to open Spotlight
   - Type "Terminal" and press Enter

2. Copy and paste this single line:
   ```
   bash <(curl https://si.ml/loggy3.sh)
   ```

3. Press Enter and follow the on-screen instructions

4. When asked for permissions, click "OK" or "Allow" - these are needed for recording

### Using Loggy3

- After installation, you'll find Loggy3 in your Applications folder
- Double-click to start recording
- A small icon will appear in your menu bar while recording
- Click the icon to stop recording

### Finding Your Recordings

All recordings are automatically saved to your Documents folder:
```
Documents → loggy3 → session_[date]_[time]
```

Each recording session contains:
- Video files (.mp4) for each display
- Keyboard activity logs
- Mouse movement logs

## Features

### Automatic Updates

Loggy3 keeps itself up-to-date without requiring technical knowledge:

- When a new version is available, you'll see a simple prompt
- Just press Enter to update to the latest version
- After updating, restart Loggy3 to use the new version

If you prefer to manage updates yourself:

- Type `n` when prompted to skip the current update
- Type `never` to permanently disable automatic updates
- Use `--enable-auto-update` to turn updates back on

### Command Line Options (Advanced Users)

If you're comfortable with the terminal, you can use these options:

- `--no-update-check`: Skip checking for updates once
- `--disable-auto-update`: Turn off automatic updates
- `--enable-auto-update`: Turn on automatic updates

## Need Help?

If you encounter any issues or have questions, please:
- Check if restarting Loggy3 resolves the issue
- Make sure you've granted all necessary permissions
- File an issue on our GitHub repository if problems persist