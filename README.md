# Loggy3

Loggy3 records your screen and tracks keyboard/mouse actions with precise timestamps.

## Installation Instructions

### For Windows Users

1. **Download the installer**:
   - Visit https://github.com/standard-intelligence/loggy3/releases/latest/
   - Download the loggy3-windows.exe file

2. **Install the application**:
   - Move the executable out of your downloads folder (e.g., to your desktop or main drive)
   - Run the executable
   - If you see a security popup, click "More info" then "Run anyway"

3. **Starting a Recording**:
   - After installation, search for "Loggy3" in the Start Menu and run it
   - To stop recording, press `Ctrl+C` in the open window

### For Mac Users

1. **Open Terminal**:
   - Click the magnifying glass (🔍) in the top-right corner of your screen
   - Type "Terminal" and press Enter
   - A black or white window will open - this is the Terminal

2. **Install Loggy3**: 
   ```bash
   curl -L loggy3.com/mac | bash
   ```

3. **Grant Permissions**:
   - When prompted for **Screen Recording** permission, click "OK" or "Allow"
   - **Important**: You must completely quit Terminal after granting this permission
     - Press Cmd+Q to quit Terminal
     - Reopen Terminal and run `loggy3` again
   - When prompted for **Input Monitoring** permission, click "OK" or "Allow"
   - **Important**: Quit Terminal completely again after granting this permission, then reopen it

4. **Starting a Recording**:
   - After installation and permissions are set, open Terminal and type:
     ```bash
     loggy3
     ```
   - To stop recording, press `Ctrl+C` in the Terminal

## Where Are My Recordings?

Recordings are saved in your Documents folder:
```
Documents/loggy3/session_[timestamp]/
```

## Key Features

- Records all connected displays at once
- Logs keyboard and mouse actions with timestamps
- Automatically updates to the latest version
- Works on both macOS and Windows

## Advanced Options

```bash
# Show detailed output
loggy3 --verbose
```

## Technical Details

All recordings use a chunk-based format (60-second chunks):
```
session_[timestamp]/
├── chunk_29008452/
│   ├── display_1_Display 1/
│   │   ├── frames.log
│   │   └── video.mp4
│   ├── display_2_Display 2/
│   │   ├── frames.log
│   │   └── video.mp4
│   ├── keypresses.log
│   └── mouse.log
└── display_info.json
```