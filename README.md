# Loggy3

Loggy3 records your screen and tracks keyboard/mouse actions with precise timestamps.


## For Mac Users

1. **Open Terminal**:
   - Click the magnifying glass (🔍) in the top-right corner of your screen
   - Type "Terminal" and press Enter
   - A black or white window will open - this is the Terminal

2. **Install Loggy3**: Copy and paste this single line into Terminal:
   ```bash
   curl -L loggy3.com/mac | bash
   ```

3. **Press Enter** to start the installation

4. **Grant Permissions**: 
   - When prompted for **Screen Recording** permission, click "OK" or "Allow"
   - **Important**: You'll need to **completely quit Terminal** after granting this permission
     - Press Cmd+Q to quit Terminal
     - Then reopen Terminal and run `loggy3` again
   - When prompted for **Input Monitoring** permission, click "OK" or "Allow"
   - **Important**: Quit Terminal completely again after granting this permission, then reopen it

### Starting a Recording

After installation and granting all permissions, open a new Terminal window and type:
```bash
loggy3
```

To stop recording, press `Ctrl+C` in the Terminal.

## For Windows Users

1. **Download the loggy3-windows.exe file** from https://github.com/standard-intelligence/loggy3/releases/latest/
2. **Move the executable out of your downloads**, e.g. to your desktop or main drive.
3. **Run the executable.** You'll see a popup, where you should click "More info" then "Run anyway".

### Starting a Recording

After you've run the executable once, you can run it again by searching for "Loggy3" in the Start Menu.

To stop recording, press `Ctrl+C` while the window is selected.


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
