# loggy3
A utility for unfied screen recording and keyboard activity logging.

## Installation
```
bash <(curl https://si.ml/loggy3.sh)
```

logs are all stored in ~/Documents/loggy3

example structure:
```
/Users/username/Documents/loggy3
├── session_20250223_204746
│   ├── display_1_Display 1
│   │   ├── chunk_00000.mp4
│   │   ├── chunk_00001.mp4
│   │   └── frames.log
│   ├── display_2_Display 2
│   │   ├── chunk_00000.mp4
│   │   └── frames.log
│   ├── display_info.json
│   ├── keypresses.log
│   └── mouse.log
└── session_20250223_205034
    ├── display_1_Display 1
    │   ├── chunk_00000.mp4
    │   └── frames.log
    ├── display_2_Display 2
    │   ├── chunk_00000.mp4
    │   └── frames.log
    ├── display_info.json
    ├── keypresses.log
    └── mouse.log
```

## Features

### Auto-Update
Loggy3 automatically checks for updates on each launch. When a new version is available, you'll be prompted to update. The app will handle the download and installation process automatically.

You can control the auto-update behavior with these flags:
- `--no-update-check`: Skip the update check for this launch only
- `--disable-auto-update`: Disable auto-updates permanently
- `--enable-auto-update`: Re-enable auto-updates if previously disabled

When prompted during an update, you can:
- Press Enter or "y" to install the update immediately
- Type "n" to skip this update but continue checking for future updates
- Type "never" to permanently disable auto-updates