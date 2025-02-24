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
│   ├── display_1_Display 1
│   │   ├── chunk_00000.mp4
│   │   ├── chunk_00001.mp4
│   │   └── frames.log
│   ├── display_2_Display 2
│   │   ├── chunk_00000.mp4
│   │   └── frames.log
│   ├── display_info.json
│   ├── keypresses.log
│   └── mouse.log
└── session_20250223_205034
    ├── display_1_Display 1
    │   ├── chunk_00000.mp4
    │   └── frames.log
    ├── display_2_Display 2
    │   ├── chunk_00000.mp4
    │   └── frames.log
    ├── display_info.json
    ├── keypresses.log
    └── mouse.log
```
