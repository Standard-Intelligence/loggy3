# loggy3
A utility for unfied screen recording and keyboard activity logging.


logs are all stored in ~/loggy3

example structure:
```
/Users/username/loggy3
├── session_20250223_112859
│   ├── display_1_Display 1
│   │   ├── chunks_metadata.log
│   │   ├── frames.log
│   │   ├── output_20250223_112859.mp4
│   │   ├── output_20250223_113005.mp4
│   │   └── output_20250223_113106.mp4
│   ├── display_2_Display 2
│   │   ├── chunks_metadata.log
│   │   ├── frames.log
│   │   └── output_20250223_112859.mp4
│   ├── display_info.json
│   ├── keypresses.log
│   └── mouse.log
├── session_20250223_113242
│   ├── display_1_Display 1
│   │   ├── chunks_metadata.log
│   │   ├── frames.log
│   │   └── output_20250223_113243.mp4
│   ├── display_2_Display 2
│   │   ├── chunks_metadata.log
│   │   ├── frames.log
│   │   └── output_20250223_113243.mp4
│   ├── display_info.json
│   ├── keypresses.log
│   └── mouse.log
└── session_20250223_113424
    ├── display_1_Display 1
    │   ├── frames.log
    │   ├── output_20250223_113424.mp4
    │   ├── output_20250223_113530.mp4
    │   ├── output_20250223_113641.mp4
    │   ├── output_20250223_113804.mp4
    │   └── segment_metadata.log
    ├── display_2_Display 2
    │   ├── frames.log
    │   ├── output_20250223_113424.mp4
    │   └── segment_metadata.log
    ├── display_info.json
    ├── keypresses.log
    └── mouse.log
```