#!/usr/bin/env python3

import os
import sys
import json
import subprocess
from pathlib import Path

def count_video_frames(video_path):
    """Count the number of frames in a video file using ffprobe"""
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-select_streams', 'v:0', 
        '-count_packets', 
        '-show_entries', 'stream=nb_read_packets', 
        '-of', 'json', 
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return int(data.get('streams', [{}])[0].get('nb_read_packets', 0))
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing video {video_path}: {e}")
        return 0
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error parsing ffprobe output for {video_path}: {e}")
        return 0

def count_log_lines(log_path):
    """Count the number of lines in a log file"""
    try:
        with open(log_path, 'r') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Error reading log file {log_path}: {e}")
        return 0

def analyze_session(session_dir):
    """Analyze all chunks in a session directory"""
    session_path = Path(session_dir)
    
    if not session_path.exists() or not session_path.is_dir():
        print(f"Error: {session_dir} is not a valid directory")
        sys.exit(1)
    
    print(f"Analyzing session: {session_path.name}")
    print("-" * 80)

    # Find all chunk directories
    chunk_dirs = sorted([d for d in session_path.glob("chunk_*") if d.is_dir()])
    
    if not chunk_dirs:
        print("No chunk directories found")
        return
    
    print(f"Found {len(chunk_dirs)} chunk directories")
    
    total_video_frames = 0
    total_log_lines = 0
    mismatches = []
    
    for chunk_dir in chunk_dirs:
        print(f"\nProcessing {chunk_dir.name}:")
        
        # Find all display directories
        display_dirs = [d for d in chunk_dir.glob("display_*") if d.is_dir()]
        
        for display_dir in display_dirs:
            video_path = display_dir / "video.mp4"
            log_path = display_dir / "frames.log"
            
            if not video_path.exists() or not log_path.exists():
                continue
            
            video_frames = count_video_frames(str(video_path))
            log_lines = count_log_lines(str(log_path))
            
            total_video_frames += video_frames
            total_log_lines += log_lines
            
            match = video_frames == log_lines
            status = "✓ MATCH" if match else "✗ MISMATCH"
            color = "\033[92m" if match else "\033[91m"  # Green for match, red for mismatch
            
            print(f"  {display_dir.name}:")
            print(f"    Video frames: {video_frames}")
            print(f"    Log lines:    {log_lines}")
            print(f"    Status:       {color}{status}\033[0m")
            
            if not match:
                mismatches.append((str(display_dir), video_frames, log_lines))
    
    print("\n" + "=" * 80)
    print(f"SUMMARY:")
    print(f"Total video frames: {total_video_frames}")
    print(f"Total log lines:    {total_log_lines}")
    
    if total_video_frames == total_log_lines:
        print("\033[92mOverall: MATCH\033[0m")
    else:
        diff = abs(total_video_frames - total_log_lines)
        diff_percent = (diff / max(total_video_frames, total_log_lines)) * 100
        print(f"\033[91mOverall: MISMATCH - Difference: {diff} frames ({diff_percent:.2f}%)\033[0m")
    
    if mismatches:
        print("\nMismatches detected in:")
        for path, v_frames, l_lines in mismatches:
            diff = l_lines - v_frames
            print(f"  {path}: Log has {diff:+d} entries compared to video")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <session_directory>")
        sys.exit(1)
    
    analyze_session(sys.argv[1]) 