#!/usr/bin/env python3

import os
import sys
import re
import json
import cv2
import ast  # safer than eval for dicts in mouse.log
import math
import subprocess
import tempfile
from typing import List, Tuple, Optional, Dict
from datetime import datetime

def parse_mouse_log(mouse_log_path: str):
    """
    Returns a list of (timestamp_ms, event) where event is either:
    - A tuple of (x, y) for position updates
    - A dict for press/release events or wheel events
    """
    events = []
    if not os.path.exists(mouse_log_path):
        print(f"[WARNING] No mouse.log found at {mouse_log_path}")
        return events

    with open(mouse_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            match = re.match(r'^\((\d+),\s*(\{.*\})\)$', line)
            if not match:
                continue
            ts_str, dict_str = match.groups()
            try:
                ts = int(ts_str)
                data = ast.literal_eval(dict_str)

                # Handle the new format based on 'type' field
                if data.get('type') == 'move':
                    # Position update
                    events.append((ts, (data['x'], data['y'])))
                elif data.get('type') == 'button':
                    # For button events, just record the event but DON'T store as a new position
                    # This prevents cursor teleporting on clicks
                    button_event = {}
                    if data['action'] == 'press':
                        button_event['press'] = data['button']
                    elif data['action'] == 'release':
                        button_event['release'] = data['button']
                    events.append((ts, button_event))
                elif data.get('type') == 'delta':
                    # Delta update
                    events.append((ts, {'deltaX': data['deltaX'], 'deltaY': data['deltaY']}))
                elif data.get('type') == 'wheel':
                    # Wheel event
                    events.append((ts, {'wheel': True, 'deltaX': data['deltaX'], 'deltaY': data['deltaY']}))
            except Exception as e:
                print(f"[WARNING] Failed to parse mouse log line: {line}")
                print(f"[WARNING] Error: {e}")
                continue

    events.sort(key=lambda x: x[0])

    # DEBUG: Print global min/max
    positions = [ev for _, ev in events if isinstance(ev, tuple) and len(ev) == 2]
    if positions:
        min_gx = min(p[0] for p in positions)
        max_gx = max(p[0] for p in positions)
        min_gy = min(p[1] for p in positions)
        max_gy = max(p[1] for p in positions)
        print(f"[DEBUG] Global mouse X range: {min_gx} to {max_gx}")
        print(f"[DEBUG] Global mouse Y range: {min_gy} to {max_gy}")

    return events

def parse_keypress_log(keypress_log_path: str):
    """
    Returns a list of (timestamp_ms, keys_string).
    Lines look like:
      (1686428953042, '+CTRL+SHIFT')
      (1686428953100, 'none')
    """
    events = []
    if not os.path.exists(keypress_log_path):
        print(f"[WARNING] No keypresses.log found at {keypress_log_path}")
        return events

    with open(keypress_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            match = re.match(r'^\((\d+),\s*\'(.*)\'\)$', line)
            if match:
                ts_str, keys_str = match.groups()
                try:
                    ts = int(ts_str)
                    events.append((ts, keys_str))
                except Exception:
                    pass

    events.sort(key=lambda x: x[0])
    return events

def parse_frames_log(frames_log_path: str):
    """
    Returns a list of integer timestamps (ms) from frames.log.
    Each line in frames.log is a single millisecond timestamp.
    """
    frame_ts = []
    if not os.path.exists(frames_log_path):
        print(f"[ERROR] frames.log not found at {frames_log_path}")
        return frame_ts

    with open(frames_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                frame_ts.append(int(line))

    return frame_ts

def load_display_info(session_dir: str):
    """
    Reads 'display_info.json' from the session directory.
    Returns a dict of { display_id: info_dict } keyed by the "id" field.
    """
    info_path = os.path.join(session_dir, "display_info.json")
    if not os.path.exists(info_path):
        print(f"[WARNING] display_info.json not found at {info_path}")
        return {}

    with open(info_path, 'r', encoding='utf-8') as f:
        display_array = json.load(f)

    info_dict = {}
    for d in display_array:
        info_dict[d["id"]] = d
    return info_dict

def find_display_id_from_dir(display_dir: str) -> Optional[int]:
    """
    For display directory names like:
      display_1_Built-in Retina Display
      display_2_Display 2
    we capture the number after 'display_' before the second underscore.
    """
    display_basename = os.path.basename(display_dir)
    match = re.search(r'^display_(\d+)_', display_basename)
    if match:
        return int(match.group(1))
    return None

def find_latest_value(timestamp: int, events: List[Tuple[int, any]], start_idx: int) -> Tuple[Optional[any], int]:
    """
    For a sorted list of (ts, val), find the val with ts <= timestamp,
    starting from start_idx. Returns (val, new_index).
    """
    val = None
    idx = start_idx
    while idx < len(events) and events[idx][0] <= timestamp:
        val = events[idx][1]
        idx += 1
    # If we advanced idx by 1 too far, step back
    return val, idx - 1

def extract_chunk_number(chunk_dir: str) -> str:
    """
    Extracts the chunk number from a chunk directory name like 'chunk_00123'.
    Returns the chunk number as a string.
    """
    chunk_basename = os.path.basename(chunk_dir)
    match = re.search(r'chunk_(\d+)', chunk_basename)
    if match:
        return match.group(1)
    return "unknown"

def format_timestamp(timestamp_ms: int) -> str:
    """
    Formats a millisecond timestamp into a human-readable date/time string.
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def process_display_chunk(
    chunk_dir: str,
    display_dir: str,
    mouse_events: List[Tuple[int, any]],
    key_events: List[Tuple[int, str]],
    display_map: dict,
    output_fps: int = 30
) -> Optional[str]:
    """
    Processes a single chunk for a display with variable timing.
    Returns the path to the overlaid video if successful, None if failed.
    """
    video_path = os.path.join(display_dir, "video.mp4")
    frames_log_path = os.path.join(display_dir, "frames.log")
    overlaid_mp4_path = os.path.join(display_dir, "overlaid.mp4")

    if not os.path.exists(video_path):
        print(f"[WARNING] No video.mp4 found at {video_path}")
        return None

    # Parse the frame timestamps
    frame_ts = parse_frames_log(frames_log_path)
    if not frame_ts:
        print(f"[ERROR] No frames could be parsed from {frames_log_path}, skipping {display_dir}.")
        return None

    # Determine display ID from directory name, then find matching info in JSON
    disp_id = find_display_id_from_dir(display_dir)
    if disp_id is None or disp_id not in display_map:
        print(f"[ERROR] Could not match display info for {display_dir}")
        return None

    di = display_map[disp_id]
    
    # Get chunk number for overlay
    chunk_number = extract_chunk_number(chunk_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # We'll use a fixed output FPS for the video file itself
    # But we'll duplicate frames as needed to match the original timing
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1'
    out = cv2.VideoWriter(overlaid_mp4_path, fourcc, output_fps, (width, height))

    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(frame_ts))
    frame_idx = 0

    mouse_idx = 0
    key_idx = 0
    last_mouse_pos = None
    last_key_str = "none"
    is_dragging = False
    accumulated_dx = 0.0
    accumulated_dy = 0.0
    is_left_pressed = False

    # DEBUG: Track min/max local and scaled
    min_lx = math.inf
    max_lx = -math.inf
    min_ly = math.inf
    max_ly = -math.inf
    min_sx = math.inf
    max_sx = -math.inf
    min_sy = math.inf
    max_sy = -math.inf

    # Add debug counters
    drag_updates = 0
    total_deltas_used = 0
    last_drag_start_pos = None

    # Extract delta events from mouse events
    delta_events = [(ts, data['deltaX'], data['deltaY']) 
                   for ts, data in mouse_events 
                   if isinstance(data, dict) and 'deltaX' in data and 'deltaY' in data and not data.get('wheel', False)]
    delta_idx = 0

    # Store processed frames to interpolate for dynamic timing
    processed_frames = []
    
    # Loop through video frames
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        current_ts = frame_ts[frame_idx]
        
        # Format timestamp for display
        formatted_time = format_timestamp(current_ts)

        # Update pressed-key string
        new_key_str, new_key_idx = find_latest_value(current_ts, key_events, key_idx)
        if new_key_str is not None:
            last_key_str = new_key_str
            key_idx = new_key_idx

        # Process all mouse events up to this frame's timestamp
        while mouse_idx < len(mouse_events):
            ts, event = mouse_events[mouse_idx]
            if ts > current_ts:
                break
            
            if isinstance(event, tuple) and len(event) == 2:
                # Position update - Only these events update the cursor position
                last_mouse_pos = event
                if is_dragging:
                    # Store this as new base position for deltas
                    last_drag_start_pos = event
                accumulated_dx = 0.0
                accumulated_dy = 0.0
            elif isinstance(event, dict) and 'press' in event:
                if event['press'] == 'Left':
                    is_dragging = True
                    is_left_pressed = True
                    if last_mouse_pos:
                        last_drag_start_pos = last_mouse_pos
            elif isinstance(event, dict) and 'release' in event:
                if event['release'] == 'Left':
                    is_dragging = False
                    is_left_pressed = False
                    last_drag_start_pos = None
                    drag_updates = 0
                    total_deltas_used = 0
            
            mouse_idx += 1

        # If dragging, accumulate delta movements from last known position
        if is_dragging and last_drag_start_pos is not None:
            start_dx = accumulated_dx  # Track starting deltas to see if they changed
            start_dy = accumulated_dy
            
            while delta_idx < len(delta_events):
                ts, dx, dy = delta_events[delta_idx]
                if ts > current_ts:
                    break
                accumulated_dx += dx
                accumulated_dy += dy
                delta_idx += 1
                total_deltas_used += 1

            if accumulated_dx != start_dx or accumulated_dy != start_dy:
                # Only update if deltas actually changed
                base_x, base_y = last_drag_start_pos
                last_mouse_pos = (base_x + accumulated_dx, base_y + accumulated_dy)
                drag_updates += 1

        # Create a copy of the frame for overlay
        overlay_frame = frame.copy()

        # If we have a valid mouse position, transform to local scaled coords
        if last_mouse_pos is not None:
            gx, gy = last_mouse_pos
            # Convert from global to display-local
            local_x = gx - di["x"]  # di.x can be negative
            local_y = gy - di["y"]

            # Scale if the capture is smaller/larger than original
            scaled_x = int(local_x * di["capture_width"] / di["original_width"])
            scaled_y = int(local_y * di["capture_height"] / di["original_height"])

            # DEBUG: update min/max trackers
            if local_x < min_lx:
                min_lx = local_x
            if local_x > max_lx:
                max_lx = local_x
            if local_y < min_ly:
                min_ly = local_y
            if local_y > max_ly:
                max_ly = local_y
            if scaled_x < min_sx:
                min_sx = scaled_x
            if scaled_x > max_sx:
                max_sx = scaled_x
            if scaled_y < min_sy:
                min_sy = scaled_y
            if scaled_y > max_sy:
                max_sy = scaled_y

            # Draw a red circle if within bounds (filled if mouse button down)
            if 0 <= scaled_x < width and 0 <= scaled_y < height:
                # Draw different cursor depending on if mouse button is down
                if is_left_pressed:
                    # Filled circle for mouse down
                    cv2.circle(overlay_frame, (scaled_x, scaled_y), 8, (0, 0, 255), -1)
                else:
                    # Hollow circle with border for normal cursor
                    cv2.circle(overlay_frame, (scaled_x, scaled_y), 8, (0, 0, 255), 2)
                    cv2.circle(overlay_frame, (scaled_x, scaled_y), 2, (0, 0, 255), -1)
                
                # Add mouse position text slightly offset from the cursor
                mouse_text = f"({scaled_x}, {scaled_y})"
                text_offset_x = scaled_x + 15
                text_offset_y = scaled_y + 25
                cv2.putText(
                    overlay_frame,
                    mouse_text,
                    (text_offset_x, text_offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,  # smaller font size than the key display
                    (255, 255, 255),  # white text
                    2
                )

        # Draw the pressed keys text in the top-left
        if not last_key_str:
            last_key_str = "none"
        label_str = f"Keys: {last_key_str}"
        cv2.putText(
            overlay_frame,
            label_str,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Add chunk number and timestamp overlay
        chunk_text = f"Chunk: {chunk_number}"
        time_text = f"Time: {formatted_time}"
        
        # Put chunk number in top-right corner, more to the left to prevent clipping
        cv2.putText(
            overlay_frame,
            chunk_text,
            (width - 350, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Put timestamp below it
        cv2.putText(
            overlay_frame,
            time_text,
            (width - 450, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Store the processed frame and its timestamp
        processed_frames.append((overlay_frame, current_ts))
        frame_idx += 1

    cap.release()
    
    # Now write frames to output with appropriate timing
    if processed_frames:
        # First, calculate all output frames needed based on relative timing
        first_ts = processed_frames[0][1]
        last_ts = processed_frames[-1][1]
        duration_ms = last_ts - first_ts
        
        # Calculate how many frames we need for the target FPS
        total_output_frames = int(round(duration_ms / 1000 * output_fps))
        if total_output_frames <= 0:
            total_output_frames = 1  # Ensure at least one frame
            
        # Build mapping of timestamps to processed frames
        timestamps = [ts for _, ts in processed_frames]
        
        # Generate and write output frames
        for i in range(total_output_frames):
            # Calculate the timestamp for this output frame based on duration
            target_ts = first_ts + (i * duration_ms / total_output_frames)
            
            # Find the closest frame to this timestamp
            closest_idx = min(range(len(timestamps)), key=lambda idx: abs(timestamps[idx] - target_ts))
            frame = processed_frames[closest_idx][0]
            
            # Write the frame
            out.write(frame)
    
    out.release()

    print(f"[INFO] Created overlaid video at {overlaid_mp4_path} for Display ID {disp_id}")
    return overlaid_mp4_path

def concat_videos(video_files: List[str], output_path: str):
    """
    Concatenates multiple video files into one using ffmpeg.
    """
    if not video_files:
        return False
    
    # Create a temporary file with the list of videos to concatenate
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        list_file = f.name
        for video in video_files:
            f.write(f"file '{os.path.abspath(video)}'\n")
    
    try:
        # Run ffmpeg to concatenate the videos
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
            '-i', list_file, '-c', 'copy', output_path
        ]
        
        process = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.returncode != 0:
            print(f"[ERROR] FFmpeg concat failed: {process.stderr}")
            return False
            
        print(f"[INFO] Successfully concatenated {len(video_files)} videos into {output_path}")
        return True
    
    finally:
        # Clean up the temporary file
        if os.path.exists(list_file):
            os.unlink(list_file)

def main():
    if len(sys.argv) < 2:
        print("Usage: python render.py <session_dir>")
        sys.exit(1)

    session_dir = sys.argv[1]
    if not os.path.isdir(session_dir):
        print(f"[ERROR] {session_dir} is not a directory.")
        sys.exit(1)

    # Load top-level display info
    display_map = load_display_info(session_dir)
    if not display_map:
        print(f"[ERROR] Could not load display_info.json from {session_dir}")
        sys.exit(1)

    # Find all chunk directories, sorted by name
    chunk_dirs = []
    for item in os.listdir(session_dir):
        if item.startswith("chunk_"):
            chunk_path = os.path.join(session_dir, item)
            if os.path.isdir(chunk_path):
                chunk_dirs.append(chunk_path)
    
    chunk_dirs.sort()  # Ensure chronological order
    
    if not chunk_dirs:
        print(f"[ERROR] No chunk directories found in {session_dir}")
        sys.exit(1)
    
    print(f"[INFO] Found {len(chunk_dirs)} chunks to process")

    # Dict to track all display videos by display ID
    display_videos: Dict[int, List[str]] = {}
    
    # Process each chunk
    for chunk_dir in chunk_dirs:
        chunk_name = os.path.basename(chunk_dir)
        print(f"[INFO] Processing chunk: {chunk_name}")
        
        # Parse logs for this chunk
        mouse_log_path = os.path.join(chunk_dir, "mouse.log")
        keypress_log_path = os.path.join(chunk_dir, "keypresses.log")
        
        mouse_events = parse_mouse_log(mouse_log_path)
        key_events = parse_keypress_log(keypress_log_path)
        
        # Process all display directories in this chunk
        for display_item in os.listdir(chunk_dir):
            if not display_item.startswith("display_"):
                continue
                
            display_dir = os.path.join(chunk_dir, display_item)
            if not os.path.isdir(display_dir):
                continue
                
            if os.path.exists(os.path.join(display_dir, "video.mp4")):
                print(f"[INFO] Processing display: {display_item}")
                
                # Get display ID to track videos
                disp_id = find_display_id_from_dir(display_dir)
                if disp_id is None:
                    print(f"[WARNING] Could not parse display ID from {display_item}, skipping")
                    continue
                
                # Process this chunk for this display with a fixed output fps
                overlaid_path = process_display_chunk(chunk_dir, display_dir, mouse_events, key_events, display_map, output_fps=30)
                
                if overlaid_path:
                    # Add to the list of videos for this display
                    if disp_id not in display_videos:
                        display_videos[disp_id] = []
                    display_videos[disp_id].append(overlaid_path)
    
    # Now concatenate all videos for each display
    for disp_id, video_paths in display_videos.items():
        if len(video_paths) == 0:
            continue
            
        display_name = f"Display {disp_id}"
        # Try to get a better display name from the display info
        if disp_id in display_map:
            display_name = display_map[disp_id].get("title", display_name)
        
        # Create output filename
        sanitized_display_name = re.sub(r'[^\w\-_]', '_', display_name)
        output_path = os.path.join(session_dir, f"{sanitized_display_name}_combined.mp4")
        
        print(f"[INFO] Combining {len(video_paths)} videos for {display_name}")
        if len(video_paths) == 1:
            # Just copy the single video
            print(f"[INFO] Only one video segment for {display_name}, copying directly")
            try:
                cmd = ['cp', video_paths[0], output_path]
                subprocess.run(cmd, check=True)
                print(f"[INFO] Created combined video at {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to copy video: {e}")
        else:
            # Concatenate multiple videos
            if concat_videos(video_paths, output_path):
                print(f"[INFO] Created combined video at {output_path}")
            else:
                print(f"[ERROR] Failed to create combined video for {display_name}")

if __name__ == "__main__":
    main()