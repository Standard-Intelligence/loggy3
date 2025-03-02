#!/usr/bin/env python3

import os
import sys
import re
import json
import cv2
import ast  # safer than eval for dicts in mouse.log
import math
from typing import List, Tuple, Optional

def parse_mouse_log(mouse_log_path: str):
    """
    Returns a list of (timestamp_ms, event) where event is either:
    - A tuple of (x, y) for position updates
    - A dict for press/release events like {'press': 'Left'} or {'release': 'Left'}
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
                if 'x' in data and 'y' in data:
                    # Position update
                    events.append((ts, (data['x'], data['y'])))
                elif 'press' in data or 'release' in data:
                    # Button event
                    events.append((ts, data))
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

def parse_mouse_delta_log(mouse_delta_log_path: str):
    """
    Returns a list of (timestamp_ms, delta_x, delta_y).
    The lines in mouse_delta.log look like:
      (1740004286196, {'deltaX': 1.5, 'deltaY': 2.3})
    """
    events = []
    if not os.path.exists(mouse_delta_log_path):
        print(f"[WARNING] No mouse_delta.log found at {mouse_delta_log_path}")
        return events

    with open(mouse_delta_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            match = re.match(r'^\((\d+),\s*(\{.*\})\)$', line)
            if not match:
                continue
            ts_str, dict_str = match.groups()
            try:
                ts = int(ts_str)
                data = ast.literal_eval(dict_str)
                if 'deltaX' in data and 'deltaY' in data:
                    events.append((ts, data['deltaX'], data['deltaY']))
            except Exception:
                continue

    events.sort(key=lambda x: x[0])
    print(f"[DEBUG] Loaded {len(events)} mouse delta events")
    if events:
        print(f"[DEBUG] First delta event: {events[0]}")
        print(f"[DEBUG] Last delta event: {events[-1]}")
    return events

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

def find_display_id_from_chunk_dir(chunk_dir: str) -> Optional[int]:
    """
    For chunk dir names like:
      1_Built-in Retina Display_20250208_123256
      2_PA329CRV_20250208_123256
    we capture the leading digits (1 or 2) before the first underscore.
    """
    chunk_basename = os.path.basename(chunk_dir)
    match = re.search(r'^(\d+)_', chunk_basename)
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

def overlay_video_chunk(
    chunk_dir: str,
    mouse_events: List[Tuple[int, any]],  # Changed type hint
    mouse_delta_events: List[Tuple[int, float, float]],
    key_events: List[Tuple[int, str]],
    display_map: dict
):
    """
    Overlays:
      - A red dot for the mouse cursor position.
      - The pressed key state in the top-left corner.
    Writes 'overlaid.mp4' next to 'output.mp4'.
    """
    output_mp4_path = os.path.join(chunk_dir, "output.mp4")
    frames_log_path = os.path.join(chunk_dir, "frames.log")
    overlaid_mp4_path = os.path.join(chunk_dir, "overlaid.mp4")

    if not os.path.exists(output_mp4_path):
        return  # nothing to do

    frame_ts = parse_frames_log(frames_log_path)
    if not frame_ts:
        print(f"[ERROR] No frames could be parsed from {frames_log_path}, skipping {chunk_dir}.")
        return

    # Determine display ID from chunk dir name, then find matching info in JSON
    disp_id = find_display_id_from_chunk_dir(chunk_dir)
    if disp_id is None or disp_id not in display_map:
        print(f"[ERROR] Could not match display info for {chunk_dir}")
        return

    di = display_map[disp_id]
    # di includes:
    # {
    #   "id": 2,
    #   "title": "Display 2",
    #   "x": -180,
    #   "y": -1080,
    #   "original_width": 1920,
    #   "original_height": 1080,
    #   "capture_width": 1920,
    #   "capture_height": 1080,
    #   ...
    # }

    cap = cv2.VideoCapture(output_mp4_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open {output_mp4_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1:
        fps = 30.0  # fallback if missing

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1'
    out = cv2.VideoWriter(overlaid_mp4_path, fourcc, fps, (width, height))

    total_frames = len(frame_ts)
    frame_idx = 0

    mouse_idx = 0
    delta_idx = 0
    key_idx = 0
    last_mouse_pos = None
    last_key_str = "none"
    is_dragging = False
    accumulated_dx = 0.0
    accumulated_dy = 0.0

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

    # Loop through video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= total_frames:
            break

        current_ts = frame_ts[frame_idx]

        # Update pressed-key string
        new_key_str, new_key_idx = find_latest_value(current_ts, key_events, key_idx)
        if new_key_str is not None:
            last_key_str = new_key_str
            key_idx = new_key_idx

        # Check for mouse events
        while mouse_idx < len(mouse_events):
            ts, event = mouse_events[mouse_idx]
            if ts > current_ts:
                break
            
            if isinstance(event, tuple) and len(event) == 2:
                # Position update
                last_mouse_pos = event
                if is_dragging:
                    print(f"[DEBUG] Reset position during drag: {event}")
                    # Store this as new base position for deltas
                    last_drag_start_pos = event
                accumulated_dx = 0.0
                accumulated_dy = 0.0
            elif isinstance(event, dict):
                if 'press' in event and event['press'] == 'Left':
                    is_dragging = True
                    print(f"[DEBUG] Started dragging at ts={ts}")
                    if last_mouse_pos:
                        last_drag_start_pos = last_mouse_pos
                elif 'release' in event and event['release'] == 'Left':
                    is_dragging = False
                    last_drag_start_pos = None
                    print(f"[DEBUG] Stopped dragging at ts={ts}")
                    print(f"[DEBUG] Total delta updates during drag: {drag_updates}")
                    print(f"[DEBUG] Total deltas accumulated: {total_deltas_used}")
                    drag_updates = 0
                    total_deltas_used = 0
            
            mouse_idx += 1

        # If dragging, accumulate delta movements from last known position
        if is_dragging and last_drag_start_pos is not None:
            deltas_this_frame = 0
            start_dx = accumulated_dx  # Track starting deltas to see if they changed
            start_dy = accumulated_dy
            
            while delta_idx < len(mouse_delta_events):
                ts, dx, dy = mouse_delta_events[delta_idx]
                if ts > current_ts:
                    break
                accumulated_dx += dx
                accumulated_dy += dy
                delta_idx += 1
                deltas_this_frame += 1
                total_deltas_used += 1

            if accumulated_dx != start_dx or accumulated_dy != start_dy:
                # Only update if deltas actually changed
                base_x, base_y = last_drag_start_pos
                last_mouse_pos = (base_x + accumulated_dx, base_y + accumulated_dy)
                drag_updates += 1
                print(f"[DEBUG] Frame {frame_idx}: Deltas changed by ({accumulated_dx-start_dx}, {accumulated_dy-start_dy}), new pos: {last_mouse_pos}")

        # If we have a valid mouse position, transform to local scaled coords
        if last_mouse_pos is not None:
            gx, gy = last_mouse_pos
            # Convert from global to display-local
            local_x = gx - di["x"]  # di.x can be negative
            local_y = gy - di["y"]

            # Scale if the capture is smaller/larger than original
            # (In your case might be 1:1 or 720p, etc.)
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

            # Draw a red circle if within bounds
            if 0 <= scaled_x < width and 0 <= scaled_y < height:
                cv2.circle(frame, (scaled_x, scaled_y), 8, (0, 0, 255), -1)
                # Add mouse position text slightly offset from the cursor
                mouse_text = f"({scaled_x}, {scaled_y})"
                text_offset_x = scaled_x + 15
                text_offset_y = scaled_y + 25
                cv2.putText(
                    frame,
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
            frame,
            label_str,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # DEBUG: Print local/scaled min/max
    if min_lx != math.inf:
        print(f"[DEBUG] Chunk: {chunk_dir}")
        print(f"        Display {disp_id} local_x range: {min_lx} → {max_lx}")
        print(f"        Display {disp_id} local_y range: {min_ly} → {max_ly}")
        print(f"        Display {disp_id} scaled_x range: {min_sx} → {max_sx}")
        print(f"        Display {disp_id} scaled_y range: {min_sy} → {max_sy}")

    print(f"[INFO] Created overlaid video at {overlaid_mp4_path} for Display ID {disp_id}")

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

    # Parse global logs in session folder
    mouse_log_path = os.path.join(session_dir, "mouse.log")
    keypress_log_path = os.path.join(session_dir, "keypresses.log")
    mouse_delta_log_path = os.path.join(session_dir, "mouse_delta.log")
    
    mouse_events = parse_mouse_log(mouse_log_path)
    key_events = parse_keypress_log(keypress_log_path)
    mouse_delta_events = parse_mouse_delta_log(mouse_delta_log_path)

    # Walk subdirectories, find where "output.mp4" & "frames.log" exist
    for root, dirs, files in os.walk(session_dir):
        if "output.mp4" in files and "frames.log" in files:
            print(f"[INFO] Found chunk in {root}")
            overlay_video_chunk(root, mouse_events, mouse_delt   a_events, key_events, display_map)

if __name__ == "__main__":
    main()