#!/usr/bin/env python3

import os
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union

@dataclass
class FrameEvent:
    """Frame captured from a display"""
    seq: int
    monotonic_time: int  # Monotonic timestamp in ms
    display_id: int
    
@dataclass
class KeypressEvent:
    """Keyboard event (key press or release)"""
    seq: int
    monotonic_time: int  # Monotonic timestamp in ms
    action: str          # "down" or "up"
    key_code: int        # macOS CGKeyCode value
    modifiers: List[str] = None  # List of active modifiers: shift, control, option/alt, command
    
@dataclass
class MouseEvent:
    """Mouse event (movement, click, scroll)"""
    seq: int
    monotonic_time: int  # Monotonic timestamp in ms
    action: str          # "move", "down", "up", "scroll"
    x: float             # Screen X coordinate
    y: float             # Screen Y coordinate
    button: Optional[int] = None    # Button number (0=left, 1=right, 2=middle)
    delta_x: Optional[float] = None  # Movement in X direction
    delta_y: Optional[float] = None  # Movement in Y direction

@dataclass
class TimelineEvent:
    """Unified timeline event containing any event type"""
    seq: int
    monotonic_time: int
    event_type: str  # "frame", "keypress", "mouse"
    data: Union[FrameEvent, KeypressEvent, MouseEvent]

class Timeline:
    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.events: List[TimelineEvent] = []
        self.load_session()
        
    def load_session(self):
        # Get information about displays
        display_info = self._load_display_info()
        
        # Find all chunk directories in the session
        chunk_dirs = sorted([d for d in os.listdir(self.session_dir) 
                            if os.path.isdir(os.path.join(self.session_dir, d)) and d.startswith('chunk_')])
        
        for chunk_dir in chunk_dirs:
            chunk_path = os.path.join(self.session_dir, chunk_dir)
            
            # Load frames for each display
            for display_id, display_data in display_info.items():
                # Find the display directory in this chunk
                display_dir_prefix = f"display_{display_id}_"
                display_dirs = [d for d in os.listdir(chunk_path) 
                               if os.path.isdir(os.path.join(chunk_path, d)) and d.startswith(display_dir_prefix)]
                
                if not display_dirs:
                    continue
                    
                display_dir = display_dirs[0]  # Should only be one directory per display
                frames_log = os.path.join(chunk_path, display_dir, "frames.log")
                
                if os.path.exists(frames_log):
                    self._load_frames(frames_log, display_id, display_data['title'])
            
            # Load keypress events
            keypresses_log = os.path.join(chunk_path, "keypresses.log")
            if os.path.exists(keypresses_log):
                self._load_keypresses(keypresses_log)
                
            # Load mouse events
            mouse_log = os.path.join(chunk_path, "mouse.log")
            if os.path.exists(mouse_log):
                self._load_mouse_events(mouse_log)
        
        # Sort all events by monotonic time to ensure correct order
        self.events.sort(key=lambda x: x.monotonic_time)
        
    def _load_display_info(self) -> Dict[int, Dict[str, Any]]:
        display_info_path = os.path.join(self.session_dir, "display_info.json")
        if not os.path.exists(display_info_path):
            return {}
            
        with open(display_info_path, 'r') as f:
            display_data = json.load(f)
            
        # Create a dictionary with display ID as key
        return {d['id']: d for d in display_data}
    
    def _load_frames(self, frames_log_path: str, display_id: int, display_title: str):
        with open(frames_log_path, 'r') as f:
            for line in f:
                try:
                    # Format is: seq, wall_time, monotonic_time
                    seq, wall_time, monotonic_time = map(int, line.strip().split(', '))
                    
                    # Simplified frame event with just essential data for model training
                    frame_event = FrameEvent(
                        seq=seq,
                        monotonic_time=monotonic_time,
                        display_id=display_id
                    )
                    
                    timeline_event = TimelineEvent(
                        seq=seq,
                        monotonic_time=monotonic_time,
                        event_type="frame",
                        data=frame_event
                    )
                    
                    self.events.append(timeline_event)
                except Exception as e:
                    print(f"Error parsing frame log line: {line.strip()} - {str(e)}")
    
    def _load_keypresses(self, keypresses_log_path: str):
        with open(keypresses_log_path, 'r') as f:
            for line in f:
                try:
                    # Parse the line with regex to handle the JSON data
                    match = re.match(r'\[\((\d+), (\d+), (\d+)\), \'(.*)\'\]', line.strip())
                    if match:
                        seq, wall_time, monotonic_time, data_str = match.groups()
                        seq, wall_time, monotonic_time = int(seq), int(wall_time), int(monotonic_time)
                        
                        # Parse the JSON data
                        event_data = json.loads(data_str)
                        event_type = event_data['type']
                        
                        # Skip events we don't need for the model
                        if event_type not in ['key_down', 'key_up']:
                            continue
                        
                        # Extract modifiers from flags
                        modifiers = []
                        flags_detail = event_data.get('flagsDetail', {})
                        if flags_detail:
                            if flags_detail.get('shift'):
                                modifiers.append('shift')
                            if flags_detail.get('control'):
                                modifiers.append('control')
                            if flags_detail.get('alternate'):  # Option/Alt key
                                modifiers.append('option')
                            if flags_detail.get('command'):
                                modifiers.append('command')
                        
                        # Map event type to simpler action
                        action = "down" if event_type == "key_down" else "up"
                        
                        # Create simplified keypress event
                        keypress_event = KeypressEvent(
                            seq=seq,
                            monotonic_time=monotonic_time,
                            action=action,
                            key_code=event_data.get('keycode', 0),
                            modifiers=modifiers
                        )
                        
                        timeline_event = TimelineEvent(
                            seq=seq,
                            monotonic_time=monotonic_time,
                            event_type="keypress",
                            data=keypress_event
                        )
                        
                        self.events.append(timeline_event)
                except Exception as e:
                    print(f"Error parsing keypress log line: {line.strip()} - {str(e)}")
    
    def _load_mouse_events(self, mouse_log_path: str):
        with open(mouse_log_path, 'r') as f:
            for line in f:
                try:
                    # Parse the line with regex to handle the JSON data
                    match = re.match(r'\[\((\d+), (\d+), (\d+)\), \'(.*)\'\]', line.strip())
                    if match:
                        seq, wall_time, monotonic_time, data_str = match.groups()
                        seq, wall_time, monotonic_time = int(seq), int(wall_time), int(monotonic_time)
                        
                        # Parse the JSON data
                        event_data = json.loads(data_str)
                        event_type = event_data['type']
                        
                        # Get location data
                        location = event_data.get('location', {'x': 0, 'y': 0})
                        x = location.get('x', 0)
                        y = location.get('y', 0)
                        
                        # Map event types to simplified actions
                        action_map = {
                            'mouse_movement': 'move',
                            'mouse_down': 'down',
                            'mouse_up': 'up',
                            'scroll_wheel': 'scroll',
                            'left_mouse_dragged': 'move',
                            'right_mouse_dragged': 'move'
                        }
                        
                        action = action_map.get(event_type, 'unknown')
                        
                        # Create simplified mouse event
                        mouse_event = MouseEvent(
                            seq=seq,
                            monotonic_time=monotonic_time,
                            action=action,
                            x=x,
                            y=y,
                            button=event_data.get('buttonNumber'),
                            delta_x=event_data.get('deltaX'),
                            delta_y=event_data.get('deltaY')
                        )
                        
                        timeline_event = TimelineEvent(
                            seq=seq,
                            monotonic_time=monotonic_time,
                            event_type="mouse",
                            data=mouse_event
                        )
                        
                        self.events.append(timeline_event)
                except Exception as e:
                    print(f"Error parsing mouse log line: {line.strip()} - {str(e)}")
    
    def get_events_in_time_range(self, start_time: int, end_time: int) -> List[TimelineEvent]:
        """Get all events within a specific monotonic time range"""
        return [event for event in self.events 
                if start_time <= event.monotonic_time <= end_time]
    
    def get_events_by_type(self, event_type: str) -> List[TimelineEvent]:
        """Get all events of a specific type (frame, keypress, mouse)"""
        return [event for event in self.events if event.event_type == event_type]
    
    def get_frame_events_for_display(self, display_id: int) -> List[TimelineEvent]:
        """Get all frame events for a specific display"""
        return [event for event in self.events 
                if event.event_type == "frame" and 
                isinstance(event.data, FrameEvent) and 
                event.data.display_id == display_id]
                
    def export_for_training(self, output_path: str):
        """Export the timeline in a format suitable for model training"""
        training_data = []
        
        for event in self.events:
            if event.event_type == "frame":
                data = {
                    "type": "frame",
                    "time": event.monotonic_time,
                    "display_id": event.data.display_id,
                    "seq": event.seq
                }
            elif event.event_type == "keypress":
                data = {
                    "type": "keypress",
                    "time": event.monotonic_time,
                    "action": event.data.action,
                    "key_code": event.data.key_code,
                    "modifiers": event.data.modifiers or []
                }
            elif event.event_type == "mouse":
                data = {
                    "type": "mouse",
                    "time": event.monotonic_time,
                    "action": event.data.action,
                    "x": event.data.x,
                    "y": event.data.y,
                    "button": event.data.button,
                    "delta_x": event.data.delta_x,
                    "delta_y": event.data.delta_y
                }
            else:
                continue
                
            training_data.append(data)
            
        with open(output_path, 'w') as f:
            json.dump(training_data, f)
            
        print(f"Exported {len(training_data)} events to {output_path}")
        
    def create_sequence_dataset(self, sequence_length: int = 100, stride: int = 10) -> List[Dict]:
        """Create a dataset of fixed-length sequences for training sequential models
        
        Args:
            sequence_length: Number of events in each sequence
            stride: Step size between sequences
            
        Returns:
            List of sequences, each containing events and target information
        """
        sequences = []
        
        if len(self.events) < sequence_length:
            print(f"Warning: Timeline has fewer events ({len(self.events)}) than requested sequence length ({sequence_length})")
            return sequences
            
        for i in range(0, len(self.events) - sequence_length, stride):
            # Get a window of events
            window = self.events[i:i + sequence_length]
            
            # Extract relevant features
            sequence_data = {
                "events": [],
                "start_time": window[0].monotonic_time,
                "end_time": window[-1].monotonic_time,
                "duration_ms": window[-1].monotonic_time - window[0].monotonic_time
            }
            
            # Process each event in the window
            for event in window:
                if event.event_type == "frame":
                    event_data = {
                        "type": "frame",
                        "time": event.monotonic_time,
                        "display_id": event.data.display_id
                    }
                elif event.event_type == "keypress":
                    event_data = {
                        "type": "keypress",
                        "time": event.monotonic_time,
                        "action": event.data.action,
                        "key_code": event.data.key_code,
                        "modifiers": event.data.modifiers or []
                    }
                elif event.event_type == "mouse":
                    event_data = {
                        "type": "mouse",
                        "time": event.monotonic_time,
                        "action": event.data.action,
                        "x": event.data.x,
                        "y": event.data.y,
                        "delta_x": event.data.delta_x if event.data.delta_x is not None else 0,
                        "delta_y": event.data.delta_y if event.data.delta_y is not None else 0
                    }
                else:
                    continue
                    
                sequence_data["events"].append(event_data)
            
            # Count event types in this sequence
            event_types = [e["type"] for e in sequence_data["events"]]
            sequence_data["counts"] = {
                "frame": event_types.count("frame"),
                "keypress": event_types.count("keypress"),
                "mouse": event_types.count("mouse")
            }
            
            sequences.append(sequence_data)
            
        return sequences
    
    def get_closest_frame(self, time: int, display_id: Optional[int] = None) -> Optional[TimelineEvent]:
        """Find the closest frame to a specific monotonic time, optionally for a specific display"""
        frames = self.get_events_by_type("frame")
        if display_id is not None:
            frames = [f for f in frames if isinstance(f.data, FrameEvent) and f.data.display_id == display_id]
            
        if not frames:
            return None
            
        return min(frames, key=lambda x: abs(x.monotonic_time - time))
    
    def __len__(self):
        return len(self.events)
    
    def __getitem__(self, index):
        return self.events[index]

# Example usage
def render_timeline_to_video(timeline, output_path, display_id=None, start_time=None, end_time=None, output_fps=30, max_frames=None):
    """Render timeline events as a video overlay on the captured frames
    
    Args:
        timeline: Timeline object containing all events
        output_path: Path to save the rendered video
        display_id: Optional display ID to render (if None, render first available display)
        start_time: Optional start time in ms (if None, start from beginning)
        end_time: Optional end time in ms (if None, continue to end)
        output_fps: FPS for the output video
        max_frames: Maximum number of frames to render
        
    Returns:
        Path to the rendered video if successful, None otherwise
    """
    import cv2
    import os
    from datetime import datetime
    
    # First, filter events by time range if specified
    events = timeline.events
    if start_time is not None or end_time is not None:
        start = start_time if start_time is not None else 0
        end = end_time if end_time is not None else float('inf')
        events = [e for e in events if start <= e.monotonic_time <= end]
        
    if not events:
        print("No events to render in the specified time range")
        return None
        
    # Get all frame events
    frame_events = [e for e in events if e.event_type == "frame"]
    if not frame_events:
        print("No frame events found to render")
        return None
        
    # Get display information
    display_info_path = os.path.join(timeline.session_dir, "display_info.json")
    if not os.path.exists(display_info_path):
        print(f"Display info not found at {display_info_path}")
        return None
        
    with open(display_info_path, 'r') as f:
        display_info = json.load(f)
        
    # Filter frames by display_id if specified
    if display_id is not None:
        frame_events = [e for e in frame_events if isinstance(e.data, FrameEvent) and e.data.display_id == display_id]
        display_data = next((d for d in display_info if d['id'] == display_id), None)
        if not display_data:
            print(f"Display ID {display_id} not found in display info")
            return None
    else:
        # Use the first available display
        if not frame_events:
            print("No frame events found for any display")
            return None
        display_id = frame_events[0].data.display_id
        display_data = next((d for d in display_info if d['id'] == display_id), None)
        if not display_data:
            print(f"Display ID {display_id} not found in display info")
            return None
    
    # Find all video.mp4 files for this display
    video_files = []
    for chunk_dir in sorted([d for d in os.listdir(timeline.session_dir) if d.startswith('chunk_')]):
        chunk_path = os.path.join(timeline.session_dir, chunk_dir)
        if not os.path.isdir(chunk_path):
            continue
            
        display_dir_prefix = f"display_{display_id}_"
        display_dirs = [d for d in os.listdir(chunk_path) if os.path.isdir(os.path.join(chunk_path, d)) and d.startswith(display_dir_prefix)]
        
        if not display_dirs:
            continue
            
        display_dir = display_dirs[0]
        video_path = os.path.join(chunk_path, display_dir, "video.mp4")
        if os.path.exists(video_path):
            video_files.append(video_path)
    
    if not video_files:
        print(f"No video files found for display {display_id}")
        return None
        
    # Get mouse and keypress events
    mouse_events = [e for e in events if e.event_type == "mouse"]
    keypress_events = [e for e in events if e.event_type == "keypress"]
    
    # Create video writer
    if not video_files:
        print(f"No video files found for display {display_id}")
        return None
        
    # Start with the first video to get dimensions
    cap = cv2.VideoCapture(video_files[0])
    if not cap.isOpened():
        print(f"Failed to open video file {video_files[0]}")
        return None
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    # Since we have multiple video files, we need to track frame timestamps
    frame_timestamps = []
    for i, event in enumerate(frame_events):
        frame_timestamps.append(event.monotonic_time)
        
    # Sort frame events by timestamp
    frame_events.sort(key=lambda x: x.monotonic_time)
    
    # Limit to max frames if specified
    if max_frames is not None and max_frames > 0 and len(frame_events) > max_frames:
        print(f"Limiting to first {max_frames} frames (out of {len(frame_events)} total)")
        frame_events = frame_events[:max_frames]
    
    # Process mouse and keypress events
    mouse_idx = 0
    keypress_idx = 0
    last_mouse_pos = None
    last_key_info = {'key_code': None, 'action': None, 'modifiers': []}
    is_button_pressed = False
    
    # Keep track of pressed modifier keys
    active_modifiers = set()
    
    # Track drag state
    is_dragging = False
    
    # Process frames
    for frame_idx, frame_event in enumerate(frame_events):
        # Determine which video file we need to read from
        video_idx = frame_idx // len(frame_events) * len(video_files)
        if video_idx >= len(video_files):
            video_idx = len(video_files) - 1
            
        cap = cv2.VideoCapture(video_files[video_idx])
        if not cap.isOpened():
            print(f"Failed to open video file {video_files[video_idx]}")
            continue
            
        # Seek to the right frame (approximate)
        frames_per_file = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        rel_frame_idx = frame_idx % frames_per_file
        cap.set(cv2.CAP_PROP_POS_FRAMES, rel_frame_idx)
        
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {rel_frame_idx} from {video_files[video_idx]}")
            cap.release()
            continue
            
        cap.release()
        
        # Get timestamp for this frame
        current_ts = frame_event.monotonic_time
        
        # Format timestamp for display
        dt = datetime.fromtimestamp(current_ts / 1000)
        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Include milliseconds
        
        # Process all mouse events up to this frame's timestamp
        while mouse_idx < len(mouse_events):
            mouse_event = mouse_events[mouse_idx]
            if mouse_event.monotonic_time > current_ts:
                break
                
            # Update mouse position based on event type
            if mouse_event.data.action == 'move':
                last_mouse_pos = (mouse_event.data.x, mouse_event.data.y)
            elif mouse_event.data.action == 'down':
                is_button_pressed = True
                if mouse_event.data.button == 0:  # Left button
                    is_dragging = True
            elif mouse_event.data.action == 'up':
                is_button_pressed = False
                if mouse_event.data.button == 0:  # Left button
                    is_dragging = False
                    
            mouse_idx += 1
            
        # Process all keypress events up to this frame's timestamp
        while keypress_idx < len(keypress_events):
            keypress_event = keypress_events[keypress_idx]
            if keypress_event.monotonic_time > current_ts:
                break
                
            # Update key information
            last_key_info = {
                'key_code': keypress_event.data.key_code,
                'action': keypress_event.data.action,
                'modifiers': keypress_event.data.modifiers or []
            }
            
            # Update active modifiers
            if keypress_event.data.modifiers:
                if keypress_event.data.action == 'down':
                    for mod in keypress_event.data.modifiers:
                        active_modifiers.add(mod)
                elif keypress_event.data.action == 'up':
                    for mod in keypress_event.data.modifiers:
                        if mod in active_modifiers:
                            active_modifiers.remove(mod)
                            
            keypress_idx += 1
            
        # Create frame copy for overlay
        overlay_frame = frame.copy()
        
        # Draw mouse cursor if we have a position
        if last_mouse_pos:
            # Convert global coordinates to local video coordinates
            gx, gy = last_mouse_pos
            
            # Convert from global to display-local
            local_x = gx - display_data["x"]
            local_y = gy - display_data["y"]
            
            # Scale to match video dimensions
            scaled_x = int(local_x * display_data["capture_width"] / display_data["original_width"])
            scaled_y = int(local_y * display_data["capture_height"] / display_data["original_height"])
            
            # Draw mouse cursor
            if 0 <= scaled_x < width and 0 <= scaled_y < height:
                if is_button_pressed:
                    # Filled circle for mouse button down
                    cv2.circle(overlay_frame, (scaled_x, scaled_y), 8, (0, 0, 255), -1)
                else:
                    # Hollow circle for normal cursor
                    cv2.circle(overlay_frame, (scaled_x, scaled_y), 8, (0, 0, 255), 2)
                    cv2.circle(overlay_frame, (scaled_x, scaled_y), 2, (0, 0, 255), -1)
                    
                # Add position text
                pos_text = f"({scaled_x}, {scaled_y})"
                cv2.putText(
                    overlay_frame,
                    pos_text,
                    (scaled_x + 15, scaled_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
        
        # Draw key information
        mods_str = '+'.join(active_modifiers) if active_modifiers else "none"
        key_str = f"Key: {last_key_info['key_code']} ({last_key_info['action']}), Mods: {mods_str}"
        cv2.putText(
            overlay_frame,
            key_str,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Draw timestamp
        cv2.putText(
            overlay_frame,
            f"Time: {formatted_time}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Draw display info
        cv2.putText(
            overlay_frame,
            f"Display: {display_data['title']} (ID: {display_id})",
            (width - 400, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Write frame to output
        out.write(overlay_frame)
        
    # Release resources
    out.release()
    
    print(f"Timeline visualization rendered to {output_path}")
    return output_path

if __name__ == "__main__":
    import sys
    import argparse
    from collections import Counter
    import math
    import cv2
    import os
    
    parser = argparse.ArgumentParser(description='Process loggy3 session data into a timeline.')
    parser.add_argument('session_dir', help='Path to the session directory')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize timeline in terminal')
    parser.add_argument('--width', '-w', type=int, default=120, help='Width of visualization (default: 120 chars)')
    parser.add_argument('--sample-count', '-s', type=int, default=1000, help='Number of samples for visualization (default: 1000)')
    parser.add_argument('--export', '-e', type=str, help='Export timeline to JSON file for training')
    parser.add_argument('--sequences', type=str, help='Export timeline as sequences for model training')
    parser.add_argument('--seq-length', type=int, default=100, help='Length of sequences for training (default: 100)')
    parser.add_argument('--stride', type=int, default=10, help='Stride between sequences (default: 10)')
    parser.add_argument('--render', type=str, help='Render timeline visualization to video file')
    parser.add_argument('--display-id', type=int, help='Render only the specified display ID')
    parser.add_argument('--output-fps', type=int, default=30, help='FPS for rendered video (default: 30)')
    parser.add_argument('--start-time', type=int, help='Start time for rendering (in ms)')
    parser.add_argument('--end-time', type=int, help='End time for rendering (in ms)')
    parser.add_argument('--max-frames', type=int, help='Maximum number of frames to render')
    args = parser.parse_args()
    
    timeline = Timeline(args.session_dir)
    
    print(f"Loaded {len(timeline)} events")
    
    # Print breakdown by event type
    frame_events = timeline.get_events_by_type("frame")
    keypress_events = timeline.get_events_by_type("keypress")
    mouse_events = timeline.get_events_by_type("mouse")
    
    print(f"Frame events: {len(frame_events)}")
    print(f"Keypress events: {len(keypress_events)}")
    print(f"Mouse events: {len(mouse_events)}")
    
    # Export for model training if requested
    if args.export:
        timeline.export_for_training(args.export)
        
    # Export as sequences if requested
    if args.sequences:
        sequences = timeline.create_sequence_dataset(args.seq_length, args.stride)
        with open(args.sequences, 'w') as f:
            json.dump(sequences, f)
        print(f"Exported {len(sequences)} sequences to {args.sequences}")
        
    # Render video visualization if requested
    if args.render:
        output_path = args.render
        render_timeline_to_video(
            timeline, 
            output_path, 
            display_id=args.display_id,
            start_time=args.start_time,
            end_time=args.end_time,
            output_fps=args.output_fps,
            max_frames=args.max_frames
        )
    
    # Print sample of each event type
    if frame_events:
        print("\nSample frame event:")
        print(f"  Type: frame")
        print(f"  Time: {frame_events[0].monotonic_time}")
        print(f"  Display ID: {frame_events[0].data.display_id}")
        
    if keypress_events:
        print("\nSample keypress event:")
        print(f"  Type: keypress")
        print(f"  Time: {keypress_events[0].monotonic_time}")
        print(f"  Action: {keypress_events[0].data.action}")
        print(f"  Key code: {keypress_events[0].data.key_code}")
        print(f"  Modifiers: {keypress_events[0].data.modifiers}")
        
    if mouse_events:
        print("\nSample mouse event:")
        print(f"  Type: mouse")
        print(f"  Time: {mouse_events[0].monotonic_time}")
        print(f"  Action: {mouse_events[0].data.action}")
        print(f"  Position: ({mouse_events[0].data.x}, {mouse_events[0].data.y})")
        if mouse_events[0].data.delta_x is not None or mouse_events[0].data.delta_y is not None:
            print(f"  Delta: ({mouse_events[0].data.delta_x}, {mouse_events[0].data.delta_y})")
    
    # CLI visualization of the timeline
    if args.visualize and timeline.events:
        print("\nTimeline Visualization:")
        print("-" * args.width)
        
        # Determine the time range
        start_time = min(event.monotonic_time for event in timeline.events)
        end_time = max(event.monotonic_time for event in timeline.events)
        time_range = end_time - start_time
        
        # Get even sample distribution across the timeline
        sample_indices = [int(i * (len(timeline.events) - 1) / (args.sample_count - 1)) for i in range(args.sample_count)]
        sampled_events = [timeline.events[i] for i in sample_indices]
        
        # Visualization characters for each event type
        vis_chars = {
            "frame": "█",  # Frame events (filled block)
            "keypress": "▲",  # Keypress events (triangle)
            "mouse": "●"   # Mouse events (circle)
        }
        
        # Group events by time slot
        num_slots = args.width
        slots = [[] for _ in range(num_slots)]
        
        for event in sampled_events:
            # Determine which slot this event belongs to
            relative_time = event.monotonic_time - start_time
            slot_idx = min(int((relative_time / time_range) * num_slots), num_slots - 1)
            slots[slot_idx].append(event)
        
        # Create visualization line
        visualization = ['·'] * num_slots  # Default is a dot for empty slots
        
        for i, slot_events in enumerate(slots):
            if not slot_events:
                continue
                
            # Count event types in this slot
            event_types = Counter(event.event_type for event in slot_events)
            most_common = event_types.most_common(1)
            if most_common:
                most_common_type = most_common[0][0]
                visualization[i] = vis_chars.get(most_common_type, '·')
        
        # Print the visualization with legend
        print(''.join(visualization))
        print("-" * args.width)
        print(f"Legend: {vis_chars['frame']} = Frame, {vis_chars['keypress']} = Keypress, {vis_chars['mouse']} = Mouse, · = Empty")
        print(f"Time range: {start_time}ms - {end_time}ms ({(end_time - start_time) / 1000:.2f} seconds)")
        
        # Print event density graph by type
        print("\nEvent Density:")
        max_count = max(len(slot) for slot in slots) if slots else 0
        if max_count > 0:
            scale_factor = min(10, max_count)  # Scale to a maximum of 10 rows
            
            for event_type, char in vis_chars.items():
                densities = []
                for slot in slots:
                    count = sum(1 for event in slot if event.event_type == event_type)
                    scaled_count = math.ceil((count / max_count) * scale_factor) if max_count > 0 else 0
                    densities.append(scaled_count)
                
                # Print event type name and its density graph
                print(f"{event_type.ljust(10)}: ", end="")
                for d in densities:
                    print(char * d, end="")
                print()
        
        # Print timeline analytics
        if timeline.events:
            # Calculate events per second
            total_duration_sec = time_range / 1000
            if total_duration_sec > 0:
                frames_per_sec = len(frame_events) / total_duration_sec
                keypresses_per_sec = len(keypress_events) / total_duration_sec
                mouse_events_per_sec = len(mouse_events) / total_duration_sec
                
                print(f"\nAnalytics (per second):")
                print(f"Frames: {frames_per_sec:.2f}/s")
                print(f"Keypresses: {keypresses_per_sec:.2f}/s")
                print(f"Mouse events: {mouse_events_per_sec:.2f}/s")
                
            # Calculate time between consecutive events of the same type
            if len(frame_events) > 1:
                frame_intervals = [frame_events[i+1].monotonic_time - frame_events[i].monotonic_time 
                                 for i in range(len(frame_events)-1)]
                avg_frame_interval = sum(frame_intervals) / len(frame_intervals)
                print(f"Average time between frames: {avg_frame_interval:.2f}ms ({1000/avg_frame_interval:.2f} fps)")
            
            # If we have displays, show them
            display_ids = set()
            for event in frame_events:
                if isinstance(event.data, FrameEvent):
                    display_ids.add(event.data.display_id)
            
            if display_ids:
                print(f"\nDisplays detected: {len(display_ids)}")
                for display_id in display_ids:
                    display_frames = timeline.get_frame_events_for_display(display_id)
                    print(f"  Display {display_id}: {len(display_frames)} frames")