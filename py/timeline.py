#!/usr/bin/env python3

import os
import json
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import torch as T
from einops import rearrange
import cv2
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from datetime import timedelta
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('timeline')

@dataclass
class TimelineEvent:
    seq: int # temporal sequence position
    time: int # milliseconds since start of chunk

@dataclass
class FrameEvent(TimelineEvent):
    pixels: T.Tensor # [H, W, C]

@dataclass
class PositionEvent(TimelineEvent):
    x: float # screen coordinates
    y: float
    delta_x: float # raw mouse deltas
    delta_y: float
    
@dataclass
class TimelineScrollEvent(TimelineEvent):
    scroll_x: int # scroll deltas
    scroll_y: int

@dataclass
class TimelinePressEvent(TimelineEvent):
    key: int # key code or mouse button
    down: bool # down or up


@dataclass
class ComputerEvent:
    seq: int # temporal sequence position
    dur: float # [0,1.0) = [milliseconds till next event] / 33ms

@dataclass
class PatchEvent(ComputerEvent):
    pixels: T.Tensor # [H, W, C]
    grid_x: int # patch grid position
    grid_y: int

@dataclass
class MoveEvent(ComputerEvent):
    delta_x: float # raw mouse deltas
    delta_y: float
    pos_dx: float # screen coordinate deltas
    pos_dy: float


@dataclass
class ScrollEvent(ComputerEvent):
    scroll_x: int # scroll deltas
    scroll_y: int

@dataclass
class PressEvent(ComputerEvent):
    key: int # key code or mouse button
    down: bool # down or up

class NullEvent(ComputerEvent):
    pass

SPECIAL_KEYS = {
    # outside of the normal keycode range
    "alphaShift": 1001,
    "shift": 1002,
    "control": 1003,
    "alternate": 1004,
    "command": 1005,
    "help": 1006,
    "secondaryFn": 1007,
    "numericPad": 1008,
    "nonCoalesced": 1009,

    # mouse buttons
    "LeftMouse": 1010,
    "RightMouse": 1011,    
}

# Reverse mapping for analytics
SPECIAL_KEYS_REVERSE = {v: k for k, v in SPECIAL_KEYS.items()}
PATCH_SIZE = 10

def timeline_to_computer_events(timeline_events: List[FrameEvent | PositionEvent | TimelineScrollEvent | TimelinePressEvent]) -> List[ComputerEvent]:
    logger.info(f"Converting {len(timeline_events)} timeline events to computer events")
    timeline_events = sorted(timeline_events, key=lambda x: x.seq)
    computer_events = []
    mouse_pos = None
    screenstate = None
    
    for i, event in enumerate(timeline_events):
        event_dur = 1.0
        if i < len(timeline_events) - 1:
            event_dur = (timeline_events[i+1].time - event.time) / 33.0
            if event_dur > 1.0:
                event_dur = 1.0
        match event:
            case FrameEvent():
                pixels = event.pixels.float()
                
                # Handle non-divisible dimensions by cropping to the nearest multiple of PATCH_SIZE
                orig_h, orig_w = pixels.shape[0], pixels.shape[1]
                new_h = (orig_h // PATCH_SIZE) * PATCH_SIZE
                new_w = (orig_w // PATCH_SIZE) * PATCH_SIZE
                
                if new_h != orig_h or new_w != orig_w:
                    logger.info(f"Cropping frame from {orig_h}x{orig_w} to {new_h}x{new_w} to be divisible by patch size {PATCH_SIZE}")
                    pixels = pixels[:new_h, :new_w, :]
                
                grid_height = pixels.shape[0] // PATCH_SIZE
                grid_width = pixels.shape[1] // PATCH_SIZE
                
                pixels = rearrange(pixels, "(h p1) (w p2) c -> (h w) p1 p2 c", p1=PATCH_SIZE, p2=PATCH_SIZE)
                if screenstate is None:
                    screenstate = T.zeros_like(pixels)
                
                state_diff = T.abs(pixels - screenstate).mean(dim=(1,2,3))
                active_patches = state_diff > 10.0
                active_patches_idx = T.nonzero(active_patches, as_tuple=False)
                
                computer_events.append(NullEvent(seq=event.seq, dur=event_dur))
                
                for patch_idx in active_patches_idx:
                    patch_idx_val = int(patch_idx.item())  # Convert tensor to int
                    patch_y = patch_idx_val // grid_width
                    patch_x = patch_idx_val % grid_width
                    
                    computer_events.append(PatchEvent(
                        seq=event.seq, dur=event_dur, pixels=pixels[patch_idx_val], grid_x=int(patch_x), grid_y=int(patch_y)))
                
                screenstate = pixels
            case PositionEvent():
                if mouse_pos is None:
                    mouse_pos = (event.x, event.y)
                computer_events.append(MoveEvent(
                    seq=event.seq, dur=event_dur, delta_x=event.delta_x, delta_y=event.delta_y, pos_dx=event.x - mouse_pos[0], pos_dy=event.y - mouse_pos[1]))
                mouse_pos = (event.x, event.y)
            case TimelineScrollEvent():
                computer_events.append(ScrollEvent(
                    seq=event.seq, dur=event_dur, scroll_x=event.scroll_x, scroll_y=event.scroll_y))
            case TimelinePressEvent():
                computer_events.append(PressEvent(
                    seq=event.seq, dur=event_dur, key=event.key, down=event.down))
    
    return computer_events

def collect_timeline_analytics(timeline_events: List[FrameEvent | PositionEvent | TimelineScrollEvent | TimelinePressEvent]) -> dict:
    """
    Collect analytics data from timeline events.
    This function is separate from timeline_to_computer_events to keep the production code clean.
    
    Args:
        timeline_events: List of timeline events
        
    Returns:
        Dictionary containing analytics data
    """
    timeline_events = sorted(timeline_events, key=lambda x: x.seq)
    mouse_pos = None
    screenstate = None
    
    # Tracking variables for patching analytics
    frame_count = 0
    total_patches = 0
    active_patches_per_frame = []
    patch_activity_heatmap = defaultdict(int)
    
    for i, event in enumerate(timeline_events):
        match event:
            case FrameEvent():
                frame_count += 1
                pixels = event.pixels.float()
                logger.debug(f"Processing frame {frame_count} with shape {pixels.shape}")
                
                # Calculate grid dimensions for logging
                grid_height = pixels.shape[0] // PATCH_SIZE
                grid_width = pixels.shape[1] // PATCH_SIZE
                logger.debug(f"Frame grid dimensions: {grid_height}x{grid_width} patches")
                
                pixels = rearrange(pixels, "(h p1) (w p2) c -> (h w) p1 p2 c", p1=PATCH_SIZE, p2=PATCH_SIZE)
                if screenstate is None:
                    screenstate = T.zeros_like(pixels)
                    logger.info(f"Initialized screenstate with shape {screenstate.shape}")
                
                state_diff = T.abs(pixels - screenstate).mean(dim=(1,2,3))
                active_patches = state_diff > 10.0
                active_patches_idx = T.nonzero(active_patches, as_tuple=False)
                
                # Log patch activity for this frame
                active_count = len(active_patches_idx)
                total_patches += active_count
                active_patches_per_frame.append(active_count)
                logger.debug(f"Frame {frame_count}: {active_count} active patches out of {len(state_diff)} total patches ({active_count/len(state_diff)*100:.2f}%)")
                
                for patch_idx in active_patches_idx:
                    patch_idx_val = int(patch_idx.item())  # Convert tensor to int
                    patch_y = patch_idx_val // grid_width
                    patch_x = patch_idx_val % grid_width
                    
                    # Update heatmap
                    patch_activity_heatmap[(patch_x, patch_y)] += 1
                
                screenstate = pixels
    
    # Log summary statistics
    if frame_count > 0:
        logger.info(f"Processed {frame_count} frames with {total_patches} active patches")
        logger.info(f"Average active patches per frame: {total_patches/frame_count:.2f}")
        if active_patches_per_frame:
            logger.info(f"Min active patches: {min(active_patches_per_frame)}, Max: {max(active_patches_per_frame)}")
    
    return {
        'frame_count': frame_count,
        'total_patches': total_patches,
        'active_patches_per_frame': active_patches_per_frame,
        'patch_activity_heatmap': dict(patch_activity_heatmap)
    }

class Timeline:
    def __init__(self, chunk_dir: str, output_dir: Optional[str] = None, collect_analytics: bool = False):
        self.chunk_dir = chunk_dir
        
        # Set up output directory
        if output_dir is None:
            # Extract chunk name from path
            chunk_name = os.path.basename(os.path.normpath(chunk_dir))
            # Create default output directory
            self.output_dir = os.path.join("./data", f"{chunk_name}_analysis")
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Analysis outputs will be saved to: {self.output_dir}")
        
        self.timeline_events = self.load_chunk()
        self.events = timeline_to_computer_events(self.timeline_events)
        
        # Detect video FPS
        self.video_fps = self._detect_video_fps()
        logger.info(f"Detected video FPS: {self.video_fps:.2f}")
        
        # Collect analytics data if requested
        self.patching_analytics = {}
        if collect_analytics:
            self.patching_analytics = collect_timeline_analytics(self.timeline_events)

    def _detect_video_fps(self) -> float:
        """
        Detect FPS from the original video file or calculate it from frame timings.
        
        Returns:
            Detected FPS, or 30.0 if it cannot be determined
        """
        # Try to get FPS from video file metadata first
        display_dirs = [d for d in os.listdir(self.chunk_dir) 
                       if os.path.isdir(os.path.join(self.chunk_dir, d)) and d.startswith("display_")]
        if display_dirs:
            main_display_dir = os.path.join(self.chunk_dir, display_dirs[0])
            video_path = os.path.join(main_display_dir, "video.mp4")
            
            if os.path.exists(video_path):
                video = cv2.VideoCapture(video_path)
                if video.isOpened():
                    fps = video.get(cv2.CAP_PROP_FPS)
                    video.release()
                    if fps > 0:
                        return fps
        
        # Fallback: Calculate FPS from frame event timing if available
        frame_events = sorted([event for event in self.timeline_events if isinstance(event, FrameEvent)], 
                             key=lambda x: x.seq)
        
        if len(frame_events) > 1:
            frame_durations = [frame_events[i+1].time - frame_events[i].time 
                              for i in range(len(frame_events)-1)]
            avg_frame_duration_ms = sum(frame_durations) / len(frame_durations)
            if avg_frame_duration_ms > 0:
                return 1000 / avg_frame_duration_ms
        
        # Default fallback
        return 30.0

    def load_chunk(self):
        display_dirs = [d for d in os.listdir(self.chunk_dir) 
                       if os.path.isdir(os.path.join(self.chunk_dir, d)) and d.startswith("display_")]
        if not display_dirs: raise ValueError(f"No display directories found in {self.chunk_dir}")
        main_display_dir = os.path.join(self.chunk_dir, display_dirs[0])
        frames = self.load_frames(main_display_dir)

        keypresses_log = os.path.join(self.chunk_dir, "keypresses.log")
        if not os.path.exists(keypresses_log): raise ValueError(f"No keypresses log found in {self.chunk_dir}")
        keypresses = self.load_keypresses(keypresses_log)

        mouse_log = os.path.join(self.chunk_dir, "mouse.log")
        if not os.path.exists(mouse_log): raise ValueError(f"No mouse log found in {self.chunk_dir}")
        mouse = self.load_mouse(mouse_log)

        return frames + keypresses + mouse

    def load_frames(self, display_dir: str) -> List[FrameEvent]:
        video_path = os.path.join(display_dir, "video.mp4")
        if not os.path.exists(video_path): raise ValueError(f"No video found in {display_dir}")
        video = cv2.VideoCapture(video_path)
        frames = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret: break
            frames.append(T.from_numpy(frame))
        video.release()

        frames_log_path = os.path.join(display_dir, "frames.log")
        if not os.path.exists(frames_log_path): raise ValueError(f"No frames log found in {display_dir}")
        timings = []
        with open(frames_log_path, "r") as f:
            for line in f:
                seq, _, monotonic_time = map(int, line.strip().split(', '))
                timings.append((seq, monotonic_time))
        assert timings == sorted(timings, key=lambda x: x[0])
        assert len(timings) == len(frames)

        return [FrameEvent(seq=seq, time=time, pixels=frame) for (seq, time), frame in zip(timings, frames)]
    
    def load_keypresses(self, keypresses_log: str) -> List[TimelinePressEvent]:
        keypress_raws = []
        with open(keypresses_log, "r") as f:
            for line in f:
                keypress_raws.append(eval(line.strip()))
        keypress_timings = []
        keypress_jsons = []
        for keypress_raw in keypress_raws:
            seq, _, monotonic_time = keypress_raw[0]
            keypress_timings.append((seq, monotonic_time))
            keypress_jsons.append(json.loads(keypress_raw[1]))
        assert len(keypress_timings) == len(keypress_jsons)

        keypress_types: List[Tuple[int, bool]] = []
        for keypress_json in keypress_jsons:
            match keypress_json["type"]:
                case "key_down":
                    keypress_types.append((keypress_json["keycode"], True))
                case "key_up":
                    keypress_types.append((keypress_json["keycode"], False))
                case "flags_changed":
                    if len(keypress_json["flagsChanged"].keys()) != 1: 
                        logger.warning(f"Expected only one key in flagsChanged, but got {keypress_json['flagsChanged'].keys()}")
                    flag_key, flag_value = next(iter(keypress_json["flagsChanged"].items()))
                    is_pressed = flag_value == "pressed"
                    keypress_types.append((SPECIAL_KEYS[flag_key], is_pressed))
                case _:
                    raise ValueError(f"Unknown keypress type: {keypress_json['type']}")
        assert len(keypress_timings) == len(keypress_types)

        return [TimelinePressEvent(
            seq=seq, 
            time=monotonic_time, 
            key=key, 
            down=down) for (seq, monotonic_time), (key, down) in zip(keypress_timings, keypress_types)]

    def load_mouse(self, mouse_log: str) -> List[Union[PositionEvent, TimelineScrollEvent, TimelinePressEvent]]:
        mouse_raws = []
        with open(mouse_log, "r") as f:
            for line in f:
                mouse_raws.append(eval(line.strip()))
        mouse_timings = []
        mouse_jsons = []
        for mouse_raw in mouse_raws:
            seq, _, monotonic_time = mouse_raw[0]
            mouse_timings.append((seq, monotonic_time))
            mouse_jsons.append(json.loads(mouse_raw[1]))
        assert len(mouse_timings) == len(mouse_jsons)

        mouse_types: List[Union[Tuple[str, float, float], Tuple[str, int, bool], Tuple[str, int, int]]] = []
        for mouse_json in mouse_jsons:
            match mouse_json["type"]:
                case "mouse_movement":
                    mouse_types.append(('position', mouse_json["location"]["x"], mouse_json["location"]["y"], mouse_json["deltaX"], mouse_json["deltaY"]))
                case "mouse_down":
                    mouse_code = SPECIAL_KEYS[mouse_json["eventType"].replace("Down", "")]
                    mouse_types.append(('press', mouse_code, True))
                case "mouse_up":
                    mouse_code = SPECIAL_KEYS[mouse_json["eventType"].replace("Up", "")]
                    mouse_types.append(('press', mouse_code, False))
                case "scroll_wheel":
                    mouse_types.append(('scroll', mouse_json["pointDeltaAxis1"], mouse_json["pointDeltaAxis2"]))
                case _:
                    raise ValueError(f"Unknown mouse type: {mouse_json['type']}")
        assert len(mouse_timings) == len(mouse_types)

        events = []
        for (seq, monotonic_time), (event_type, *args) in zip(mouse_timings, mouse_types):
            match event_type:
                case 'position':
                    events.append(PositionEvent(seq=seq, time=monotonic_time, x=args[0], y=args[1], delta_x=args[2], delta_y=args[3]))
                case 'press':
                    events.append(TimelinePressEvent(seq=seq, time=monotonic_time, key=int(args[0]), down=bool(args[1])))
                case 'scroll':
                    events.append(TimelineScrollEvent(seq=seq, time=monotonic_time, scroll_x=int(args[0]), scroll_y=int(args[1])))
        return events

    def __len__(self):
        return len(self.events)
    
    def __getitem__(self, index):
        return self.events[index]
    
    def render_reconstructed_video(self, output_path=None, fps=30, highlight_active=True, highlight_color=(0, 165, 255), highlight_opacity=0.3, show_events_overlay=True):
        """
        Render a video reconstructed from the active patches.
        
        Args:
            output_path: Path to save the video. If None, will save to the output directory.
            fps: Frames per second for the output video.
            highlight_active: Whether to highlight active patches with an overlay.
            highlight_color: BGR color tuple for the highlight overlay (default is orange in BGR).
            highlight_opacity: Opacity of the highlight overlay (0.0 to 1.0).
            show_events_overlay: Whether to display keyboard and mouse events as an overlay.
        
        Returns:
            Path to the saved video file.
        """
        if not output_path:
            output_path = os.path.join(self.output_dir, "reconstructed_video.mp4")
        
        # Find all FrameEvents to get original dimensions
        frame_events = [event for event in self.timeline_events if isinstance(event, FrameEvent)]
        if not frame_events:
            logger.error("No frame events found, cannot reconstruct video")
            return None
        
        # Get dimensions from the first frame
        first_frame = frame_events[0].pixels
        height, width = first_frame.shape[0], first_frame.shape[1]
        
        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Sort events by sequence number
        patch_events = sorted([event for event in self.events if isinstance(event, PatchEvent)], 
                             key=lambda x: x.seq)
        
        # Group patch events by sequence number (frame)
        patch_events_by_seq = defaultdict(list)
        for event in patch_events:
            patch_events_by_seq[event.seq].append(event)
        
        # Sort all events by sequence for the overlay
        all_events = sorted(self.events, key=lambda x: x.seq)
        
        # Initialize the screen state
        screen = np.zeros((height, width, 3), dtype=np.uint8)
        grid_height = height // PATCH_SIZE
        grid_width = width // PATCH_SIZE
        
        # Initialize event state tracking for overlay
        current_keys_down = set()
        last_mouse_pos = (0, 0)
        last_scroll = (0, 0)
        
        # Event history for the current frame
        current_frame_events = []
        
        logger.info(f"Reconstructing video with dimensions {height}x{width} ({grid_height}x{grid_width} patches)")
        
        # Process each frame
        frame_count = 0
        current_seq = None
        last_event_idx = 0
        
        # Go through all unique sequence numbers
        for seq in sorted(patch_events_by_seq.keys()):
            events = patch_events_by_seq[seq]
            current_seq = seq
            current_frame_events = []
            
            # Find all events that happened since the last frame up to this one
            while last_event_idx < len(all_events) and all_events[last_event_idx].seq <= seq:
                event = all_events[last_event_idx]
                
                # Track event state for overlay
                if isinstance(event, PressEvent):
                    if event.down:
                        current_keys_down.add(event.key)
                        current_frame_events.append(f"Key down: {SPECIAL_KEYS_REVERSE.get(event.key, str(event.key))}")
                    else:
                        if event.key in current_keys_down:
                            current_keys_down.remove(event.key)
                        current_frame_events.append(f"Key up: {SPECIAL_KEYS_REVERSE.get(event.key, str(event.key))}")
                
                elif isinstance(event, MoveEvent):
                    last_mouse_pos = (last_mouse_pos[0] + event.delta_x, last_mouse_pos[1] + event.delta_y)
                    current_frame_events.append(f"Mouse move: ({int(last_mouse_pos[0])}, {int(last_mouse_pos[1])})")
                
                elif isinstance(event, ScrollEvent):
                    last_scroll = (event.scroll_x, event.scroll_y)
                    if event.scroll_x != 0 or event.scroll_y != 0:
                        current_frame_events.append(f"Scroll: ({event.scroll_x}, {event.scroll_y})")
                
                last_event_idx += 1
            
            # Apply all patches for this frame
            for event in events:
                # Convert patch to numpy and place it in the correct position
                patch = event.pixels.numpy()
                y_start = event.grid_y * PATCH_SIZE
                y_end = y_start + PATCH_SIZE
                x_start = event.grid_x * PATCH_SIZE
                x_end = x_start + PATCH_SIZE
                
                # Ensure we don't go out of bounds
                if y_end <= height and x_end <= width:
                    # Update the screen state with the patch
                    screen[y_start:y_end, x_start:x_end] = patch
            
            # Create a copy of the screen for this frame
            frame = screen.copy()
            
            # Add orange overlay to active patches if highlighting is enabled
            if highlight_active and events:
                # Create an overlay mask for active patches
                overlay = np.zeros_like(frame, dtype=np.uint8)
                
                # Fill in the overlay for each active patch
                for event in events:
                    y_start = event.grid_y * PATCH_SIZE
                    y_end = y_start + PATCH_SIZE
                    x_start = event.grid_x * PATCH_SIZE
                    x_end = x_start + PATCH_SIZE
                    
                    if y_end <= height and x_end <= width:
                        # Fill the patch area with the highlight color
                        overlay[y_start:y_end, x_start:x_end] = highlight_color
                
                # Apply the overlay with the specified opacity
                cv2.addWeighted(overlay, highlight_opacity, frame, 1.0, 0, frame)
            
            # Add event overlay if enabled
            if show_events_overlay:
                # Create semi-transparent background for text
                text_bg = np.zeros_like(frame)
                # Move the background rectangle to the right side
                right_margin = 10
                bg_width = 350
                bg_height = 170
                bg_x = width - bg_width - right_margin
                cv2.rectangle(text_bg, (bg_x, 0), (width - right_margin, bg_height), (0, 0, 0), -1)
                
                # Apply the semi-transparent background
                alpha = 0.7
                cv2.addWeighted(text_bg, alpha, frame, 1.0, 0, frame)
                
                # Add header - moved to right side
                cv2.putText(frame, f"Frame #{frame_count} (Seq: {current_seq})", 
                           (bg_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show mouse position - moved to right side
                cv2.putText(frame, f"Mouse: ({int(last_mouse_pos[0])}, {int(last_mouse_pos[1])})", 
                           (bg_x + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Show keys currently held down - moved to right side
                keys_text = "Keys down: "
                if current_keys_down:
                    keys_text += ", ".join([SPECIAL_KEYS_REVERSE.get(k, str(k)) for k in current_keys_down])
                else:
                    keys_text += "None"
                
                cv2.putText(frame, keys_text, (bg_x + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Show recent events for this frame (last 3) - moved to right side
                cv2.putText(frame, "Recent events:", (bg_x + 10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                for i, event_text in enumerate(current_frame_events[-3:]):
                    cv2.putText(frame, event_text, (bg_x + 20, 145 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write the frame
            video_writer.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        video_writer.release()
        logger.info(f"Reconstructed video saved to {output_path} with {frame_count} frames")
        return output_path
    
    def generate_patch_heatmap(self, output_path=None):
        """
        Generate a heatmap showing which patches were most active.
        
        Args:
            output_path: Path to save the heatmap image. If None, will save to the output directory.
            
        Returns:
            Path to the saved heatmap image.
        """
        if not output_path:
            output_path = os.path.join(self.output_dir, "patch_activity_heatmap.png")
        
        # Find all FrameEvents to get original dimensions
        frame_events = [event for event in self.timeline_events if isinstance(event, FrameEvent)]
        if not frame_events:
            logger.error("No frame events found, cannot create heatmap")
            return None
        
        # Get dimensions from the first frame
        first_frame = frame_events[0].pixels
        height, width = first_frame.shape[0], first_frame.shape[1]
        grid_height = height // PATCH_SIZE
        grid_width = width // PATCH_SIZE
        
        # Create a heatmap matrix
        heatmap = np.zeros((grid_height, grid_width))
        
        # Fill the heatmap from the analytics data
        patch_activity = self.patching_analytics.get('patch_activity_heatmap', {})
        for (x, y), count in patch_activity.items():
            if 0 <= x < grid_width and 0 <= y < grid_height:
                heatmap[y, x] = count
        
        # Plot the heatmap
        plt.figure(figsize=(12, 10))
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Patch Activity Count')
        plt.title('Patch Activity Heatmap')
        plt.xlabel('X Grid Position')
        plt.ylabel('Y Grid Position')
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Patch activity heatmap saved to {output_path}")
        return output_path

    def print_analytics(self, save_plots=False):
        """
        Print comprehensive analytics about the timeline data to help identify issues
        and understand the structure of the data.
        
        Args:
            save_plots: If True, save plots to the output directory
        """
        # Define colors for better readability
        COLORS = {
            'header': '\033[1;36m',  # Cyan, bold
            'subheader': '\033[1;34m',  # Blue, bold
            'normal': '\033[0m',  # Reset
            'value': '\033[1;32m',  # Green, bold
            'warning': '\033[1;31m',  # Red, bold
            'highlight': '\033[1;33m',  # Yellow, bold
            'section': '\033[1;35m',  # Magenta, bold
            'percent': '\033[1;96m',  # Bright cyan, bold
        }
        
        print(f"\n{COLORS['header']}{'='*80}{COLORS['normal']}")
        print(f"{COLORS['header']}TIMELINE ANALYTICS REPORT{COLORS['normal']}")
        print(f"{COLORS['header']}{'='*80}{COLORS['normal']}")
        print(f"{COLORS['header']}Output directory: {self.output_dir}{COLORS['normal']}")
        
        # Basic statistics
        total_events = len(self.events)
        total_timeline_events = len(self.timeline_events)
        print(f"\n{COLORS['section']}1. BASIC STATISTICS:{COLORS['normal']}")
        print(f"   Total timeline events: {COLORS['value']}{total_timeline_events}{COLORS['normal']}")
        print(f"   Total computer events: {COLORS['value']}{total_events}{COLORS['normal']}")
        
        # Count event types
        event_types = Counter([type(event).__name__ for event in self.events])
        timeline_event_types = Counter([type(event).__name__ for event in self.timeline_events])
        
        print(f"\n{COLORS['section']}2. EVENT TYPE DISTRIBUTION:{COLORS['normal']}")
        print(f"{COLORS['subheader']}   Timeline Events:{COLORS['normal']}")
        for event_type, count in timeline_event_types.most_common():
            percentage = count/total_timeline_events*100
            print(f"   - {COLORS['highlight']}{event_type}{COLORS['normal']}: {COLORS['value']}{count}{COLORS['normal']} ({COLORS['percent']}{percentage:.1f}%{COLORS['normal']})")
        
        print(f"\n{COLORS['subheader']}   Computer Events:{COLORS['normal']}")
        for event_type, count in event_types.most_common():
            percentage = count/total_events*100
            print(f"   - {COLORS['highlight']}{event_type}{COLORS['normal']}: {COLORS['value']}{count}{COLORS['normal']} ({COLORS['percent']}{percentage:.1f}%{COLORS['normal']})")
        
        # Add patching analytics section
        print(f"\n{COLORS['section']}3. PATCHING ANALYTICS:{COLORS['normal']}")
        if self.patching_analytics:
            frame_count = self.patching_analytics.get('frame_count', 0)
            total_patches = self.patching_analytics.get('total_patches', 0)
            active_patches_per_frame = self.patching_analytics.get('active_patches_per_frame', [])
            
            print(f"   Total frames processed: {COLORS['value']}{frame_count}{COLORS['normal']}")
            print(f"   Total active patches: {COLORS['value']}{total_patches}{COLORS['normal']}")
            
            if frame_count > 0:
                avg_patches = total_patches / frame_count
                print(f"   Average active patches per frame: {COLORS['value']}{avg_patches:.2f}{COLORS['normal']}")
                
                # Get first frame for compression ratio calculation
                frame_events = [event for event in self.timeline_events if isinstance(event, FrameEvent)]
                if frame_events:
                    first_frame = frame_events[0].pixels
                    print(f"   Compression ratio (patches/total): {COLORS['value']}{avg_patches/(frame_count * (first_frame.shape[0] // PATCH_SIZE) * (first_frame.shape[1] // PATCH_SIZE)):.4f}{COLORS['normal']}")
            
            if active_patches_per_frame:
                print(f"   Min active patches in a frame: {COLORS['value']}{min(active_patches_per_frame)}{COLORS['normal']}")
                print(f"   Max active patches in a frame: {COLORS['value']}{max(active_patches_per_frame)}{COLORS['normal']}")
                
                # Plot histogram of active patches per frame
                if save_plots:
                    plt.figure(figsize=(10, 6))
                    plt.hist(active_patches_per_frame, bins=50)
                    plt.yscale('log')
                    plt.title('Distribution of Active Patches per Frame')
                    plt.xlabel('Number of Active Patches')
                    plt.ylabel('Frequency')
                    histogram_path = os.path.join(self.output_dir, 'active_patches_histogram.png')
                    plt.savefig(histogram_path)
                    plt.close()
                    print(f"   Active patches histogram saved to {COLORS['highlight']}{os.path.basename(histogram_path)}{COLORS['normal']}")
                
                # Generate patch activity heatmap
                if save_plots:
                    heatmap_path = self.generate_patch_heatmap()
                    if heatmap_path:
                        print(f"   Patch activity heatmap saved to {COLORS['highlight']}{os.path.basename(heatmap_path)}{COLORS['normal']}")
            
            # Render reconstructed video if requested
            if save_plots:
                video_path = self.render_reconstructed_video(highlight_active=True)
                if video_path:
                    print(f"   Reconstructed video saved to {COLORS['highlight']}{os.path.basename(video_path)}{COLORS['normal']} (with orange overlay on active patches)")
        else:
            print(f"   {COLORS['warning']}No patching analytics available{COLORS['normal']}")
        
        # Time analysis
        if self.timeline_events:
            start_time = min(event.time for event in self.timeline_events)
            end_time = max(event.time for event in self.timeline_events)
            duration_ms = end_time - start_time
            
            print(f"\n{COLORS['section']}4. TIMELINE DURATION:{COLORS['normal']}")
            print(f"   Start time: {COLORS['value']}{start_time}{COLORS['normal']} ms")
            print(f"   End time: {COLORS['value']}{end_time}{COLORS['normal']} ms")
            print(f"   Duration: {COLORS['value']}{duration_ms}{COLORS['normal']} ms ({COLORS['highlight']}{timedelta(milliseconds=duration_ms)}{COLORS['normal']})")
            
            # Analyze time gaps
            time_sorted = sorted(self.timeline_events, key=lambda x: x.time)
            time_gaps = [time_sorted[i+1].time - time_sorted[i].time for i in range(len(time_sorted)-1)]
            
            if time_gaps:
                print(f"\n{COLORS['section']}5. TIME GAP ANALYSIS:{COLORS['normal']}")
                print(f"   Min gap: {COLORS['value']}{min(time_gaps)}{COLORS['normal']} ms")
                print(f"   Max gap: {COLORS['value']}{max(time_gaps)}{COLORS['normal']} ms")
                print(f"   Mean gap: {COLORS['value']}{sum(time_gaps)/len(time_gaps):.2f}{COLORS['normal']} ms")
                print(f"   Median gap: {COLORS['value']}{sorted(time_gaps)[len(time_gaps)//2]}{COLORS['normal']} ms")
                
                # Identify large gaps
                large_gaps = [(i, gap) for i, gap in enumerate(time_gaps) if gap > 1000]  # gaps > 1 second
                if large_gaps:
                    print(f"   Found {COLORS['warning']}{len(large_gaps)}{COLORS['normal']} large gaps (>1000ms):")
                    for i, (idx, gap) in enumerate(large_gaps[:5]):  # Show first 5 large gaps
                        event_before = time_sorted[idx]
                        event_after = time_sorted[idx+1]
                        print(f"     Gap {i+1}: {COLORS['warning']}{gap}{COLORS['normal']} ms between {COLORS['highlight']}{type(event_before).__name__}{COLORS['normal']} and {COLORS['highlight']}{type(event_after).__name__}{COLORS['normal']}")
                    if len(large_gaps) > 5:
                        print(f"     ... and {COLORS['warning']}{len(large_gaps)-5}{COLORS['normal']} more large gaps")
        
        # Sequence analysis
        seq_values = [event.seq for event in self.events]
        if seq_values:
            print(f"\n{COLORS['section']}6. SEQUENCE ANALYSIS:{COLORS['normal']}")
            print(f"   Min seq: {COLORS['value']}{min(seq_values)}{COLORS['normal']}")
            print(f"   Max seq: {COLORS['value']}{max(seq_values)}{COLORS['normal']}")
            
            # Check for sequence gaps
            seq_sorted = sorted(seq_values)
            expected_range = set(range(min(seq_sorted), max(seq_sorted) + 1))
            missing_seq = expected_range - set(seq_sorted)
            if missing_seq:
                print(f"   Missing {COLORS['warning']}{len(missing_seq)}{COLORS['normal']} sequence numbers")
                if len(missing_seq) < 10:
                    print(f"   Missing sequences: {COLORS['warning']}{sorted(missing_seq)}{COLORS['normal']}")
                else:
                    print(f"   First 10 missing sequences: {COLORS['warning']}{sorted(list(missing_seq))[:10]}{COLORS['normal']}")
            
            # Check for duplicate sequences
            seq_counts = Counter(seq_values)
            duplicates = {seq: count for seq, count in seq_counts.items() if count > 1}
            if duplicates:
                print(f"   Found {COLORS['warning']}{len(duplicates)}{COLORS['normal']} duplicate sequence numbers")
                if len(duplicates) < 10:
                    for seq, count in sorted(duplicates.items()):
                        print(f"     Seq {COLORS['highlight']}{seq}{COLORS['normal']} appears {COLORS['warning']}{count}{COLORS['normal']} times")
                else:
                    print(f"   First 10 duplicates: {COLORS['warning']}{dict(sorted(duplicates.items())[:10])}{COLORS['normal']}")
        
        # Duration analysis
        # Separate analysis for PatchEvents and non-PatchEvents
        patch_events_dur = [event.dur for event in self.events if isinstance(event, PatchEvent)]
        non_patch_events_dur = [event.dur for event in self.events if not isinstance(event, PatchEvent)]
        
        print(f"\n{COLORS['section']}7. DURATION ANALYSIS:{COLORS['normal']}")
        
        # PatchEvents duration analysis
        if patch_events_dur:
            print(f"\n{COLORS['subheader']}   7.1 PATCH EVENTS DURATION:{COLORS['normal']}")
            print(f"      Total patch events: {COLORS['value']}{len(patch_events_dur)}{COLORS['normal']}")
            print(f"      Min dur: {COLORS['value']}{min(patch_events_dur):.6f}{COLORS['normal']}")
            print(f"      Max dur: {COLORS['value']}{max(patch_events_dur):.6f}{COLORS['normal']}")
            print(f"      Mean dur: {COLORS['value']}{sum(patch_events_dur)/len(patch_events_dur):.6f}{COLORS['normal']}")
            
            # Check for constant durations
            dur_counts = Counter([round(d, 6) for d in patch_events_dur])
            most_common_dur = dur_counts.most_common(1)[0]
            percentage = most_common_dur[1]/len(patch_events_dur)*100
            print(f"      Most common dur: {COLORS['value']}{most_common_dur[0]:.6f}{COLORS['normal']} (appears {COLORS['highlight']}{most_common_dur[1]}{COLORS['normal']} times, {COLORS['percent']}{percentage:.1f}%{COLORS['normal']})")
            
            if most_common_dur[1] / len(patch_events_dur) > 0.5:
                print(f"      {COLORS['warning']}WARNING: Duration values appear to be constant for {percentage:.1f}% of patch events{COLORS['normal']}")
            
            # Histogram of durations
            if save_plots:
                plt.figure(figsize=(10, 6))
                plt.hist(patch_events_dur, bins=50)
                plt.title('Distribution of Duration Values - Patch Events')
                plt.xlabel('Duration')
                plt.ylabel('Count')
                plt.savefig(os.path.join(self.output_dir, 'patch_events_duration_histogram.png'))
                plt.close()
        
        # Non-PatchEvents duration analysis
        if non_patch_events_dur:
            print(f"\n{COLORS['subheader']}   7.2 NON-PATCH EVENTS DURATION:{COLORS['normal']}")
            print(f"      Total non-patch events: {COLORS['value']}{len(non_patch_events_dur)}{COLORS['normal']}")
            print(f"      Min dur: {COLORS['value']}{min(non_patch_events_dur):.6f}{COLORS['normal']}")
            print(f"      Max dur: {COLORS['value']}{max(non_patch_events_dur):.6f}{COLORS['normal']}")
            print(f"      Mean dur: {COLORS['value']}{sum(non_patch_events_dur)/len(non_patch_events_dur):.6f}{COLORS['normal']}")
            
            # Check for constant durations
            dur_counts = Counter([round(d, 6) for d in non_patch_events_dur])
            most_common_dur = dur_counts.most_common(1)[0]
            percentage = most_common_dur[1]/len(non_patch_events_dur)*100
            print(f"      Most common dur: {COLORS['value']}{most_common_dur[0]:.6f}{COLORS['normal']} (appears {COLORS['highlight']}{most_common_dur[1]}{COLORS['normal']} times, {COLORS['percent']}{percentage:.1f}%{COLORS['normal']})")
            
            if most_common_dur[1] / len(non_patch_events_dur) > 0.5:
                print(f"      {COLORS['warning']}WARNING: Duration values appear to be constant for {percentage:.1f}% of non-patch events{COLORS['normal']}")
            
            # Histogram of durations
            if save_plots:
                plt.figure(figsize=(10, 6))
                plt.hist(non_patch_events_dur, bins=50)
                plt.title('Distribution of Duration Values - Non-Patch Events')
                plt.xlabel('Duration')
                plt.ylabel('Count')
                plt.savefig(os.path.join(self.output_dir, 'non_patch_events_duration_histogram.png'))
                plt.close()
        
        # Event-specific analysis
        
        # PatchEvent analysis
        patch_events = [event for event in self.events if isinstance(event, PatchEvent)]
        if patch_events:
            print(f"\n{COLORS['section']}8. PATCH EVENT ANALYSIS:{COLORS['normal']}")
            print(f"   Total patch events: {COLORS['value']}{len(patch_events)}{COLORS['normal']}")
            
            # Check grid positions
            grid_x_values = [event.grid_x for event in patch_events]
            grid_y_values = [event.grid_y for event in patch_events]
            
            grid_x_counts = Counter(grid_x_values)
            grid_y_counts = Counter(grid_y_values)
            
            print(f"   Unique grid_x values: {COLORS['value']}{len(grid_x_counts)}{COLORS['normal']}")
            print(f"   Unique grid_y values: {COLORS['value']}{len(grid_y_counts)}{COLORS['normal']}")
            
            if len(grid_x_counts) == 1:
                print(f"   {COLORS['warning']}WARNING: All grid_x values are constant ({list(grid_x_counts.keys())[0]}){COLORS['normal']}")
            
            if len(grid_y_counts) == 1:
                print(f"   {COLORS['warning']}WARNING: All grid_y values are constant ({list(grid_y_counts.keys())[0]}){COLORS['normal']}")
            
            # Check frame dimensions
            if patch_events:
                frame_shapes = Counter([tuple(event.pixels.shape) for event in patch_events])
                print(f"   Frame shapes: {COLORS['highlight']}{dict(frame_shapes)}{COLORS['normal']}")
        
        # MoveEvent analysis
        move_events = [event for event in self.events if isinstance(event, MoveEvent)]
        if move_events:
            print(f"\n{COLORS['section']}9. MOVE EVENT ANALYSIS:{COLORS['normal']}")
            print(f"   Total move events: {COLORS['value']}{len(move_events)}{COLORS['normal']}")
            
            delta_x_values = [event.delta_x for event in move_events]
            delta_y_values = [event.delta_y for event in move_events]
            
            # Check for zero deltas
            zero_delta_x = sum(1 for dx in delta_x_values if dx == 0)
            zero_delta_y = sum(1 for dy in delta_y_values if dy == 0)
            
            x_percentage = zero_delta_x/len(move_events)*100
            y_percentage = zero_delta_y/len(move_events)*100
            
            print(f"   Zero delta_x: {COLORS['value']}{zero_delta_x}{COLORS['normal']} ({COLORS['percent']}{x_percentage:.1f}%{COLORS['normal']})")
            print(f"   Zero delta_y: {COLORS['value']}{zero_delta_y}{COLORS['normal']} ({COLORS['percent']}{y_percentage:.1f}%{COLORS['normal']})")
            
            if zero_delta_x / len(move_events) > 0.9:
                print(f"   {COLORS['warning']}WARNING: delta_x is zero for {x_percentage:.1f}% of move events{COLORS['normal']}")
            
            if zero_delta_y / len(move_events) > 0.9:
                print(f"   {COLORS['warning']}WARNING: delta_y is zero for {y_percentage:.1f}% of move events{COLORS['normal']}")
            
            # Statistics on deltas
            if delta_x_values:
                print(f"   delta_x - min: {COLORS['value']}{min(delta_x_values):.2f}{COLORS['normal']}, max: {COLORS['value']}{max(delta_x_values):.2f}{COLORS['normal']}, mean: {COLORS['value']}{sum(delta_x_values)/len(delta_x_values):.2f}{COLORS['normal']}")
            
            if delta_y_values:
                print(f"   delta_y - min: {COLORS['value']}{min(delta_y_values):.2f}{COLORS['normal']}, max: {COLORS['value']}{max(delta_y_values):.2f}{COLORS['normal']}, mean: {COLORS['value']}{sum(delta_y_values)/len(delta_y_values):.2f}{COLORS['normal']}")
            
            # Reconstruct mouse path
            if move_events and save_plots:
                x_pos, y_pos = 0.0, 0.0
                positions_x, positions_y = [x_pos], [y_pos]
                
                for event in move_events:
                    x_pos += event.delta_x
                    y_pos += event.delta_y
                    positions_x.append(x_pos)
                    positions_y.append(y_pos)
                
                plt.figure(figsize=(10, 10))
                plt.plot(positions_x, positions_y, 'b-', alpha=0.5)
                plt.scatter(positions_x[0], positions_y[0], color='green', s=100, label='Start')
                plt.scatter(positions_x[-1], positions_y[-1], color='red', s=100, label='End')
                plt.title('Reconstructed Mouse Path')
                plt.xlabel('X Position')
                plt.ylabel('Y Position')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, 'mouse_path.png'))
                plt.close()
        
        # ScrollEvent analysis
        scroll_events = [event for event in self.events if isinstance(event, ScrollEvent)]
        if scroll_events:
            print(f"\n{COLORS['section']}10. SCROLL EVENT ANALYSIS:{COLORS['normal']}")
            print(f"   Total scroll events: {COLORS['value']}{len(scroll_events)}{COLORS['normal']}")
            
            scroll_x_values = [event.scroll_x for event in scroll_events]
            scroll_y_values = [event.scroll_y for event in scroll_events]
            
            # Check for zero scrolls
            zero_scroll_x = sum(1 for sx in scroll_x_values if sx == 0)
            zero_scroll_y = sum(1 for sy in scroll_y_values if sy == 0)
            
            x_percentage = zero_scroll_x/len(scroll_events)*100
            y_percentage = zero_scroll_y/len(scroll_events)*100
            
            print(f"   Zero scroll_x: {COLORS['value']}{zero_scroll_x}{COLORS['normal']} ({COLORS['percent']}{x_percentage:.1f}%{COLORS['normal']})")
            print(f"   Zero scroll_y: {COLORS['value']}{zero_scroll_y}{COLORS['normal']} ({COLORS['percent']}{y_percentage:.1f}%{COLORS['normal']})")
            
            if zero_scroll_x / len(scroll_events) > 0.9:
                print(f"   {COLORS['warning']}WARNING: scroll_x is zero for {x_percentage:.1f}% of scroll events{COLORS['normal']}")
            
            if zero_scroll_y / len(scroll_events) > 0.9:
                print(f"   {COLORS['warning']}WARNING: scroll_y is zero for {y_percentage:.1f}% of scroll events{COLORS['normal']}")
            
            # Statistics on scrolls
            if scroll_x_values:
                print(f"   scroll_x - min: {COLORS['value']}{min(scroll_x_values)}{COLORS['normal']}, max: {COLORS['value']}{max(scroll_x_values)}{COLORS['normal']}, mean: {COLORS['value']}{sum(scroll_x_values)/len(scroll_x_values):.2f}{COLORS['normal']}")
            
            if scroll_y_values:
                print(f"   scroll_y - min: {COLORS['value']}{min(scroll_y_values)}{COLORS['normal']}, max: {COLORS['value']}{max(scroll_y_values)}{COLORS['normal']}, mean: {COLORS['value']}{sum(scroll_y_values)/len(scroll_y_values):.2f}{COLORS['normal']}")
        
        # PressEvent analysis
        press_events = [event for event in self.events if isinstance(event, PressEvent)]
        if press_events:
            print(f"\n{COLORS['section']}11. PRESS EVENT ANALYSIS:{COLORS['normal']}")
            print(f"   Total press events: {COLORS['value']}{len(press_events)}{COLORS['normal']}")
            
            # Count key codes
            key_counts = Counter([event.key for event in press_events])
            print(f"   Unique keys pressed: {COLORS['value']}{len(key_counts)}{COLORS['normal']}")
            
            # Show most common keys
            print(f"{COLORS['subheader']}   Most common keys:{COLORS['normal']}")
            for key, count in key_counts.most_common(10):
                key_name = SPECIAL_KEYS_REVERSE.get(key, f"Key {key}")
                print(f"     {COLORS['highlight']}{key_name}{COLORS['normal']}: {COLORS['value']}{count}{COLORS['normal']} times")
            
            # Check down/up balance
            down_events = sum(1 for event in press_events if event.down)
            up_events = sum(1 for event in press_events if not event.down)
            
            print(f"   Key down events: {COLORS['value']}{down_events}{COLORS['normal']}")
            print(f"   Key up events: {COLORS['value']}{up_events}{COLORS['normal']}")
            
            if abs(down_events - up_events) > 5:
                print(f"   {COLORS['warning']}WARNING: Imbalance between key down ({down_events}) and key up ({up_events}) events{COLORS['normal']}")
            
            # Check for key press sequences
            key_sequence = [(event.key, event.down) for event in press_events]
            key_state = {}
            orphaned_ups = 0
            orphaned_downs = 0
            
            for key, is_down in key_sequence:
                if is_down:
                    if key in key_state and key_state[key]:
                        orphaned_downs += 1
                    key_state[key] = True
                else:
                    if key not in key_state or not key_state[key]:
                        orphaned_ups += 1
                    key_state[key] = False
            
            # Check for keys still down at the end
            keys_still_down = sum(1 for state in key_state.values() if state)
            
            if orphaned_downs > 0:
                print(f"   {COLORS['warning']}WARNING: Found {orphaned_downs} orphaned key down events (no matching up event){COLORS['normal']}")
            
            if orphaned_ups > 0:
                print(f"   {COLORS['warning']}WARNING: Found {orphaned_ups} orphaned key up events (no matching down event){COLORS['normal']}")
            
            if keys_still_down > 0:
                print(f"   {COLORS['warning']}WARNING: {keys_still_down} keys still down at the end of the timeline{COLORS['normal']}")
        
        print(f"\n{COLORS['header']}{'='*80}{COLORS['normal']}")
        print(f"{COLORS['header']}END OF ANALYTICS REPORT{COLORS['normal']}")
        print(f"{COLORS['header']}{'='*80}{COLORS['normal']}\n")

    def generate_srt_file(self, filename="timeline.srt", fps=None):
        """
        Generate an SRT file from timeline events, clustering non-frame events by frame.
        SRT timecodes use video playback time (starting from 0 at the first frame)
        with the original video's frame rate or a specified override.
        
        Args:
            filename: Name of the SRT file to generate
            fps: Frame rate to use for SRT timecodes (default: use detected FPS)
            
        Returns:
            Path to the saved SRT file.
        """
        if not self.timeline_events:
            logger.error("No timeline events found, cannot generate SRT file")
            return None
        
        # Find all frame events to serve as anchors
        frame_events = sorted([event for event in self.timeline_events if isinstance(event, FrameEvent)], 
                             key=lambda x: x.seq)
        
        if not frame_events:
            logger.error("No frame events found, cannot generate SRT file")
            return None
        
        # Get non-frame events
        non_frame_events = [event for event in self.timeline_events 
                           if not isinstance(event, FrameEvent)]
        
        # Group non-frame events by the frame they belong to
        events_by_frame = []
        for i in range(len(frame_events)):
            current_frame = frame_events[i]
            next_frame_time = float('inf')
            
            if i < len(frame_events) - 1:
                next_frame_time = frame_events[i+1].time
            
            # Find events that occurred between this frame and the next
            frame_events_group = []
            for event in non_frame_events:
                if current_frame.time <= event.time < next_frame_time:
                    frame_events_group.append(event)
            
            # Sort events within this frame by their timestamp
            frame_events_group.sort(key=lambda x: x.time)
            events_by_frame.append((current_frame, frame_events_group))
        
        # Use detected FPS or override
        if fps is None:
            fps = self.video_fps
        frame_duration_ms = 1000.0 / fps
        logger.info(f"Using {fps:.2f}fps for SRT timecodes (frame duration: {frame_duration_ms:.2f}ms)")
        
        # Generate SRT content
        srt_content = ""
        subtitle_index = 1
        
        for frame_idx, (frame, events) in enumerate(events_by_frame):
            if not events:  # Skip frames with no events
                continue
                
            # Calculate video playback timestamps (starting from 0 at first frame)
            start_time_ms = frame_idx * frame_duration_ms
            end_time_ms = (frame_idx + 1) * frame_duration_ms
            
            # Format timestamps in SRT format (HH:MM:SS,mmm)
            start_timestamp = self._format_srt_timestamp(start_time_ms)
            end_timestamp = self._format_srt_timestamp(end_time_ms)
            
            # Format events for this frame
            event_texts = []
            # Add frame number as header for better readability
            event_texts.append(f"[Frame #{frame_idx+1}] (Seq: {frame.seq}, Original Time: {self._format_timestamp_readable(frame.time)})")
            
            for event in events:
                rel_time = event.time - frame.time
                time_indicator = f"+{rel_time}ms" if rel_time > 0 else "0ms"
                
                if isinstance(event, TimelinePressEvent):
                    key_name = SPECIAL_KEYS_REVERSE.get(event.key, f"Key {event.key}")
                    action = "pressed" if event.down else "released"
                    event_texts.append(f"• {key_name} {action} [{time_indicator}]")
                elif isinstance(event, PositionEvent):
                    event_texts.append(f"• Mouse at ({int(event.x)}, {int(event.y)}, deltas: {int(event.delta_x)}, {int(event.delta_y)}) [{time_indicator}]")
                elif isinstance(event, TimelineScrollEvent):
                    event_texts.append(f"• Scroll: ({event.scroll_x}, {event.scroll_y}) [{time_indicator}]")
            
            if event_texts:
                # Format SRT entry
                srt_content += f"{subtitle_index}\n"
                srt_content += f"{start_timestamp} --> {end_timestamp}\n"
                srt_content += "\n".join(event_texts) + "\n\n"
                subtitle_index += 1
        
        # Save SRT file
        srt_path = os.path.join(self.output_dir, filename)
        with open(srt_path, "w") as f:
            f.write(srt_content)
        
        logger.info(f"SRT file with {subtitle_index-1} subtitles saved to {srt_path}")
        return srt_path
    
    def _format_srt_timestamp(self, time_ms):
        """
        Format milliseconds time into SRT timestamp format: HH:MM:SS,mmm
        
        Args:
            time_ms: Time in milliseconds
            
        Returns:
            Formatted timestamp string
        """
        # Convert to components
        total_seconds = int(time_ms / 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        milliseconds = int(time_ms % 1000)
        
        # Format as SRT timestamp
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _format_timestamp_readable(self, time_ms):
        """
        Format milliseconds time into a human-readable format
        
        Args:
            time_ms: Time in milliseconds
            
        Returns:
            Formatted timestamp string
        """
        seconds = time_ms / 1000.0
        return f"{seconds:.3f}s"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, help="Directory to save analysis outputs (default: ./data/{chunkname}_analysis)")
    
    parser.add_argument("--no-plots", action="store_true", help="Disable analytics plots")
    parser.add_argument("--no-highlight", action="store_true", help="Disable highlighting of active patches in the reconstructed video")
    parser.add_argument("--no-events-overlay", action="store_true", help="Disable keyboard and mouse events overlay on the reconstructed video")
    parser.add_argument("--no-srt", action="store_true", help="Disable generation of SRT subtitle file with keyboard and mouse events grouped by frame")
    
    parser.add_argument("--highlight-color", type=str, default="orange", help="Color for highlighting active patches (orange, red, blue, yellow, cyan, magenta, green)")
    parser.add_argument("--highlight-opacity", type=float, default=0.3, help="Opacity of the highlight overlay (0.0 to 1.0)")
    parser.add_argument("--srt-filename", type=str, default="timeline.srt", help="Filename for the generated SRT file (default: timeline.srt)")
    parser.add_argument("--srt-fps", type=float, default=30.0, help="Frame rate to use for SRT timecodes (default: auto-detect from video, fallback to 30.0)")
    args = parser.parse_args()
    
    color_map = {
        "orange": (0, 165, 255),
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "green": (0, 255, 0)
    }
    highlight_color = color_map.get(args.highlight_color.lower(), (0, 165, 255))  # Default to orange
    
    timeline = Timeline(args.chunk_dir, args.output_dir, collect_analytics=True)
    
    if not args.no_plots:
        highlight_active = not args.no_highlight
        show_events_overlay = not args.no_events_overlay
        video_path = timeline.render_reconstructed_video(
            highlight_active=highlight_active,
            highlight_color=highlight_color,
            highlight_opacity=args.highlight_opacity,
            show_events_overlay=show_events_overlay
        )
        if video_path:
            print(f"Reconstructed video saved to {video_path}" + 
                  (" with active patch highlighting" if highlight_active else "") +
                  (" and events overlay" if show_events_overlay else ""))
    
    timeline.print_analytics(save_plots=not args.no_plots)

    # Generate SRT file if requested
    if not args.no_srt:
        srt_path = timeline.generate_srt_file(
            filename=args.srt_filename,
            fps=args.srt_fps if args.srt_fps != 30.0 else None
        )
        if srt_path:
            fps_source = "specified" if args.srt_fps != 30.0 else "detected"
            print(f"SRT subtitle file saved to {srt_path} using {fps_source} FPS")