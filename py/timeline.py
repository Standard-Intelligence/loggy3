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
import math

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
    readable: str # human readable key name


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
    readable: str # human readable key name

class NullEvent(ComputerEvent):
    pass

SPECIAL_KEYS = {
    # outside of the normal keycode range
    "alphaShift": 128,
    "shift": 129,
    "control": 130,
    "alternate": 131,
    "command": 132,
    "help": 133,
    "secondaryFn": 134,
    "numericPad": 135,
    "nonCoalesced": 136,

    # mouse buttons
    "LeftMouse": 137,
    "RightMouse": 138,   
}

PATCH_SIZE = 64

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
                active_patches = state_diff > 1.0
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
                    seq=event.seq, dur=event_dur, key=event.key, down=event.down, readable=event.readable))
    
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
                
                # Handle non-divisible dimensions by cropping to the nearest multiple of PATCH_SIZE
                orig_h, orig_w = pixels.shape[0], pixels.shape[1]
                new_h = (orig_h // PATCH_SIZE) * PATCH_SIZE
                new_w = (orig_w // PATCH_SIZE) * PATCH_SIZE
                
                if new_h != orig_h or new_w != orig_w:
                    logger.info(f"Cropping frame from {orig_h}x{orig_w} to {new_h}x{new_w} to be divisible by patch size {PATCH_SIZE}")
                    pixels = pixels[:new_h, :new_w, :]
                
                # Calculate grid dimensions for logging
                grid_height = pixels.shape[0] // PATCH_SIZE
                grid_width = pixels.shape[1] // PATCH_SIZE
                logger.debug(f"Frame grid dimensions: {grid_height}x{grid_width} patches")
                
                pixels = rearrange(pixels, "(h p1) (w p2) c -> (h w) p1 p2 c", p1=PATCH_SIZE, p2=PATCH_SIZE)
                if screenstate is None:
                    screenstate = T.zeros_like(pixels)
                    logger.info(f"Initialized screenstate with shape {screenstate.shape}")
                
                state_diff = T.abs(pixels - screenstate).mean(dim=(1,2,3))
                active_patches = state_diff > 1.0
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
        
        # Initialize tokenized events as None until requested
        self.tokenized_events = None
        self.detokenized_events = None

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

        keypress_types: List[Tuple[int, bool, str]] = []
        for keypress_json in keypress_jsons:
            match keypress_json["type"]:
                case "key_down":
                    keypress_types.append((keypress_json["keycode"], True, keypress_json["keycodeStr"]))
                case "key_up":
                    keypress_types.append((keypress_json["keycode"], False, keypress_json["keycodeStr"]))
                case "flags_changed":
                    if len(keypress_json["flagsChanged"].keys()) != 1: 
                        logger.warning(f"Expected only one key in flagsChanged, but got {keypress_json['flagsChanged'].keys()}")
                    flag_key, flag_value = next(iter(keypress_json["flagsChanged"].items()))
                    is_pressed = flag_value == "pressed"
                    keypress_types.append((SPECIAL_KEYS[flag_key], is_pressed, flag_key))
                case _:
                    raise ValueError(f"Unknown keypress type: {keypress_json['type']}")
        assert len(keypress_timings) == len(keypress_types)

        return [TimelinePressEvent(
            seq=seq, 
            time=monotonic_time, 
            key=key, 
            down=down, 
            readable=readable) for (seq, monotonic_time), (key, down, readable) in zip(keypress_timings, keypress_types)]

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

        mouse_types: List[Union[Tuple[str, float, float], Tuple[str, int, bool, str], Tuple[str, int, int]]] = []
        for mouse_json in mouse_jsons:
            match mouse_json["type"]:
                case "mouse_movement":
                    mouse_types.append(('position', mouse_json["location"]["x"], mouse_json["location"]["y"], mouse_json["deltaX"], mouse_json["deltaY"]))
                case "mouse_down":
                    mouse_code = SPECIAL_KEYS[mouse_json["eventType"].replace("Down", "")]
                    mouse_types.append(('press', mouse_code, True, mouse_json["eventType"].replace("Down", "")))
                case "mouse_up":
                    mouse_code = SPECIAL_KEYS[mouse_json["eventType"].replace("Up", "")]
                    mouse_types.append(('press', mouse_code, False, mouse_json["eventType"].replace("Up", "")))
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
                    events.append(TimelinePressEvent(seq=seq, time=monotonic_time, key=int(args[0]), down=bool(args[1]), readable=args[2]))
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
        
        # For mouse delta visualization
        mouse_deltas = []  # Keep track of recent mouse deltas
        max_deltas_history = 10  # Number of recent deltas to track
        
        # For scroll visualization
        scroll_history = []  # Keep track of recent scroll events
        max_scroll_history = 20  # Number of recent scroll events to track
        
        # For key event visualization
        recent_key_events = []  # Keep track of recent key events (down/up, key, time)
        max_key_events = 10  # Number of recent key events to track
        key_event_duration = 20  # Number of frames a key event is highlighted
        
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
            
            # Current frame mouse delta
            current_delta_x = 0
            current_delta_y = 0
            
            # Find all events that happened since the last frame up to this one
            while last_event_idx < len(all_events) and all_events[last_event_idx].seq <= seq:
                event = all_events[last_event_idx]
                
                # Track event state for overlay
                if isinstance(event, PressEvent):
                    if event.down:
                        current_keys_down.add(event.readable)
                        current_frame_events.append(f"Key down: {event.readable}")
                        # Add to recent key events for visualization
                        recent_key_events.append((True, event.readable, frame_count))
                        if len(recent_key_events) > max_key_events:
                            recent_key_events.pop(0)
                    else:
                        if event.readable in current_keys_down:
                            current_keys_down.remove(event.readable)
                        current_frame_events.append(f"Key up: {event.readable}")
                        # Add to recent key events for visualization
                        recent_key_events.append((False, event.readable, frame_count))
                        if len(recent_key_events) > max_key_events:
                            recent_key_events.pop(0)
                
                elif isinstance(event, MoveEvent):
                    current_delta_x = event.delta_x
                    current_delta_y = event.delta_y
                    last_mouse_pos = (last_mouse_pos[0] + event.delta_x, last_mouse_pos[1] + event.delta_y)
                    mouse_deltas.append((current_delta_x, current_delta_y))
                    if len(mouse_deltas) > max_deltas_history:
                        mouse_deltas.pop(0)
                    current_frame_events.append(f"Mouse move: ({event.pos_dx}, {event.pos_dy}, {event.delta_x}, {event.delta_y})")
                
                elif isinstance(event, ScrollEvent):
                    last_scroll = (event.scroll_x, event.scroll_y)
                    scroll_history.append(last_scroll)
                    if len(scroll_history) > max_scroll_history:
                        scroll_history.pop(0)
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
                
                cv2.addWeighted(overlay, highlight_opacity, frame, 1.0, 0, frame)
            
            # Add event overlay if enabled
            if show_events_overlay:
                # Create semi-transparent background for text
                text_bg = np.zeros_like(frame)
                
                # Background for the text overlay (top right)
                right_margin = 10
                bg_width = 350
                bg_height = 170
                bg_x = width - bg_width - right_margin
                cv2.rectangle(text_bg, (bg_x, 0), (width - right_margin, bg_height), (0, 0, 0), -1)
                
                # Background for mouse delta arrow (center)
                arrow_bg_size = 200
                arrow_bg_x = (width - arrow_bg_size) // 2
                arrow_bg_y = (height - arrow_bg_size) // 2
                cv2.rectangle(text_bg, 
                             (arrow_bg_x, arrow_bg_y), 
                             (arrow_bg_x + arrow_bg_size, arrow_bg_y + arrow_bg_size), 
                             (0, 0, 0), -1)
                
                # Background for scroll visualization (bottom right)
                scroll_bg_size = 150
                scroll_bg_x = width - scroll_bg_size - right_margin
                scroll_bg_y = height - scroll_bg_size - 10
                cv2.rectangle(text_bg, 
                             (scroll_bg_x, scroll_bg_y), 
                             (scroll_bg_x + scroll_bg_size, scroll_bg_y + scroll_bg_size), 
                             (0, 0, 0), -1)
                
                # Background for keyboard visualization (bottom left of scroll)
                keyboard_bg_width = 300
                keyboard_bg_height = 150
                keyboard_bg_x = scroll_bg_x - keyboard_bg_width - 10
                keyboard_bg_y = height - keyboard_bg_height - 10
                cv2.rectangle(text_bg, 
                             (keyboard_bg_x, keyboard_bg_y), 
                             (keyboard_bg_x + keyboard_bg_width, keyboard_bg_y + keyboard_bg_height), 
                             (0, 0, 0), -1)
                
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
                    keys_text += ", ".join(current_keys_down)
                else:
                    keys_text += "None"
                
                cv2.putText(frame, keys_text, (bg_x + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Show recent events for this frame (last 3) - moved to right side
                cv2.putText(frame, "Recent events:", (bg_x + 10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                for i, event_text in enumerate(current_frame_events[-3:]):
                    cv2.putText(frame, event_text, (bg_x + 20, 145 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw mouse delta arrow in the center
                arrow_center_x = arrow_bg_x + arrow_bg_size // 2
                arrow_center_y = arrow_bg_y + arrow_bg_size // 2
                
                # Draw coordinate system
                cv2.line(frame, 
                        (arrow_center_x, arrow_bg_y + 10), 
                        (arrow_center_x, arrow_bg_y + arrow_bg_size - 10), 
                        (100, 100, 100), 1)  # Vertical line
                cv2.line(frame, 
                        (arrow_bg_x + 10, arrow_center_y), 
                        (arrow_bg_x + arrow_bg_size - 10, arrow_center_y), 
                        (100, 100, 100), 1)  # Horizontal line
                
                # Scale factor for arrow (to make it visible)
                scale = 3.0
                
                # Draw mouse delta arrow
                if current_delta_x != 0 or current_delta_y != 0:
                    # Calculate arrow endpoint with scaling
                    arrow_end_x = int(arrow_center_x + current_delta_x * scale)
                    arrow_end_y = int(arrow_center_y + current_delta_y * scale)
                    
                    # Draw the main arrow
                    cv2.arrowedLine(frame, 
                                   (arrow_center_x, arrow_center_y), 
                                   (arrow_end_x, arrow_end_y), 
                                   (0, 255, 0), 2, tipLength=0.3)
                
                # Draw historical mouse deltas with fading color
                for i, (dx, dy) in enumerate(mouse_deltas[:-1]):  # Skip the most recent one (already drawn)
                    # Calculate color intensity based on recency (older = more faded)
                    intensity = int(255 * (i / len(mouse_deltas)))
                    color = (0, intensity, 0)
                    
                    # Calculate arrow endpoint with scaling
                    hist_arrow_end_x = int(arrow_center_x + dx * scale)
                    hist_arrow_end_y = int(arrow_center_y + dy * scale)
                    
                    # Draw historical arrow (thinner)
                    cv2.arrowedLine(frame, 
                                   (arrow_center_x, arrow_center_y), 
                                   (hist_arrow_end_x, hist_arrow_end_y), 
                                   color, 1, tipLength=0.2)
                
                # Add label for mouse delta visualization
                cv2.putText(frame, "Mouse Movement", 
                           (arrow_bg_x + 10, arrow_bg_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Draw scroll visualization in bottom right
                # Create a small coordinate system in the scroll area
                scroll_center_x = scroll_bg_x + scroll_bg_size // 2
                scroll_center_y = scroll_bg_y + scroll_bg_size // 2
                
                # Draw coordinate axes
                cv2.line(frame, 
                        (scroll_center_x, scroll_bg_y + 10), 
                        (scroll_center_x, scroll_bg_y + scroll_bg_size - 10), 
                        (100, 100, 100), 1)  # Vertical line
                cv2.line(frame, 
                        (scroll_bg_x + 10, scroll_center_y), 
                        (scroll_bg_x + scroll_bg_size - 10, scroll_center_y), 
                        (100, 100, 100), 1)  # Horizontal line
                
                # Mark the center point
                cv2.circle(frame, (scroll_center_x, scroll_center_y), 2, (150, 150, 150), -1)
                
                # Draw axis labels
                cv2.putText(frame, "+Y", 
                           (scroll_center_x + 5, scroll_bg_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, "-Y", 
                           (scroll_center_x + 5, scroll_bg_y + scroll_bg_size - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, "-X", 
                           (scroll_bg_x + 15, scroll_center_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, "+X", 
                           (scroll_bg_x + scroll_bg_size - 25, scroll_center_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Draw scroll history points
                scroll_scale = 1.5  # Scale factor for scroll values - increased from 1.0
                
                # Symmetric log (symlog) transformation function
                def symlog_transform(val, linear_threshold=5.0, min_visual_val=3.0):
                    if val == 0:
                        return 0
                    
                    # For small non-zero values, ensure minimum visual displacement
                    if abs(val) <= 1:
                        return min_visual_val * (1 if val > 0 else -1)
                        
                    # For other small values, apply linear scaling with minimum visual effect
                    elif abs(val) < linear_threshold:
                        # Scale linearly but ensure it's visibly larger than the minimum
                        scaled = (abs(val) / linear_threshold) * (linear_threshold - min_visual_val) + min_visual_val
                        return scaled * (1 if val > 0 else -1)
                    else:
                        # Log region with sign preservation for large values
                        sign = 1 if val >= 0 else -1
                        log_val = math.log10(abs(val) - linear_threshold + 1) + linear_threshold
                        return sign * log_val
                
                for i, (sx, sy) in enumerate(scroll_history):
                    # Calculate color intensity based on recency (newer = brighter)
                    intensity = int(155 + 100 * (i / len(scroll_history)))
                    color = (0, 0, intensity)
                    
                    # Apply symlog transformation to scroll values
                    transformed_sx = symlog_transform(sx) * scroll_scale
                    transformed_sy = symlog_transform(sy) * scroll_scale
                    
                    # Invert Y since positive scroll Y is usually scrolling up
                    point_x = scroll_center_x + int(transformed_sx)
                    point_y = scroll_center_y - int(transformed_sy)
                    
                    # Draw point
                    point_size = 3
                    # Make the most recent point larger and with a different color
                    if i == len(scroll_history) - 1 and (sx != 0 or sy != 0):
                        point_size = 5
                        # Draw a bright circle for latest non-zero scroll
                        cv2.circle(frame, (point_x, point_y), point_size, (0, 255, 255), -1)
                        # Draw small text showing the actual scroll values
                        cv2.putText(frame, f"({sx},{sy})", 
                                   (point_x + 7, point_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                    else:
                        cv2.circle(frame, (point_x, point_y), point_size, color, -1)
                
                # Add label for scroll visualization
                cv2.putText(frame, "Scroll History (Symlog)", 
                           (scroll_bg_x + 10, scroll_bg_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Draw keyboard visualization
                # Define keyboard layout (simplified)
                keyboard_rows = [
                    ["Esc", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12"],
                    ["`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", "Backspace"],
                    ["Tab", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]", "\\"],
                    ["Caps", "A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'", "Enter"],
                    ["Shift", "Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "Shift"],
                    ["Ctrl", "Win", "Alt", "Space", "Alt", "Fn", "Menu", "Ctrl"]
                ]
                
                # Draw keyboard
                key_width = 20
                key_height = 20
                key_margin = 2
                row_margin = 2
                
                # Create a mapping of all key positions for event animation
                key_positions = {}  # key -> (x, y, width, height)
                
                for row_idx, row in enumerate(keyboard_rows):
                    # Adjust starting position for each row to center it
                    row_width = len(row) * (key_width + key_margin) - key_margin
                    row_start_x = keyboard_bg_x + (keyboard_bg_width - row_width) // 2
                    row_start_y = keyboard_bg_y + 30 + row_idx * (key_height + row_margin)
                    
                    for key_idx, key in enumerate(row):
                        key_x = row_start_x + key_idx * (key_width + key_margin)
                        
                        # Adjust width for special keys
                        current_key_width = key_width
                        if key in ["Backspace", "Tab", "Enter", "Shift", "Ctrl", "Space"]:
                            if key == "Backspace":
                                current_key_width = key_width * 2
                            elif key == "Tab":
                                current_key_width = key_width * 1.5
                            elif key == "Enter":
                                current_key_width = key_width * 2
                            elif key == "Shift":
                                current_key_width = key_width * 2
                            elif key == "Ctrl":
                                current_key_width = key_width * 1.5
                            elif key == "Space":
                                current_key_width = key_width * 5
                        
                        # Store key position for event animation
                        key_positions[key.upper()] = (key_x, row_start_y, int(current_key_width), key_height)
                        
                        # Determine if key is pressed - improved to handle different key formats
                        key_pressed = False
                        key_upper = key.upper()
                        
                        # Map key names to possible variations in event readable strings
                        key_variations = {
                            'SHIFT': ['SHIFT', 'shift'],
                            'CTRL': ['CTRL', 'control', 'CONTROL'],
                            'ALT': ['ALT', 'option', 'OPTION'],
                            'ENTER': ['ENTER', 'return', 'RETURN'],
                            'SPACE': ['SPACE', ' '],
                            'BACKSPACE': ['BACKSPACE', 'DELETE'],
                            'ESC': ['ESC', 'ESCAPE'],
                            'TAB': ['TAB'],
                        }
                        
                        # Check if the key is pressed using various possible names
                        if key_upper in key_variations:
                            for variation in key_variations[key_upper]:
                                if variation in current_keys_down:
                                    key_pressed = True
                                    break
                        else:
                            # For simple keys, just check if it exists in the current_keys_down set
                            key_pressed = key_upper in current_keys_down or key in current_keys_down
                        
                        # Set color based on pressed state
                        key_color = (255, 100, 100) if key_pressed else (70, 70, 70)
                        text_color = (255, 255, 255)
                        
                        # Draw key rectangle
                        cv2.rectangle(frame, 
                                     (key_x, row_start_y), 
                                     (key_x + int(current_key_width), row_start_y + key_height), 
                                     key_color, -1)
                        
                        # Draw key text (if it fits)
                        if len(key) <= 1:
                            cv2.putText(frame, key, 
                                       (key_x + int(current_key_width) // 2 - 4, row_start_y + key_height // 2 + 4), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                
                # Draw animations for recent key events
                for is_down, key_name, event_frame in recent_key_events:
                    # Calculate age of event in frames
                    age = frame_count - event_frame
                    if age < key_event_duration:
                        # Find the correct key position
                        normalized_key = key_name.upper()
                        
                        # Map special keys to their keyboard representation
                        special_key_mapping = {
                            ' ': 'SPACE',
                            'RETURN': 'ENTER',
                            'DELETE': 'BACKSPACE',
                            'ESCAPE': 'ESC',
                            'CONTROL': 'CTRL',
                            'OPTION': 'ALT',
                            'shift': 'SHIFT',
                            'control': 'CTRL',
                            'option': 'ALT',
                        }
                        
                        if normalized_key in special_key_mapping:
                            normalized_key = special_key_mapping[normalized_key]
                            
                        # Find key position
                        key_pos = None
                        # First try exact match
                        if normalized_key in key_positions:
                            key_pos = key_positions[normalized_key]
                        # Then try just the first character (for single keys)
                        elif len(normalized_key) == 1:
                            key_pos = key_positions.get(normalized_key, None)
                        
                        if key_pos:
                            x, y, w, h = key_pos
                            # Calculate animation effect (pulsing glow)
                            animation_progress = 1.0 - (age / key_event_duration)
                            glow_size = int(animation_progress * 5)
                            
                            # Color for key press/release
                            if is_down:
                                glow_color = (0, 255, 0)  # Green for key down
                            else:
                                glow_color = (0, 0, 255)  # Red for key up
                            
                            # Draw glow effect around key
                            cv2.rectangle(frame, 
                                         (x - glow_size, y - glow_size), 
                                         (x + w + glow_size, y + h + glow_size), 
                                         glow_color, 2)
                
                # Improved key press detection by checking case-insensitive and handling special keys
                # Debug information - show what keys are actually being tracked
                debug_y_pos = keyboard_bg_y + keyboard_bg_height - 20
                active_keys_text = ", ".join(current_keys_down)
                if len(active_keys_text) > 0:
                    # Split into multiple lines if too long
                    if len(active_keys_text) > 30:
                        chunks = [active_keys_text[i:i+30] for i in range(0, len(active_keys_text), 30)]
                        for i, chunk in enumerate(chunks):
                            cv2.putText(frame, chunk, 
                                      (keyboard_bg_x + 10, debug_y_pos - (len(chunks) - i - 1) * 15), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    else:
                        cv2.putText(frame, active_keys_text, 
                                   (keyboard_bg_x + 10, debug_y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Add label for keyboard visualization
                cv2.putText(frame, "Keyboard", 
                           (keyboard_bg_x + 10, keyboard_bg_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
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

    def generate_mouse_delta_scatter_plot(self, output_path=None, percentiles=[5, 25, 50, 75, 95]):
        """
        Generate a scatter plot of mouse movement deltas with percentile bars and density-based point sizing.
        
        Args:
            output_path: Path to save the plot image. If None, will save to the output directory.
            percentiles: List of percentiles to show as horizontal and vertical bars.
            
        Returns:
            Path to the saved plot image.
        """
        if not output_path:
            output_path = os.path.join(self.output_dir, "mouse_delta_scatter.png")
        
        # Extract all move events
        move_events = [event for event in self.events if isinstance(event, MoveEvent)]
        if not move_events:
            logger.error("No move events found, cannot create mouse delta scatter plot")
            return None
        
        # Extract delta values
        delta_x_values = [event.delta_x for event in move_events]
        delta_y_values = [event.delta_y for event in move_events]
        
        # Calculate percentiles for both axes
        x_percentiles = np.percentile(delta_x_values, percentiles)
        y_percentiles = np.percentile(delta_y_values, percentiles)
        
        # Create the scatter plot
        plt.figure(figsize=(12, 12))
        
        # Create a symmetric log scale for better visualization of small and large values
        plt.xscale('symlog', linthresh=1)  # Linear scale for -1 < x < 1, log scale elsewhere
        plt.yscale('symlog', linthresh=1)  # Linear scale for -1 < y < 1, log scale elsewhere
        
        # Plot the grid centered at 0,0
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        # Add percentile lines for x-axis
        for i, p in enumerate(percentiles):
            plt.axvline(x=x_percentiles[i], color='red', linestyle='--', alpha=0.5, 
                       label=f'X {p}th percentile: {x_percentiles[i]:.2f}' if i == 0 else "")
        
        # Add percentile lines for y-axis
        for i, p in enumerate(percentiles):
            plt.axhline(y=y_percentiles[i], color='blue', linestyle='--', alpha=0.5,
                       label=f'Y {p}th percentile: {y_percentiles[i]:.2f}' if i == 0 else "")
        
        # Round values to group nearby points
        # Lower decimals = more aggressive grouping
        decimals = 1
        coords = [(round(x, decimals), round(y, decimals)) for x, y in zip(delta_x_values, delta_y_values)]
        
        # Count occurrences of each coordinate pair
        coord_counts = Counter(coords)
        
        # Extract unique coordinates and their counts
        unique_coords = list(coord_counts.keys())
        counts = list(coord_counts.values())
        
        # Convert to arrays for plotting
        x_coords = [coord[0] for coord in unique_coords]
        y_coords = [coord[1] for coord in unique_coords]
        
        # Scale marker sizes based on counts
        # Use a non-linear scaling to make differences more visible
        min_size = 10
        max_size = 500
        min_count = min(counts) if counts else 1
        max_count = max(counts) if counts else 1
        
        # Use a logarithmic scale for better visibility of size differences
        if max_count > min_count:
            sizes = [min_size + (max_size - min_size) * (np.log(count) - np.log(min_count)) / 
                    (np.log(max_count) - np.log(min_count)) for count in counts]
        else:
            sizes = [min_size for _ in counts]
        
        # Create scatter plot with size and color representing density
        scatter = plt.scatter(x_coords, y_coords, s=sizes, alpha=0.5, 
                             c=counts, cmap='viridis', edgecolors='none')
        
        # Add a colorbar to show the count scale
        cbar = plt.colorbar(scatter, label='Point Density')
        
        # Set plot limits to ensure it's centered and includes the data
        max_abs_x = max(abs(min(delta_x_values)), abs(max(delta_x_values)))
        max_abs_y = max(abs(min(delta_y_values)), abs(max(delta_y_values)))
        max_abs = max(max_abs_x, max_abs_y) * 1.1  # Add 10% margin
        plt.xlim(-max_abs, max_abs)
        plt.ylim(-max_abs, max_abs)
        
        # Add labels and title
        plt.title('Mouse Movement Deltas with Density-Based Sizing (Log Scale)')
        plt.xlabel('Delta X (symlog scale)')
        plt.ylabel('Delta Y (symlog scale)')
        
        # Add statistical information to the plot
        plt.figtext(0.02, 0.02, 
                   f"X range: [{min(delta_x_values):.2f}, {max(delta_x_values):.2f}]\n"
                   f"Y range: [{min(delta_y_values):.2f}, {max(delta_y_values):.2f}]\n"
                   f"X median: {np.median(delta_x_values):.2f}\n"
                   f"Y median: {np.median(delta_y_values):.2f}\n"
                   f"Total points: {len(delta_x_values)}\n"
                   f"Unique coordinates: {len(unique_coords)}\n"
                   f"Max density: {max_count} points",
                   bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        # Add a legend
        plt.legend(loc='upper right')
        
        # Add percentile values table
        table_text = []
        for i, p in enumerate(percentiles):
            table_text.append([f"{p}th", f"{x_percentiles[i]:.2f}", f"{y_percentiles[i]:.2f}"])
        
        the_table = plt.table(cellText=table_text,
                             colLabels=['Percentile', 'X Value', 'Y Value'],
                             loc='upper left', bbox=[0.0, 0.8, 0.3, 0.2])
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        
        # Save the plot
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"Mouse delta scatter plot saved to {output_path}")
        return output_path

    def generate_mouse_scroll_scatter_plot(self, output_path=None, percentiles=[5, 25, 50, 75, 95]):
        """
        Generate a scatter plot of mouse scroll events with percentile bars and density-based point sizing.
        
        Args:
            output_path: Path to save the plot image. If None, will save to the output directory.
            percentiles: List of percentiles to show as horizontal and vertical bars.
            
        Returns:
            Path to the saved plot image.
        """
        if not output_path:
            output_path = os.path.join(self.output_dir, "mouse_scroll_scatter.png")
        
        # Extract all scroll events
        scroll_events = [event for event in self.events if isinstance(event, ScrollEvent)]
        if not scroll_events:
            logger.error("No scroll events found, cannot create mouse scroll scatter plot")
            return None
        
        # Extract scroll values
        scroll_x_values = [event.scroll_x for event in scroll_events]
        scroll_y_values = [event.scroll_y for event in scroll_events]
        
        # Calculate percentiles for both axes
        x_percentiles = np.percentile(scroll_x_values, percentiles)
        y_percentiles = np.percentile(scroll_y_values, percentiles)
        
        # Create the scatter plot
        plt.figure(figsize=(12, 12))
        
        # Create a symmetric log scale for better visualization of small and large values
        plt.xscale('symlog', linthresh=1)  # Linear scale for -1 < x < 1, log scale elsewhere
        plt.yscale('symlog', linthresh=1)  # Linear scale for -1 < y < 1, log scale elsewhere
        
        # Plot the grid centered at 0,0
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        # Add percentile lines for x-axis
        for i, p in enumerate(percentiles):
            plt.axvline(x=x_percentiles[i], color='red', linestyle='--', alpha=0.5, 
                       label=f'X {p}th percentile: {x_percentiles[i]:.2f}' if i == 0 else "")
        
        # Add percentile lines for y-axis
        for i, p in enumerate(percentiles):
            plt.axhline(y=y_percentiles[i], color='blue', linestyle='--', alpha=0.5,
                       label=f'Y {p}th percentile: {y_percentiles[i]:.2f}' if i == 0 else "")
        
        # Round values to group nearby points
        # Lower decimals = more aggressive grouping
        decimals = 0  # Scroll values are integers, so no need for decimals
        coords = [(round(x, decimals), round(y, decimals)) for x, y in zip(scroll_x_values, scroll_y_values)]
        
        # Count occurrences of each coordinate pair
        coord_counts = Counter(coords)
        
        # Extract unique coordinates and their counts
        unique_coords = list(coord_counts.keys())
        counts = list(coord_counts.values())
        
        # Convert to arrays for plotting
        x_coords = [coord[0] for coord in unique_coords]
        y_coords = [coord[1] for coord in unique_coords]
        
        # Scale marker sizes based on counts
        # Use a non-linear scaling to make differences more visible
        min_size = 10
        max_size = 500
        min_count = min(counts) if counts else 1
        max_count = max(counts) if counts else 1
        
        # Use a logarithmic scale for better visibility of size differences
        if max_count > min_count:
            sizes = [min_size + (max_size - min_size) * (np.log(count) - np.log(min_count)) / 
                    (np.log(max_count) - np.log(min_count)) for count in counts]
        else:
            sizes = [min_size for _ in counts]
        
        # Create scatter plot with size and color representing density
        scatter = plt.scatter(x_coords, y_coords, s=sizes, alpha=0.5, 
                             c=counts, cmap='viridis', edgecolors='none')
        
        # Add a colorbar to show the count scale
        cbar = plt.colorbar(scatter, label='Point Density')
        
        # Set plot limits to ensure it's centered and includes the data
        max_abs_x = max(abs(min(scroll_x_values)), abs(max(scroll_x_values)))
        max_abs_y = max(abs(min(scroll_y_values)), abs(max(scroll_y_values)))
        max_abs = max(max_abs_x, max_abs_y) * 1.1  # Add 10% margin
        plt.xlim(-max_abs, max_abs)
        plt.ylim(-max_abs, max_abs)
        
        # Add labels and title
        plt.title('Mouse Scroll Values with Density-Based Sizing (Log Scale)')
        plt.xlabel('Scroll X (symlog scale)')
        plt.ylabel('Scroll Y (symlog scale)')
        
        # Add statistical information to the plot
        plt.figtext(0.02, 0.02, 
                   f"X range: [{min(scroll_x_values)}, {max(scroll_x_values)}]\n"
                   f"Y range: [{min(scroll_y_values)}, {max(scroll_y_values)}]\n"
                   f"X median: {np.median(scroll_x_values)}\n"
                   f"Y median: {np.median(scroll_y_values)}\n"
                   f"Total points: {len(scroll_x_values)}\n"
                   f"Unique coordinates: {len(unique_coords)}\n"
                   f"Max density: {max_count} points",
                   bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        # Add a legend
        plt.legend(loc='upper right')
        
        # Add percentile values table
        table_text = []
        for i, p in enumerate(percentiles):
            table_text.append([f"{p}th", f"{x_percentiles[i]:.2f}", f"{y_percentiles[i]:.2f}"])
        
        the_table = plt.table(cellText=table_text,
                             colLabels=['Percentile', 'X Value', 'Y Value'],
                             loc='upper left', bbox=[0.0, 0.8, 0.3, 0.2])
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        
        # Save the plot
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"Mouse scroll scatter plot saved to {output_path}")
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
                
                # Generate mouse delta scatter plot
                delta_plot_path = self.generate_mouse_delta_scatter_plot()
                if delta_plot_path:
                    print(f"   Mouse delta scatter plot saved to {COLORS['highlight']}{os.path.basename(delta_plot_path)}{COLORS['normal']}")
        
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
            
            # Generate scroll scatter plot if save_plots is enabled
            if save_plots:
                scroll_plot_path = self.generate_mouse_scroll_scatter_plot()
                if scroll_plot_path:
                    print(f"   Mouse scroll scatter plot saved to {COLORS['highlight']}{os.path.basename(scroll_plot_path)}{COLORS['normal']}")
        
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
            # Group events by key to access the readable property
            events_by_key = defaultdict(list)
            for event in press_events:
                events_by_key[event.key].append(event)
            
            for key, count in key_counts.most_common(10):
                key_name = events_by_key[key][0].readable
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
                    key_name = f"Key {event.readable}"
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

    def tokenize_events(self):
        """
        Tokenize computer events into discrete integer tokens.
        
        The tokenization scheme:
        - 0: Null token
        - 1: Patch event token
        - 2-279: KeyPress events (128 keycodes + 11 special codes) × 2 states (up/down)
        - 280-360: Mouse movement events in a 9×9 grid
        - 361-441: Scroll events in a 9×9 grid
        
        Returns:
            List of integer tokens
        """
        logger.info("Tokenizing computer events")
        tokens = []
        
        # Constants for token ranges
        NULL_TOKEN = 0
        PATCH_TOKEN = 1
        KEY_TOKEN_START = 2
        MOVE_TOKEN_START = KEY_TOKEN_START + (128 + 11) * 2  # 280
        SCROLL_TOKEN_START = MOVE_TOKEN_START + 9 * 9  # 361
        
        # Total tokens: 1 (null) + 1 (patch) + 278 (keys) + 81 (move) + 81 (scroll) = 442
        
        for event in self.events:
            if isinstance(event, NullEvent):
                tokens.append(NULL_TOKEN)
            elif isinstance(event, PatchEvent):
                tokens.append(PATCH_TOKEN)
            elif isinstance(event, PressEvent):
                # Keys are mapped to 2-279
                # Formula: KEY_TOKEN_START + key_code * 2 + (1 if down else 0)
                down_offset = 1 if event.down else 0
                tokens.append(KEY_TOKEN_START + event.key * 2 + down_offset)
            elif isinstance(event, MoveEvent):
                # Mouse movement is quantized to a 9×9 grid (-3,-3) to (3,3) plus outliers
                # Formula: MOVE_TOKEN_START + quantized_x * 9 + quantized_y
                
                def quantize_delta(delta):
                    if delta < -3:
                        return 0
                    elif delta > 3:
                        return 8
                    else:
                        return int(delta) + 4  # Map -3..3 to 1..7
                
                quantized_x = quantize_delta(event.delta_x)
                quantized_y = quantize_delta(event.delta_y)
                
                tokens.append(MOVE_TOKEN_START + quantized_x * 9 + quantized_y)
            elif isinstance(event, ScrollEvent):
                # Scroll is quantized to a 9×9 grid (-3,-3) to (3,3) plus outliers
                # Formula: SCROLL_TOKEN_START + quantized_x * 9 + quantized_y
                
                def quantize_scroll(scroll):
                    if scroll < -3:
                        return 0
                    elif scroll > 3:
                        return 8
                    else:
                        return int(scroll) + 4  # Map -3..3 to 1..7
                
                quantized_x = quantize_scroll(event.scroll_x)
                quantized_y = quantize_scroll(event.scroll_y)
                
                tokens.append(SCROLL_TOKEN_START + quantized_x * 9 + quantized_y)
        
        self.tokenized_events = tokens
        logger.info(f"Tokenized {len(tokens)} computer events")
        return tokens
    
    def detokenize_events(self):
        """
        Convert tokenized events back to ComputerEvents to visualize the effect of quantization.
        
        Returns:
            List of ComputerEvents
        """
        if self.tokenized_events is None:
            logger.warning("No tokenized events found. Tokenizing events first.")
            self.tokenize_events()
        
        logger.info("Detokenizing events")
        detokenized_events = []
        
        # Constants for token ranges (must match those in tokenize_events)
        NULL_TOKEN = 0
        PATCH_TOKEN = 1
        KEY_TOKEN_START = 2
        MOVE_TOKEN_START = KEY_TOKEN_START + (128 + 11) * 2  # 280
        SCROLL_TOKEN_START = MOVE_TOKEN_START + 9 * 9  # 361
        
        # Build a map of original key codes to readable names from the original events
        key_to_readable = {}
        for event in self.events:
            if isinstance(event, PressEvent):
                key_to_readable[event.key] = event.readable
        
        for i, token in enumerate(self.tokenized_events):
            # Get original event for seq and dur
            orig_event = self.events[i]
            seq = orig_event.seq
            dur = orig_event.dur
            
            if token == NULL_TOKEN:
                detokenized_events.append(NullEvent(seq=seq, dur=dur))
            elif token == PATCH_TOKEN:
                # For PatchEvent, we don't have the pixel data, but we can use the grid positions
                if isinstance(orig_event, PatchEvent):
                    # Use the original event's data
                    detokenized_events.append(PatchEvent(
                        seq=seq, dur=dur, pixels=orig_event.pixels, 
                        grid_x=orig_event.grid_x, grid_y=orig_event.grid_y
                    ))
                else:
                    # This shouldn't happen in correct tokenization
                    logger.warning(f"Unexpected token {token} at position {i}")
                    detokenized_events.append(NullEvent(seq=seq, dur=dur))
            elif KEY_TOKEN_START <= token < MOVE_TOKEN_START:
                # KeyPress events
                token_offset = token - KEY_TOKEN_START
                key_code = token_offset // 2
                is_down = (token_offset % 2) == 1
                
                # Find readable name by checking both special keys and our map
                readable = "Unknown"
                
                # First check if it's a special key
                for key, value in SPECIAL_KEYS.items():
                    if value == key_code:
                        readable = key
                        break
                
                # If not found in special keys, check our map from original events
                if readable == "Unknown" and key_code in key_to_readable:
                    readable = key_to_readable[key_code]
                
                detokenized_events.append(PressEvent(
                    seq=seq, dur=dur, key=key_code, down=is_down, readable=readable
                ))
            elif MOVE_TOKEN_START <= token < SCROLL_TOKEN_START:
                # Mouse movement events
                token_offset = token - MOVE_TOKEN_START
                quantized_x = token_offset // 9
                quantized_y = token_offset % 9
                
                # Convert quantized values back to deltas
                def dequantize_delta(quantized):
                    if quantized == 0:
                        return -4.0
                    elif quantized == 8:
                        return 4.0
                    else:
                        return float(quantized - 4)
                
                delta_x = dequantize_delta(quantized_x)
                delta_y = dequantize_delta(quantized_y)
                
                # For simplicity, we'll set pos_dx and pos_dy equal to delta_x and delta_y
                detokenized_events.append(MoveEvent(
                    seq=seq, dur=dur, delta_x=delta_x, delta_y=delta_y, 
                    pos_dx=delta_x, pos_dy=delta_y
                ))
            elif SCROLL_TOKEN_START <= token < SCROLL_TOKEN_START + 81:
                # Scroll events
                token_offset = token - SCROLL_TOKEN_START
                quantized_x = token_offset // 9
                quantized_y = token_offset % 9
                
                # Convert quantized values back to scroll values
                def dequantize_scroll(quantized):
                    if quantized == 0:
                        return -4
                    elif quantized == 8:
                        return 4
                    else:
                        return quantized - 4
                
                scroll_x = dequantize_scroll(quantized_x)
                scroll_y = dequantize_scroll(quantized_y)
                
                detokenized_events.append(ScrollEvent(
                    seq=seq, dur=dur, scroll_x=scroll_x, scroll_y=scroll_y
                ))
            else:
                # Invalid token
                logger.warning(f"Invalid token {token} at position {i}")
                detokenized_events.append(NullEvent(seq=seq, dur=dur))
        
        self.detokenized_events = detokenized_events
        logger.info(f"Detokenized {len(detokenized_events)} events")
        return detokenized_events
    
    def render_quantized_video(self, output_path=None, fps=30):
        """
        Render a video reconstructed from the tokenized and then detokenized events
        to visualize the effect of quantization.
        
        Args:
            output_path: Path to save the video. If None, will save to the output directory.
            fps: Frames per second for the output video.
        
        Returns:
            Path to the saved video file.
        """
        if not output_path:
            output_path = os.path.join(self.output_dir, "quantized_video.mp4")
        
        if self.detokenized_events is None:
            logger.info("Detokenized events not found. Detokenizing events first.")
            self.detokenize_events()
        
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
        patch_events = sorted([event for event in self.detokenized_events if isinstance(event, PatchEvent)], 
                             key=lambda x: x.seq)
        
        # Group patch events by sequence number (frame)
        patch_events_by_seq = defaultdict(list)
        for event in patch_events:
            patch_events_by_seq[event.seq].append(event)
        
        # Sort all events by sequence for the overlay
        all_events = sorted(self.detokenized_events, key=lambda x: x.seq)
        
        # Initialize the screen state
        screen = np.zeros((height, width, 3), dtype=np.uint8)
        grid_height = height // PATCH_SIZE
        grid_width = width // PATCH_SIZE
        
        # Initialize event state tracking for overlay
        current_keys_down = set()
        current_keys_readable = {}  # Map key code to readable name
        mouse_pos = (width // 2, height // 2)  # Start in the middle
        
        # For visualizing the last mouse delta
        last_delta_x = 0
        last_delta_y = 0
        
        # For visualizing the last scroll
        last_scroll_x = 0
        last_scroll_y = 0
        
        # Create a visualization area for controls at the bottom
        control_panel_height = 160
        control_panel_width = width
        
        # Calculate dimensions for each section of the control panel
        section_width = control_panel_width // 3
        
        # Event history for the current frame
        current_frame_events = []
        
        logger.info(f"Reconstructing quantized video with dimensions {height}x{width} ({grid_height}x{grid_width} patches)")
        
        # Add a title banner to indicate this is the quantized version
        banner_height = 40
        banner = np.zeros((banner_height, width, 3), dtype=np.uint8)
        cv2.putText(banner, "QUANTIZED VIDEO (442 Token Types)", 
                   (width // 2 - 200, banner_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
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
                
                # Track mouse movement
                if isinstance(event, MoveEvent):
                    # Store the last delta for visualization
                    last_delta_x = event.delta_x
                    last_delta_y = event.delta_y
                    
                    # Update absolute position
                    new_x = mouse_pos[0] + event.pos_dx
                    new_y = mouse_pos[1] + event.pos_dy
                    
                    # Clamp to screen boundaries
                    new_x = max(0, min(width - 1, new_x))
                    new_y = max(0, min(height - 1, new_y))
                    
                    mouse_pos = (new_x, new_y)
                
                # Track scroll events
                elif isinstance(event, ScrollEvent):
                    last_scroll_x = event.scroll_x
                    last_scroll_y = event.scroll_y
                
                # Track key presses
                elif isinstance(event, PressEvent):
                    if event.down:
                        current_keys_down.add(event.key)
                        current_keys_readable[event.key] = event.readable
                    else:
                        if event.key in current_keys_down:
                            current_keys_down.remove(event.key)
                            if event.key in current_keys_readable:
                                del current_keys_readable[event.key]
                
                current_frame_events.append(event)
                last_event_idx += 1
            
            # Process patch events for this frame
            for event in events:
                # Convert grid position to pixel coordinates
                x1 = event.grid_x * PATCH_SIZE
                y1 = event.grid_y * PATCH_SIZE
                x2 = x1 + PATCH_SIZE
                y2 = y1 + PATCH_SIZE
                
                # Update the screen with the patch's pixels
                patch_np = event.pixels.numpy()
                screen[y1:y2, x1:x2] = patch_np
            
            # Create a copy of the current screen for drawing overlays
            frame = screen.copy()
            
            # Create control panel
            control_panel = np.zeros((control_panel_height, control_panel_width, 3), dtype=np.uint8)
            
            # ----- 1. Mouse Movement Section -----
            mouse_section = control_panel[:, 0:section_width]
            
            # Draw a grid for the mouse delta visualization
            grid_center_x = section_width // 2
            grid_center_y = control_panel_height // 2
            grid_size = 20  # Size of each grid cell
            grid_span = 3   # Number of cells in each direction from center
            
            # Draw grid lines
            for i in range(-grid_span, grid_span + 1):
                # Vertical lines
                cv2.line(mouse_section, 
                        (grid_center_x + i * grid_size, grid_center_y - grid_span * grid_size),
                        (grid_center_x + i * grid_size, grid_center_y + grid_span * grid_size),
                        (50, 50, 50), 1)
                # Horizontal lines
                cv2.line(mouse_section,
                        (grid_center_x - grid_span * grid_size, grid_center_y + i * grid_size),
                        (grid_center_x + grid_span * grid_size, grid_center_y + i * grid_size),
                        (50, 50, 50), 1)
            
            # Draw coordinate axes
            cv2.line(mouse_section,
                    (grid_center_x - grid_span * grid_size, grid_center_y),
                    (grid_center_x + grid_span * grid_size, grid_center_y),
                    (0, 255, 255), 1)  # Yellow X-axis
            cv2.line(mouse_section,
                    (grid_center_x, grid_center_y - grid_span * grid_size),
                    (grid_center_x, grid_center_y + grid_span * grid_size),
                    (0, 255, 255), 1)  # Yellow Y-axis
            
            # Draw the mouse delta arrow
            arrow_end_x = grid_center_x + int(last_delta_x * grid_size)
            arrow_end_y = grid_center_y + int(last_delta_y * grid_size)
            
            # Ensure arrow stays in bounds
            arrow_end_x = max(grid_center_x - grid_span * grid_size, min(grid_center_x + grid_span * grid_size, arrow_end_x))
            arrow_end_y = max(grid_center_y - grid_span * grid_size, min(grid_center_y + grid_span * grid_size, arrow_end_y))
            
            # Draw the arrow
            cv2.arrowedLine(mouse_section, 
                           (grid_center_x, grid_center_y), 
                           (arrow_end_x, arrow_end_y), 
                           (0, 165, 255), 2)  # Orange arrow
            
            # Add labels
            cv2.putText(mouse_section, "Mouse Delta", 
                       (grid_center_x - 50, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show the actual delta values
            cv2.putText(mouse_section, f"∆x: {last_delta_x:.1f}, ∆y: {last_delta_y:.1f}", 
                       (grid_center_x - 70, control_panel_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # ----- 2. Scroll Section -----
            scroll_section = control_panel[:, section_width:section_width*2]
            
            # Draw a grid for the scroll visualization
            scroll_center_x = section_width // 2
            scroll_center_y = control_panel_height // 2
            
            # Draw grid lines (same as mouse section)
            for i in range(-grid_span, grid_span + 1):
                # Vertical lines
                cv2.line(scroll_section, 
                        (scroll_center_x + i * grid_size, scroll_center_y - grid_span * grid_size),
                        (scroll_center_x + i * grid_size, scroll_center_y + grid_span * grid_size),
                        (50, 50, 50), 1)
                # Horizontal lines
                cv2.line(scroll_section,
                        (scroll_center_x - grid_span * grid_size, scroll_center_y + i * grid_size),
                        (scroll_center_x + grid_span * grid_size, scroll_center_y + i * grid_size),
                        (50, 50, 50), 1)
            
            # Draw coordinate axes
            cv2.line(scroll_section,
                    (scroll_center_x - grid_span * grid_size, scroll_center_y),
                    (scroll_center_x + grid_span * grid_size, scroll_center_y),
                    (0, 255, 255), 1)  # Yellow X-axis
            cv2.line(scroll_section,
                    (scroll_center_x, scroll_center_y - grid_span * grid_size),
                    (scroll_center_x, scroll_center_y + grid_span * grid_size),
                    (0, 255, 255), 1)  # Yellow Y-axis
            
            # Draw the scroll indicator
            scroll_indicator_x = scroll_center_x + last_scroll_x * grid_size
            scroll_indicator_y = scroll_center_y + last_scroll_y * grid_size
            
            # Ensure indicator stays in bounds
            scroll_indicator_x = max(scroll_center_x - grid_span * grid_size, 
                                    min(scroll_center_x + grid_span * grid_size, scroll_indicator_x))
            scroll_indicator_y = max(scroll_center_y - grid_span * grid_size, 
                                    min(scroll_center_y + grid_span * grid_size, scroll_indicator_y))
            
            # Draw a circle to represent scroll position
            cv2.circle(scroll_section, 
                      (int(scroll_indicator_x), int(scroll_indicator_y)), 
                      5, (0, 255, 0), -1)  # Green circle
            
            # Add labels
            cv2.putText(scroll_section, "Scroll", 
                       (scroll_center_x - 25, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show the actual scroll values
            cv2.putText(scroll_section, f"x: {last_scroll_x}, y: {last_scroll_y}", 
                       (scroll_center_x - 50, control_panel_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # ----- 3. Keyboard Section -----
            key_section = control_panel[:, section_width*2:section_width*3]
            
            # Draw header
            cv2.putText(key_section, "Active Keys", 
                       (section_width // 2 - 40, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Display active keys with both ID and readable name
            if current_keys_down:
                y_pos = 50
                for key_id in current_keys_down:
                    readable = current_keys_readable.get(key_id, "Unknown")
                    key_text = f"Key {key_id} ({readable})"
                    
                    # Wrap text if too long
                    if len(key_text) > 25:
                        key_text = key_text[:22] + "..."
                    
                    cv2.putText(key_section, key_text, 
                               (20, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_pos += 25
                    
                    # Limit to prevent overflow
                    if y_pos > control_panel_height - 30:
                        cv2.putText(key_section, "...", 
                                   (20, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        break
            else:
                cv2.putText(key_section, "No keys pressed", 
                           (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Combine control panel sections
            control_panel[:, 0:section_width] = mouse_section
            control_panel[:, section_width:section_width*2] = scroll_section
            control_panel[:, section_width*2:section_width*3] = key_section
            
            # Draw separator lines between sections
            cv2.line(control_panel, 
                    (section_width, 0), 
                    (section_width, control_panel_height), 
                    (100, 100, 100), 1)
            cv2.line(control_panel, 
                    (section_width*2, 0), 
                    (section_width*2, control_panel_height), 
                    (100, 100, 100), 1)
            
            # Draw a mouse cursor at the current position
            cv2.circle(frame, (int(mouse_pos[0]), int(mouse_pos[1])), 5, (0, 165, 255), -1)
            
            # Add the title banner
            frame_with_banner = np.vstack([banner, frame])
            
            # Add the control panel at the bottom
            frame_with_controls = np.vstack([frame_with_banner, control_panel])
            
            # Resize back to original dimensions (removing the banner and control panel)
            frame_final = cv2.resize(frame_with_controls, (width, height))
            
            # Write the frame
            video_writer.write(frame_final)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        video_writer.release()
        logger.info(f"Quantized video saved to {output_path} with {frame_count} frames")
        return output_path

    def visualize_token_distribution(self, output_path=None):
        """
        Generate a visualization of the token distribution to understand the tokenization.
        
        Args:
            output_path: Path to save the visualization. If None, will save to the output directory.
            
        Returns:
            Path to the saved visualization file.
        """
        if not output_path:
            output_path = os.path.join(self.output_dir, "token_distribution.png")
        
        if self.tokenized_events is None:
            logger.info("Tokenized events not found. Tokenizing events first.")
            self.tokenize_events()
        
        # Constants for token ranges
        NULL_TOKEN = 0
        PATCH_TOKEN = 1
        KEY_TOKEN_START = 2
        MOVE_TOKEN_START = KEY_TOKEN_START + (128 + 11) * 2  # 280
        SCROLL_TOKEN_START = MOVE_TOKEN_START + 9 * 9  # 361
        
        # Count token frequencies
        token_counts = Counter(self.tokenized_events)
        
        # Create categories for the tokens
        categories = {
            "Null": [NULL_TOKEN],
            "Keys (down)": range(KEY_TOKEN_START, MOVE_TOKEN_START, 2),
            "Keys (up)": range(KEY_TOKEN_START + 1, MOVE_TOKEN_START, 2),
            "Mouse Movement": range(MOVE_TOKEN_START, SCROLL_TOKEN_START),
            "Scroll": range(SCROLL_TOKEN_START, SCROLL_TOKEN_START + 81)
        }
        
        # Count tokens by category
        category_counts = {}
        for category, token_range in categories.items():
            category_counts[category] = sum(token_counts.get(token, 0) for token in token_range)
        
        # Filter tokens to exclude patch tokens for visualization
        tokens_without_patches = [token for token in self.tokenized_events if token != PATCH_TOKEN]
        
        # Most common tokens (excluding patch tokens)
        most_common = [item for item in token_counts.most_common(15) if item[0] != PATCH_TOKEN][:10]
        
        # Create figures for visualization
        plt.figure(figsize=(15, 10))
        
        # 1. Token categories pie chart
        plt.subplot(2, 2, 1)
        plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        plt.title('Token Categories Distribution')
        
        # 2. Top 10 most common tokens (excluding patch tokens)
        plt.subplot(2, 2, 2)
        token_labels = []
        for token, count in most_common:
            if token == NULL_TOKEN:
                token_labels.append("Null")
            elif KEY_TOKEN_START <= token < MOVE_TOKEN_START:
                key_code = (token - KEY_TOKEN_START) // 2
                state = "down" if (token - KEY_TOKEN_START) % 2 == 1 else "up"
                # Get key name if it's a special key
                key_name = "Unknown"
                for k, v in SPECIAL_KEYS.items():
                    if v == key_code:
                        key_name = k
                        break
                token_labels.append(f"Key {key_name if key_name != 'Unknown' else key_code} ({state})")
            elif MOVE_TOKEN_START <= token < SCROLL_TOKEN_START:
                token_offset = token - MOVE_TOKEN_START
                x = token_offset // 9
                y = token_offset % 9
                x_val = x - 4 if x < 4 else ">" + str(x - 4) if x > 4 else "0"
                y_val = y - 4 if y < 4 else ">" + str(y - 4) if y > 4 else "0"
                token_labels.append(f"Move ({x_val}, {y_val})")
            elif SCROLL_TOKEN_START <= token < SCROLL_TOKEN_START + 81:
                token_offset = token - SCROLL_TOKEN_START
                x = token_offset // 9
                y = token_offset % 9
                x_val = x - 4 if x < 4 else ">" + str(x - 4) if x > 4 else "0"
                y_val = y - 4 if y < 4 else ">" + str(y - 4) if y > 4 else "0"
                token_labels.append(f"Scroll ({x_val}, {y_val})")
            else:
                token_labels.append(f"Token {token}")
        
        plt.barh(range(len(most_common)), [count for _, count in most_common])
        plt.yticks(range(len(most_common)), token_labels)
        plt.title('Top 10 Most Common Tokens (Excluding Patch Tokens)')
        plt.xlabel('Count')
        
        # 3. Token histogram (excluding patch tokens)
        plt.subplot(2, 1, 2)
        tokens_without_patches_or_null = [token for token in tokens_without_patches if token != NULL_TOKEN]
        plt.hist(tokens_without_patches_or_null, bins=442, alpha=0.75)
        plt.title('Token Distribution Histogram (Excluding Patch and Null Tokens)')
        plt.xlabel('Token ID')
        plt.ylabel('Frequency')
        
        # Add vertical lines to separate token categories
        category_boundaries = [KEY_TOKEN_START, MOVE_TOKEN_START, SCROLL_TOKEN_START, SCROLL_TOKEN_START + 81]
        category_names = ["", "Keys", "Mouse", "Scroll", ""]
        
        for i in range(1, len(category_boundaries)):
            plt.axvline(x=category_boundaries[i], color='r', linestyle='--', alpha=0.5)
            # Add annotations for the categories
            midpoint = (category_boundaries[i-1] + category_boundaries[i]) / 2
            plt.text(midpoint, plt.ylim()[1] * 0.9, category_names[i], 
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Token distribution visualization saved to {output_path}")
        
        # Print some statistics about the tokenization
        total_tokens = len(self.tokenized_events)
        unique_tokens = len(token_counts)
        patch_tokens = token_counts.get(PATCH_TOKEN, 0)
        
        logger.info(f"Total tokens: {total_tokens}")
        logger.info(f"Unique tokens used: {unique_tokens} out of 442 possible tokens")
        logger.info(f"Patch tokens: {patch_tokens} ({patch_tokens/total_tokens*100:.2f}%)")
        
        # Print token category statistics
        for category, count in category_counts.items():
            logger.info(f"{category}: {count} tokens ({count/total_tokens*100:.2f}%)")
        
        return output_path
    
    def export_token_sequence(self, output_path=None):
        """
        Export the token sequence to a text file for use in other applications.
        
        Args:
            output_path: Path to save the token sequence. If None, will save to the output directory.
            
        Returns:
            Path to the saved token sequence file.
        """
        if not output_path:
            output_path = os.path.join(self.output_dir, "token_sequence.txt")
        
        if self.tokenized_events is None:
            logger.info("Tokenized events not found. Tokenizing events first.")
            self.tokenize_events()
        
        with open(output_path, 'w') as f:
            # Write header with information about the tokenization
            f.write("# Timeline token sequence\n")
            f.write("# Token format: 442 possible token types\n")
            f.write("# - 0: Null token\n")
            f.write("# - 1: Patch event token\n")
            f.write("# - 2-279: KeyPress events (128 keycodes + 11 special codes) × 2 states (up/down)\n")
            f.write("# - 280-360: Mouse movement events in a 9×9 grid\n")
            f.write("# - 361-441: Scroll events in a 9×9 grid\n\n")
            
            # Write each token on a new line
            for token in self.tokenized_events:
                f.write(f"{token}\n")
        
        logger.info(f"Token sequence exported to {output_path}")
        return output_path
        
    def export_tensors(self, patches_path=None, events_path=None):
        """
        Export tensor data for patches and events in the requested format.
        
        Args:
            patches_path: Path to save the patches tensor file. If None, will save to patches.pt in the output directory.
            events_path: Path to save the events tensor file. If None, will save to events.pt in the output directory.
            
        Returns:
            Tuple of paths to the saved tensor files (patches_path, events_path).
        """
        import torch
        
        if not patches_path:
            patches_path = os.path.join(self.output_dir, "patches.pt")
        
        if not events_path:
            events_path = os.path.join(self.output_dir, "events.pt")
        
        # Ensure events are tokenized
        if self.tokenized_events is None:
            logger.info("Tokenized events not found. Tokenizing events first.")
            self.tokenize_events()
        
        # Extract patch events for the patches.pt file
        patch_events = [event for event in self.events if isinstance(event, PatchEvent)]
        
        if not patch_events:
            logger.warning("No patch events found. Cannot create patches.pt")
            return None, None
        
        # Get dimensions from the first patch
        first_patch = patch_events[0].pixels
        patch_size = first_patch.shape[0]  # Assuming square patches
        channels = first_patch.shape[2]
        
        # Create tensors for patches.pt
        num_patches = len(patch_events)
        
        # Create pixels tensor: (num_frames) x (patch_size^2 * channels)
        flattened_size = patch_size * patch_size * channels
        pixels_tensor = torch.zeros((num_patches, flattened_size))
        
        # Create position info tensor: (num_frames) x (temporal_seq, x, y)
        position_tensor = torch.zeros((num_patches, 3), dtype=torch.long)
        
        # Fill the tensors
        for i, patch in enumerate(patch_events):
            # Flatten patch pixels from [H, W, C] to [H*W*C]
            flat_pixels = patch.pixels.reshape(-1)
            pixels_tensor[i] = flat_pixels
            
            # Store position information
            position_tensor[i, 0] = patch.seq  # temporal sequence
            position_tensor[i, 1] = patch.grid_x  # x position
            position_tensor[i, 2] = patch.grid_y  # y position
        
        # Constants for token identification
        PATCH_TOKEN = 1
        
        # Filter out patch tokens from the events tensor
        non_patch_indices = [i for i, token in enumerate(self.tokenized_events) if token != PATCH_TOKEN]
        filtered_tokens = [self.tokenized_events[i] for i in non_patch_indices]
        filtered_events = [self.events[i] for i in non_patch_indices]
        
        # Create tensor for events.pt: (num_non_patch_events) x (event_type_token, temporal_seq)
        num_events = len(filtered_tokens)
        events_tensor = torch.zeros((num_events, 2), dtype=torch.long)
        
        # Fill the events tensor (excluding patch events)
        for i, (token, event) in enumerate(zip(filtered_tokens, filtered_events)):
            events_tensor[i, 0] = token  # event type token
            events_tensor[i, 1] = event.seq  # temporal sequence
        
        # Save the tensors
        torch.save({
            'pixels': pixels_tensor,
            'position': position_tensor
        }, patches_path)
        
        torch.save({
            'tokens': events_tensor
        }, events_path)
        
        logger.info(f"Patches data saved to {patches_path} ({num_patches} patches)")
        logger.info(f"Events data saved to {events_path} ({num_events} non-patch events)")
        
        return patches_path, events_path
        
    def validate_exported_tensors(self, patches_path=None, events_path=None, output_dir=None):
        """
        Decode and validate the exported tensor files by creating visualizations.
        
        Args:
            patches_path: Path to the patches tensor file. If None, will use patches.pt in the output directory.
            events_path: Path to the events tensor file. If None, will use events.pt in the output directory.
            output_dir: Directory to save validation outputs. If None, will use the class output directory.
            
        Returns:
            Tuple of paths to the saved validation files (frame_validation_path, events_validation_path).
        """
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        
        if not patches_path:
            patches_path = os.path.join(self.output_dir, "patches.pt")
        
        if not events_path:
            events_path = os.path.join(self.output_dir, "events.pt")
        
        if not output_dir:
            output_dir = self.output_dir
        
        if not os.path.exists(patches_path) or not os.path.exists(events_path):
            logger.error(f"Tensor files not found. Run export_tensors first.")
            return None, None
        
        # Load the tensor files
        patches_data = torch.load(patches_path)
        events_data = torch.load(events_path)
        
        # Get the dimensions of the data
        pixels_tensor = patches_data['pixels']
        position_tensor = patches_data['position']
        events_tensor = events_data['tokens']
        
        logger.info(f"Loaded patches tensor with shape {pixels_tensor.shape}")
        logger.info(f"Loaded position tensor with shape {position_tensor.shape}")
        logger.info(f"Loaded events tensor with shape {events_tensor.shape} (non-patch events only)")
        
        # 1. Validate patches: Reconstruct a sample frame from patches
        
        # Find frame events in the original data to get dimensions
        frame_events = [event for event in self.timeline_events if isinstance(event, FrameEvent)]
        if not frame_events:
            logger.error("No frame events found, cannot reconstruct for validation")
            return None, None
        
        # Get dimensions from the first frame
        first_frame = frame_events[0].pixels
        height, width = first_frame.shape[0], first_frame.shape[1]
        
        # Determine patch dimensions
        patch_events = [event for event in self.events if isinstance(event, PatchEvent)]
        if not patch_events:
            logger.error("No patch events found, cannot reconstruct for validation")
            return None, None
        
        first_patch = patch_events[0].pixels
        patch_size = first_patch.shape[0]  # Assuming square patches
        channels = first_patch.shape[2]
        grid_height = height // patch_size
        grid_width = width // patch_size
        
        # Group patches by seq (frame number)
        patches_by_seq = defaultdict(list)
        for i in range(position_tensor.shape[0]):
            seq = position_tensor[i, 0].item()
            x = position_tensor[i, 1].item()
            y = position_tensor[i, 2].item()
            pixels = pixels_tensor[i].reshape(patch_size, patch_size, channels)
            patches_by_seq[seq].append((x, y, pixels))
        
        # Take the first few sequences to visualize
        sequences_to_show = min(6, len(patches_by_seq))
        sample_seqs = sorted(list(patches_by_seq.keys()))[:sequences_to_show]
        
        plt.figure(figsize=(15, 5 * sequences_to_show))
        
        for idx, seq in enumerate(sample_seqs):
            # Initialize a blank canvas for this frame
            frame = np.zeros((height, width, channels))
            
            # Place patches on the canvas
            for x, y, pixels in patches_by_seq[seq]:
                y_start = y * patch_size
                y_end = y_start + patch_size
                x_start = x * patch_size
                x_end = x_start + patch_size
                
                # Ensure we don't go out of bounds
                if y_end <= height and x_end <= width:
                    frame[y_start:y_end, x_start:x_end] = pixels.numpy()
            
            # Convert to uint8 for display
            frame = (frame * 255).astype(np.uint8)
            
            plt.subplot(sequences_to_show, 1, idx + 1)
            plt.imshow(frame)
            plt.title(f"Reconstructed Frame from Sequence {seq}")
            plt.axis('off')
        
        # Save the visualization
        frame_validation_path = os.path.join(output_dir, "frame_validation.png")
        plt.tight_layout()
        plt.savefig(frame_validation_path)
        plt.close()
        
        # 2. Validate events: Decode tokens to event types
        
        # Constants for token ranges (must match those in tokenize_events)
        NULL_TOKEN = 0
        PATCH_TOKEN = 1  # Note: PATCH_TOKEN should not appear in the events tensor now
        KEY_TOKEN_START = 2
        MOVE_TOKEN_START = KEY_TOKEN_START + (128 + 11) * 2  # 280
        SCROLL_TOKEN_START = MOVE_TOKEN_START + 9 * 9  # 361
        
        # Decode a sample of tokens
        max_events_to_show = 100
        decoded_events = []
        
        # Show at most max_events_to_show events
        for i in range(min(max_events_to_show, events_tensor.shape[0])):
            token = events_tensor[i, 0].item()
            seq = events_tensor[i, 1].item()
            
            if token == NULL_TOKEN:
                event_type = "Null"
            elif token == PATCH_TOKEN:
                # This should not happen as we've filtered out patch tokens
                event_type = "Patch (unexpected - should be in patches.pt)"
            elif KEY_TOKEN_START <= token < MOVE_TOKEN_START:
                token_offset = token - KEY_TOKEN_START
                key_code = token_offset // 2
                is_down = (token_offset % 2) == 1
                state = "Down" if is_down else "Up"
                
                # Try to get readable key name
                readable = "Unknown"
                for key, value in SPECIAL_KEYS.items():
                    if value == key_code:
                        readable = key
                        break
                
                event_type = f"Key({readable}, {state})"
            elif MOVE_TOKEN_START <= token < SCROLL_TOKEN_START:
                token_offset = token - MOVE_TOKEN_START
                quantized_x = token_offset // 9
                quantized_y = token_offset % 9
                
                # Dequantize deltas
                def dequantize_delta(quantized):
                    if quantized == 0:
                        return -4.0
                    elif quantized == 8:
                        return 4.0
                    else:
                        return float(quantized - 4)
                
                delta_x = dequantize_delta(quantized_x)
                delta_y = dequantize_delta(quantized_y)
                
                event_type = f"Move(dx={delta_x:.1f}, dy={delta_y:.1f})"
            elif SCROLL_TOKEN_START <= token < SCROLL_TOKEN_START + 81:
                token_offset = token - SCROLL_TOKEN_START
                quantized_x = token_offset // 9
                quantized_y = token_offset % 9
                
                # Dequantize scroll values
                def dequantize_scroll(quantized):
                    if quantized == 0:
                        return -4
                    elif quantized == 8:
                        return 4
                    else:
                        return quantized - 4
                
                scroll_x = dequantize_scroll(quantized_x)
                scroll_y = dequantize_scroll(quantized_y)
                
                event_type = f"Scroll(sx={scroll_x}, sy={scroll_y})"
            else:
                event_type = f"Unknown({token})"
            
            decoded_events.append((seq, token, event_type))
        
        # Save the decoded events to a text file
        events_validation_path = os.path.join(output_dir, "events_validation.txt")
        with open(events_validation_path, 'w') as f:
            f.write("Sequence\tToken\tDecoded Event\n")
            f.write("-" * 60 + "\n")
            f.write("Note: Patch events are stored separately in patches.pt\n\n")
            for seq, token, event_type in decoded_events:
                f.write(f"{seq}\t{token}\t{event_type}\n")
        
        logger.info(f"Frame validation image saved to {frame_validation_path}")
        logger.info(f"Events validation report saved to {events_validation_path}")
        
        return frame_validation_path, events_validation_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, help="Directory to save analysis outputs (default: ./data/{chunkname}_analysis)")
    
    parser.add_argument("--no-plots", action="store_true", help="Disable analytics plots")
    parser.add_argument("--no-highlight", action="store_true", help="Disable highlighting of active patches in the reconstructed video")
    parser.add_argument("--no-events-overlay", action="store_true", help="Disable keyboard and mouse events overlay on the reconstructed video")
    parser.add_argument("--no-srt", action="store_true", help="Disable generation of SRT subtitle file with keyboard and mouse events grouped by frame")
    parser.add_argument("--generate-quantized", action="store_true", help="Generate a video showing the quantization effects of tokenization")
    parser.add_argument("--no-token-export", action="store_true", help="Disable exporting token sequence to text file when using --generate-quantized")
    
    parser.add_argument("--highlight-color", type=str, default="blue", help="Color for highlighting active patches (orange, red, blue, yellow, cyan, magenta, green)")
    parser.add_argument("--highlight-opacity", type=float, default=0.1, help="Opacity of the highlight overlay (0.0 to 1.0)")
    parser.add_argument("--srt-filename", type=str, default="timeline.srt", help="Filename for the generated SRT file (default: timeline.srt)")
    parser.add_argument("--srt-fps", type=float, default=30.0, help="Frame rate to use for SRT timecodes (default: auto-detect from video, fallback to 30.0)")
    parser.add_argument("--export-tensors", action="store_true", help="Export patches.pt and events.pt tensor files for ML training")
    parser.add_argument("--validate-tensors", action="store_true", help="Validate exported tensor files by generating visualizations")
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
    
    # Generate tokenized/quantized video if requested
    if args.generate_quantized:
        # First tokenize events
        timeline.tokenize_events()
        
        # Generate visualization of token distribution
        token_dist_path = timeline.visualize_token_distribution()
        if token_dist_path:
            print(f"Token distribution visualization saved to {token_dist_path}")
        
        # Export token sequence unless disabled
        if not args.no_token_export:
            token_seq_path = timeline.export_token_sequence()
            if token_seq_path:
                print(f"Token sequence exported to {token_seq_path}")
        
        # Then render the quantized video
        quantized_video_path = timeline.render_quantized_video()
        if quantized_video_path:
            print(f"Quantized video saved to {quantized_video_path} (shows effects of 442-token quantization)")
    
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
    
    # Export tensor files if requested
    if args.export_tensors:
        patches_path, events_path = timeline.export_tensors()
        if patches_path and events_path:
            print(f"Patches tensor saved to {patches_path}")
            print(f"Events tensor saved to {events_path}")
    
    # Validate tensor files if requested
    if args.validate_tensors:
        if not args.export_tensors:
            print("Loading previously exported tensor files for validation...")
        frame_validation_path, events_validation_path = timeline.validate_exported_tensors()
        if frame_validation_path and events_validation_path:
            print(f"Frame validation image saved to {frame_validation_path}")
            print(f"Events validation report saved to {events_validation_path}")