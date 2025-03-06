import time
import glob
import multiprocessing as mp
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchcodec.decoders import VideoDecoder
from queue import Empty
from functools import partial
import sys
import os
from loggy3 import chunk_tokenizer, token_to_readable, print_token_sequence

class PatchDeduplicator(nn.Module):
    def __init__(self, patch_size=16, similarity_threshold=0.01):
        super().__init__()
        self.patch_size = patch_size
        self.similarity_threshold = similarity_threshold

    def forward(self, frames_tensor):
        S, C, H, W = frames_tensor.shape
        
        original_dtype = frames_tensor.dtype
        if frames_tensor.dtype != T.uint8:
            if frames_tensor.max() <= 1.0:
                frames_tensor = (frames_tensor * 255).to(T.uint8)
            else:
                frames_tensor = frames_tensor.to(T.uint8)
        
        input_threshold = int(self.similarity_threshold * 255) if self.similarity_threshold < 1 else self.similarity_threshold
        
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Patch size {self.patch_size} must divide frame dimensions evenly. "
                f"Got frame shape (H={H}, W={W})"
            )
        
        n_patches_h = H // self.patch_size
        n_patches_w = W // self.patch_size
        total_patches = n_patches_h * n_patches_w
        
        # [S, C, H, W] -> [S, C, n_patches_h, patch_size, n_patches_w, patch_size]
        patches = frames_tensor.reshape(S, C, n_patches_h, self.patch_size, n_patches_w, self.patch_size)
        
        y_coords = T.arange(n_patches_h, device=frames_tensor.device).view(n_patches_h, 1).repeat(1, n_patches_w).reshape(-1)
        x_coords = T.arange(n_patches_w, device=frames_tensor.device).view(1, n_patches_w).repeat(n_patches_h, 1).reshape(-1)
        
        coords = T.stack([x_coords, y_coords], dim=1).to(T.uint8)
        
        patch_flat_size = C * self.patch_size * self.patch_size
        first_frame_patches = []
        
        for idx in range(total_patches):
            h_idx = idx // n_patches_w
            w_idx = idx % n_patches_w
            
            patch = patches[0, :, h_idx, :, w_idx, :].reshape(-1)  # [C*patch_size*patch_size]
            first_frame_patches.append(patch)
            
        first_frame_patches = T.stack(first_frame_patches)
        
        first_frame_with_pos = T.cat([first_frame_patches, coords], dim=1)
        
        compressed_frames = [first_frame_with_pos]
        
        patches_float = patches.float()
        downsampled = patches_float.mean(dim=(3, 5))  # [S, C, n_patches_h, n_patches_w]
        
        patch_diffs = (downsampled[1:] - downsampled[:-1]).abs().mean(dim=1)  # [S-1, n_patches_h, n_patches_w]
        
        for t in range(1, S):
            active_mask = patch_diffs[t-1] > input_threshold  # [n_patches_h, n_patches_w]
            y_indices, x_indices = T.where(active_mask)
            
            if len(y_indices) > 0:
                active_patches = []
                for i in range(len(y_indices)):
                    h_idx = y_indices[i]
                    w_idx = x_indices[i]
                    patch = patches[t, :, h_idx, :, w_idx, :].reshape(-1)
                    active_patches.append(patch)
                
                active_patches = T.stack(active_patches)
                
                active_coords = T.stack([x_indices, y_indices], dim=1).to(T.uint8)
                
                frame_with_pos = T.cat([active_patches, active_coords], dim=1)
                compressed_frames.append(frame_with_pos)
            else:
                compressed_frames.append(T.empty((0, first_frame_with_pos.size(1)), 
                                                   dtype=T.uint8, 
                                                   device=frames_tensor.device))
        
        return compressed_frames
    
    def reconstruct(self, compressed_frames):
        C = 3
        S = len(compressed_frames)
        H = 720
        W = 1280
        
        reconstructed = T.zeros((S, C, H, W), dtype=T.uint8, 
                                   device=compressed_frames[0].device if len(compressed_frames) > 0 else 'cpu')
        
        first_frame = compressed_frames[0]
        for i in range(first_frame.size(0)):
            patch = first_frame[i]
            patch_data = patch[:-2]
            x, y = patch[-2].item(), patch[-1].item()
            
            patch_reshaped = patch_data.reshape(C, self.patch_size, self.patch_size)
            
            h_start = y * self.patch_size
            w_start = x * self.patch_size
            reconstructed[0, :, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size] = patch_reshaped
        
        for t in range(1, S):
            reconstructed[t] = reconstructed[t-1].clone()
            frame_patches = compressed_frames[t]
            for i in range(frame_patches.size(0)):
                patch = frame_patches[i]
                patch_data = patch[:-2]
                x, y = patch[-2].item(), patch[-1].item()
                
                patch_reshaped = patch_data.reshape(C, self.patch_size, self.patch_size)
                
                h_start = y * self.patch_size
                w_start = x * self.patch_size
                reconstructed[t, :, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size] = patch_reshaped
        
        return reconstructed

deduplicator = PatchDeduplicator(patch_size=16, similarity_threshold=0.01)

def worker_process(task_queue, output_queue, frames_per_clip, done_event):
    while not done_event.is_set():
        try:
            chunk_path = task_queue.get(block=False)
        except Empty:
            if task_queue.empty():
                break
            time.sleep(0.1)
            continue

        chunk_tokens = chunk_tokenizer(chunk_path, os_type=None)
        video_paths = []
        display_ids = set()
        try:
            for item in os.listdir(chunk_path):
                display_path = os.path.join(chunk_path, item)
                display_prefix = "display_"
                prechars = len(display_prefix)
                if os.path.isdir(display_path) and item.startswith(display_prefix):
                    video_path = os.path.join(display_path, "video.mp4")
                    display_id = item[prechars:]
                    if display_id not in display_ids:
                        if len(display_ids) < 4:
                            display_ids.add(display_id)
                        else:
                            continue
                    if os.path.exists(video_path):
                        video_paths.append(video_path)
        except Exception as e:
            print(f"Error finding video in {chunk_path}: {e}")
            continue

        frame_tensors = []
        for video_path in video_paths:
            try:
                decoder = VideoDecoder(video_path, num_ffmpeg_threads=1, device="cpu")
                total_frames = decoder.metadata.num_frames
                frames_tensor = decoder[:].cpu()
                frames_tensor = pad_frames(frames_tensor)
                compressed_frames = deduplicator(frames_tensor)
                frame_tensors.append(compressed_frames)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
        output_queue.put({
            "chunk_path": chunk_path,
            "tokens": chunk_tokens,
            "frame_tensors": frame_tensors
        })

def pad_frames(frames_tensor):
    # frames_tensor shape: (M, C, H, W)
    M, C, H, W = frames_tensor.shape
    target_h = 720
    
    # First resize height to 720
    if H != target_h:
        # Calculate new width maintaining aspect ratio
        new_w = int(W * (target_h / H))
        frames_tensor = F.interpolate(frames_tensor, size=(target_h, new_w), mode='bilinear', align_corners=False)
        _, _, H, W = frames_tensor.shape
    
    # Now handle width - target is 16:9 aspect ratio for 720p
    target_w = 1280
    
    if W > target_w:
        # Crop width if larger than target
        excess_w = W - target_w
        crop_left = excess_w // 2
        frames_tensor = frames_tensor[:, :, :, crop_left:crop_left+target_w]
    elif W < target_w:
        # Pad width if smaller than target
        pad_w = target_w - W
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        print(f"Padding width with: left={pad_left}, right={pad_right}")
        
        frames_tensor = F.pad(
            frames_tensor, (pad_left, pad_right, 0, 0)
        )
    
    return frames_tensor
class VideoStreamLoader:
    def __init__(self, chunk_paths, frames_per_clip, batch_size, workers=16):
        self.chunk_paths = chunk_paths
        self.frames_per_clip = frames_per_clip
        self.batch_size = batch_size
        self.workers = workers

        self.manager = mp.Manager()
        self.task_queue = self.manager.Queue()
        self.output_queue = self.manager.Queue(maxsize=32)
        self.done_event = self.manager.Event()
        
        for path in self.chunk_paths:
            self.task_queue.put(path)

        self.processes = []
        worker_func = partial(
            worker_process, 
            self.task_queue, 
            self.output_queue, 
            self.frames_per_clip, 
            self.done_event
        )
        
        for _ in range(self.workers):
            p = mp.Process(target=worker_func, daemon=True)
            p.start()
            self.processes.append(p)

    def __iter__(self):
        return self

    def __next__(self):
        batch_clips = []
        try:
            for _ in range(self.batch_size):
                if self.output_queue.empty() and self.task_queue.empty():
                    if len(batch_clips) == 0:
                        self.cleanup()
                        raise StopIteration
                    break
                try:
                    clip = self.output_queue.get(timeout=10)
                    batch_clips.append(clip)
                except Exception:
                    if self.output_queue.empty() and self.task_queue.empty():
                        if len(batch_clips) == 0:
                            self.cleanup()
                            raise StopIteration
                        break
            
            if not batch_clips:
                self.cleanup()
                raise StopIteration
                
            return batch_clips
        except Exception as e:
            self.cleanup()
            raise e

    def cleanup(self):
        self.done_event.set()
        
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                
        for p in self.processes:
            p.join(timeout=1.0)
        
        self.manager.shutdown()

def render_video_with_tokens(clip, output_path):
    """
    Render a video with tokens overlaid on frames.
    
    Args:
        clip: A tuple of (tensor, tokens) where tensor is a frames tensor and tokens is a list of (seq, token) pairs
        output_path: Path to save the rendered video
    """
    import numpy as np
    import cv2
    import torch
    
    # Unpack the clip
    first_tensor, tokens = clip
    
    # Setup video writer
    height, width = first_tensor.shape[2], first_tensor.shape[3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    # Sort tokens by sequence number
    sorted_tokens = sorted(tokens, key=lambda x: x[0])
    
    # Map tokens to frames based on sequence
    token_map = {}
    curr_frame = -1
    token_map[curr_frame] = []
    for seq, token in sorted_tokens:
        if 'Display 1' in token_to_readable(token):
            curr_frame += 1
            token_map[curr_frame] = []
        token_map[curr_frame].append((seq, token))
    
    # Process and write frames
    for i in range(first_tensor.shape[0]):
        # Get the frame as a numpy array - ensuring it's in the right format for OpenCV
        if torch.is_tensor(first_tensor[i]):
            # Clone and detach to avoid modifying the original tensor
            frame_tensor = first_tensor[i].clone().detach().cpu()
            # Convert from CxHxW to HxWxC and ensure it's contiguous
            frame = frame_tensor.permute(1, 2, 0).contiguous().numpy()
        else:
            # If it's already a numpy array, just make a copy
            frame = np.array(first_tensor[i])
        
        # Ensure we have the right data type
        frame = frame.astype(np.uint8)
        
        # Make sure the array is contiguous in memory
        frame = np.ascontiguousarray(frame)
        
        # Add tokens to the frame
        if i in token_map:
            frame_tokens = token_map[i]
            # Start from bottom right
            y_offset = height - 30  # Start from bottom with some margin
            for seq, token in frame_tokens:
                token_text = token_to_readable(token)
                
                # Replace emojis with text that OpenCV can render
                token_text = (token_text.replace("🎥", "[VIDEO]")
                                        .replace("🔑", "[KEY]")
                                        .replace("📜", "[SCROLL]")
                                        .replace("🐭", "[MOUSE]"))
                
                token_text += f" pos: {seq}"
                
                # Get text size for positioning
                text_size = cv2.getTextSize(token_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Calculate x position from right side
                x_position = width - text_size[0] - 10  # 10px margin from right
                
                # Add semi-transparent background for text
                cv2.rectangle(
                    frame, 
                    (x_position - 10, y_offset - 20), 
                    (x_position + text_size[0], y_offset + 5), 
                    (0, 0, 0), 
                    -1
                )
                # Add text
                cv2.putText(
                    frame, 
                    token_text, 
                    (x_position, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
                y_offset -= 30  # Move up for next token
        
        # Write the frame to video
        out.write(frame)
    
    # Release the video writer
    out.release()
    print(f"Video with overlaid tokens saved to {output_path}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    session_dir = sys.argv[1]
    chunk_folders = glob.glob(os.path.join(session_dir, "chunk_*"))
    frames_per_clip = 30*10
    batch_size = 32
    frames_per_batch = frames_per_clip * batch_size
    loader = VideoStreamLoader(chunk_folders, frames_per_clip=frames_per_clip, batch_size=batch_size, workers=32)
    t0 = time.time()
    last_log = t0
    total_frames = 0
    total_batches = 0
    
    try:
        for batch in loader:
            total_batches += 1
            total_frames += frames_per_batch
            
            # Render the first batch as a video with tokens
            if total_batches == 1 and len(batch) > 0:
                print("Rendering clips with tokens overlaid...")
                output_dir = 'data/clip_visualizations'
                os.makedirs(output_dir, exist_ok=True)
                
                for i, item in enumerate(batch):
                    chunk_name = os.path.basename(item["chunk_path"])
                    for j, frame_tensor in enumerate(item["frame_tensors"]):
                        frame_tensor = deduplicator.reconstruct(frame_tensor)
                        print(frame_tensor.shape, frame_tensor.dtype, frame_tensor.min(), frame_tensor.max())
                        output_path = os.path.join(output_dir, f"{chunk_name}_video_{i}_{j}_with_tokens.mp4")
                        print(f'rendering chunk {item["chunk_path"]} video {i} {j}')
                        render_video_with_tokens((frame_tensor, item["tokens"]), output_path)

            current_time = time.time()
            if current_time - last_log >= 1.0:
                print(f'batch has {len(batch)} clips')
                print(f'first clip has {len(batch[0]["tokens"])} tokens')
                print('first 20 tokens:')
                print_token_sequence(batch[0]["tokens"][:20])
                print(f'first clip has {len(batch[0]["frame_tensors"])} videos')
                print(f'1st clip 1st video shape: {batch[0]["frame_tensors"][0][0].shape}')

                elapsed = current_time - t0
                running_fps = total_frames / elapsed if elapsed > 0 else 0
                print(f"\nBatch {total_batches}: {running_fps:.1f} FPS ({total_frames} frames in {elapsed:.1f}s)")
                last_log = current_time
                
    except KeyboardInterrupt:
        loader.cleanup()
    
    elapsed = time.time() - t0
    fps = total_frames / elapsed if elapsed > 0 else 0
    print("\nFinal Summary:")
    print(f"Processed {total_frames} frames in {elapsed:.2f} seconds")
    print(f"Average speed: {fps:.1f} FPS")
    print(f"Total batches: {total_batches}")
