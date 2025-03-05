import time
import glob
import threading
import queue
import torch
from torchcodec.decoders import VideoDecoder

class VideoStreamLoader:
    def __init__(self, video_paths, frames_per_clip, batch_size, workers=16):
        self.video_paths = video_paths
        self.frames_per_clip = frames_per_clip
        self.batch_size = batch_size
        self.workers = workers

        self.task_queue = queue.Queue()
        self.output_queue = queue.Queue(maxsize=32)

        for path in self.video_paths:
            self.task_queue.put(path)

        for _ in range(self.workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()

    def _worker(self):
        """Worker thread: decode videos and produce frame clip tensors."""
        while True:
            try:
                video_path = self.task_queue.get(block=False)
            except queue.Empty:
                break
            decoder = VideoDecoder(video_path, num_ffmpeg_threads=1, device="cpu")
            total_frames = decoder.metadata.num_frames
            for start in range(0, total_frames, self.frames_per_clip):
                end = min(start + self.frames_per_clip, total_frames)
                if end - start < self.frames_per_clip:
                    break
                frames_tensor = decoder[start : end].cpu()
                frames_tensor = self._pad_frames(frames_tensor)
                self.output_queue.put(frames_tensor)
            self.task_queue.task_done()

    def _pad_frames(self, frames_tensor):
        # frames_tensor shape: (M, C, H, W)
        M, C, H, W = frames_tensor.shape
        pad_h = max(0, 720 - H)
        pad_w = max(0, 1280 - W)
        if pad_h > 0 or pad_w > 0:
            frames_tensor = torch.nn.functional.pad(frames_tensor, (0, pad_w, 0, pad_h))
        return frames_tensor

    def __iter__(self):
        return self

    def __next__(self):
        batch_clips = []
        for _ in range(self.batch_size):
            if self.output_queue.empty() and self.task_queue.empty():
                raise StopIteration
            clip = self.output_queue.get(timeout=100)
            batch_clips.append(clip)
            self.output_queue.task_done()
        batch_tensor = torch.stack(batch_clips, dim=0)
        return batch_tensor

if __name__ == "__main__":
    video_files = glob.glob("./data/vids/*.mp4")
    frames_per_clip = 30*10
    batch_size = 32
    frames_per_batch = frames_per_clip * batch_size
    loader = VideoStreamLoader(video_files, frames_per_clip=frames_per_clip, batch_size=batch_size, workers=32)
    t0 = time.time()
    last_log = t0
    total_frames = 0
    total_batches = 0
    
    for batch in loader:
        total_batches += 1
        total_frames += frames_per_batch
        
        current_time = time.time()
        if current_time - last_log >= 1.0:
            print(f"\nBatch tensor shape: {batch.shape}")
            
            elapsed = current_time - t0
            running_fps = total_frames / elapsed if elapsed > 0 else 0
            print(f"\nBatch {total_batches}: {running_fps:.1f} FPS ({total_frames} frames in {elapsed:.1f}s)")
            last_log = current_time
    
    elapsed = time.time() - t0
    fps = total_frames / elapsed if elapsed > 0 else 0
    print("\nFinal Summary:")
    print(f"Processed {total_frames} frames in {elapsed:.2f} seconds")
    print(f"Average speed: {fps:.1f} FPS")
    print(f"Total batches: {total_batches}")
