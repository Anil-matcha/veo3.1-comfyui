"""
Veo 3.1 Video Saver Node
==========================
Downloads a video from a URL, saves it to ComfyUI's output directory,
and returns the frames as an IMAGE tensor for further processing.
"""

import os

import cv2
import numpy as np
import requests
import torch
from PIL import Image

try:
    import folder_paths
except ImportError:
    class folder_paths:  # noqa: N801
        @staticmethod
        def get_output_directory():
            return os.path.join(os.path.expanduser("~"), "comfyui_output")


class Veo31VideoSaver:
    """
    Downloads a Veo 3.1 video URL, saves it to ComfyUI's output directory,
    and returns the frames as an IMAGE tensor for preview / further processing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url":       ("STRING", {"multiline": False, "default": ""}),
                "save_subfolder":  ("STRING", {"default": "veo31_videos"}),
                "filename_prefix": ("STRING", {"default": "veo31"}),
            },
            "optional": {
                "frame_load_cap": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999,
                     "tooltip": "Max frames to load (0 = all)"},
                ),
                "skip_first_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 500},
                ),
                "select_every_nth": (
                    "INT",
                    {"default": 1, "min": 1, "max": 30,
                     "tooltip": "Return every N-th frame (1 = every frame)"},
                ),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING", "INT")
    RETURN_NAMES  = ("frames", "filepath", "frame_count")
    FUNCTION      = "save_and_load"
    CATEGORY      = "🎬 Veo 3.1"
    OUTPUT_NODE   = True

    def save_and_load(
        self,
        video_url,
        save_subfolder,
        filename_prefix,
        frame_load_cap=0,
        skip_first_frames=0,
        select_every_nth=1,
    ):
        if not video_url or not video_url.strip().startswith("http"):
            return self._error("Invalid or empty video URL")

        output_dir = folder_paths.get_output_directory()
        save_dir   = os.path.join(output_dir, save_subfolder)
        os.makedirs(save_dir, exist_ok=True)

        counter  = 1
        filepath = os.path.join(save_dir, f"{filename_prefix}_{counter:05d}.mp4")
        while os.path.exists(filepath):
            counter += 1
            filepath = os.path.join(save_dir, f"{filename_prefix}_{counter:05d}.mp4")

        try:
            print(f"[Veo3.1 Saver] Downloading {video_url[:80]}...")
            resp = requests.get(video_url, stream=True, timeout=300)
            resp.raise_for_status()
            with open(filepath, "wb") as fh:
                for chunk in resp.iter_content(8192):
                    if chunk:
                        fh.write(chunk)
            print(f"[Veo3.1 Saver] Saved → {filepath}")
        except Exception as exc:
            return self._error(f"Download failed: {exc}")

        frames_tensor, frame_count = self._load_frames(
            filepath, frame_load_cap, skip_first_frames, select_every_nth
        )

        filename = os.path.basename(filepath)
        preview  = {
            "filename":  filename,
            "subfolder": save_subfolder,
            "type":      "output",
            "format":    "video/mp4",
        }
        print(f"[Veo3.1 Saver] Loaded {frame_count} frames")
        return {"ui": {"gifs": [preview]}, "result": (frames_tensor, filepath, frame_count)}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_frames(self, video_path, frame_load_cap, skip_first_frames, select_every_nth):
        try:
            cap     = cv2.VideoCapture(video_path)
            frames  = []
            raw_idx = 0
            loaded  = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if raw_idx < skip_first_frames:
                    raw_idx += 1
                    continue

                if (raw_idx - skip_first_frames) % select_every_nth == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    frames.append(rgb)
                    loaded += 1
                    if frame_load_cap > 0 and loaded >= frame_load_cap:
                        break

                raw_idx += 1

            cap.release()

            if not frames:
                raise RuntimeError("No frames extracted from video")

            return torch.from_numpy(np.stack(frames, axis=0)), len(frames)

        except ImportError:
            print("[Veo3.1 Saver] opencv-python not installed — falling back to ffmpeg")
            return self._ffmpeg_first_frame(video_path)
        except Exception as exc:
            print(f"[Veo3.1 Saver] WARNING — frame load error: {exc}")
            return self._dummy_frame(), 1

    def _ffmpeg_first_frame(self, video_path):
        try:
            import subprocess
            thumb = video_path.replace(".mp4", "_thumb.jpg")
            subprocess.run(
                ["ffmpeg", "-i", video_path, "-vframes", "1", "-y", thumb],
                check=True, capture_output=True,
            )
            img = Image.open(thumb)
            arr = np.array(img).astype(np.float32) / 255.0
            os.remove(thumb)
            return torch.from_numpy(arr).unsqueeze(0), 1
        except Exception as exc:
            print(f"[Veo3.1 Saver] ffmpeg fallback failed: {exc}")
            return self._dummy_frame(), 1

    @staticmethod
    def _dummy_frame():
        return torch.zeros(1, 64, 64, 3)

    def _error(self, msg):
        print(f"[Veo3.1 Saver] ERROR — {msg}")
        return {
            "ui":     {"text": [msg]},
            "result": (self._dummy_frame(), "ERROR", 0),
        }


# ─── Node registry ────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "Veo31VideoSaver": Veo31VideoSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Veo31VideoSaver": "🎬 Veo 3.1 Save Video",
}
