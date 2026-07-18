"""
Veo 3.1 ComfyUI Nodes
========================
Dedicated nodes for every Veo 3.1 variant.

Nodes
-----
  Veo31TextToVideo       — T2V (standard / fast / lite)
  Veo31ImageToVideo      — I2V (standard / fast / lite, optional last frame)
  Veo31ReferenceToVideo  — Reference-guided generation (up to 4 images)
  Veo31ExtendVideo       — Extend a previous Veo 3.1 generation
  Veo314KUpscale         — Upscale a previous generation to 4K
"""

import io
import json
import os
import time

import numpy as np
import requests
import torch
from PIL import Image

BASE_URL      = "https://api.muapi.ai/api/v1"
POLL_INTERVAL = 10
MAX_WAIT      = 900

# ─── Model endpoint lists ─────────────────────────────────────────────────────

T2V_MODELS = [
    "veo3.1-text-to-video",
    "veo3.1-fast-text-to-video",
    "veo3.1-lite-text-to-video",
]

I2V_MODELS = [
    "veo3.1-image-to-video",
    "veo3.1-fast-image-to-video",
    "veo3.1-lite-image-to-video",
]

RESOLUTION_OPTS = ["720p", "1080p", "4k"]
ASPECT_OPTS     = ["16:9", "9:16"]


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _upload_image(api_key: str, image_tensor: torch.Tensor) -> str:
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    img_arr = (image_tensor.cpu().numpy() * 255).astype("uint8")
    img = Image.fromarray(img_arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    resp = requests.post(
        f"{BASE_URL}/upload_file",
        headers={"x-api-key": api_key},
        files={"file": ("image.jpg", buf, "image/jpeg")},
        timeout=120,
    )
    _raise_for_status(resp)
    return _extract_url(resp.json())


def _extract_url(data: dict) -> str:
    url = data.get("url") or data.get("file_url") or data.get("output")
    if not url:
        raise RuntimeError(f"Upload response missing URL: {data}")
    return str(url)


def _submit_job(api_key: str, endpoint: str, payload: dict) -> str:
    resp = requests.post(
        f"{BASE_URL}/{endpoint}",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    _raise_for_status(resp)
    data = resp.json()
    request_id = data.get("request_id")
    if not request_id:
        raise RuntimeError(f"No request_id in response: {data}")
    return request_id


def _poll_result(api_key: str, request_id: str) -> dict:
    deadline = time.time() + MAX_WAIT
    while time.time() < deadline:
        resp = requests.get(
            f"{BASE_URL}/predictions/{request_id}/result",
            headers={"x-api-key": api_key},
            timeout=30,
        )
        _raise_for_status(resp)
        data   = resp.json()
        status = data.get("status")
        print(f"[Veo3.1] status={status}  id={request_id}")
        if status == "completed":
            return data
        if status == "failed":
            raise RuntimeError(f"Generation failed: {data.get('error', 'unknown')}")
        time.sleep(POLL_INTERVAL)
    raise RuntimeError(f"Timed out after {MAX_WAIT}s for request_id={request_id}")


def _extract_output_url(result: dict) -> str:
    outputs = result.get("outputs") or result.get("output") or []
    if isinstance(outputs, list) and outputs:
        return str(outputs[0])
    if isinstance(outputs, str):
        return outputs
    for key in ("video_url", "image_url", "url"):
        if result.get(key):
            return str(result[key])
    raise RuntimeError(f"Cannot find output URL in result: {result}")


def _raise_for_status(resp: requests.Response) -> None:
    if resp.status_code == 401:
        raise RuntimeError("Authentication failed — check your API key.")
    if resp.status_code == 402:
        raise RuntimeError("Insufficient credits.")
    if resp.status_code == 429:
        raise RuntimeError("Rate limit exceeded — please wait and retry.")
    resp.raise_for_status()


def _first_frame_from_url(video_url: str) -> torch.Tensor:
    """Download a video URL → first frame as IMAGE tensor [1,H,W,C]."""
    try:
        import tempfile
        import cv2

        resp = requests.get(video_url, timeout=180, stream=True)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            for chunk in resp.iter_content(8192):
                if chunk:
                    tmp.write(chunk)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        cap.release()
        os.remove(tmp_path)
        if not ret:
            raise RuntimeError("Could not decode first frame")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).unsqueeze(0)
    except Exception as exc:
        print(f"[Veo3.1] WARNING — first-frame preview failed: {exc}")
        return torch.zeros(1, 64, 64, 3)


def _parse_extra(extra_params_json: str) -> dict:
    try:
        return json.loads(extra_params_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"extra_params_json is not valid JSON: {exc}") from exc


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1 — Veo 3.1 Text-to-Video
# ═══════════════════════════════════════════════════════════════════════════════

class Veo31TextToVideo:
    """
    Veo 3.1 — Text to Video
    -------------------------
    Generate an 8-second video from a text prompt using Veo 3.1.

    Models
    ------
    • veo3.1-text-to-video       — Standard quality, with audio
    • veo3.1-fast-text-to-video  — Faster, slightly lower quality
    • veo3.1-lite-text-to-video  — Lightweight / lower cost

    Resolution: 720p / 1080p / 4k
    Aspect ratio: 16:9 / 9:16
    Duration: fixed 8 s (Veo 3.1 constraint)

    Returns
    -------
    video_url   STRING  — CDN URL of the generated MP4
    first_frame IMAGE   — First frame tensor for previewing in ComfyUI
    request_id  STRING  — Original request ID (use for extend / 4K upscale)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":      ("STRING",   {"multiline": False, "default": ""}),
                "model":        (T2V_MODELS, {"default": "veo3.1-text-to-video"}),
                "prompt":       ("STRING",   {"multiline": True,
                                              "default": "Scene: Old clockmaker's studio filled with ticking clocks and dust motes.\n"
                                                         "Characters: Elderly clockmaker tightening a gear through magnifying glass.\n"
                                                         "Action: Macro focus on ticking hands then slow pullback revealing full room.\n"
                                                         "Mood: Intimate, timeless craftsmanship."}),
                "aspect_ratio": (ASPECT_OPTS,     {"default": "16:9"}),
                "resolution":   (RESOLUTION_OPTS, {"default": "720p"}),
            },
            "optional": {
                "extra_params_json": ("STRING", {"multiline": True, "default": "{}"}),
            },
        }

    RETURN_TYPES  = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES  = ("video_url", "first_frame", "request_id")
    FUNCTION      = "generate"
    CATEGORY      = "🎬 Veo 3.1"

    def generate(self, api_key, model, prompt, aspect_ratio, resolution,
                 extra_params_json="{}"):
        if not api_key.strip():
            raise ValueError("api_key is required.")

        extra   = _parse_extra(extra_params_json)
        payload = {
            "prompt":       prompt,
            "aspect_ratio": aspect_ratio,
            "duration":     8,
            "resolution":   resolution,
            **extra,
        }

        print(f"[Veo3.1 T2V] model={model}  submitting...")
        request_id = _submit_job(api_key, model, payload)
        print(f"[Veo3.1 T2V] request_id={request_id}  polling...")
        result    = _poll_result(api_key, request_id)
        video_url = _extract_output_url(result)
        print(f"[Veo3.1 T2V] Done → {video_url}")
        return (video_url, _first_frame_from_url(video_url), request_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2 — Veo 3.1 Image-to-Video
# ═══════════════════════════════════════════════════════════════════════════════

class Veo31ImageToVideo:
    """
    Veo 3.1 — Image to Video
    --------------------------
    Animate a static image into an 8-second Veo 3.1 video.
    Optionally provide a last_image to anchor the ending frame.

    Models
    ------
    • veo3.1-image-to-video       — Standard quality, with audio
    • veo3.1-fast-image-to-video  — Faster variant
    • veo3.1-lite-image-to-video  — Lightweight / lower cost

    Inputs
    ------
    image       IMAGE (required) — Starting frame
    last_image  IMAGE (optional) — Ending frame (first–last frame mode)

    Returns
    -------
    video_url   STRING
    first_frame IMAGE
    request_id  STRING
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":      ("STRING",   {"multiline": False, "default": ""}),
                "model":        (I2V_MODELS, {"default": "veo3.1-image-to-video"}),
                "image":        ("IMAGE",),
                "prompt":       ("STRING",   {"multiline": True,
                                              "default": "The scene comes to life with smooth, cinematic motion. "
                                                         "Camera slowly pulls back revealing the full environment."}),
                "aspect_ratio": (ASPECT_OPTS,     {"default": "16:9"}),
                "resolution":   (RESOLUTION_OPTS, {"default": "720p"}),
            },
            "optional": {
                "last_image":        ("IMAGE",),
                "extra_params_json": ("STRING", {"multiline": True, "default": "{}"}),
            },
        }

    RETURN_TYPES  = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES  = ("video_url", "first_frame", "request_id")
    FUNCTION      = "generate"
    CATEGORY      = "🎬 Veo 3.1"

    def generate(self, api_key, model, image, prompt, aspect_ratio, resolution,
                 last_image=None, extra_params_json="{}"):
        if not api_key.strip():
            raise ValueError("api_key is required.")

        extra = _parse_extra(extra_params_json)

        print(f"[Veo3.1 I2V] Uploading start frame...")
        image_url = _upload_image(api_key, image)

        payload = {
            "prompt":       prompt,
            "image_url":    image_url,
            "aspect_ratio": aspect_ratio,
            "duration":     8,
            "resolution":   resolution,
            **extra,
        }

        if last_image is not None:
            print(f"[Veo3.1 I2V] Uploading last frame...")
            payload["last_image"] = _upload_image(api_key, last_image)

        print(f"[Veo3.1 I2V] model={model}  submitting...")
        request_id = _submit_job(api_key, model, payload)
        print(f"[Veo3.1 I2V] request_id={request_id}  polling...")
        result    = _poll_result(api_key, request_id)
        video_url = _extract_output_url(result)
        print(f"[Veo3.1 I2V] Done → {video_url}")
        return (video_url, _first_frame_from_url(video_url), request_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 3 — Veo 3.1 Reference-to-Video
# ═══════════════════════════════════════════════════════════════════════════════

class Veo31ReferenceToVideo:
    """
    Veo 3.1 — Reference to Video
    ------------------------------
    Generate a video guided by up to 4 reference images.
    Images are uploaded and sent as images_list[].

    Returns
    -------
    video_url   STRING
    first_frame IMAGE
    request_id  STRING
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":    ("STRING", {"multiline": False, "default": ""}),
                "prompt":     ("STRING", {"multiline": True,
                                          "default": "A dynamic cinematic video inspired by the reference images, "
                                                     "with smooth motion and professional lighting."}),
                "image_1":    ("IMAGE",),
                "resolution": (RESOLUTION_OPTS, {"default": "720p"}),
            },
            "optional": {
                "image_2":           ("IMAGE",),
                "image_3":           ("IMAGE",),
                "image_4":           ("IMAGE",),
                "generate_audio":    ("BOOLEAN", {"default": True}),
                "extra_params_json": ("STRING",  {"multiline": True, "default": "{}"}),
            },
        }

    RETURN_TYPES  = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES  = ("video_url", "first_frame", "request_id")
    FUNCTION      = "generate"
    CATEGORY      = "🎬 Veo 3.1"

    def generate(self, api_key, prompt, image_1, resolution,
                 image_2=None, image_3=None, image_4=None,
                 generate_audio=True, extra_params_json="{}"):
        if not api_key.strip():
            raise ValueError("api_key is required.")

        extra = _parse_extra(extra_params_json)

        images_list = []
        for idx, img in enumerate([image_1, image_2, image_3, image_4], start=1):
            if img is not None:
                print(f"[Veo3.1 Ref2V] Uploading image {idx}...")
                images_list.append(_upload_image(api_key, img))

        payload = {
            "prompt":         prompt,
            "images_list":    images_list,
            "resolution":     resolution,
            "duration":       8,
            "generate_audio": generate_audio,
            **extra,
        }

        endpoint = "veo3.1-reference-to-video"
        print(f"[Veo3.1 Ref2V] {len(images_list)} image(s)  submitting...")
        request_id = _submit_job(api_key, endpoint, payload)
        print(f"[Veo3.1 Ref2V] request_id={request_id}  polling...")
        result    = _poll_result(api_key, request_id)
        video_url = _extract_output_url(result)
        print(f"[Veo3.1 Ref2V] Done → {video_url}")
        return (video_url, _first_frame_from_url(video_url), request_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 4 — Veo 3.1 Extend Video
# ═══════════════════════════════════════════════════════════════════════════════

class Veo31ExtendVideo:
    """
    Veo 3.1 — Extend Video
    -----------------------
    Seamlessly continue a previous Veo 3.1 generation.

    Pass the request_id from any Veo 3.1 T2V / I2V / Ref2V node and a prompt
    describing what should happen next.

    Returns
    -------
    video_url      STRING — CDN URL of the extended video
    first_frame    IMAGE  — First frame preview
    new_request_id STRING — New request ID for further chaining
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":    ("STRING", {"multiline": False, "default": ""}),
                "request_id": ("STRING", {"multiline": False, "default": "",
                                          "tooltip": "request_id from a previous Veo 3.1 generation"}),
                "prompt":     ("STRING", {"multiline": True,
                                          "default": "Continue the scene with dramatic camera movement and ambient sound."}),
            },
            "optional": {
                "extra_params_json": ("STRING", {"multiline": True, "default": "{}"}),
            },
        }

    RETURN_TYPES  = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES  = ("video_url", "first_frame", "new_request_id")
    FUNCTION      = "extend"
    CATEGORY      = "🎬 Veo 3.1"

    def extend(self, api_key, request_id, prompt, extra_params_json="{}"):
        if not api_key.strip():
            raise ValueError("api_key is required.")
        if not request_id.strip():
            raise ValueError("request_id is required.")
        if not prompt.strip():
            raise ValueError("prompt is required for extend.")

        extra   = _parse_extra(extra_params_json)
        payload = {
            "request_id": request_id.strip(),
            "prompt":     prompt.strip(),
            **extra,
        }

        endpoint = "veo3.1-extend-video"
        print(f"[Veo3.1 Extend] source={request_id}  submitting...")
        new_id    = _submit_job(api_key, endpoint, payload)
        print(f"[Veo3.1 Extend] new_request_id={new_id}  polling...")
        result    = _poll_result(api_key, new_id)
        video_url = _extract_output_url(result)
        print(f"[Veo3.1 Extend] Done → {video_url}")
        return (video_url, _first_frame_from_url(video_url), new_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 5 — Veo 3.1 4K Upscale
# ═══════════════════════════════════════════════════════════════════════════════

class Veo314KUpscale:
    """
    Veo 3.1 — 4K Upscale
    ----------------------
    Upscale a previously generated Veo 3.1 video to 4K resolution.
    Pass the request_id from any Veo 3.1 generation node.

    Returns
    -------
    video_url      STRING — CDN URL of the 4K upscaled video
    first_frame    IMAGE  — First frame preview
    new_request_id STRING — New request ID
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":    ("STRING", {"multiline": False, "default": ""}),
                "request_id": ("STRING", {"multiline": False, "default": "",
                                          "tooltip": "request_id from a previous Veo 3.1 generation to upscale to 4K"}),
            },
            "optional": {
                "extra_params_json": ("STRING", {"multiline": True, "default": "{}"}),
            },
        }

    RETURN_TYPES  = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES  = ("video_url", "first_frame", "new_request_id")
    FUNCTION      = "upscale"
    CATEGORY      = "🎬 Veo 3.1"

    def upscale(self, api_key, request_id, extra_params_json="{}"):
        if not api_key.strip():
            raise ValueError("api_key is required.")
        if not request_id.strip():
            raise ValueError("request_id is required.")

        extra   = _parse_extra(extra_params_json)
        payload = {"request_id": request_id.strip(), **extra}

        endpoint = "veo3.1-4k-video"
        print(f"[Veo3.1 4K] source={request_id}  submitting 4K upscale...")
        new_id    = _submit_job(api_key, endpoint, payload)
        print(f"[Veo3.1 4K] new_request_id={new_id}  polling...")
        result    = _poll_result(api_key, new_id)
        video_url = _extract_output_url(result)
        print(f"[Veo3.1 4K] Done → {video_url}")
        return (video_url, _first_frame_from_url(video_url), new_id)


# ─── Node registry ────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "Veo31TextToVideo":      Veo31TextToVideo,
    "Veo31ImageToVideo":     Veo31ImageToVideo,
    "Veo31ReferenceToVideo": Veo31ReferenceToVideo,
    "Veo31ExtendVideo":      Veo31ExtendVideo,
    "Veo314KUpscale":        Veo314KUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Veo31TextToVideo":      "🎬 Veo 3.1 Text to Video",
    "Veo31ImageToVideo":     "🎬 Veo 3.1 Image to Video",
    "Veo31ReferenceToVideo": "🎬 Veo 3.1 Reference to Video",
    "Veo31ExtendVideo":      "🎬 Veo 3.1 Extend Video",
    "Veo314KUpscale":        "🎬 Veo 3.1 4K Upscale",
}
