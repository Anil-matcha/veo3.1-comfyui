#!/usr/bin/env python3
"""
Muapi API client for muapi-director.
Thin, dependency-free wrapper around the Muapi Media Generation API.
"""
import json
import os
import subprocess
import time
import urllib.request
import urllib.error

class MuapiError(RuntimeError):
    pass

def _key() -> str:
    k = os.environ.get("MUAPI_API_KEY")
    if not k:
        raise MuapiError("MUAPI_API_KEY is not set. Please set it in your environment.")
    return k

def _headers(json_body: bool = True) -> dict:
    h = {
        "x-api-key": _key(),
        "Accept": "application/json"
    }
    if json_body:
        h["Content-Type"] = "application/json"
    return h

def map_aspect_ratio_to_size(aspect: str, resolution: str = "1k") -> tuple[int, int]:
    """Helper to map aspect ratios to exact width and height pixel boundaries."""
    sizes = {
        "16:9": (1024, 576),
        "9:16": (576, 1024),
        "1:1": (1024, 1024),
        "3:4": (768, 1024),
        "4:3": (1024, 768)
    }
    w, h = sizes.get(aspect, (1024, 1024))
    if resolution == "2k":
        w = int(w * 1.5)
        h = int(h * 1.5)
    return w, h

def _post(endpoint: str, payload: dict, timeout: int = 120) -> dict:
    base = os.environ.get("MUAPI_BASE_URL", "https://api.muapi.ai/api/v1").rstrip("/")
    # Map model shortcut to specific route if needed
    if endpoint == "flux-dev":
        endpoint = "flux-dev-image"
    elif endpoint == "runway-image-to-video":
        endpoint = "runway-image-to-video"
    elif endpoint == "minimax-speech-2.6-turbo":
        endpoint = "minimax-speech-2.6-turbo"
    elif endpoint == "suno-create-music":
        endpoint = "suno-create-music"
        
    url = f"{base}/{endpoint.lstrip('/')}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=_headers(),
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise MuapiError(f"POST {url} -> {e.code}: {e.read().decode()[:400]}") from e
    except Exception as e:
        raise MuapiError(f"POST {url} failed: {e}") from e

def _get(endpoint: str, timeout: int = 60, retries: int = 3) -> dict:
    base = os.environ.get("MUAPI_BASE_URL", "https://api.muapi.ai/api/v1").rstrip("/")
    url = f"{base}/{endpoint.lstrip('/')}"
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=_headers(json_body=False))
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(2 ** i)
    raise MuapiError(f"GET {url} failed after {retries} tries: {last}")

# ---------------------------------------------------------------- generation

def submit(model: str, payload: dict) -> str:
    """Submit a task to a specific model endpoint; returns the request_id."""
    # Pre-process image payloads to convert aspect_ratio + resolution into width + height for Flux
    if "flux" in model:
        if "aspect_ratio" in payload:
            aspect = payload.pop("aspect_ratio")
            resolution = payload.pop("resolution", "1k")
            w, h = map_aspect_ratio_to_size(aspect, resolution)
            payload["width"] = w
            payload["height"] = h
            
    res = _post(model, payload)
    rid = res.get("request_id") or res.get("id")
    if not rid:
        raise MuapiError(f"No request_id returned from submit: {res}")
    return rid

def poll(request_id: str, interval: int = 3, timeout_s: int = 900) -> dict:
    """Poll a prediction result until it reaches a terminal status."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(interval)
        try:
            res = _get(f"/predictions/{request_id}/result")
        except MuapiError as e:
            raise e
        
        status = (res.get("status") or "").lower()
        if status in ("completed", "success"):
            return res
        if status in ("failed", "cancelled"):
            raise MuapiError(f"Prediction {request_id} {status}: {res.get('error') or res}")
            
    raise MuapiError(f"Prediction {request_id} timed out after {timeout_s}s")

def extract_url(result: dict) -> str | None:
    """Extract the primary output URL from a poll result, handling diverse schemas."""
    if not isinstance(result, dict):
        return None
    candidates = []
    outputs = result.get("outputs")
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, str):
            candidates.append(first)
        elif isinstance(first, dict):
            for k in ("url", "image_url", "video_url", "audio_url"):
                if first.get(k):
                    candidates.append(first[k])
    for k in ("image_url", "video_url", "audio_url", "url"):
        if isinstance(result.get(k), str):
            candidates.append(result[k])
    images = result.get("images")
    if isinstance(images, list) and images:
        candidates.append(images[0] if isinstance(images[0], str) else images[0].get("url"))
    od = result.get("output_data")
    if isinstance(od, dict):
        for k in ("outputs", "images", "videos"):
            v = od.get(k)
            if isinstance(v, list) and v:
                candidates.append(v[0] if isinstance(v[0], str) else v[0].get("url"))
        for k in ("image_url", "video_url", "audio_url", "url"):
            if isinstance(od.get(k), str):
                candidates.append(od[k])
    return next((c for c in candidates if c), None)

def generate_image(model: str, prompt: str, **params) -> str:
    """Helper to run synchronous image generation."""
    payload = {"prompt": prompt, **params}
    rid = submit(model, payload)
    res = poll(rid, interval=3, timeout_s=180)
    url = extract_url(res)
    if not url:
        raise MuapiError(f"Image generation succeeded but no URL found in: {res}")
    return url

def generate_video(model: str, prompt: str, **params) -> str:
    """Helper to run synchronous video generation."""
    payload = {"prompt": prompt, **params}
    rid = submit(model, payload)
    res = poll(rid, interval=4, timeout_s=900)
    url = extract_url(res)
    if not url:
        raise MuapiError(f"Video generation succeeded but no URL found in: {res}")
    return url

def generate_audio(model: str, prompt: str, **params) -> str:
    """Helper to run synchronous audio/narration generation."""
    payload = {"prompt": prompt, **params}
    rid = submit(model, payload)
    res = poll(rid, interval=3, timeout_s=300)
    url = extract_url(res)
    if not url:
        raise MuapiError(f"Audio generation succeeded but no URL found in: {res}")
    return url

# ---------------------------------------------------------------- upload / download

def upload(file_path: str) -> str:
    """Upload a local file, returning its public hosted URL. Uses system curl."""
    base = os.environ.get("MUAPI_BASE_URL", "https://api.muapi.ai/api/v1").rstrip("/")
    key = _key()
    
    cmd = [
        "curl", "-s", "-X", "POST", f"{base}/upload_file",
        "-H", f"x-api-key: {key}",
        "-F", f"file=@{file_path}"
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        data = json.loads(out)
        url = data.get("url") or data.get("file_url") or (data.get("data") or {}).get("url")
        if not url:
            raise MuapiError(f"Upload returned no URL: {out[:400]}")
        return url
    except Exception as e:
        raise MuapiError(f"Upload failed for {file_path}: {e}")

def download(url: str, dest: str) -> str:
    """Download a remote URL to a local destination file. Uses curl for reliability."""
    try:
        subprocess.run(["curl", "-s", "-L", "--retry", "3", "-o", dest, url], check=True)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            raise MuapiError(f"Download produced an empty file: {url}")
        return dest
    except Exception as e:
        raise MuapiError(f"Download failed for {url} -> {dest}: {e}")

# ---------------------------------------------------------------- llm

def chat(messages: list, model: str = "gpt-4o-mini", **params) -> str:
    """Submit a chat completion request to OpenAI using OPENAI_API_KEY."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise MuapiError("OPENAI_API_KEY is not set in environment.")
    
    url = "https://api.openai.com/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps({"model": model, "messages": messages, **params}).encode(),
        headers={
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.load(r)
            return resp["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        raise MuapiError(f"OpenAI chat completion failed: {e.code} - {e.read().decode()[:400]}") from e
    except Exception as e:
        raise MuapiError(f"OpenAI chat completion failed: {e}") from e

if __name__ == "__main__":
    print("Muapi Client loaded successfully.")
    print("MUAPI_API_KEY:", "Set" if os.environ.get("MUAPI_API_KEY") else "MISSING")
    print("OPENAI_API_KEY:", "Set" if os.environ.get("OPENAI_API_KEY") else "MISSING")
