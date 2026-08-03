#!/usr/bin/env python3
"""
MuAPI Client — API Wrapper for https://api.muapi.ai/api/v1
Handles image generation, video motion, audio synthesis, background removal,
file upload/download, and LLM chat completions.
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request


class MuapiError(RuntimeError):
    pass


def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() not in os.environ:
                        os.environ[k.strip()] = v.strip().strip('"').strip("'")


def _key() -> str:
    _load_env()
    key = os.environ.get("MUAPI_API_KEY")
    if not key:
        raise MuapiError(
            "MUAPI_API_KEY environment variable is not set. "
            "Get a key from https://muapi.ai"
        )
    return key


def _headers(json_body: bool = True) -> dict:
    h = {"x-api-key": _key()}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def map_aspect_ratio_to_size(aspect_ratio: str, resolution: str = "1k") -> tuple[int, int]:
    mapping = {
        "16:9": (1280, 720) if resolution == "720p" else (1920, 1080),
        "9:16": (720, 1280) if resolution == "720p" else (1080, 1920),
        "1:1": (1024, 1024),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
    }
    return mapping.get(aspect_ratio, (1280, 720))


def _post(endpoint: str, payload: dict, timeout: int = 60) -> dict:
    base = os.environ.get("MUAPI_BASE_URL", "https://api.muapi.ai/api/v1").rstrip("/")
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


def submit(model: str, payload: dict) -> str:
    """Submit a task to a model endpoint; returns the job request_id."""
    if "flux" in model or "imagen" in model or "nano-banana" in model:
        if "aspect_ratio" in payload and ("width" not in payload or "height" not in payload):
            aspect = payload.pop("aspect_ratio")
            resolution = payload.pop("resolution", "1k")
            w, h = map_aspect_ratio_to_size(aspect, resolution)
            payload["width"] = w
            payload["height"] = h

    res = _post(model, payload)
    rid = res.get("request_id") or res.get("id") or (res.get("data") or {}).get("id")
    if not rid:
        raise MuapiError(f"No request_id returned from submit: {res}")
    return rid


def poll(request_id: str, interval: int = 3, timeout_s: int = 900) -> dict:
    """Poll a prediction result until completion."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(interval)
        try:
            res = _get(f"/predictions/{request_id}/result")
        except MuapiError:
            try:
                res = _get(f"/model/prediction/{request_id}")
            except MuapiError as e:
                raise e

        status = (res.get("status") or (res.get("data") or {}).get("status") or "").lower()
        if status in ("completed", "success", "succeeded"):
            return res
        if status in ("failed", "cancelled"):
            err = res.get("error") or (res.get("data") or {}).get("error") or res
            raise MuapiError(f"Prediction {request_id} {status}: {err}")

    raise MuapiError(f"Prediction {request_id} timed out after {timeout_s}s")


def extract_url(result: dict) -> str | None:
    """Extract primary output URL from API response."""
    if not isinstance(result, dict):
        return None

    if "data" in result and isinstance(result["data"], dict):
        result = result["data"]

    candidates = []
    outputs = result.get("outputs") or result.get("output")
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, str):
            candidates.append(first)
        elif isinstance(first, dict):
            for k in ("url", "image_url", "video_url", "audio_url"):
                if first.get(k):
                    candidates.append(first[k])
    elif isinstance(outputs, str):
        candidates.append(outputs)

    for k in ("image_url", "video_url", "audio_url", "url"):
        if isinstance(result.get(k), str):
            candidates.append(result[k])

    images = result.get("images")
    if isinstance(images, list) and images:
        candidates.append(images[0] if isinstance(images[0], str) else images[0].get("url"))

    return next((c for c in candidates if c), None)


def generate_image(model: str, prompt: str, **params) -> str:
    """Synchronous image generation helper."""
    payload = {"prompt": prompt, **params}
    rid = submit(model, payload)
    res = poll(rid, interval=3, timeout_s=180)
    url = extract_url(res)
    if not url:
        raise MuapiError(f"Image generation succeeded but no URL found in: {res}")
    return url


def generate_video(model: str, prompt: str, **params) -> str:
    """Synchronous video generation helper."""
    payload = {"prompt": prompt, **params}
    rid = submit(model, payload)
    res = poll(rid, interval=4, timeout_s=900)
    url = extract_url(res)
    if not url:
        raise MuapiError(f"Video generation succeeded but no URL found in: {res}")
    return url


def generate_audio(model: str, prompt: str, **params) -> str:
    """Synchronous audio generation helper."""
    payload = {"prompt": prompt, **params}
    rid = submit(model, payload)
    res = poll(rid, interval=3, timeout_s=300)
    url = extract_url(res)
    if not url:
        raise MuapiError(f"Audio generation succeeded but no URL found in: {res}")
    return url


def upload(file_path: str) -> str:
    """Upload a local file using curl, returning its hosted URL."""
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
    """Download a remote URL to a local destination file using curl."""
    try:
        subprocess.run(["curl", "-s", "-L", "--retry", "3", "-o", dest, url], check=True)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            raise MuapiError(f"Download produced an empty file: {url}")
        return dest
    except Exception as e:
        raise MuapiError(f"Download failed for {url} -> {dest}: {e}")


def chat(messages: list, model: str = "gpt-4o-mini", **params) -> str:
    """Chat completion request using OPENAI_API_KEY if available."""
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
    print("MuAPI Client loaded successfully.")
