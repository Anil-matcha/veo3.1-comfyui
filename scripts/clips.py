#!/usr/bin/env python3
"""
Clip stage: animate each SHOT's keyframe into a short clip using Muapi's image-to-video models.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import muapi_client
from styles import resolve_theme

VIDEO_MODEL = "runway-image-to-video"

def shots_of(beat):
    if beat.get("shots"):
        for s in beat["shots"]:
            yield s, f"{beat['id']}{s.get('id','')}"
    else:
        yield beat, f"{beat['id']}"

CAMERA_VOCAB = {
    "static":   "a locked-off static camera (no camera move)",
    "push_in":  "one very slow smooth push-in (uniform scale-up, Ken-Burns)",
    "pull_out": "one slow smooth pull-out (uniform scale-down) revealing the full scene",
    "pan":      "one slow horizontal pan across the frame (flat translate, no perspective shift)",
    "tilt":     "one slow vertical tilt (flat translate, no perspective shift)",
    "parallax": "a gentle multi-layer parallax drift (paper layers moving at slightly different speeds), the camera otherwise steady",
    # bold / experimental:
    "orbit":     "a slow 3D orbit / camera arc around the scene",
    "dolly_zoom": "a dolly-zoom (vertigo) — push in while the background pulls back",
    "roll":      "a slow camera roll / canted rotation",
    "whip":      "a fast whip-pan sweep that settles",
}

AMPLITUDE = {
    "calm":   "subtle, restrained amplitude",
    "punchy": "lively, energetic amplitude with clear, bold movement",
    "max":    "high-energy amplitude — elements burst, scatter and fly boldly",
}

DEFAULT_ELEMENT = ("several cut-out elements move naturally with the scene — they bob, tilt, "
                   "slide, drift and scatter; halftone dots pulse")

def collage_prompt(camera, element_motion=None, feel=None, palette=None, amplitude="punchy",
                   has_title=True, constraints="strict"):
    cam = CAMERA_VOCAB.get(camera, camera)
    element = element_motion or DEFAULT_ELEMENT
    feel = feel or "tactile, editorial, a page from a scrapbook"
    palette = palette or "the still's existing palette, high contrast"
    amp = AMPLITUDE.get(amplitude, AMPLITUDE["punchy"])
    text_lock = ("Keep the HEADLINE TEXT sharp, legible and stable — do not warp or wobble the lettering. " if has_title else "")
    
    if constraints == "loose":
        guard = (text_lock + "Keep the paper-collage look (printed cut-outs, not photoreal); "
                 "avoid a subject melting into shapeless goo. Otherwise explore freely.")
    else:  # strict — defect guards for clean, text-heavy explainers
        guard = (text_lock + "Keep the layout stable. Stay flat 2D — no 3D rotation, no "
                 "perspective change, camera parallel to the poster. ONE continuous move that "
                 "does not loop, retract or reset. Rigid paper — no morph/melt. Animate the "
                 "motion only; don't re-render the picture.")
                 
    return (
        "Animate this still into a mixed-media paper-collage MOTION GRAPHIC, printed cut-outs, not photoreal.\n"
        f"CAMERA (one move only): {cam}.\n"
        f"ELEMENT MOTION (rich — this is the energy, {amp}): {element}. Elements move as paper cut-outs (slide, flap, hinge, pop, scatter, fly).\n"
        "AESTHETIC: keep the torn-paper, tape, halftone, newsprint and paper-stencil textures and the bold flat background.\n"
        f"FEEL: {feel}.\nCOLOR: {palette}.\n"
        f"CONSTRAINTS: {guard}"
    )

def generate_shot_clip(key, shot, prompt, model, aspect, vid_res, clip_dir):
    try:
        url = shot.get("keyframe_url")
        dur = int(shot.get("dur", 5))
        
        # Build the payload according to specific model requirements
        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect
        }
        
        # Map input parameters
        if "gemini-omni" in model:
            payload["image_urls"] = [url]
            # gemini-omni only supports duration in literal [4, 6, 8, 10] seconds
            if dur <= 5:
                payload["duration"] = 4
            elif dur <= 7:
                payload["duration"] = 6
            elif dur <= 9:
                payload["duration"] = 8
            else:
                payload["duration"] = 10
        elif "veo3" in model:
            payload["images_list"] = [url]
        else:
            payload["image_url"] = url
            payload["duration"] = dur
            
        print(f"[{key}] Generating video clip ({dur}s)...")
        rid = muapi_client.submit(model, payload)
        res = muapi_client.poll(rid, interval=5, timeout_s=900)
        video_url = muapi_client.extract_url(res)
        
        if not video_url:
            raise muapi_client.MuapiError(f"Video generated but no output URL was found: {res}")
            
        dest = os.path.join(clip_dir, f"clip_{key}.mp4")
        print(f"[{key}] Downloading video clip...")
        muapi_client.download(video_url, dest)
        print(f"[{key}] Success -> {dest}")
        return key, video_url, dest
    except Exception as e:
        print(f"[{key}] Failed: {e}")
        return key, None, None

def run(project_dir, only=None):
    bpath = os.path.join(project_dir, "beats.json")
    if not os.path.exists(bpath):
        print(f"Error: beats.json not found at {bpath}")
        return
        
    with open(bpath) as f:
        doc = json.load(f)
        
    aspect = doc.get("aspect", "16:9")
    motion_style = doc.get("motion_style") or "punchy"
    constraints = doc.get("constraints", "strict")
    model = doc.get("video_model", VIDEO_MODEL)
    vid_res = doc.get("video_resolution", "720p")
    
    clip_dir = os.path.join(project_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)
    
    jobs = []
    by_key = {}
    
    for beat in doc["beats"]:
        for shot, key in shots_of(beat):
            if only and key not in only:
                continue
                
            if shot.get("clip_url"):
                print(f"[{key}] Clip already generated. Skipping.")
                continue
                
            url = shot.get("keyframe_url")
            if not url and shot.get("keyframe_path") and os.path.exists(shot["keyframe_path"]):
                print(f"[{key}] Uploading local keyframe...")
                url = muapi_client.upload(shot["keyframe_path"])
                shot["keyframe_url"] = url
                
            if not url:
                print(f"[{key}] No keyframe found. Generate keyframes first.")
                continue
                
            camera = shot.get("camera_move") or "push_in"
            element = shot.get("element_motion")
            
            prompt = collage_prompt(
                camera=camera,
                element_motion=element,
                feel=beat.get("feel"),
                palette=beat.get("bg"),
                amplitude=motion_style,
                has_title=shot.get("title", True),
                constraints=constraints
            )
            shot["clip_prompt"] = prompt
            jobs.append((key, shot, prompt))
            by_key[key] = shot
            
    if not jobs:
        print("No new video clips to generate.")
        return
        
    print(f"Animating {len(jobs)} clips in parallel using model '{model}'...")
    with ThreadPoolExecutor(max_workers=min(len(jobs), 3)) as executor:
        futures = [
            executor.submit(generate_shot_clip, key, shot, prompt, model, aspect, vid_res, clip_dir)
            for key, shot, prompt in jobs
        ]
        for f in futures:
            key, url, dest = f.result()
            if url:
                shot = by_key[key]
                shot["clip_url"] = url
                shot["clip_path"] = dest
                
    with open(bpath, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print("Updated", bpath)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 clips.py <project_dir> [only_keys]")
        sys.exit(1)
        
    proj = os.path.abspath(sys.argv[1])
    only_keys = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    run(proj, only_keys)
