#!/usr/bin/env python3
"""
Keyframe stage: generate one styled keyframe per shot.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import muapi_client
from styles import compose_collage_prompt, resolve_theme

IMAGE_MODEL = "flux-dev"

def shots_of(beat):
    """Yield (shot_dict, shot_key) for a beat; synthesize one shot if none."""
    if beat.get("shots"):
        for s in beat["shots"]:
            yield s, f"{beat['id']}{s.get('id','')}"
    else:
        yield beat, f"{beat['id']}"

def generate_shot_keyframe(key, shot, prompt, img_model, img_res, kf_dir):
    try:
        print(f"[{key}] Generating image...")
        url = muapi_client.generate_image(
            model=img_model,
            prompt=prompt,
            aspect_ratio=shot.get("aspect", "16:9"),  # Fallback to shot-level or default
            resolution=img_res
        )
        dest = os.path.join(kf_dir, f"kf_{key}.jpg")
        print(f"[{key}] Downloading image...")
        muapi_client.download(url, dest)
        print(f"[{key}] Success -> {dest}")
        return key, url, dest
    except Exception as e:
        print(f"[{key}] Failed: {e}")
        return key, None, None

def run(project_dir):
    bpath = os.path.join(project_dir, "beats.json")
    if not os.path.exists(bpath):
        print(f"Error: beats.json not found at {bpath}")
        return
        
    with open(bpath) as f:
        doc = json.load(f)
        
    aspect = doc.get("aspect", "16:9")
    img_model = doc.get("image_model", IMAGE_MODEL)
    img_res = doc.get("image_resolution", "1k")
    theme = resolve_theme(doc.get("theme")) or {}
    collage_style = theme.get("idiom") or doc.get("collage_style", "american-retro")
    
    t_palette = theme.get("palette") or doc.get("palette")
    t_type = theme.get("type_style") or doc.get("type_style")
    t_finish = theme.get("finish") or doc.get("finish")
    
    kf_dir = os.path.join(project_dir, "keyframes")
    os.makedirs(kf_dir, exist_ok=True)
    
    jobs = []
    by_key = {}
    
    for beat in doc["beats"]:
        for shot, key in shots_of(beat):
            if shot.get("keyframe_url"):
                print(f"[{key}] Already has keyframe. Skipping.")
                continue
                
            scene = shot["scene"]
            # Enforce aspect ratio at shot level or fall back to document level
            shot_aspect = shot.get("aspect", aspect)
            shot["aspect"] = shot_aspect
            
            prompt = compose_collage_prompt(
                scene=scene,
                title_en=beat.get("title_en", ""),
                bg=beat.get("bg", "warm ochre"),
                aspect=shot_aspect,
                with_title=shot.get("title", True),
                style=collage_style,
                palette=t_palette,
                type_style=t_type,
                finish=t_finish
            )
            shot["keyframe_prompt"] = prompt
            jobs.append((key, shot, prompt))
            by_key[key] = shot
            
    if not jobs:
        print("No new keyframes to generate.")
        return
        
    print(f"Generating {len(jobs)} keyframes in parallel...")
    with ThreadPoolExecutor(max_workers=min(len(jobs), 4)) as executor:
        futures = [
            executor.submit(generate_shot_keyframe, key, shot, prompt, img_model, img_res, kf_dir)
            for key, shot, prompt in jobs
        ]
        for f in futures:
            key, url, dest = f.result()
            if url:
                shot = by_key[key]
                shot["keyframe_url"] = url
                shot["keyframe_path"] = dest
                
    with open(bpath, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print("Updated", bpath)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 keyframes.py <project_dir>")
        sys.exit(1)
        
    proj = os.path.abspath(sys.argv[1])
    run(proj)
