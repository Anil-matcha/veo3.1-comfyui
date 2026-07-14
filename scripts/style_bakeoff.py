#!/usr/bin/env python3
"""
Style bake-off: render ONE representative beat in several candidate collage styles so
the user can pick the visual idiom before committing the whole film.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import muapi_client
from styles import compose_collage_prompt, resolve_theme

DEFAULT_CANDIDATES = ["american-retro", "swiss-modern", "punk-zine", "atomic-age"]
IMAGE_MODEL = "flux-dev"

def first_shot(beat):
    return beat["shots"][0] if beat.get("shots") else beat

def bake_style(name, scene, ten, bg, aspect, img_model, img_res, out_dir):
    try:
        tp = resolve_theme(name) or {}
        prompt = compose_collage_prompt(
            scene=scene,
            title_en=ten,
            bg=bg,
            aspect=aspect,
            style=tp.get("idiom", name),
            palette=tp.get("palette"),
            type_style=tp.get("type_style"),
            finish=tp.get("finish")
        )
        print(f"[{name}] Generating image...")
        url = muapi_client.generate_image(
            model=img_model,
            prompt=prompt,
            aspect_ratio=aspect,
            resolution=img_res
        )
        dest_path = os.path.join(out_dir, f"{name}.jpg")
        print(f"[{name}] Downloading image...")
        muapi_client.download(url, dest_path)
        print(f"[{name}] Success -> {dest_path}")
        return name, True
    except Exception as e:
        print(f"[{name}] Failed: {e}")
        return name, False

def run(project_dir, styles=None, beat_index=0):
    styles = styles or DEFAULT_CANDIDATES
    beats_path = os.path.join(project_dir, "beats.json")
    if not os.path.exists(beats_path):
        print(f"Error: beats.json not found at {beats_path}")
        return
        
    with open(beats_path) as f:
        doc = json.load(f)
        
    aspect = doc.get("aspect", "16:9")
    img_model = doc.get("image_model", IMAGE_MODEL)
    img_res = doc.get("image_resolution", "1k")
    beat = doc["beats"][beat_index]
    shot = first_shot(beat)
    scene = shot["scene"]
    bg = beat.get("bg", "warm ochre")
    ten = beat.get("title_en", "")
    
    out_dir = os.path.join(project_dir, "style-bakeoff")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Starting bake-off for {len(styles)} styles using model '{img_model}'...")
    
    # Run styles in parallel using a ThreadPoolExecutor
    results = {}
    with ThreadPoolExecutor(max_workers=len(styles)) as executor:
        futures = [
            executor.submit(bake_style, name, scene, ten, bg, aspect, img_model, img_res, out_dir)
            for name in styles
        ]
        for f in futures:
            name, success = f.result()
            results[name] = success
            
    print(f"\nBakeoff finished. Results saved to: {out_dir}")
    print("Please review the candidates and update 'collage_style' in your beats.json.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 style_bakeoff.py <project_dir> [style1,style2,...] [beat_index]")
        sys.exit(1)
        
    proj = os.path.abspath(sys.argv[1])
    styles_list = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    bi = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    run(proj, styles_list, bi)
