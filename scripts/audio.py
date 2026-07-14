#!/usr/bin/env python3
"""
Audio stage: generate per-beat narration via Minimax Speech 2.6 and background music via Suno.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import muapi_client

VOICE_MODEL = "minimax-speech-2.6-turbo"
MUSIC_MODEL = "suno-create-music"

def probe_dur(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", path
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        return float(out.strip())
    except Exception:
        return 0.0

def generate_beat_narration(beat_id, text, voice_id, speed, emotion, adir):
    try:
        print(f"[narr_{beat_id}] Generating narration...")
        payload = {
            "voice_id": voice_id,
            "speed": speed,
            "volume": 1.0,
            "pitch": 0,
            "emotion": emotion,
            "format": "mp3"
        }
        url = muapi_client.generate_audio(VOICE_MODEL, text, **payload)
        dest = os.path.join(adir, f"narr_{beat_id}.mp3")
        print(f"[narr_{beat_id}] Downloading audio file...")
        muapi_client.download(url, dest)
        dur = round(probe_dur(dest), 2)
        print(f"[narr_{beat_id}] Success: {dur}s -> {dest}")
        return beat_id, url, dest, dur
    except Exception as e:
        print(f"[narr_{beat_id}] Failed: {e}")
        return beat_id, None, None, 0.0

def generate_bgm(music_prompt, adir):
    try:
        print("[bgm] Generating background music via Suno...")
        payload = {
            "custom_mode": True,
            "instrumental": True,
            "title": "BGM",
            "style": music_prompt
        }
        url = muapi_client.generate_audio(MUSIC_MODEL, music_prompt, **payload)
        dest = os.path.join(adir, "bgm.mp3")
        print("[bgm] Downloading BGM file...")
        muapi_client.download(url, dest)
        dur = round(probe_dur(dest), 2)
        print(f"[bgm] Success: {dur}s -> {dest}")
        return url, dest, dur
    except Exception as e:
        print(f"[bgm] Failed: {e}")
        return None, None, 0.0

def run(project_dir):
    bpath = os.path.join(project_dir, "beats.json")
    if not os.path.exists(bpath):
        print(f"Error: beats.json not found at {bpath}")
        return
        
    with open(bpath) as f:
        doc = json.load(f)
        
    adir = os.path.join(project_dir, "audio")
    os.makedirs(adir, exist_ok=True)
    
    voice = doc.get("voice", {})
    voice_id = voice.get("voice_id", "Friendly_Person")
    speed = float(voice.get("speed", 1.0))
    emotion = voice.get("emotion", "happy")
    
    jobs = []
    for beat in doc["beats"]:
        if beat.get("narration_audio"):
            print(f"[narr_{beat['id']}] Already generated. Skipping.")
            continue
        jobs.append((beat["id"], beat["narration"], voice_id, speed, emotion, adir))
        
    bgm_path = os.path.join(adir, "bgm.mp3")
    generate_bgm_job = not os.path.exists(bgm_path)
    if not generate_bgm_job:
        print(f"[bgm] Reusing existing background music at {bgm_path}")
        
    if not jobs and not generate_bgm_job:
        print("No new audio to generate.")
        return
        
    # Run all audio jobs concurrently in a thread pool
    narr_results = {}
    bgm_result = (None, None, 0.0)
    
    total_workers = len(jobs) + (1 if generate_bgm_job else 0)
    print(f"Submitting {total_workers} audio generation tasks in parallel...")
    
    with ThreadPoolExecutor(max_workers=max(total_workers, 1)) as executor:
        futures = []
        for beat_id, text, v_id, sp, em, folder in jobs:
            f = executor.submit(generate_beat_narration, beat_id, text, v_id, sp, em, folder)
            futures.append(("narr", f))
            
        if generate_bgm_job:
            music_prompt = doc.get("music", "lofi acoustic guitar background music, calm, instrumental")
            f = executor.submit(generate_bgm, music_prompt, adir)
            futures.append(("bgm", f))
            
        for job_type, f in futures:
            if job_type == "narr":
                beat_id, url, dest, dur = f.result()
                if url:
                    narr_results[beat_id] = (url, dest, dur)
            elif job_type == "bgm":
                bgm_result = f.result()
                
    # Update JSON documentation
    for beat in doc["beats"]:
        beat_id = beat["id"]
        if beat_id in narr_results:
            url, dest, dur = narr_results[beat_id]
            beat["narration_audio"] = dest
            beat["narration_dur"] = dur
            
    bgm_url, bgm_dest, bgm_dur = bgm_result
    if bgm_url:
        doc["bgm_path"] = bgm_dest
        doc["bgm_dur"] = bgm_dur
        
    with open(bpath, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print("Updated", bpath)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 audio.py <project_dir>")
        sys.exit(1)
        
    proj = os.path.abspath(sys.argv[1])
    run(proj)
