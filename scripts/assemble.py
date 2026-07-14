#!/usr/bin/env python3
"""
Assembly stage (ffmpeg): multi-shot clips + per-beat narration + music -> final.mp4
"""
import json
import os
import subprocess
import sys

# Ensure scripts directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import text_overlay

FPS, TAIL = 24, 0.5
WATERMARK = "Made with Muapi Director"
RES = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}

def ff(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)

def probe_dur(path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", path
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        return float(out.strip())
    except Exception:
        return 0.0

def is_valid_audio(path):
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(100)
            if b"ftypavif" in header or b"JFIF" in header or b"PNG" in header:
                return False
    except Exception:
        return False
    return True

def shots_of(beat):
    if beat.get("shots"):
        for s in beat["shots"]:
            yield s
    else:
        yield beat

def run(project_dir):
    bpath = os.path.join(project_dir, "beats.json")
    if not os.path.exists(bpath):
        print(f"Error: beats.json not found at {bpath}")
        return
        
    with open(bpath) as f:
        doc = json.load(f)
        
    beats = doc["beats"]
    W, H = RES.get(doc.get("aspect", "16:9"), (1920, 1080))
    wm_text = doc.get("watermark", WATERMARK)
    mix = doc.get("mix", {})
    music_vol = float(mix.get("music", 0.6))
    voice_vol = float(mix.get("voice", 1.25))
    
    tmp = os.path.join(project_dir, "_seg")
    os.makedirs(tmp, exist_ok=True)
    
    # 1. Flatten shots into timed segments
    segs = []
    beat_spans = []
    t = 0.0
    
    for beat in beats:
        beat_start = t
        shot_list = list(shots_of(beat))
        durs = [float(s.get("dur", 5)) for s in shot_list]
        need = float(beat.get("narration_dur", sum(durs))) + TAIL
        
        if sum(durs) < need:
            durs[-1] += need - sum(durs)
            
        for s, d in zip(shot_list, durs):
            clip_path = s.get("clip_path")
            if not clip_path or not os.path.exists(clip_path):
                print(f"Error: clip path missing or file not found for shot {s.get('id', '')} in beat {beat['id']}")
                return
            segs.append({"clip": clip_path, "dur": round(d, 2)})
            t += d
            
        beat_spans.append({"start": beat_start, "dur": round(t - beat_start, 2), "beat": beat})
        
    total = round(t, 2)
    
    # 2. Normalise each shot to a silent segment of exactly its dur
    seg_files = []
    for i, s in enumerate(segs):
        out = os.path.join(tmp, f"seg_{i:02d}.mp4")
        cd = probe_dur(s["clip"])
        factor = s["dur"] / cd if cd > 0 else 1.0
        pre = f"setpts={factor:.4f}*PTS," if factor > 1.02 else ""
        
        # Blurred-fill background so off-aspect clips fit nicely
        fc = (f"[0:v]{pre}split[s0][s1];"
              f"[s0]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
              f"boxblur=26:2,eq=brightness=-0.05[bg];"
              f"[s1]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={FPS},"
              f"tpad=stop_mode=clone:stop_duration=1[v]")
              
        ff(["-i", s["clip"], "-an", "-filter_complex", fc, "-map", "[v]", "-t", f"{s['dur']}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", out])
        seg_files.append(out)
        
    # 3. Concat all shot segments (video only)
    listf = os.path.join(tmp, "list.txt")
    with open(listf, "w") as f:
        for sf in seg_files:
            f.write(f"file '{os.path.abspath(sf)}'\n")
            
    body = os.path.join(tmp, "body_silent.mp4")
    ff(["-f", "concat", "-safe", "0", "-i", listf, "-c", "copy", body])
    
    # 4. Captions (per beat) + watermark PNGs
    cap_pngs = []
    for bs in beat_spans:
        beat = bs["beat"]
        p = os.path.join(tmp, f"cap_{beat['id']}.png")
        kf = next((s["keyframe_path"] for s in shots_of(beat)
                   if s.get("keyframe_path") and os.path.exists(s["keyframe_path"])), None)
        acc = text_overlay.accent_color(kf) if kf else None
        text_overlay.render_caption(beat["narration"], p, W, H, accent=acc)
        cap_pngs.append(p)
        
    wm_png = text_overlay.render_watermark(wm_text, os.path.join(tmp, "wm.png"), W, H)
    
    # 5. One pass: overlay captions+wm, mix per-beat narration, duck BGM
    has_audio = True
    for bs in beat_spans:
        if not is_valid_audio(bs["beat"].get("narration_audio")):
            has_audio = False
    if not is_valid_audio(doc.get("bgm_path")):
        has_audio = False

    nb = len(beat_spans)
    inputs = ["-i", body]
    for p in cap_pngs:
        inputs += ["-i", p]
    inputs += ["-i", wm_png]
    
    chain, prev = [], "[0:v]"
    for i, bs in enumerate(beat_spans):
        s, e = bs["start"] + 0.2, bs["start"] + bs["dur"] - 0.1
        lbl = f"[v{i+1}]"
        chain.append(f"{prev}[{i+1}:v]overlay=0:0:enable='between(t,{s:.2f},{e:.2f})'{lbl}")
        prev = lbl
    chain.append(f"{prev}[{nb+1}:v]overlay=0:0[v]")
    
    final = os.path.join(project_dir, "final.mp4")

    if has_audio:
        narr_base = nb + 2
        for bs in beat_spans:
            inputs += ["-i", bs["beat"]["narration_audio"]]
            
        bgm_idx = narr_base + nb
        inputs += ["-i", doc["bgm_path"]]

        # Per-beat narration delayed to its start, then mixed
        nlabels = []
        for i, bs in enumerate(beat_spans):
            ms = int(bs["start"] * 1000)
            chain.append(f"[{narr_base+i}:a]adelay={ms}:all=1[n{i}]")
            nlabels.append(f"[n{i}]")
            
        # Pad narration to full duration
        chain.append(f"{''.join(nlabels)}amix=inputs={nb}:normalize=0:duration=longest,volume={voice_vol},apad,atrim=0:{total}[narrmix]")
        chain.append("[narrmix]asplit=2[narrA][narrB]")
        chain.append(f"[{bgm_idx}:a]atrim=0:{total},volume={music_vol},afade=t=out:st={max(total-2,0):.2f}:d=2[bgt]")
        chain.append("[bgt][narrA]sidechaincompress=threshold=0.02:ratio=12:attack=5:release=350[bgd]")
        chain.append(f"[narrB][bgd]amix=inputs=2:normalize=0:duration=longest,volume=1.4,atrim=0:{total}[a]")
        filt = ";".join(chain)
        
        ff([*inputs, "-filter_complex", filt, "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", final])
    else:
        print("[Warning] Audio files are invalid/images (Sandbox Key returned mock AVIF files). Compiling silent video...")
        filt = ";".join(chain)
        ff([*inputs, "-filter_complex", filt, "-map", "[v]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", final])
        
    print(f"Success! FINAL VIDEO -> {final} (~{total}s, {len(segs)} shots)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 assemble.py <project_dir>")
        sys.exit(1)
        
    proj = os.path.abspath(sys.argv[1])
    run(proj)
