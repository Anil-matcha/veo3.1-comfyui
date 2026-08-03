#!/usr/bin/env python3
"""
A-roll clip stage: cut each ASR beat's time range out of the source recording
and re-style it as a paper-collage clip via MuAPI video models.
"""
import json
import os
import subprocess
import sys

from provider import get_provider, run_jobs
from styles import resolve_theme, resolve_video_aspect

PRIMARY_MODEL = "veo31-image-to-video"


def cut_segment(src, start, end, dest, pad=0.15):
    dur = max(end - start, 0.1) + pad
    ss = max(start - pad / 2, 0)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{ss:.2f}", "-i", src,
                    "-t", f"{dur:.2f}", "-c:v", "libx264", "-c:a", "aac",
                    "-pix_fmt", "yuv420p", dest], check=True)


def build_prompt(theme, content_beats=""):
    tp = theme or {}
    idiom = tp.get("idiom", "american-retro")
    palette = tp.get("palette", "bold retro primaries")
    type_style = tp.get("type_style", "bold condensed headlines")
    finish = tp.get("finish", "aged paper, high contrast")
    beats_line = f"\n\nCONTENT BEATS: {content_beats}" if content_beats else ""
    return (
        "Transform my raw talking-head video into a mixed-media collage motion graphic.\n\n"
        "TALKING-HEAD LOCK (most important):\n"
        "The presenter is a PHOTOGRAPHIC paper cut-out sticker with a bold black printed "
        "outline and a torn-paper silhouette edge -- her actual photographic likeness, "
        "facial performance, LIP MOVEMENTS, eye-line and hand gestures follow the source "
        "video frame-for-frame, unmodified. Do NOT re-texture, redraw, or halftone-print the "
        "face or skin itself -- only the silhouette edge and the world around her are paper-"
        "collage. Keep ALL motion, timing and framing from the source.\n\n"
        f"STYLE (background & props): {idiom}. Palette: {palette}. Typography: {type_style}. "
        f"Finish: {finish}. Clearly layered hand-cut paper cut-outs with visible torn and "
        "scissor-cut edges, tape corners and soft real paper drop shadows, on a bold flat "
        f"paper background.{beats_line}\n\n"
        "CONSTRAINTS: background elements and stamps never cover her face or hands."
    )


def run(project_dir, only=None):
    bpath = os.path.join(project_dir, "beats.json")
    with open(bpath) as f:
        doc = json.load(f)
    if doc.get("mode") != "aroll":
        raise SystemExit(f"{bpath} is not an A-roll project (mode != 'aroll')")

    src = doc["source_video"]
    target_aspect = doc.get("aspect", "9:16")
    model = doc.get("video_model", PRIMARY_MODEL)
    resolved_aspect, exact = resolve_video_aspect(target_aspect, model)

    theme = resolve_theme(doc.get("theme")) or {}
    prov = get_provider(doc.get("provider"))

    seg_dir = os.path.join(project_dir, "_aroll_segs")
    clip_dir = os.path.join(project_dir, "clips")
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)

    specs, by_id = {}, {}
    for beat in doc["beats"]:
        bid = str(beat["id"])
        if only and bid not in only:
            continue
        if beat.get("clip_url") and os.path.exists(beat.get("clip_path", "")):
            continue

        seg_file = os.path.join(seg_dir, f"seg_{bid}.mp4")
        if not os.path.exists(seg_file):
            cut_segment(src, beat["start"], beat["end"], seg_file)

        seg_url = prov.upload(seg_file)
        beat["source_clip_url"] = seg_url

        prompt = build_prompt(theme, beat.get("content_beats", ""))
        beat["clip_prompt"] = prompt

        params = {"image_url": seg_url, "duration": int(beat.get("dur", 5))}
        if resolved_aspect:
            params["aspect_ratio"] = resolved_aspect

        specs[bid] = (lambda p=prompt, par=params: prov.submit_video(model, p, **par))
        by_id[bid] = beat

    done = run_jobs(prov, specs, poll_s=4, stall_s=120, max_retries=2, deadline_s=900)

    for bid, url in done.items():
        if not url:
            continue
        dest = os.path.join(clip_dir, f"clip_{bid}.mp4")
        prov.download(url, dest)
        beat = by_id[bid]
        beat["clip_url"] = url
        beat["clip_path"] = dest
        print(f"[{bid}] saved {dest}")

    with open(bpath, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print("Updated", bpath)


if __name__ == "__main__":
    proj = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "out", "aroll-demo")
    only_ids = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    run(os.path.abspath(proj), only_ids)
