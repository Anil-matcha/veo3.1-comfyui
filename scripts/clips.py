#!/usr/bin/env python3
"""
Clip stage: animate each SHOT's keyframe into a short video clip via MuAPI.
"""
import json
import os
import sys

from provider import get_provider, run_jobs
from styles import resolve_theme, resolve_video_aspect

VIDEO_MODEL = "veo3.1-image-to-video"


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
    "parallax": "a gentle multi-layer parallax drift (paper layers moving at slightly different speeds)",
    "orbit":     "a slow 3D orbit / camera arc around the scene",
    "dolly_zoom": "a dolly-zoom — push in while the background pulls back",
    "roll":      "a slow camera roll",
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
                   has_title=True, constraints="strict", anchor_freeze=None):
    cam = CAMERA_VOCAB.get(camera, camera)
    element = element_motion or DEFAULT_ELEMENT
    feel = feel or "tactile, editorial, a page from a scrapbook"
    palette = palette or "the still's existing palette, high contrast"
    amp = AMPLITUDE.get(amplitude, AMPLITUDE["punchy"])
    text_lock = ("Keep the HEADLINE TEXT sharp, legible and stable — do not warp or wobble the lettering. " if has_title else "")
    if anchor_freeze:
        text_lock = f"{anchor_freeze} " + text_lock
    if constraints == "loose":
        guard = (text_lock + "Keep the paper-collage look (printed cut-outs, not photoreal); "
                 "avoid a subject melting into shapeless goo. Otherwise explore freely.")
    else:
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


def run(project_dir):
    bpath = os.path.join(project_dir, "beats.json")
    with open(bpath) as f:
        doc = json.load(f)
    model = doc.get("video_model", VIDEO_MODEL)
    target_aspect = doc.get("aspect", "16:9")
    resolved_aspect, exact = resolve_video_aspect(target_aspect, model)
    theme = resolve_theme(doc.get("theme")) or {}
    t_amp = theme.get("motion_style", "punchy")

    clip_dir = os.path.join(project_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)

    prov = get_provider(doc.get("provider"))
    specs, by_key = {}, {}
    for beat in doc["beats"]:
        b_feel = beat.get("feel")
        for shot, key in shots_of(beat):
            if shot.get("clip_url") and os.path.exists(shot.get("clip_path", "")):
                continue
            kf = shot.get("keyframe_url")
            if not kf and shot.get("keyframe_path"):
                kf = prov.upload(shot["keyframe_path"])
                shot["keyframe_url"] = kf
            if not kf:
                print(f"[{key}] no keyframe_url, skipping")
                continue

            cam = shot.get("camera_move", "push_in")
            elem = shot.get("element_motion")
            amp = shot.get("motion_style") or t_amp
            constr = shot.get("constraints", "strict")
            prompt = collage_prompt(cam, elem, feel=b_feel, amplitude=amp,
                                    has_title=shot.get("title", True), constraints=constr,
                                    anchor_freeze=beat.get("anchor_freeze"))
            shot["clip_prompt"] = prompt

            dur = 8 if "veo3" in model else int(shot.get("duration", 5))
            params = {"image_url": kf, "duration": dur}
            if resolved_aspect:
                params["aspect_ratio"] = resolved_aspect

            specs[key] = (lambda p=prompt, par=params: prov.submit_video(model, p, **par))
            by_key[key] = shot

    done = run_jobs(prov, specs, poll_s=4, stall_s=120, max_retries=2, deadline_s=900)

    for key, url in done.items():
        if not url:
            continue
        dest = os.path.join(clip_dir, f"clip_{key}.mp4")
        prov.download(url, dest)
        shot = by_key[key]
        shot["clip_url"] = url
        shot["clip_path"] = dest
        print(f"[{key}] saved {dest}")

    with open(bpath, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print("Updated", bpath)


if __name__ == "__main__":
    proj = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "out", "demo")
    run(os.path.abspath(proj))
