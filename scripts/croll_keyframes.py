#!/usr/bin/env python3
"""
C-roll keyframe stage: anchors ONE still photo inside generated collage posters ("cutout roll").
"""
import json
import os
import sys

from provider import get_provider, run_jobs
from styles import compose_collage_prompt, resolve_theme, image_params

EDIT_MODEL = "flux-dev"

FACE_LOCK = (
    "The person's face and hair from the attached photo are cut out as a PHOTOGRAPHIC "
    "sticker with a torn white paper border — keep the facial identity, features and the "
    "exact expression from the photo pixel-faithful; do not redraw, repaint or stylize the "
    "face; NO halftone dots, print texture or ink treatment on the face or hair — the face "
    "stays a clean photographic print. All poses and gestures are expressed by the body "
    "only. From the neck down the body is a hand-drawn paper-doll illustration jointed "
    "like a vintage paper puppet with visible cut edges, FULLY CLOTHED in {wardrobe}. "
)

PRODUCT_LOCK = (
    "{subject} from the attached photo is cut out as a PHOTOGRAPHIC sticker with a clean "
    "scissor-cut edge and a soft real paper drop shadow — keep its exact shape, materials, "
    "surface reflections and every word of its label typography pixel-faithful. Do not "
    "redraw, restyle or repaint the subject or its label. Re-style ONLY the world around "
    "it as printed paper collage. "
)

CROLL_GUARDS = (
    " Halftone dots and print textures live on the BACKGROUND only. Newspaper scraps carry "
    "completely UNREADABLE blurred micro-text. No readable text anywhere in the image."
)

FREEZE = {
    "portrait": ("FREEZE the photographic face sticker — it is a frozen layer, "
                 "pixel-identical to the still for the entire duration; never redraw, warp "
                 "or animate the face; the paper-doll body may shift slightly at its joints."),
    "product": ("FREEZE the photographic subject sticker and its label — a frozen layer, "
                "pixel-identical to the still for the entire duration; every letter of the "
                "label stays exactly as in the still; it may only settle gently with its "
                "drop shadow."),
}


def build_lock(doc):
    kind = doc.get("croll_subject", "portrait")
    if kind == "product":
        return PRODUCT_LOCK.format(subject=doc.get("subject_desc", "The product")), kind
    wardrobe = doc.get("subject_wardrobe", "the same outfit as in the photo")
    return FACE_LOCK.format(wardrobe=wardrobe), kind


def shots_of(beat):
    if beat.get("shots"):
        for s in beat["shots"]:
            yield s, f"{beat['id']}{s.get('id','')}"
    else:
        yield beat, f"{beat['id']}"


def run(project_dir):
    bpath = os.path.join(project_dir, "beats.json")
    with open(bpath) as f:
        doc = json.load(f)
    photo = doc["anchor_photo"]
    aspect = doc.get("aspect", "9:16")
    img_model = doc.get("image_model", EDIT_MODEL)
    img_res = doc.get("image_resolution", "1k")
    theme = resolve_theme(doc.get("theme")) or {}
    collage_style = theme.get("idiom") or doc.get("collage_style", "american-retro")
    t_palette = theme.get("palette") or doc.get("palette")
    t_type = theme.get("type_style") or doc.get("type_style")
    t_finish = theme.get("finish") or doc.get("finish")

    lock_text, kind = build_lock(doc)
    doc["anchor_freeze"] = FREEZE[kind]

    prov = get_provider(doc.get("provider"))
    if not photo.startswith("http://") and not photo.startswith("https://"):
        photo_url = prov.upload(photo)
        doc["anchor_photo_url"] = photo_url
    else:
        photo_url = photo

    kf_dir = os.path.join(project_dir, "keyframes")
    os.makedirs(kf_dir, exist_ok=True)

    specs, by_key = {}, {}
    for beat in doc["beats"]:
        for shot, key in shots_of(beat):
            if shot.get("keyframe_url"):
                continue
            scene = shot["scene"]
            base_prompt = compose_collage_prompt(scene, "", "", beat.get("bg", "warm ochre"),
                                                 aspect, with_title=False, style=collage_style,
                                                 palette=t_palette, type_style=t_type, finish=t_finish)
            prompt = f"{lock_text} {base_prompt}{CROLL_GUARDS}"
            shot["keyframe_prompt"] = prompt

            params = {"image_url": photo_url, **image_params(img_model, aspect, img_res)}
            specs[key] = (lambda p=prompt, par=params: prov.submit_image(img_model, p, **par))
            by_key[key] = shot

    done = run_jobs(prov, specs, poll_s=3, stall_s=75, max_retries=2, deadline_s=300)

    for key, url in done.items():
        if not url:
            continue
        dest = os.path.join(kf_dir, f"kf_{key}.jpg")
        prov.download(url, dest)
        shot = by_key[key]
        shot["keyframe_url"] = url
        shot["keyframe_path"] = dest
        print(f"[{key}] saved {dest}")

    with open(bpath, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print("Updated", bpath)


if __name__ == "__main__":
    proj = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "out", "croll-demo")
    run(os.path.abspath(proj))
