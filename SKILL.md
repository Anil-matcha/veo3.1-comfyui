---
name: muapi-director
description: >
  Turn ONE topic, talking-head video, or photo into a finished Vox-style paper-collage
  explainer / ad video on the MuAPI platform (api.muapi.ai) + local ffmpeg — script,
  collage keyframes, motion, voice-over, music, captions, all automated.
  Three input modalities: a topic (B-roll), a talking-head video (A-roll mode), or a single
  photo of a person/product anchored into the collage (C-roll mode).
---

# MuAPI Director

Turn a topic, talking-head video, or photo into a finished **Vox-style paper-collage video**: a bold, punchy, narrated explainer/ad where each beat is a torn-paper collage poster that comes alive, with voice-over, music and captions. Powered by **MuAPI** (`MUAPI_API_KEY`) + local **ffmpeg**.

## Input Modalities

1. **B-roll (topic → film):** Pass a topic (e.g. *"A brief history of Silicon Valley"*). It drafts a script, builds keyframe posters, animates them, synthesizes narration/BGM, and assembles the video.
2. **A-roll (talking-head video → collage):** Pass a presenter talking-head recording. It runs ASR, segments the transcript, and re-styles the video into paper-collage graphics while preserving facial features, lip-sync, and gestures frame-for-frame.
3. **C-roll (one photo → collage):** Pass a selfie or product shot. The subject is cut out as an unedited photographic sticker, anchored into generated collage posters per beat, and animated.

## Prerequisites

- `MUAPI_API_KEY` set in your environment (get one from https://muapi.ai).
- `ffmpeg` + `ffprobe` installed on system path.
- Python 3 with `Pillow` (`pip install pillow`).

## Workflow Steps

### B-roll Mode (Topic)
1. Write `out/<project>/beats.json` containing topic, language, and beat map.
2. `python3 scripts/style_bakeoff.py out/<project>` (Pick theme from candidates).
3. `python3 scripts/keyframes.py out/<project>` (Generates collage posters).
4. `python3 scripts/clips.py out/<project>` (Animates posters into motion clips).
5. `python3 scripts/audio.py out/<project>` (Generates TTS narration and instrumental BGM).
6. `python3 scripts/assemble.py out/<project>` (Muxes audio, ducking, captions, watermark -> `final.mp4`).

### A-roll Mode (Talking Head)
1. `python3 scripts/asr_beats.py out/<project> <source.mp4>`
2. `python3 scripts/aroll_clips.py out/<project>`
3. `python3 scripts/aroll_assemble.py out/<project>`

### C-roll Mode (Photo Anchor)
1. Add `"mode": "croll"`, `"anchor_photo": "photo.png"`, `"croll_subject": "portrait"|"product"` to `beats.json`.
2. `python3 scripts/croll_keyframes.py out/<project>`
3. `python3 scripts/clips.py out/<project>`
4. `python3 scripts/audio.py out/<project>`
5. `python3 scripts/assemble.py out/<project>`
