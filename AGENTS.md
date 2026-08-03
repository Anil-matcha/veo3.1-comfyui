# MuAPI Director — Agent Guide

This repository is an **Agent Skill**: a self-contained workflow that turns a topic, talking-head video, or photo into a finished Vox-style paper-collage video (script → keyframes → motion → voice-over → music → captions) powered by **MuAPI** (`api.muapi.ai`). It also includes custom ComfyUI nodes for Google Veo 3.1.

## How to use it (for the agent)

1. Read **`SKILL.md`** — the full workflow covering B-roll, A-roll, and C-roll modes and approval gates.
2. Work one project at a time under `out/<project>/`, driven by a single `beats.json`.
3. Run the pipeline stages in **`scripts/`**:
   - **B-roll (topic → film):** `style_bakeoff.py → keyframes.py → clips.py → audio.py → assemble.py`
   - **A-roll (talking-head → collage):** `asr_beats.py → aroll_clips.py → aroll_assemble.py`
   - **C-roll (photo anchor → collage):** `croll_keyframes.py → clips.py → audio.py → assemble.py`

## Requirements

- `MUAPI_API_KEY` in the environment — https://muapi.ai
- `ffmpeg` + `ffprobe`
- Python 3 with `Pillow`
