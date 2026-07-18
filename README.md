# 🎬 Vox AI Motion Graphics Generator

**Turn any topic into a finished Vox-style paper-collage explainer or advertisement video — script, collage keyframes, animation, voice-over, music, and captions, all automated.**

> Inspired by the Vox-style paper-collage motion-graphics workflow — one topic in, a styled `final.mp4` out.

An **agent skill** designed to run end-to-end with a single hosted API key + local `ffmpeg`. Simply provide a one-line topic, and the tool produces a styled `final.mp4`.

---

## What It Is

The aesthetic is the modern editorial **paper-collage** popularized by Vox explainers: hand-cut paper cutouts, torn edges, tape, halftone dots, newspaper clippings, bold flat colors, and big cutout headlines — brought to life with dynamic motion, a narrator voice-over, music, and burned-in captions.

---

## How It Works

A single topic flows through a 6-stage pipeline driven by a single `beats.json` configuration file:

```
topic
  │
  ├─ 1. Beat Map        Pick a narrative arc → write beats.json (OpenAI key)  ◀── GATE 1: User approves beat map
  ├─ 2. Style Bake-Off  Render the same beat in 3–4 themes (Flux Dev)         ◀── GATE 2: User picks the look
  ├─ 3. Keyframes       One collage poster per beat (Flux Dev)
  ├─ 4. Motion          Animate each poster (Runway / Veo3 / Wan2.1)
  ├─ 5. Voice & Music   Narration (Minimax Speech 2.6) + BGM (Suno)
  ├─ 6. Assemble        Stitch clips, duck music under narration, burn captions (ffmpeg)
  └─ final.mp4
```

Two human decision gates keep you in control:
1. Approve the story beat map.
2. Select the visual theme preset.

Everything else is fully automated.

---

## Core Technologies

| Pipeline Job | Model Endpoint |
| :--- | :--- |
| **Keyframes (Text-to-Image)** | `flux-dev` |
| **Motion (Image-to-Video)** | `runway-image-to-video` or `veo3-image-to-video` |
| **Narration (TTS)** | `minimax-speech-2.6-turbo` |
| **Music** | `suno-create-music` |

---

## Installation

This is an **agent skill** — it works with any coding agent (like Claude Code, Codex, etc.) that can read a workflow and run scripts.

1. Install local dependencies:
   * **ffmpeg** + **ffprobe**
   * **Python 3** with **Pillow** (`pip install pillow`)

2. Configure environment keys:
   ```bash
   export MUAPI_API_KEY="your-api-key"   # image/video/voice/music models — key from muapi.ai
   export OPENAI_API_KEY="your-openai-key"
   ```

---

## Quick Start

Just ask your coding agent (with this skill loaded):

> *"Make me a 15-second Vox-style collage video introducing the history of coffee."*

The agent will draft a beat map, perform a style bake-off, generate keyframes, animate the clips, generate speech/music, and assemble the final product under `out/<project>/final.mp4`.
