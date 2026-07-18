<div align="center">

<h1 align="center">Vox AI Motion Graphics Generator</h1>
<h3 align="center">Turn any topic into a finished Vox-style paper-collage explainer video</h3>

<p align="center">
  <img src="https://img.shields.io/badge/🐍Python-3.10+-00d9ff?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e">
  <img src="https://img.shields.io/badge/🎬ffmpeg-Powered-ff6b6b?style=for-the-badge&logo=ffmpeg&logoColor=white&labelColor=1a1a2e">
  <img src="https://img.shields.io/badge/License-MIT-4ecdc4?style=for-the-badge&logo=opensourceinitiative&logoColor=white&labelColor=1a1a2e">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Agent_Skill-Claude_Code_·_Codex-7c3aed?style=for-the-badge&logoColor=white&labelColor=1a1a2e">
  <img src="https://img.shields.io/badge/One_Key-Setup-FFC107?style=for-the-badge&logoColor=white&labelColor=1a1a2e">
</p>

</div>

---

### 🚨 The problem with making explainer videos today:
- ❌ **Fragmented workflow** — separate tools for scripting, image gen, animation, voice-over, music, and captions
- ❌ **No narrative structure** — raw text-to-video models don't understand story arcs, hooks, or pacing
- ❌ **Inconsistent look** — the collage aesthetic drifts from shot to shot
- ❌ **Hours of manual editing** — stitching, ducking music, and burning captions by hand

### 💡 The solution:
🎬 **Screenwriter**, **Collage Artist**, **Animator**, and **Editor** — all in one automated pipeline.

Type one topic. The agent writes the story beats, renders each beat as a torn-paper collage poster, animates it, adds a narrator voice-over, music, and burned-in captions — then stitches everything into a finished `final.mp4`. You stay in control with just **two approval gates**: the story beat map and the visual theme.

---

<p align="center">
  <a href="https://github.com/Anil-matcha/awesome-generative-ai-apps">
    <img src="https://img.shields.io/badge/Part%20of-Awesome%20Generative%20AI%20Apps-FFD700?style=for-the-badge&logo=github&logoColor=black" alt="Awesome Generative AI Apps">
  </a>
</p>

> 🎨 **[Explore 50+ more open-source AI apps →](https://github.com/Anil-matcha/awesome-generative-ai-apps)**

## 🎥 Demo

<!-- Drop a demo mp4/gif here — e.g. drag a file into this section on GitHub to embed it -->

> *"Make me a 15-second Vox-style collage video on the history of coffee."* → a styled `final.mp4`, fully narrated and captioned.

---

## 📑 Table of Contents

- [✨ Key Features](#-key-features)
- [🎨 The Look](#-the-look)
- [🔄 How It Works](#-how-it-works)
- [🧩 Models](#-models)
- [🚀 Quick Start](#-quick-start)
- [🛠️ Project Structure](#️-project-structure)
- [🔗 Related Projects](#-related-projects)

---

## ✨ Key Features

- **One topic → finished film** — a single one-line prompt produces a complete, captioned, narrated video
- **Authentic Vox collage aesthetic** — torn paper, cutouts, tape, halftone dots, newspaper clippings, bold flat color, big headlines
- **Story-first** — picks a narrative arc (timeline, problem-agitate-solution, how-it-works…) and writes a hook-led beat map before generating anything
- **Style bake-off** — renders the same beat in 3–4 themes so you pick the look before committing
- **Living-poster motion** — animates each collage keyframe into dynamic motion instead of static slides
- **Voice + music + captions** — narration, background music ducked under the voice, and burned-in captions, all automated
- **Two human gates, everything else automated** — approve the beat map, pick the theme; the pipeline handles the rest
- **Agent-native** — a self-contained skill any coding agent (Claude Code, Codex, …) can read and run
- **One API key + ffmpeg** — no cluster of accounts to wire up

---

## 🎨 The Look

The aesthetic is the modern editorial **paper-collage** popularized by Vox explainers and creators like Stav Zilber and rom1trs: hand-cut paper cutouts, torn edges, tape, halftone dots, newspaper clippings, bold flat colors per beat, and big cutout headlines — brought to life with dynamic motion, a narrator voice-over, music, and burned-in captions.

The collage look is born in the **image** step (each beat is a finished collage *poster*), and the **motion** is added after — so the DNA of the style is locked in before anything moves.

---

## 🔄 How It Works

A single topic flows through a 6-stage pipeline driven by one `beats.json` per project:

```
topic
  │
  ├─ 1. Beat Map        Pick a narrative arc → write beats.json          ◀── GATE 1: approve the beat map
  ├─ 2. Style Bake-Off  Render the same beat in 3–4 themes               ◀── GATE 2: pick the look
  ├─ 3. Keyframes       One collage poster per beat
  ├─ 4. Motion          Animate each poster into a clip
  ├─ 5. Voice & Music   Narration + background music
  ├─ 6. Assemble        Stitch clips, duck music, burn captions (ffmpeg)
  └─ final.mp4
```

Two human decision gates keep you in control — **approve the story beat map** and **select the visual theme**. Everything else is fully automated.

---

## 🧩 Models

| Pipeline Job | Model |
| :--- | :--- |
| **Keyframes** (text-to-image) | `flux-dev` |
| **Motion** (image-to-video) | `runway-image-to-video` / `veo3-image-to-video` / `wan2.1` |
| **Narration** (text-to-speech) | `minimax-speech-2.6-turbo` |
| **Music** | `suno-create-music` |

---

## 🚀 Quick Start

**1. Install local dependencies**
- **ffmpeg** + **ffprobe** (`brew install ffmpeg` on macOS)
- **Python 3** with **Pillow** (`pip install pillow`)

**2. Configure environment keys**
```bash
export MUAPI_API_KEY="your-api-key"     # image / video / voice / music models — key from muapi.ai
export OPENAI_API_KEY="your-openai-key"  # story planning
```

**3. Ask your coding agent** (with this skill loaded)
> *"Make me a 15-second Vox-style collage video introducing the history of coffee."*

The agent drafts a beat map, runs a style bake-off, generates keyframes, animates the clips, generates speech and music, and assembles the result under `out/<project>/final.mp4`.

---

## 🛠️ Project Structure

```
.
├── SKILL.md            # the full agent workflow + the two approval gates
├── AGENTS.md           # entry point for any coding agent
├── scripts/            # one script per pipeline stage
│   ├── style_bakeoff.py
│   ├── keyframes.py
│   ├── clips.py
│   ├── audio.py
│   ├── assemble.py
│   └── ...
├── references/         # prompt guide, beat/story library, voices
└── examples/           # sample beats.json
```

This is an **agent skill** — Claude Code auto-loads it from `SKILL.md`; Codex and other agents follow `SKILL.md` via `AGENTS.md`. Just ask for a *"vox video"* or a *"collage video."*

---

## 🔗 Related Projects

- [Open-AI-Micro-Drama-Generator](https://github.com/Anil-matcha/Open-AI-Micro-Drama-Generator) — agentic AI micro-drama video generator
- [AI-B-roll](https://github.com/Anil-matcha/AI-B-roll) — auto-generate AI b-roll for your videos
- [Text-To-Video-AI](https://github.com/SamurAIGPT/Text-To-Video-AI) — generate full videos from text
- [AI-Youtube-Shorts-Generator](https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator) — auto-clip long videos into viral vertical shorts
- [awesome-ai-video-models](https://github.com/Anil-matcha/awesome-ai-video-models) — compare AI video models by API, price & speed

---

<div align="center">

⭐ **Star this repo if it helped you** — and [explore 50+ more open-source AI apps →](https://github.com/Anil-matcha/awesome-generative-ai-apps)

</div>
