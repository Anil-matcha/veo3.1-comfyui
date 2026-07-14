---
name: muapi-director
description: >
  Turn ONE topic into a finished Vox-style paper-collage explainer / ad video, end to end
  on the Muapi API + local ffmpeg — script, collage keyframes, motion, voice-over,
  music, captions, all automated. Use this whenever the user wants a "Vox style" video,
  a paper/torn-paper collage animation, a "motion collage", a narrated explainer or short
  ad built from AI-generated collage posters, a scrapbook-style tribute, or wants to turn
  a topic / product / person into a punchy narrated collage video — even if they don't say
  the word "Vox".
  Triggers: "vox video", "collage video", "motion collage", "paper collage
  explainer", "make a collage ad", "turn this topic into a collage video".
---

# Muapi Director

Turn a one-line topic into a finished **Vox-style paper-collage video**: a bold, punchy,
narrated explainer/ad where each beat is a torn-paper collage poster that comes alive, with
voice-over, music and captions. Runs on **one Muapi API key** + local **ffmpeg**.

The look is the modern editorial paper-collage popularized by Vox explainers and creators
like Stav Zilber / rom1trs: hand-cut paper cutouts, torn edges, tape, halftone dots,
newspaper clippings, bold flat color per beat, big cutout headlines.

## The core idea (read this first)

The Vox collage look and the collage motion are **two different steps**:

1. **The look is born in the IMAGE step.** Each beat is a finished collage *poster* made by a
   text-to-image model (**flux-dev**). All the collage DNA (torn paper, cutouts, halftone, bold color,
   headline text) lives in that image. If the image isn't a rich collage, nothing downstream
   will save it.
2. **The motion is added after.** By default an AI video model animates the whole poster (the
   "living poster" path — simple, automated, using **runway-image-to-video** or **veo3-image-to-video**).

Everything hinges on the prompts. **Before writing any image or video prompt, read
`references/prompt-guide.md`** — it has the exact prompt structures that make the difference
between "a real Vox collage" and "a moving PowerPoint".

## Prerequisites (check, don't skip)

- `echo "${MUAPI_API_KEY:+set}"` — if empty, tell the user to set it (get one from the Muapiapp Dashboard) and stop.
- `echo "${OPENAI_API_KEY:+set}"` — required for prompt planning / scripts.
- `command -v ffmpeg ffprobe` — required for assembly (`brew install ffmpeg` on macOS or standard Windows install).
- `python3 -c "import PIL"` — Pillow, for captions/watermark overlays.

## Standard workflow (topic → film)

This is the default, most-automated path. Every stage is one script, all driven by a single
`beats.json` per project under `out/<project>/`.

1. **Topic → beat map.** First **read `references/beat-layer.md`** (the story layer) and pick a
   narrative `arc` that fits the topic (`timeline` for history, `pas` for ads,
   `how_it_works` for explainers, …). Then write
   `out/<project>/beats.json` following that arc: **beat-1 headline must be a ≤3s hook**; beat
   count per duration (30s→6–8, 60s→10–12); split each beat into **2 shots** (wide+detail) with
   **per-shot `camera_move` VARIED across adjacent beats** (never repeat; `static` on the payoff)
   and **rich `element_motion`** (see step 4). Each beat: `narration`, `title_cn`/`title_en`,
   `scene`, `bg`, `feel`, `hook`. This draft is the **one mandatory approval gate** — show the
   user the beat map before generating. Examples in `examples/`.

2. **Pick the visual style (hybrid — do this BEFORE keyframes).** Do not reuse one house style
   for every topic. Read `references/prompt-guide.md` (§5 theme presets); pick 3–4 **theme presets**
   (`american-retro`, `swiss-modern`, `punk-zine`, `soviet-constructivist`, `70s-groovy`, `chinese-ink`, `atomic-age`)
   that fit the topic's era/culture/tone. Run a bake-off and let the user pick by eye — AI
   proposes, the library is the quality floor, the human decides. Set the pick as `"theme"`:
   `python3 scripts/style_bakeoff.py out/<project> american-retro,swiss-modern,punk-zine`
   Set the chosen name as `"collage_style"` in beats.json (keyframes.py reads it).

3. **Keyframes (the collage look).** `python3 scripts/keyframes.py out/<project>`
   Generates one collage poster per beat/shot with **flux-dev**, headline text baked in.
   Compose prompts with the 5-part structure in `references/prompt-guide.md`. Verify each poster
   looks like a *real layered collage* before animating.

4. **Motion.** `python3 scripts/clips.py out/<project>`
   Animates each poster with **runway-image-to-video** (or **veo3-image-to-video** / **wan2.1-image-to-video**).
   Two independent axes (see `references/beat-layer.md` §3):
   • **`camera_move`** — ONE move per shot. Safe/default: `{static, push_in, pull_out, pan, tilt, parallax}`.
   • **`element_motion`** — where the energy lives; AI writes it per beat to fit that scene. Make it RICH.

5. **Voice + music.** `python3 scripts/audio.py out/<project>`
   One consistent narrator via **minimax-speech-2.6-turbo** + instrumental BGM via **suno-create-music**.
   **Pick `voice_id` to fit the topic + language** — see `references/voices.md` for the voice roster.

6. **Assemble.** `python3 scripts/assemble.py out/<project>`
   ffmpeg: normalize + concat all shots, lay the single narration ducked under the music,
   burn captions timed per beat, add the watermark. Output `out/<project>/final.mp4`.

7. **Verify.** Extract frames to jpg and look:
   `ffmpeg -ss <t> -i final.mp4 -vf "scale=640:-1,format=yuvj420p" -frames:v 1 f.jpg`

### Cadence — how long shots should be

A common mistake is one long shot per beat. On a 9:16 / social piece especially, a static
10s shot reads as dead air. Aim for a **cut every ~4–6 seconds**:

- **Shots run 3–6s; never let a single shot exceed ~7s** — beyond that the AI motion has
  nowhere to go and it feels static.
- **A beat's narration is ~8–10s, so give each beat 2 shots** (a *wide* establishing shot with
  the headline + a *detail* cut-in without it). The narration plays continuously across both;
  the visual cuts mid-sentence.
- So a ~60s film is typically **~6 beats × 2 shots × ~5s = 12 shots**, not 6 × 10s.
- Reuse the wide keyframe as shot `a`; generate a tighter detail scene for shot `b`.

## beats.json schema

```json
{
  "project": "my-film", 
  "topic": "...", 
  "language": "en",
  "aspect": "9:16",                       // 16:9 | 9:16 | 1:1
  "style": "collage",
  "provider": "muapiapp",
  "theme": "american-retro",              // THEME_PRESET (styles.THEME_PRESETS) — the LOOK layer
  "arc": "timeline",                      // narrative arc (beat-layer.md) — the STORY skeleton
  "video_model": "runway-image-to-video", // runway-image-to-video | veo3-image-to-video
  "image_model": "flux-dev",              // keyframes
  "motion_style": "punchy",               // amplitude: calm | punchy | max
  "voice": {"voice_id": "Friendly_Person", "language": "en", "speed": 1.0, "emotion": "happy"},  // voices.md
  "music": "epic cinematic orchestral, instrumental, no vocals",
  "mix": {"music": 0.6, "voice": 1.25},   // audio balance
  "watermark": "Made with Muapi Director",
  "beats": [
    {
      "id": 1, 
      "title_en": "THE BEAT TITLE",
      "bg": "bold blue cardboard", 
      "feel": "retro, energetic",
      "narration": "Narrator line for this beat...",
      "shots": [
        {
          "id": "a", 
          "dur": 5, 
          "title": true, 
          "shot_size": "WIDE", 
          "camera_move": "push_in",
          "scene": "Wide establishing scene description...",
          "element_motion": "Element motions..."
        }
      ]
    }
  ]
}
```
