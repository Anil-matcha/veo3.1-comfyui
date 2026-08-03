<p align="right"><a href="README.md">English</a> · <b>简体中文</b></p>

# 🎬 Vox Director (拼贴动效导演) & Veo 3.1 ComfyUI 节点

**一个选题、口播视频或照片进，一条成片出——脚本、拼贴关键帧、动效、旁白、配乐、字幕全流程自动化的 Vox 风格拼贴讲解/广告视频。**

一个基于 [MuAPI](https://muapi.ai) 平台 API 和本地 `ffmpeg` 的 **Agent 技能** 与 **ComfyUI 节点套件**，可供任何编码 Agent（Claude Code、Codex、Antigravity 等）或在 ComfyUI 中使用。给它一句话选题、口播视频或产品照片，直接输出成片 `.mp4`。

![License: MIT](https://img.shields.io/badge/License-MIT-black.svg) ![Powered by MuAPI](https://img.shields.io/badge/powered%20by-MuAPI-0052FF.svg) ![Agent Skill](https://img.shields.io/badge/Agent-Skill-d97757.svg) ![ComfyUI Nodes](https://img.shields.io/badge/ComfyUI-Nodes-7B2CBF.svg)

<div align="center">

`out/demo/final.mp4`

<b>▶《咖啡简史（埃塞俄比亚公元 850 年）》· 5 秒 (B-roll)</b>

</div>

<table>
  <tr>
    <td width="50%"><a href="out/demo/final.mp4"><img src="out/demo/keyframes/kf_1a.jpg" width="100%" alt="咖啡简史"></a></td>
    <td width="50%"><a href="out/demo-croll/final.mp4"><img src="out/demo-croll/keyframes/kf_1.jpg" width="100%" alt="冷萃咖啡产品广告"></a></td>
  </tr>
  <tr>
    <td align="center"><sub>咖啡简史 · 5 秒 (B-roll)</sub></td>
    <td align="center"><sub>冷萃咖啡产品广告 · 5 秒 (C-roll)</sub></td>
  </tr>
</table>

<p align="center"><sub><em>▶ 点击任意封面/路径播放生成视频</em></sub></p>

---

## 📽️ 实测生成视频（基于 MuAPI 自动生成）

以下是使用 **MuAPI Director** 全流程自动生成的示例视频：

| 模式 | 影片 | 选题 / 主体 | 最终合成视频 | 拼贴海报封面 | 动效模型 | 旁白音色 |
|---|---|---|---|---|---|---|
| **B-roll** | **☕ 咖啡简史** | *埃塞俄比亚公元 850 年* | `out/demo/final.mp4` | `out/demo/keyframes/kf_1a.jpg` | `veo3.1-image-to-video` | MiniMax TTS (`Q19bea09caa6IRAeW7`) |
| **C-roll** | **🍾 冷萃咖啡产品广告** | *冷萃咖啡瓶照片锚定* | `out/demo-croll/final.mp4` | `out/demo-croll/keyframes/kf_1.jpg` | `veo3.1-image-to-video` | MiniMax TTS (`Q19bea09caa6IRAeW7`) |

> 🎬 **生成素材汇总：**
> - ☕ **B-roll (咖啡简史):**
>   - 🖼️ **海报:** `out/demo/keyframes/kf_1a.jpg` \| 📹 **动效:** `out/demo/clips/clip_1a.mp4` \| 🎙️ **旁白:** `out/demo/audio/narr_1.mp3` \| 🎵 **配乐:** `out/demo/audio/bgm.mp3`
> - 🍾 **C-roll (冷萃咖啡产品锚定):**
>   - 📸 **锚定原图:** `out/demo-croll/anchor_photo.jpg` \| 🖼️ **海报:** `out/demo-croll/keyframes/kf_1.jpg` \| 📹 **动效:** `out/demo-croll/clips/clip_1.mp4` \| 🎙️ **旁白:** `out/demo-croll/audio/narr_1.mp3` \| 🎵 **配乐:** `out/demo-croll/audio/bgm.mp3`

---

## 这是什么

风格是 Vox 讲解片带火的现代编辑感**纸质拼贴**:手撕纸片、毛边、胶带、半调网点、报纸剪贴、每一拍一块大胆平涂色、大号剪纸标题——再配上动效、旁白、配乐和字幕，让整张海报活过来。

## 工作原理

一个选题依次流过每个阶段脚本，全程由每个项目一份 `beats.json` 驱动:

```
选题 / 视频 / 照片
  │
  ├─ 1. 分镜脚本   选叙事弧线 → 写 beats.json          ◀── 决策点 1:你确认分镜脚本
  ├─ 2. 风格试片   同一拍渲成 3–4 种主题               ◀── 决策点 2:你看图挑风格
  ├─ 3. 关键帧     每拍一张拼贴海报   (nano-banana-2 / flux-dev)
  ├─ 4. 动效       让每张海报动起来   (veo3.1-image-to-video)
  ├─ 5. 旁白+配乐  统一旁白 (minimax-speech-2.6) + 背景乐 (suno-create-music)
  ├─ 6. 合成       ffmpeg:拼接、配乐在旁白下自动闪避、烧字幕+水印
  └─ final.mp4
```

三种输入形态复用同一套引擎:

- **B-roll——一个选题进去，画面全靠生成。** 它写脚本、生成每张拼贴海报、用 Veo 3.1 生成动效、配旁白与音乐，合成视频。
- **A-roll——你已经有一段口播视频。** 它会被 ASR 自动切成段（`openai-whisper`），再整段套上拼贴风格，真人的脸、口型、手势逐帧保留（`gemini-omni-video-edit` / `veo3.1`）。
- **C-roll——你只有一张静态照片**（自拍、产品图）。主体被抠成摄影质感的贴纸——绝不重绘——每一段的海报围着它生成（`nano-banana-2` / `flux-dev`）。旁白支持声音克隆（`minimax-voice-clone`）。

---

## 模型（已在 MuAPI 上验证）

| 用途 | 模型 |
|---|---|
| 关键帧 / 拼贴海报 | `nano-banana-2` / `flux-dev` |
| 动效 / 视频生成 | `veo3.1-image-to-video` / `veo3.1-fast-image-to-video` |
| 口播视频转拼贴 (A-roll) | `gemini-omni-video-edit` / `veo3.1-image-to-video` |
| 照片锚进拼贴 (C-roll) | `nano-banana-2` / `flux-dev` |
| 旁白 TTS | `minimax-speech-2.6-turbo` |
| 声音克隆 | `minimax-voice-clone` |
| 背景音乐 | `suno-create-music` |
| 语音识别 (A-roll ASR) | `openai-whisper` |
| 抠图背景消除 | `remove-background` |

---

## 快速开始（三种模式）

```bash
export MUAPI_API_KEY="sk-..."

# 1. B-roll (选题成片)
python scripts/style_bakeoff.py out/my-topic american-retro,swiss-modern,punk-zine
python scripts/keyframes.py out/my-topic
python scripts/clips.py out/my-topic
python scripts/audio.py out/my-topic
python scripts/assemble.py out/my-topic

# 2. A-roll (口播视频转拼贴)
python scripts/asr_beats.py out/my-aroll source_presentation.mp4
python scripts/aroll_clips.py out/my-aroll
python scripts/aroll_assemble.py out/my-aroll

# 3. C-roll (单张照片/产品锚定拼贴)
python scripts/croll_keyframes.py out/my-croll
python scripts/clips.py out/my-croll
python scripts/audio.py out/my-croll
python scripts/assemble.py out/my-croll
```

---

# Veo 3.1 ComfyUI 节点

ComfyUI custom nodes for generating videos with Google's **Veo 3.1** model via the [MuAPI](https://muapi.ai) platform.

## Related Projects

- [Veo 3 on MuAPI](https://muapi.ai/veo3) — Model landing page for Veo generation.
- [Veo 3 text-to-video playground](https://muapi.ai/playground/veo3-text-to-video) — Try the model directly in the browser.
- [veo4-video-generator](https://github.com/SamurAIGPT/veo4-video-generator) — Ready-made Next.js SaaS for Veo — no ComfyUI needed
- [Veo-4-API](https://github.com/Anil-matcha/Veo-4-API) — Python wrapper for Veo 4 API — use the latest Veo model in scripts
- [muapi-comfyui](https://github.com/SamurAIGPT/muapi-comfyui) — ComfyUI nodes for 100+ MuAPI models including Veo
- [awesome-ai-video-models](https://github.com/Anil-matcha/awesome-ai-video-models) — compare AI video models by API, price & speed

## Nodes

| Node | Description |
|------|-------------|
| 🎬 Veo 3.1 Text to Video | Generate 8-second video from a text prompt |
| 🎬 Veo 3.1 Image to Video | Animate a static image; optionally anchor the last frame |
| 🎬 Veo 3.1 Reference to Video | Generate video guided by up to 4 reference images |
| 🎬 Veo 3.1 Extend Video | Continue a previous generation with a new prompt |
| 🎬 Veo 3.1 4K Upscale | Upscale any previous Veo 3.1 generation to 4K |
| 🎬 Veo 3.1 Save Video | Download & save generated video; returns frames tensor |

All nodes live in the **🎬 Veo 3.1** category in the ComfyUI node menu.

## Available Models

### Text to Video
| Model | Speed | Quality |
|-------|-------|---------|
| `veo3.1-text-to-video` | Standard | Highest, with audio |
| `veo3.1-fast-text-to-video` | Fast | Good |
| `veo3.1-lite-text-to-video` | Fast | Lightweight |

### Image to Video
| Model | Speed | Quality |
|-------|-------|---------|
| `veo3.1-image-to-video` | Standard | Highest, with audio |
| `veo3.1-fast-image-to-video` | Fast | Good |
| `veo3.1-lite-image-to-video` | Fast | Lightweight |

### Other Variants
- `veo3.1-reference-to-video` — multi-image reference generation
- `veo3.1-extend-video` — extend a previous generation
- `veo3.1-4k-video` — upscale a previous generation to 4K

All models output **8-second** videos (Veo 3.1 fixed duration).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/muapi-veo31-comfyui
pip install -r muapi-veo31-comfyui/requirements.txt
```

Restart ComfyUI.

## Setup

1. Get an API key from [MuAPI](https://muapi.ai)
2. Paste it into the `api_key` field of any Veo 3.1 node

## Parameters

### Common
| Parameter | Description |
|-----------|-------------|
| `api_key` | Your MuAPI API key |
| `prompt` | Text description of the video |
| `aspect_ratio` | `16:9` or `9:16` |
| `resolution` | `720p`, `1080p`, or `4k` |
| `extra_params_json` | Any additional model parameters as JSON |

### Image to Video extras
| Parameter | Description |
|-----------|-------------|
| `image` | Start frame (IMAGE tensor) |
| `last_image` | Optional end frame for first–last mode |

### Reference to Video extras
| Parameter | Description |
|-----------|-------------|
| `image_1` … `image_4` | Reference images (up to 4) |
| `generate_audio` | Whether to generate audio (default: true) |

### Extend / 4K Upscale
| Parameter | Description |
|-----------|-------------|
| `request_id` | `request_id` output from a previous generation node |

## Example Workflows

| File | Description |
|------|-------------|
| `MuAPI_Veo31_T2V_Example.json` | Text → Video → Save |
| `MuAPI_Veo31_I2V_Example.json` | Image → Video → Save |
| `MuAPI_Veo31_Reference_Example.json` | 2 reference images → Video → Save |

Load any workflow via **ComfyUI → Load** (drag & drop the JSON).

## Chaining nodes

```
Veo31TextToVideo
  └─ video_url  ──► Veo31VideoSaver ──► frames ──► PreviewImage
  └─ first_frame──► PreviewImage
  └─ request_id ──► Veo31ExtendVideo
                       └─ request_id ──► Veo314KUpscale
```

## Requirements

- Python 3.8+
- ComfyUI (any recent version)
- `requests`, `Pillow`, `numpy`, `torch`, `opencv-python`

## License

MIT
