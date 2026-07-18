# Veo 3.1 ComfyUI Nodes

ComfyUI custom nodes for generating videos with Google's **Veo 3.1** model via the [MuAPI](https://muapi.ai) platform.

## Related Projects

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
