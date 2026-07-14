#!/usr/bin/env python3
"""
Text overlays via Pillow -> transparent PNG (this ffmpeg has no libass/drawtext,
so captions/watermark are rendered as images and composited with `overlay`).
"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Font lookup paths for Windows, macOS, and Linux
FONT_BOLD = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
]
FONT_REG = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
]

def _font(paths, size):
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    # Fallback to default
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def _wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        # Handle cases where draw.textlength might not exist or default font is loaded
        try:
            w_len = draw.textlength(trial, font=fnt)
        except AttributeError:
            # Fallback if textlength is not supported
            w_len = len(trial) * 10
            
        if w_len <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def accent_color(image_path, default=(214, 64, 42)):
    """Sample a vibrant color from a keyframe so the caption keyline matches the palette."""
    try:
        im = Image.open(image_path).convert("RGB").resize((80, 80)).quantize(colors=24).convert("RGB")
    except Exception:
        return default
    best, best_score = None, -1.0
    for count, (r, g, b) in (im.getcolors(80 * 80) or []):
        mx, mn = max(r, g, b), min(r, g, b)
        if mx == 0:
            continue
        sat, val = (mx - mn) / mx, mx / 255.0
        if sat < 0.45 or val < 0.35 or val > 0.92:   # skip washed-out / too dark / too light
            continue
        score = sat * val * (count ** 0.3)
        if score > best_score:
            best, best_score = (r, g, b), score
    return best or default

def render_caption(text, out_path, W=1920, H=1080, margin_v=None, accent=None):
    """Cut-out paper caption: cream letters + dark-ink edge + soft shadow, NO band."""
    size = int(min(W, H) * 0.045)
    if margin_v is None:
        margin_v = int(H * 0.06)
        
    fnt = _font(FONT_BOLD, size)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d0 = ImageDraw.Draw(img)
    lines = _wrap(d0, text, fnt, int(W * 0.8))
    lh = int(size * 1.3)
    y0 = H - margin_v - lh * len(lines)
    
    pos = []
    for i, ln in enumerate(lines):
        try:
            w_len = d0.textlength(ln, font=fnt)
        except AttributeError:
            w_len = len(ln) * 10
        pos.append(((W - w_len) / 2, y0 + i * lh, ln))
        
    ow = max(3, round(size * 0.13))                       # ink edge
    
    # Soft shadow halo
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    for x, y, t in pos:
        sd.text((x + 3, y + 4), t, font=fnt, fill=(8, 6, 3, 190),
                stroke_width=ow, stroke_fill=(8, 6, 3, 190))
    img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(5)))
    
    d = ImageDraw.Draw(img)
    if accent:                                            # per-beat colored keyline
        a = tuple(accent[:3]) + (255,)
        kw = max(4, round(size * 0.14))
        for x, y, t in pos:
            d.text((x, y), t, font=fnt, fill=a, stroke_width=ow + kw, stroke_fill=a)
            
    for x, y, t in pos:                                   # cream cut-out letters + ink edge
        d.text((x, y), t, font=fnt, fill=(252, 246, 232, 255),
               stroke_width=ow, stroke_fill=(32, 22, 18, 255))
               
    img.save(out_path)
    return out_path

def render_watermark(text, out_path, W=1920, H=1080):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    size = int(min(W, H) * 0.022)
    fnt = _font(FONT_REG, size)
    try:
        tw = d.textlength(text, font=fnt)
    except AttributeError:
        tw = len(text) * 10
        
    x, y = W - tw - 30, H - size - 30
    ow = max(2, round(size * 0.14))
    d.text((x, y), text, font=fnt, fill=(250, 244, 232, 225),
           stroke_width=ow, stroke_fill=(18, 12, 8, 205))
    img.save(out_path)
    return out_path
