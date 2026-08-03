#!/usr/bin/env python3
"""
Text overlays via Pillow -> transparent PNG for captions and watermarks in muapi-director.
"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_BOLD = [
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT_REG = [
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

CAPTION_STYLES = ("white", "paper")


def _font(paths, size):
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def accent_color(image_path, default=(214, 64, 42)):
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
        if sat < 0.45 or val < 0.35 or val > 0.92:
            continue
        score = sat * val * (count ** 0.3)
        if score > best_score:
            best, best_score = (r, g, b), score
    return best or default


def render_caption(text, out_path, W=1920, H=1080, margin_v=None, accent=None, style="white"):
    if style not in CAPTION_STYLES:
        style = "white"
    size = int(min(W, H) * 0.045)
    if margin_v is None:
        margin_v = int(H * 0.06)
    fnt = _font(FONT_BOLD, size)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d0 = ImageDraw.Draw(img)
    lines = _wrap(d0, text, fnt, int(W * 0.8))
    lh = int(size * 1.3)
    y0 = H - margin_v - lh * len(lines)
    pos = [(round((W - d0.textlength(ln, font=fnt)) / 2), y0 + i * lh, ln,
            d0.textlength(ln, font=fnt)) for i, ln in enumerate(lines)]

    if style == "paper":
        _draw_paper(img, pos, fnt, size, accent)
    else:
        _draw_white(img, pos, fnt, size)
    img.save(out_path)
    return out_path


def _draw_white(img, pos, fnt, size):
    ow = max(2, round(size * 0.08))
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    for x, y, ln, _ in pos:
        sd.text((x + 3, y + 4), ln, font=fnt, fill=(0, 0, 0, 160))
    sh = sh.filter(ImageFilter.GaussianBlur(3))
    img.alpha_composite(sh)

    draw = ImageDraw.Draw(img)
    for x, y, ln, _ in pos:
        for dx in range(-ow, ow + 1):
            for dy in range(-ow, ow + 1):
                if dx * dx + dy * dy <= ow * ow:
                    draw.text((x + dx, y + dy), ln, font=fnt, fill=(10, 10, 10, 240))
        draw.text((x, y), ln, font=fnt, fill=(255, 255, 255, 255))


def _draw_paper(img, pos, fnt, size, accent):
    pad = int(size * 0.4)
    r = int(size * 0.25)
    ac = accent or (214, 64, 42)

    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    for x, y, _, w in pos:
        box = [x - pad + 3, y - pad // 2 + 5, x + w + pad + 3, y + int(size * 1.15) + 5]
        sd.rounded_rectangle(box, radius=r, fill=(0, 0, 0, 120))
    sh = sh.filter(ImageFilter.GaussianBlur(4))
    img.alpha_composite(sh)

    draw = ImageDraw.Draw(img)
    for x, y, ln, w in pos:
        box = [x - pad, y - pad // 2, x + w + pad, y + int(size * 1.15)]
        draw.rounded_rectangle(box, radius=r, fill=(248, 246, 240, 245),
                               outline=(35, 30, 25, 255), width=2)
        if accent:
            draw.line([x - pad + r, y - pad // 2 + 2, x + w + pad - r, y - pad // 2 + 2],
                      fill=(*ac, 240), width=3)
        draw.text((x, y), ln, font=fnt, fill=(25, 22, 20, 255))


def render_watermark(text, out_path, W=1920, H=1080):
    size = int(min(W, H) * 0.024)
    fnt = _font(FONT_REG, size)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    tw = draw.textlength(text, font=fnt)
    x = W - tw - int(W * 0.03)
    y = H - int(H * 0.04)
    draw.text((x + 1, y + 1), text, font=fnt, fill=(0, 0, 0, 180))
    draw.text((x, y), text, font=fnt, fill=(255, 255, 255, 220))
    img.save(out_path)
    return out_path
