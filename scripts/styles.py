#!/usr/bin/env python3
"""
Style definitions and prompt composition for muapi-director.
"""

COLLAGE_MECHANICS = (
    "Clearly layered hand-cut paper cut-outs with visible torn and scissor-cut edges, tape "
    "corners and soft real paper drop shadows, on a bold flat {bg} paper background. Halftone "
    "print dots, newspaper-clipping scraps, paper-stencil shapes, aged paper texture, slight "
    "print misregistration, scattered geometric paper accents. Figures are printed/illustrated "
    "cut-outs, NOT CGI, NOT a 3D render — keep print grain and paper imperfections. High-contrast, tactile."
)

STYLE_LIBRARY = {
    "american-retro": (
        "Vintage 1950s-60s American advertising and pulp-magazine paper collage: bold flat "
        "retro colors, mid-century engraved and illustrated Western cut-out figures, classic "
        "Americana printed imagery, nostalgic."
    ),
    "modern-flat": (
        "Clean modern flat 2D illustrated collage in the Vox explainer motion-graphic style: "
        "bold flat-color vector-illustration cut-outs, simple confident shapes, a limited "
        "contemporary palette, crisp infographic editorial layout."
    ),
    "zine": (
        "Gritty punk zine / ransom-note collage: torn newspaper and magazine scraps, cut-out "
        "mismatched headline letters, photocopied high-contrast halftone, black-and-white with "
        "one spot color, raw hand-cut edges. DIY, analog, urgent."
    ),
    "photo-collage": (
        "Cinematic documentary photo-collage: real black-and-white archival photographs cut "
        "out and layered with soft drop shadows, one restrained accent color, sepia and "
        "monochrome vintage photography. Serious, editorial, museum-like."
    ),
    "chinese-ink": (
        "Mixed-media collage of Chinese woodblock-print and ink-mural cut-out figures, aged "
        "rice-paper and Chinese newspaper clippings, vermilion seal stamps. Oriental, historical."
    ),
}

THEME_PRESETS = {
    "american-retro": {
        "idiom": "american-retro",
        "palette": "bold retro primaries — red, mustard, teal, cream",
        "type_style": "bold wood-type / heavy slab, all-caps",
        "finish": "heavy halftone dots, aged newsprint, slight misregistration",
        "mood": "nostalgic, punchy",
        "motion_style": "punchy"
    },
    "swiss-modern": {
        "idiom": "modern-flat",
        "palette": "two-color + one red accent, lots of white",
        "type_style": "Helvetica/Akzidenz grotesque, tight caps",
        "finish": "clean flat, very subtle grain",
        "mood": "precise, confident",
        "motion_style": "calm"
    },
    "punk-zine": {
        "idiom": "zine",
        "palette": "black & white + one fluorescent spot color",
        "type_style": "ransom-note cut-out mismatched letters",
        "finish": "photocopy grain, heavy misregistration",
        "mood": "urgent, rebellious",
        "motion_style": "max"
    },
    "soviet-constructivist": {
        "idiom": "Russian Constructivist photomontage, bold diagonal geometry",
        "palette": "red, black, cream",
        "type_style": "bold condensed gothic set on strong diagonals",
        "finish": "letterpress, newsprint",
        "mood": "heroic, urgent",
        "motion_style": "punchy"
    },
    "70s-groovy": {
        "idiom": "1970s groovy print",
        "palette": "mustard, rust, avocado, cream",
        "type_style": "bulbous 70s display serif",
        "finish": "riso grain, warm",
        "mood": "warm, funky",
        "motion_style": "punchy"
    },
    "chinese-ink": {
        "idiom": "chinese-ink",
        "palette": "ink black + vermilion, aged paper",
        "type_style": "Chinese brush characters + small English + red seal",
        "finish": "rice-paper, ink bleed",
        "mood": "elegant, historical",
        "motion_style": "calm"
    },
    "atomic-age": {
        "idiom": "1950s atomic-age retro-futurism",
        "palette": "teal, orange, cream",
        "type_style": "atomic script + geometric caps",
        "finish": "halftone, starbursts",
        "mood": "optimistic, bright",
        "motion_style": "punchy"
    },
}

DEFAULT_STYLE = "american-retro"

def resolve_theme(name):
    """Resolve theme preset name to look config."""
    return THEME_PRESETS.get(name)

def _headline(title_en, style, type_style=None):
    ts = f" in {type_style}" if type_style else ""
    seal = "a vermilion seal stamp" if style == "chinese-ink" else "a small paper sticker accent"
    return (f" Include a torn-paper banner with a big bold cut-out English headline "
            f"'{title_en}'{ts} and {seal}. Keep the headline crisp and legible.")

def compose_collage_prompt(scene, title_en, bg="warm ochre", aspect="16:9",
                           with_title=True, style=DEFAULT_STYLE,
                           palette=None, type_style=None, finish=None):
    """Compose the Flux Dev prompt for the collage poster."""
    idiom = STYLE_LIBRARY.get(style, style)
    pal = f" Palette: {palette}." if palette else ""
    fin = f" Print finish: {finish}." if finish else ""
    block = f"{idiom}{pal} {COLLAGE_MECHANICS.format(bg=bg)}{fin}"
    if with_title:
        title = _headline(title_en, style, type_style)
    else:
        title = " No big headline in this shot (a small accent only); it is a cut-in detail."
    return f"{block} SCENE (as layered paper cut-outs): {scene}.{title}"
