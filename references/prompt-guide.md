# Prompt Guide — The Look Layer for Flux Dev

To generate premium, high-contrast, paper-collage styles, use this guide. Every beat is generated as an image (keyframe) first, which is later animated.

---

## 1. Image Prompt Formula
Compose every image prompt using a strict 5-part structure to ensure consistency across the video.

```
[1 STYLE BLOCK]
mixed-media hand-cut paper collage, editorial zine style. Torn paper edges, scissor-cut borders, tape corners, halftone print dot patterns, paper drop shadows. Figures are printed-texture cut-outs from vintage photography or woodblocks. NOT 3D, NOT CGI. Visible paper grain and print imperfections. High contrast.

[2 SCENE DETAILS]
SCENE as layered paper cut-outs: {subject details}, {props}, {accent design pieces}; elements have clean cut-out outlines, casting subtle drop shadows on layers below.

[3 BACKGROUND]
on a bold flat {bg_color} cardboard paper background.

[4 TEXT BANNER (Headline)]
A torn paper banner with a big bold headline "{TEXT}" written in heavy sans-serif capital letters.

[5 TECHNICAL]
flat-lay scanned look, straight-on composition, 1k resolution.
```

---

## 2. Styling Presets (Theme Catalog)
To fit different topics and historical eras, you can mix and match design styles. Below are the core presets:

### Swiss Modern (Minimalist, clean, structured)
* **Palette:** Pure red, jet black, clean white, slate gray.
* **Layout:** Asymmetric grids, heavy negative space.
* **Typography:** Heavy grotesque sans-serif (Helvetica style).
* **Finish:** Subtle print texture, very clean scissor cuts.

### American Retro (1950s/60s print advertisement)
* **Palette:** Aged sepia, mustard yellow, teal blue, cherry red.
* **Layout:** Centered hero illustrations, bold offset badge designs.
* **Typography:** Bold slab-serifs or display fonts.
* **Finish:** Ben-Day dot patterns, paper aging, coffee stains.

### Punk Zine (Rebellious, raw, handmade)
* **Palette:** Duotone (neon yellow + black, hot pink + navy, green + white).
* **Layout:** Chaotic, overlapping layers, angled elements.
* **Typography:** Stencil, ransom-note cutout letters, distressed stamp letters.
* **Finish:** Heavy xerox copy grain, high-contrast print, tape patches.

### Chinese Ink (Elegant, classic, hybrid)
* **Palette:** Charcoal black, soft parchment cream, bright vermilion accent.
* **Layout:** Dynamic sweeps, circular framing.
* **Typography:** Elegant brush-stroke lettering or block stamps.
* **Finish:** Ink washes, soft deckled edges, rice paper fiber textures.

---

## 3. Designing for Animation
To get high-quality video motion out of your keyframe:
1. **Maintain high contrast:** Clear silhouettes against a flat background allow the video model to separate layers easily.
2. **Avoid busy textures in the background:** Keep backgrounds flat and solid. Gradients or detailed backgrounds cause video morphing issues.
3. **Isolate text:** Keep text on a clean, solid paper banner. Do not place text over complex details, or it will smear during animation.
