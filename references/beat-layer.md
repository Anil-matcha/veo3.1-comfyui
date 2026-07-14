# Beat & Shot Layer Library — Narrative Arc & Shot Specifications

This reference covers the story skeletons (narrative arcs) and per-shot specifications (shot size, camera moves, and internal elements) used to structure Muapi Director projects.

---

## 1. Narrative Arc Library

Choose an arc that fits the genre and tone of your video. The agent will draft a beat structure based on this arc:

| Arc Name       | Best Used For                                  | Structure Sequence                                        |
| -------------- | ---------------------------------------------- | --------------------------------------------------------- |
| `hook_payoff`  | General concepts, educational videos (Default) | Hook → Context → Build → Payoff → Wrap-up                 |
| `pas`          | Product advertisements, sales pitches          | Problem → Agitate → Solve → Proof → Call-to-Action (CTA)  |
| `bab`          | Before/After comparisons                       | Before State → After State → Bridge → Call-to-Action      |
| `how_it_works` | Process, engineering, or tutorial videos       | Hook → Overview → Step-by-Step Details → Benefit → CTA    |
| `timeline`     | History, biographies, and journeys             | Beginning → Key Event 1 → Key Event 2 → Climax → Takeaway |
| `man_in_hole`  | Comebacks, transformations, case studies       | Baseline → Fall/Crisis → Deepening → Ascent → Resolution  |

---

## 2. Hook, Pacing, and Cadence

- **Hook in ≤ 3 seconds:** The first shot must immediately grip the user. It should showcase a bold headline and highlight a surprising statistic, mistake, or direct question.
- **Cadence (Cut every ~4–6s):** To prevent visual stagnation, split each narrative beat into 2 distinct shots (a wide establishing shot followed by a close detail cut-in).
- **Beat Counts:**
  - **30-second video:** 6–8 shots (3–4 beats)
  - **60-second video:** 10–12 shots (5–6 beats)

---

## 3. Shot Composition & Flat-Safe Camera Moves

Since we are animating flat paper collage keyframes, camera movements must be uniform to keep text and graphics from warping or distorting.

### Shot Size

`EST_WIDE` (establish system/scene) · `WIDE` (subject in environment) · `MEDIUM` (centered subject) · `CLOSE` (fills frame with one item/person) · `DETAIL` (focus on a single word, texture, or symbol).

### Valid Camera Moves

- **`static`**: Locked-off camera. Great for letting text or complex statements sink in.
- **`push_in`**: Uniform zoom-in (classic Ken Burns effect). Enhances tension and focus.
- **`pull_out`**: Uniform zoom-out. Excellent for revealing background context.
- **`pan`**: Horizontal translation. Best for scanning a timeline or list.
- **`tilt`**: Vertical translation. Ideal for vertical layouts.
- **`parallax`**: Front, middle, and background layers moving at slightly different speeds to create a 2.5D depth effect.

---

## 4. Rich Element Motion (The Energy Engine)

While the camera move shifts the overall frame, **element motion** controls what animates _inside_ the collage itself. Animate multiple elements per shot to bring the scene to life:

- Example: _"A paper aeroplane swoops across the frame, a cut-out hand bobs up and down, and halftone dots scale up."_
- Let the AI craft rich, scene-specific motions to maintain high visual interest.
