#!/usr/bin/env python3
"""
Assembly stage (ffmpeg): multi-shot clips + per-beat narration + music -> final.mp4
"""
import json
import os
import subprocess
import sys

import text_overlay

FPS, TAIL = 24, 0.5
WATERMARK = "Made with MuAPI · muapi-director"
RES = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}


def ff(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def probe_dur(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


def shots_of(beat):
    if beat.get("shots"):
        for s in beat["shots"]:
            yield s
    else:
        yield beat


def run(project_dir):
    with open(os.path.join(project_dir, "beats.json")) as f:
        doc = json.load(f)
    beats = doc["beats"]
    W, H = RES.get(doc.get("aspect", "16:9"), (1920, 1080))
    wm_text = doc.get("watermark", WATERMARK)
    mix = doc.get("mix", {})
    music_vol = float(mix.get("music", 0.6))
    voice_vol = float(mix.get("voice", 1.25))
    cap_style = doc.get("caption_style", "white")
    tmp = os.path.join(project_dir, "_seg")
    os.makedirs(tmp, exist_ok=True)

    segs = []
    beat_spans = []
    t = 0.0
    for beat in beats:
        beat_start = t
        shot_list = list(shots_of(beat))
        durs = [float(s.get("dur", 10)) for s in shot_list]
        need = float(beat.get("narration_dur", sum(durs))) + TAIL
        if sum(durs) < need:
            durs[-1] += need - sum(durs)
        for s, d in zip(shot_list, durs):
            segs.append({"clip": s["clip_path"], "dur": round(d, 2)})
            t += d
        beat_spans.append({"start": beat_start, "dur": round(t - beat_start, 2), "beat": beat})
    total = round(t, 2)

    seg_files = []
    for i, s in enumerate(segs):
        out = os.path.join(tmp, f"seg_{i:02d}.mp4")
        cd = probe_dur(s["clip"])
        factor = s["dur"] / cd if cd > 0 else 1.0
        pre = f"setpts={factor:.4f}*PTS," if factor > 1.02 else ""
        fc = (f"[0:v]{pre}split[s0][s1];"
              f"[s0]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
              f"boxblur=26:2,eq=brightness=-0.05[bg];"
              f"[s1]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={FPS},"
              f"tpad=stop_mode=clone:stop_duration=1[v]")
        ff(["-i", s["clip"], "-an", "-filter_complex", fc, "-map", "[v]", "-t", f"{s['dur']}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", out])
        seg_files.append(out)

    listf = os.path.join(tmp, "list.txt")
    with open(listf, "w") as f:
        for s in seg_files:
            f.write(f"file '{s}'\n")

    v_concat = os.path.join(tmp, "v_concat.mp4")
    ff(["-f", "concat", "-safe", "0", "-i", listf, "-c", "copy", v_concat])

    narr_inputs, amix_filters = [], []
    for i, span in enumerate(beat_spans):
        b = span["beat"]
        apath = b.get("audio_path")
        if apath and os.path.exists(apath):
            idx = (len(narr_inputs) // 2) + 1
            narr_inputs.extend(["-i", apath])
            delay_ms = int(span["start"] * 1000)
            amix_filters.append(f"[{idx}:a]volume={voice_vol},adelay={delay_ms}|{delay_ms}[a{idx}];")

    bgm_path = os.path.join(project_dir, "audio", "bgm.mp3")
    has_bgm = os.path.exists(bgm_path)
    if has_bgm:
        bgm_idx = (len(narr_inputs) // 2) + 1
        narr_inputs.extend(["-stream_loop", "-1", "-i", bgm_path])

    filter_lines = []
    if amix_filters:
        filter_lines.extend(amix_filters)
        ins = "".join(f"[a{i+1}]" for i in range(len(beat_spans)))
        filter_lines.append(f"{ins}amix=inputs={len(beat_spans)}:duration=first:dropout_transition=0.5[vo];")
        if has_bgm:
            filter_lines.append(
                f"[{bgm_idx}:a]volume={music_vol}[bgm0];"
                f"[vo]asplit=2[vo1][vo2];"
                f"[bgm0][vo1]sidechaincompress=threshold=0.08:ratio=6:attack=15:release=250[bgm_ducked];"
                f"[vo2][bgm_ducked]amix=inputs=2:duration=first:weights=1.2 0.7[aout]"
            )
        else:
            filter_lines.append(f"[vo]anull[aout]")
    elif has_bgm:
        filter_lines.append(f"[{bgm_idx}:a]volume={music_vol}[aout]")
    else:
        filter_lines.append("anullsrc=channel_layout=stereo:sample_rate=44100[aout]")

    full_filter = "".join(filter_lines)
    audio_full = os.path.join(tmp, "audio_mixed.m4a")
    ff(["-i", v_concat, *narr_inputs, "-filter_complex", full_filter,
        "-map", "[aout]", "-t", f"{total}", "-c:a", "aac", "-b:a", "192k", audio_full])

    cap_overlays = []
    for i, span in enumerate(beat_spans):
        b = span["beat"]
        t_text = b.get("narration", "")
        if not t_text:
            continue
        c_png = os.path.join(tmp, f"cap_{i:02d}.png")
        kf_p = b.get("keyframe_path") or (b.get("shots") and b["shots"][0].get("keyframe_path"))
        acc = text_overlay.accent_color(kf_p) if kf_p else None
        text_overlay.render_caption(t_text, c_png, W=W, H=H, accent=acc, style=cap_style)
        cap_overlays.append((c_png, span["start"], span["start"] + span["dur"]))

    wm_png = os.path.join(tmp, "watermark.png")
    text_overlay.render_watermark(wm_text, wm_png, W=W, H=H)

    v_in = ["-i", v_concat]
    c_inputs = []
    v_maps = ["[0:v]"]

    for i, (c_png, s_start, s_end) in enumerate(cap_overlays):
        c_idx = i + 1
        c_inputs.extend(["-i", c_png])
        last_v = v_maps[-1]
        out_v = f"[vcap{i+1}]"
        v_maps.append(out_v)
        v_in.extend(["-filter_complex", f"{last_v}[{c_idx}:v]overlay=0:0:enable='between(t,{s_start},{s_end})'{out_v}"])

    wm_idx = len(cap_overlays) + 1
    c_inputs.extend(["-i", wm_png])
    last_v = v_maps[-1]
    final_v = "[vfinal]"
    fc_wm = f"{last_v}[{wm_idx}:v]overlay=0:0{final_v}"

    final_mp4 = os.path.join(project_dir, "final.mp4")
    ff([*v_in, *c_inputs, "-i", audio_full, "-filter_complex", fc_wm,
        "-map", final_v, "-map", f"{wm_idx+1}:a", "-t", f"{total}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy", final_mp4])

    print(f"Assembly finished -> {final_mp4}")


if __name__ == "__main__":
    proj = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "out", "demo")
    run(os.path.abspath(proj))
