#!/usr/bin/env python3
"""
ASR bridge (A-roll only): turns raw talking-head recording audio into a beats.json skeleton for muapi-director.
"""
import argparse
import json
import os
import subprocess
import sys

from provider import get_provider

MAX_BEAT_DUR = 9.5
MIN_BEAT_DUR = 2.0
PAUSE_GAP_S = 0.35
SENTENCE_END = (".", "!", "?")


def probe_dims(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    if not out:
        return None
    w, h = (int(x) for x in out.split(",")[:2])
    return w, h


def nearest_named_aspect(w, h):
    named = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4}
    ratio = w / h
    return min(named, key=lambda k: abs(named[k] - ratio))


def extract_audio(src, dest):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vn",
                    "-acodec", "libmp3lame", "-q:a", "2", dest], check=True)


def segment_words(words, max_dur=MAX_BEAT_DUR, min_dur=MIN_BEAT_DUR, pause_gap=PAUSE_GAP_S):
    beats, cur = [], []

    def flush():
        if not cur:
            return
        beats.append({
            "start": round(cur[0]["start"], 2),
            "end": round(cur[-1]["end"], 2),
            "text": " ".join(w["text"] for w in cur),
        })
        cur.clear()

    for i, w in enumerate(words):
        cur.append(w)
        dur = w["end"] - cur[0]["start"]
        is_last = i == len(words) - 1
        gap_next = (words[i + 1]["start"] - w["end"]) if not is_last else None
        sentence_end = w["text"].rstrip().endswith(SENTENCE_END)
        if is_last:
            flush()
        elif dur >= max_dur:
            flush()
        elif dur >= min_dur and (sentence_end or (gap_next is not None and gap_next >= pause_gap)):
            flush()

    merged = []
    for b in beats:
        if merged and (b["end"] - b["start"]) < min_dur:
            merged[-1]["end"] = b["end"]
            merged[-1]["text"] += " " + b["text"]
        else:
            merged.append(b)
    return merged


def run(project_dir, src_path, language="en", keyterms=None, max_beat_dur=MAX_BEAT_DUR):
    os.makedirs(project_dir, exist_ok=True)
    audio_temp = os.path.join(project_dir, "_source_audio.mp3")
    extract_audio(src_path, audio_temp)

    prov = get_provider()
    audio_url = prov.upload(audio_temp)

    print("Submitting ASR job...")
    job_id = prov.submit_audio("whisper", audio_url=audio_url, language=language)
    
    status = prov.get_status(job_id)
    import time
    while status["status"] == "pending":
        time.sleep(3)
        status = prov.get_status(job_id)

    if status["status"] == "failed":
        print(f"ASR failed: {status['error']}; using fall-back dummy segment.")
        dur = 10.0
        try:
            out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                  "-of", "csv=p=0", src_path], capture_output=True, text=True).stdout
            dur = float(out.strip())
        except Exception:
            pass
        beats = [{"start": 0.0, "end": round(dur, 2), "text": "Presenting talking-head segment."}]
    else:
        out_data = status.get("output") or {}
        words = out_data.get("words", []) if isinstance(out_data, dict) else []
        if not words:
            beats = [{"start": 0.0, "end": 8.0, "text": "Talking head transcript."}]
        else:
            beats = segment_words(words, max_dur=max_beat_dur)

    dims = probe_dims(src_path)
    aspect = nearest_named_aspect(*dims) if dims else "9:16"

    doc = {
        "mode": "aroll",
        "project": os.path.basename(os.path.abspath(project_dir)),
        "source_video": os.path.abspath(src_path),
        "source_audio_url": audio_url,
        "language": language,
        "aspect": aspect,
        "theme": "american-retro",
        "video_model": "veo31-image-to-video",
        "beats": [
            {
                "id": i + 1,
                "start": b["start"],
                "end": b["end"],
                "dur": round(b["end"] - b["start"], 2),
                "narration": b["text"],
                "title_en": f"BEAT {i+1}",
                "content_beats": "",
            }
            for i, b in enumerate(beats)
        ]
    }

    out_file = os.path.join(project_dir, "beats.json")
    with open(out_file, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"ASR beats skeleton written to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR bridge for A-roll mode")
    parser.add_argument("project_dir", help="Project directory")
    parser.add_argument("source", help="Source video or audio file")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()
    run(os.path.abspath(args.project_dir), os.path.abspath(args.source), language=args.language)
