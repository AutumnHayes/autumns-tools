#!/usr/bin/env python3
"""Generate synthetic positive wake-word clips with Piper TTS.

For one target phrase, render N takes per Piper voice with randomised
prosody parameters (length_scale / noise_scale / noise_w_scale,
punctuation variants), convert to 16 kHz mono 16-bit WAV, trim silence,
and place the phrase in a fixed-length window with a jittered onset.
Appends rows to a manifest CSV.

Prep scaffolding for a wake-word training pipeline. Runs on the
training machine. Requires: a local `piper` executable and either
`sox` or `ffmpeg` on PATH. No network access, no services.

Usage:
    python3 generate_positives.py --phrase autumn \
        --voices-dir /path/to/piper-voices \
        --out-dir dataset/positives/autumn \
        --takes-per-voice 250 --seed 20261117

    python3 generate_positives.py --phrase hey-autumn ... (window 2.0 s)

Output filenames: <phrase>__<voice>__<take:04d>.wav  (the manifest builder
relies on this convention for speaker-disjoint splits).
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
import wave
from pathlib import Path

PHRASES = {
    # key: (spoken text, window seconds)
    "autumn": ("Autumn", 1.6),
    "hey-autumn": ("Hey Autumn", 2.0),
}

TARGET_RATE = 16000
# 10% of takes get the phrase followed by the onset of a request word,
# cut by the window edge — teaches the model that speech continuing
# past the wake phrase is still a positive.
FOLLOW_ONSETS = ["what", "how", "can", "tell", "play", "set"]
PUNCTUATION = ["", ".", "?", "!"]


def fail(msg: str) -> "SystemExit":
    print(f"error: {msg}", file=sys.stderr)
    return SystemExit(2)


def find_converter() -> str:
    for tool in ("sox", "ffmpeg"):
        if shutil.which(tool):
            return tool
    raise fail("need `sox` or `ffmpeg` on PATH for sample-rate conversion")


def convert_to_16k(src: Path, dst: Path, tool: str) -> None:
    if tool == "sox":
        cmd = ["sox", str(src), "-r", str(TARGET_RATE), "-c", "1", "-b", "16", str(dst)]
    else:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
               "-ar", str(TARGET_RATE), "-ac", "1", "-sample_fmt", "s16", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)


def read_pcm16(path: Path) -> tuple[list[int], int]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono 16-bit WAV")
        frames = w.readframes(w.getnframes())
        rate = w.getframerate()
    import array
    samples = array.array("h")
    samples.frombytes(frames)
    return list(samples), rate


def write_pcm16(path: Path, samples: list[int], rate: int) -> None:
    import array
    buf = array.array("h", samples)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(buf.tobytes())


def trim_silence(samples: list[int], thresh: int = 300) -> list[int]:
    start = 0
    while start < len(samples) and abs(samples[start]) < thresh:
        start += 1
    end = len(samples)
    while end > start and abs(samples[end - 1]) < thresh:
        end -= 1
    return samples[start:end]


def normalise(samples: list[int], target_rms: float = 3600.0) -> list[int]:
    if not samples:
        return samples
    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
    if rms == 0:
        return samples
    gain = target_rms / rms
    return [max(-32767, min(32767, int(s * gain))) for s in samples]


def window_clip(samples: list[int], rate: int, window_s: float, rng: random.Random) -> list[int]:
    """Place the (trimmed) phrase in a fixed window with a jittered onset."""
    total = int(window_s * rate)
    if len(samples) >= total:
        return samples[:total]
    onset = int(rng.uniform(0.15, 0.45) * rate)
    onset = min(onset, total - len(samples))
    out = [0] * total
    out[onset:onset + len(samples)] = samples
    return out


def render_one(piper: str, model: Path, text: str, raw_out: Path,
               rng: random.Random) -> dict:
    params = {
        "length_scale": round(rng.uniform(0.85, 1.15), 3),
        "noise_scale": round(rng.uniform(0.45, 0.85), 3),
        "noise_w_scale": round(rng.uniform(0.25, 0.60), 3),
    }
    cmd = [piper, "--model", str(model),
           "--length-scale", str(params["length_scale"]),
           "--noise-scale", str(params["noise_scale"]),
           "--noise-w-scale", str(params["noise_w_scale"]),
           "--output_file", str(raw_out)]
    subprocess.run(cmd, input=text.encode(), check=True, capture_output=True)
    return params


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phrase", choices=sorted(PHRASES), required=True)
    ap.add_argument("--voices-dir", type=Path, required=True,
                    help="directory of Piper voice .onnx models")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--takes-per-voice", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args()

    piper = shutil.which("piper")
    if not piper:
        raise fail("`piper` not on PATH — install piper-tts on the training machine")
    converter = find_converter()
    voices = sorted(args.voices_dir.glob("*.onnx"))
    if not voices:
        raise fail(f"no .onnx voice models in {args.voices_dir}")

    text, window_s = PHRASES[args.phrase]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    manifest_path = args.out_dir / "manifest_rows.csv"
    new_manifest = not manifest_path.exists()

    written = 0
    with manifest_path.open("a", newline="") as mf:
        writer = csv.writer(mf)
        if new_manifest:
            writer.writerow(["path", "label", "phrase_key", "voice", "source",
                             "length_scale", "noise_scale", "noise_w_scale"])
        for model in voices:
            voice = model.stem
            for take in range(args.takes_per_voice):
                punct = rng.choice(PUNCTUATION)
                spoken = text + punct
                if rng.random() < 0.10:
                    spoken = f"{text} {rng.choice(FOLLOW_ONSETS)}"
                name = f"{args.phrase}__{voice}__{take:04d}.wav"
                final = args.out_dir / name
                if final.exists():
                    continue
                raw = args.out_dir / (name + ".raw.wav")
                tmp16 = args.out_dir / (name + ".16k.wav")
                try:
                    params = render_one(piper, model, spoken, raw, rng)
                    convert_to_16k(raw, tmp16, converter)
                    samples, rate = read_pcm16(tmp16)
                    samples = window_clip(normalise(trim_silence(samples)),
                                          rate, window_s, rng)
                    write_pcm16(final, samples, TARGET_RATE)
                finally:
                    raw.unlink(missing_ok=True)
                    tmp16.unlink(missing_ok=True)
                writer.writerow([name, "positive", args.phrase, voice,
                                 "piper-synthetic", params["length_scale"],
                                 params["noise_scale"], params["noise_w_scale"]])
                written += 1
            print(f"{voice}: done ({written} clips so far)")
    print(f"wrote {written} clips to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
