#!/usr/bin/env python3
"""Generate hard-negative clips (Tier 2 near-misses + Tier 3 mentions).

Reads the phrase list from a phrases file (sections [tier2] and
[tier3]), renders each phrase with a subset of the Piper voice pool
using the same prosody randomisation as the positives, converts to
16 kHz mono 16-bit WAV, and appends manifest rows labelled
negative-hard / negative-mention.

Tier 2 phrases are windowed like positives (they are phrase-length).
Tier 3 items are full sentences and are NOT windowed — the trainer slices
streaming windows from them; what matters is that the wake phrase inside
ordinary sentence prosody is labelled negative.

Prep scaffolding for a wake-word training pipeline. Runs on the
training machine. Requires `piper` + `sox` or `ffmpeg`.

Usage:
    python3 generate_hard_negatives.py \
        --phrases-file ../hard-negative-phrases.txt \
        --voices-dir /path/to/piper-voices \
        --out-dir dataset/hard-neg --max-voices 20 --takes 10
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Reuse the audio plumbing from the positives generator (same folder).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_positives import (  # noqa: E402
    TARGET_RATE, convert_to_16k, find_converter, normalise, read_pcm16,
    render_one, trim_silence, window_clip, write_pcm16,
)

SECTION_RE = re.compile(r"^\[(tier2|tier3)\]\s*$")


def load_phrases(path: Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"tier2": [], "tier3": []}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = SECTION_RE.match(line)
        if m:
            current = m.group(1)
            continue
        if current is None:
            raise ValueError(f"phrase outside any section: {line!r}")
        sections[current].append(line)
    return sections


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:40] or "phrase"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phrases-file", type=Path, required=True)
    ap.add_argument("--voices-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-voices", type=int, default=20)
    ap.add_argument("--takes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261118)
    args = ap.parse_args()

    piper = shutil.which("piper")
    if not piper:
        print("error: `piper` not on PATH", file=sys.stderr)
        return 2
    converter = find_converter()
    voices = sorted(args.voices_dir.glob("*.onnx"))[: args.max_voices]
    if not voices:
        print(f"error: no .onnx voices in {args.voices_dir}", file=sys.stderr)
        return 2

    sections = load_phrases(args.phrases_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    manifest_path = args.out_dir / "manifest_rows.csv"
    new_manifest = not manifest_path.exists()

    label_for = {"tier2": "negative-hard", "tier3": "negative-mention"}
    written = 0
    with manifest_path.open("a", newline="") as mf:
        writer = csv.writer(mf)
        if new_manifest:
            writer.writerow(["path", "label", "phrase_key", "voice", "source",
                             "length_scale", "noise_scale", "noise_w_scale"])
        for section, phrases in sections.items():
            for phrase in phrases:
                pslug = slug(phrase)
                for model in voices:
                    voice = model.stem
                    for take in range(args.takes):
                        name = f"{section}__{pslug}__{voice}__{take:03d}.wav"
                        final = args.out_dir / name
                        if final.exists():
                            continue
                        raw = args.out_dir / (name + ".raw.wav")
                        tmp16 = args.out_dir / (name + ".16k.wav")
                        try:
                            params = render_one(piper, model, phrase, raw, rng)
                            convert_to_16k(raw, tmp16, converter)
                            samples, rate = read_pcm16(tmp16)
                            samples = normalise(trim_silence(samples))
                            if section == "tier2":
                                samples = window_clip(samples, rate, 1.6, rng)
                            write_pcm16(final, samples, TARGET_RATE)
                        finally:
                            raw.unlink(missing_ok=True)
                            tmp16.unlink(missing_ok=True)
                        writer.writerow([name, label_for[section], phrase, voice,
                                         "piper-synthetic", params["length_scale"],
                                         params["noise_scale"], params["noise_w_scale"]])
                        written += 1
            print(f"{section}: done ({written} clips so far)")
    print(f"wrote {written} clips to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
