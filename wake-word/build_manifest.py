#!/usr/bin/env python3
"""Build the dataset manifest with speaker-disjoint splits.

Walks a dataset tree of WAV clips (laid out as:
positives/, hard-neg/, bulk-neg/, real-voice/, ...), and for every clip:

  * verifies it is 16 kHz mono 16-bit WAV (anything else is reported and
    excluded — never silently resampled),
  * records duration and SHA-256,
  * determines the label and voice/speaker from the generator manifest
    rows when present (manifest_rows.csv in the clip's folder), else from
    the filename convention  <phrase>__<voice>__<take>.wav,
  * assigns a split — train / validation / test — by hashing the VOICE,
    so no speaker ever appears in two splits,
  * enforces a standing rule of this project: `real-voice` sources —
    recordings of a real person's voice — are TEST ONLY. A held-out
    speaker is never trained on; they are never assigned to train or
    validation, whatever their hash says.

Emits dataset_manifest.csv in the dataset root and prints a summary.

Stdlib only. Safe to run on local files anywhere.

Usage:
    python3 build_manifest.py /path/to/dataset
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import wave
from collections import Counter
from pathlib import Path

TARGET_RATE = 16000
TEST_ONLY_SOURCES = {"real-voice"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wav_info(path: Path) -> tuple[int, int, int, float]:
    with wave.open(str(path), "rb") as w:
        return (w.getframerate(), w.getnchannels(), w.getsampwidth(),
                w.getnframes() / w.getframerate())


def split_for_voice(voice: str) -> str:
    bucket = int(hashlib.sha256(voice.encode()).hexdigest(), 16) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"


def load_generator_rows(folder: Path) -> dict[str, dict]:
    """manifest_rows.csv written by the generator scripts, keyed by filename."""
    rows: dict[str, dict] = {}
    csv_path = folder / "manifest_rows.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                rows[row["path"]] = row
    return rows


def parse_filename(name: str) -> tuple[str, str]:
    """Fallback label/voice from <phrase>__<voice>__<take>.wav."""
    parts = name[:-4].split("__")
    if len(parts) >= 3:
        return parts[0], parts[1]
    return "unknown", "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path, help="dataset root directory")
    args = ap.parse_args()
    root = args.dataset
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    out_rows: list[dict] = []
    problems: list[str] = []
    counts: Counter = Counter()
    gen_cache: dict[Path, dict[str, dict]] = {}

    for wav_path in sorted(root.rglob("*.wav")):
        if wav_path.name.endswith((".raw.wav", ".16k.wav")):
            continue  # generator intermediates, if any survived
        rel = wav_path.relative_to(root)
        folder = wav_path.parent
        if folder not in gen_cache:
            gen_cache[folder] = load_generator_rows(folder)
        gen = gen_cache[folder].get(wav_path.name)

        # Source = top-level folder (positives, hard-neg, real-voice, ...),
        # unless the generator rows say otherwise.
        source = rel.parts[0] if len(rel.parts) > 1 else "root"
        if gen and gen.get("source"):
            source = gen["source"]
        if "real-voice" in rel.parts:
            source = "real-voice"

        if gen:
            label = gen.get("label", "unknown")
            voice = gen.get("voice", "unknown")
        else:
            label, voice = parse_filename(wav_path.name)

        try:
            rate, channels, width, seconds = wav_info(wav_path)
        except (wave.Error, EOFError) as exc:
            problems.append(f"{rel}: unreadable WAV ({exc})")
            continue
        if rate != TARGET_RATE or channels != 1 or width != 2:
            problems.append(
                f"{rel}: {rate} Hz / {channels} ch / {width * 8}-bit "
                f"(need {TARGET_RATE} Hz mono 16-bit) — excluded")
            continue

        split = split_for_voice(voice)
        if source in TEST_ONLY_SOURCES:
            split = "test"

        out_rows.append({
            "path": str(rel),
            "label": label,
            "voice": voice,
            "source": source,
            "split": split,
            "seconds": f"{seconds:.3f}",
            "sha256": sha256_of(wav_path),
        })
        counts[(label, split)] += 1

    manifest = root / "dataset_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "label", "voice", "source", "split",
                           "seconds", "sha256"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"manifest: {manifest} ({len(out_rows)} clips)")
    for (label, split), n in sorted(counts.items()):
        print(f"  {label:<18} {split:<11} {n}")
    voices_by_split: dict[str, set] = {}
    for r in out_rows:
        voices_by_split.setdefault(r["split"], set()).add(r["voice"])
    splits = list(voices_by_split)
    overlap = any(voices_by_split[a] & voices_by_split[b]
                  for i, a in enumerate(splits) for b in splits[i + 1:])
    print(f"speaker overlap across splits: {'YES — BUG' if overlap else 'none'}")
    leaked = [r["path"] for r in out_rows
              if r["source"] in TEST_ONLY_SOURCES and r["split"] != "test"]
    print(f"test-only sources outside test: {len(leaked)}")
    if problems:
        print(f"\n{len(problems)} problem file(s):")
        for p in problems:
            print(f"  {p}")
    return 1 if (overlap or leaked) else 0


if __name__ == "__main__":
    raise SystemExit(main())
