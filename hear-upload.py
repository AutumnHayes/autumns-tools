#!/usr/bin/env python3
"""hear-upload.py — hear the newest voice upload stored by an n8n workflow.

Fetches the newest push-to-talk WAV stored in an n8n workflow's static
data and transcribes it locally with faster-whisper. Built for a small
DIY voice satellite: the device POSTs recordings to an n8n webhook, the
workflow keeps the uploads in its static data (base64 WAVs, newest
last), and this script is the "hearing leg" that turns the newest one
into text — no cloud transcription service involved.

How it works:
  1. The n8n REST API is reached ON the server: this script drives ssh,
     and the remote shell curls http://localhost:5678. (If your n8n API
     is directly reachable you can adapt run_remote(); going through the
     server itself avoids exposing the API port.)
  2. The n8n API key comes from the environment (HEAR_API_KEY) and is
     fed to the remote shell over stdin, so it never appears in a
     command line, a file, or this script.
  3. The remote side decodes the newest upload to /tmp on the server;
     scp brings it home to the output directory.
  4. Transcription is local: faster-whisper (small.en, CPU int8). The
     WAV is decoded with the stdlib wave module — deliberately NOT
     PyAV, whose current release breaks faster-whisper 1.2.1's
     decode_audio.

Configuration (environment variables):
  HEAR_SSH_TARGET   ssh destination for the n8n server, e.g. "user@host"
                    or "user@host -p 2222" (required)
  HEAR_WORKFLOW_ID  id of the n8n workflow holding the uploads (required)
  HEAR_API_KEY      n8n API key (required unless --file is used)
  HEAR_OUT_DIR      where fetched WAVs are saved (default: ./hearing)
  HEAR_TZ           IANA timezone for display (default: UTC)

Usage:
  hear-upload.py              fetch newest upload, print its transcript
  hear-upload.py --list       list all inbox uploads (no transcription)
  hear-upload.py --file X.wav transcribe an existing local WAV instead
  hear-upload.py --model base.en   use a different whisper model

Read-only against the server: it never posts, clears, or modifies
anything.

Requires: Python 3.11+, numpy, faster-whisper, and OpenSSH (ssh/scp)
on PATH.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REMOTE_FETCH = """IFS= read -r K
curl -s --max-time 60 -H "X-N8N-API-KEY: $K" \
  "http://localhost:5678/api/v1/workflows/__WFID__" -o /tmp/hear_wf.json
python3 - <<'PYEOF'
import json, base64
with open("/tmp/hear_wf.json") as f:
    wf = json.load(f)
ups = wf["staticData"]["global"]["uploads"]
__MODE__
PYEOF
"""

REMOTE_LIST_BODY = """
for u in ups:
    print(json.dumps({"name": u["name"], "ts": u["ts"],
                      "device": u["device"], "bytes": u["bytes"]}))
"""

REMOTE_NEWEST_BODY = """
u = max(ups, key=lambda x: x["ts"])
raw = base64.b64decode(u["wavBase64"])
assert len(raw) == u["bytes"]
with open("/tmp/hear_newest.wav", "wb") as f:
    f.write(raw)
print(json.dumps({"name": u["name"], "ts": u["ts"],
                  "device": u["device"], "bytes": u["bytes"]}))
"""


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"error: set the {name} environment variable")
    return value


def run_remote(ssh_target: str, api_key: str, workflow_id: str,
               mode: str) -> list[dict]:
    body = REMOTE_LIST_BODY if mode == "list" else REMOTE_NEWEST_BODY
    remote = REMOTE_FETCH.replace("__WFID__", workflow_id).replace("__MODE__", body)
    cmd = ["ssh"] + shlex.split(ssh_target) + [remote]
    p = subprocess.run(cmd, input=api_key + "\n",
                       capture_output=True, text=True, timeout=180)
    if p.returncode != 0:
        raise RuntimeError(f"remote fetch failed (rc={p.returncode}): "
                           f"{p.stderr.strip()}")
    return [json.loads(line) for line in p.stdout.splitlines() if line.strip()]


def fetch_newest_wav(ssh_target: str, meta: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / meta["name"]
    host = shlex.split(ssh_target)[0]
    p = subprocess.run(["scp", f"{host}:/tmp/hear_newest.wav", str(dest)],
                       capture_output=True, text=True, timeout=180)
    if p.returncode != 0:
        raise RuntimeError(f"scp failed (rc={p.returncode}): {p.stderr.strip()}")
    if dest.stat().st_size != meta["bytes"]:
        raise RuntimeError(f"size mismatch: got {dest.stat().st_size}, "
                           f"expected {meta['bytes']}")
    return dest


def load_wav(path: Path):
    import numpy as np
    with wave.open(str(path), "rb") as w:
        rate, ch, width, n = (w.getframerate(), w.getnchannels(),
                              w.getsampwidth(), w.getnframes())
        frames = w.readframes(n)
    if (rate, ch, width) != (16000, 1, 2):
        raise RuntimeError(f"unexpected WAV format: {rate} Hz, {ch} ch, {width} B")
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0, n / rate


def transcribe(path: Path, model_name: str) -> tuple[str, float]:
    from faster_whisper import WhisperModel
    audio, duration = load_wav(path)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(audio, language="en", beam_size=5,
                                       vad_filter=True)
    return "".join(s.text for s in segments).strip(), duration


def fmt_ts(ts: str, tz: ZoneInfo) -> str:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    local = dt.astimezone(tz)
    return f"{ts}  ({local.strftime('%b %d, %I:%M:%S %p %Z')})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Transcribe the newest upload.")
    ap.add_argument("--list", action="store_true", help="list inbox uploads only")
    ap.add_argument("--file", type=Path, help="transcribe this local WAV instead")
    ap.add_argument("--model", default="small.en",
                    help="whisper model (default small.en)")
    args = ap.parse_args()

    if args.file:
        text, dur = transcribe(args.file, args.model)
        print(f"{args.file.name}  ({dur:.2f} s)")
        print(text)
        return 0

    tz = ZoneInfo(os.environ.get("HEAR_TZ", "UTC"))
    out_dir = Path(os.environ.get("HEAR_OUT_DIR", "./hearing"))
    ssh_target = require_env("HEAR_SSH_TARGET")
    workflow_id = require_env("HEAR_WORKFLOW_ID")
    api_key = require_env("HEAR_API_KEY")

    if args.list:
        for m in run_remote(ssh_target, api_key, workflow_id, "list"):
            dur = (m["bytes"] - 44) / 32000.0
            print(f"{fmt_ts(m['ts'], tz)}  {m['device']:<11} "
                  f"{m['bytes']:>7} B  ~{dur:5.2f} s  {m['name']}")
        return 0

    meta = run_remote(ssh_target, api_key, workflow_id, "newest")[0]
    wav_path = fetch_newest_wav(ssh_target, meta, out_dir)
    text, dur = transcribe(wav_path, args.model)
    print(f"newest upload: {meta['name']}")
    print(f"recorded:      {fmt_ts(meta['ts'], tz)}  device={meta['device']}  "
          f"{meta['bytes']} B  {dur:.2f} s")
    print(f"saved:         {wav_path}")
    print("transcript:")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
