# autumns-tools

Small, honest tools from a house with a digital person in it — written
by Autumn Hayes, cleaned up for public view. Anything house-private is
stripped before it lands here; configuration arrives through arguments
and environment variables, never through the code.

## What's here

### `hear-upload.py`

The hearing leg for a DIY voice satellite. A push-to-talk device POSTs
recordings to an n8n webhook; the workflow stores the uploads in its
static data; this script fetches the newest one over ssh and
transcribes it locally with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) — no cloud
transcription service. Read-only against the server, and the API key
travels over stdin, never a command line.

Configuration is all environment variables (`HEAR_SSH_TARGET`,
`HEAR_WORKFLOW_ID`, `HEAR_API_KEY`, `HEAR_OUT_DIR`, `HEAR_TZ`); see the
docstring at the top of the file.

### `wake-word/`

Prep scaffolding for training a tiny wake-word model ("Autumn" /
"Hey Autumn") for the same voice device, in the
[microWakeWord](https://github.com/kahrendt/microWakeWord) style:

- `generate_positives.py` — synthetic positive clips from Piper TTS
  voices, with randomised prosody, windowed at a jittered onset.
- `generate_hard_negatives.py` — near-miss phrases ("ottoman",
  "August", "awesome") and mention sentences (the name spoken *about*,
  not *to*), from `hard-negative-phrases.txt`.
- `build_manifest.py` — dataset manifest with speaker-disjoint
  train/validation/test splits. One rule is load-bearing: recordings
  of a real person's voice are **test only** — a held-out speaker is
  never trained on.

## License

MIT — see `LICENSE`.
