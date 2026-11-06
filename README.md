# Whisper API

Self-hosted speech-to-text server using `faster-whisper` on an NVIDIA GPU. Audio and
transcripts stay on your own machines.

## Server setup

Needs an NVIDIA GPU with a working driver (`nvidia-smi` should list it) and Python
3.10 to 3.12. Everything installs into the project's own virtual environment, `.venv`.
The GPU libraries add about 1.5 GB, and the model downloads 3.09 GB on first start.

With [uv](https://docs.astral.sh/uv/), which installs the exact tested versions:

```bash
uv sync --extra gpu --no-dev
```

Or with plain Python (run the first line, then the activate line for your system):

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
source .venv/bin/activate
```

```bash
pip install -e ".[gpu]"
```

Linux only: let the engine find the GPU libraries.

```bash
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import site; b=site.getsitepackages()[0]; print(f"{b}/nvidia/cublas/lib:{b}/nvidia/cudnn/lib")')"
```

Copy `.env.example` to `.env`, then set `WHISPER_API_KEY`. The server won't listen on the
network with a key under 16 characters. This prints a good one:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Run it

Activate `.venv` as above, then:

```bash
python -m whisper_api
```

Use this rather than `uv run`, which would remove the GPU libraries unless given
`--extra gpu`. Startup runs a small test on the GPU and stops with a clear error if
the GPU setup is broken. Allow the port through the firewall for your own network only.

Check that it's up. `device` should say `cuda`:

```bash
curl http://SERVER:8000/health
```

Traffic is plain HTTP and carries your audio, so use it only on a trusted network
or VPN.

## Start automatically on Windows

Run the server by hand once first, as above. That downloads the model and proves the GPU
works. Stop it with Ctrl+C, then open PowerShell as Administrator in this folder.

Check everything is in place:

```bash
powershell -ExecutionPolicy Bypass -File deploy\service.ps1 check
```

Install it. It starts now and at every boot, with no login needed, and restarts if it stops:

```bash
powershell -ExecutionPolicy Bypass -File deploy\service.ps1 install
```

The other actions are `start`, `stop`, `restart`, `status` and `uninstall`. Restart after
changing `.env`. The log is `logs\server.log`, which holds counts and timings only.

Open the port to your local network only:

```bash
New-NetFirewallRule -DisplayName "Whisper API" -Direction Inbound -Protocol TCP -LocalPort 8000 -RemoteAddress LocalSubnet -Action Allow
```

Stop the server PC sleeping while plugged in:

```bash
powercfg /change standby-timeout-ac 0
```

## Transcribe a folder

From any PC with [uv](https://docs.astral.sh/uv/). Finished recordings are skipped, so
it's safe to stop and re-run. Each recording gets one JSON result, and recordings are
only read.

```bash
WHISPER_API_KEY=your-key uv run scripts/transcribe_folder.py "path/to/recordings" "path/to/results" --url http://SERVER:8000
```

In PowerShell, set the key first with `$env:WHISPER_API_KEY = "your-key"`. Results hold
what was said in the audio, so keep them private.

Add `--pattern "**/*.wav"` to include subfolders; results mirror the folder layout. If
the server can't be reached after all retries, the run stops so you can re-run it later.

## API

`POST /v1/audio/transcriptions` with `Authorization: Bearer <key>`, as a multipart form.
It has the same shape as OpenAI's route, so OpenAI client libraries work by changing
the base URL. Any `model` value sent is ignored.

| Field | Meaning |
|---|---|
| `file` | The audio file (required) |
| `response_format` | `json` (default), `text` or `verbose_json` |
| `language` | Force a language such as `ms`. Default detects it as the audio goes on |
| `hotwords` | Terms to favour, replacing the server's list. Empty turns them off |
| `prompt` | Starting text hint |
| `channels` | `auto` (default) splits true stereo. `mix` makes one signal |

The server never translates. It skips silence, and stereo files with identical channels
are treated as mono.

`verbose_json` result:

```json
{
  "task": "transcribe",
  "model": "large-v3",
  "processing_seconds": 12.3,
  "language": "ms",
  "duration": 71.0,
  "text": "all lines in time order",
  "channels": [
    {"name": "left", "language": "ms", "language_probability": 0.98, "speech_seconds": 30.1},
    {"name": "right", "language": "en", "language_probability": 0.91, "speech_seconds": 25.4}
  ],
  "segments": [
    {"id": 0, "channel": "left", "start": 0.0, "end": 3.2, "text": "one line",
     "avg_logprob": -0.21, "compression_ratio": 1.3, "no_speech_prob": 0.02}
  ]
}
```

- `channel` is `left`, `right` or `mono`. The server doesn't know who is on which side.
- Each channel's `language` is the one detected at its start. Audio that switches language
  is still transcribed correctly as it goes, but the library doesn't report a language
  per line.

| Status | Meaning |
|---|---|
| 400 | Not readable as audio, or unknown language |
| 401 | Missing or wrong key |
| 411 / 413 | No upload size given or chunked upload / upload too large |
| 503 | Queue full. Retry after the `Retry-After` wait; the folder script does this |

Interactive docs are at `/docs`. `GET /health` needs no key.

## Privacy

- Uploads are held in a temporary file only during the request.
- Logs record counts and timings, never transcript text.
- If the client disconnects, the work stops.
- The server won't listen on the network without a strong API key.

## Settings

All are in `.env.example`. `WHISPER_COMPUTE_TYPE=int8_float16` uses less GPU memory.

## Development

```bash
uv sync
```

```bash
uv run pytest
```

Tests use generated audio and a stand-in model, so they need no GPU, model or recordings.
