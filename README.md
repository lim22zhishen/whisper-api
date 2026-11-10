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
| `language` | Force a language such as `ms`. Default detects it for each 30 seconds of speech |
| `hotwords` | Terms or a sample sentence to favour, sent with every 30 seconds of audio. Replaces the server's list; empty turns them off |
| `prompt` | Text hint for the first 30 seconds only |
| `channels` | `auto` (default) splits true stereo. `mix` makes one signal |

The server never asks the model to translate, though mixed-language speech can still
come out translated (see [Mixed-language speech](#mixed-language-speech)). It skips
silence, and stereo files with identical channels are treated as mono.

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
- A line ends where the speaker pauses for `WHISPER_LINE_GAP_SECONDS` (default 1) or more.
  Without this, short replies far apart come out as one long line at the first one's
  time, because the model writes over speech with the silence removed. The parts of a
  split line share the model's scores for the whole line. It makes transcription
  somewhat slower; `0` turns it off.
- Each channel's `language` is the one detected at its start. Detection runs again for
  each 30 seconds of speech, but the library doesn't report a language per line. See
  [Mixed-language speech](#mixed-language-speech).

| Status | Meaning |
|---|---|
| 400 | Not readable as audio, or unknown language |
| 401 | Missing or wrong key |
| 411 / 413 | No upload size given or chunked upload / upload too large |
| 503 | Queue full. Retry after the `Retry-After` wait; the folder script does this |

Interactive docs are at `/docs`. `GET /health` needs no key.

## Mixed-language speech

The model picks one language for each 30 seconds of speech and writes that whole stretch
in it. When speakers switch language mid-sentence, the other language can come out
translated, and sometimes the pick is a language nobody spoke. Cheapest fixes first:

1. Limit the picks to the languages your speakers use, e.g. `WHISPER_LANGUAGES=ms,en,zh`.
   Include every language that is spoken: speech in a language left out comes out as a
   garbled translation.
2. Add a short sample sentence to `WHISPER_HOTWORDS`, mixing the languages the way your
   speakers do. It is sent with every 30 seconds of audio and nudges the model to keep
   the mix. `prompt` only reaches the first 30 seconds. If the sentence starts showing up
   in transcripts, shorten it.
3. Try another model. Whisper models differ in how well they keep mixed speech as spoken,
   so compare a few on your own recordings. `WHISPER_MODEL` takes a name such as
   `large-v3-turbo` (smaller and faster), the Hugging Face ID of a CTranslate2
   conversion, or a folder. To convert a fine-tuned Whisper model yourself:

   ```bash
   pip install ctranslate2 "transformers[torch]"
   ct2-transformers-converter --model ORG/MODEL --output_dir models/my-model --copy_files tokenizer.json preprocessor_config.json --quantization float16
   ```

Restart the server after changing `.env`. `/health` shows the model, language limit
and line gap in use. Test a change on a few recordings before re-running everything.

## Speed

Two settings make a run faster. Try one at a time on a few recordings, because both
use more GPU memory and together they use the most. Watch memory with `nvidia-smi`
during a run.

- `WHISPER_BATCH_SIZE=8` runs up to 8 pieces of speech (up to 30 seconds each) from one
  recording on the GPU together. In faster-whisper's own test this was about 3.7 times
  faster and took about 1.5 GB more GPU memory. Lower it if the GPU runs out of memory;
  those requests fail with status 500. Compared with the default `0`:
  - A piece that comes out garbled or stuck repeating isn't tried again. One at a
    time, the model retries such pieces with more randomness.
  - `prompt` is sent with every piece, not only the first 30 seconds.
  - The language limit, hotwords and line splitting work the same.
- `WHISPER_WORKERS=2`, with `--workers 2` on the folder script, transcribes two
  recordings at the same time. On one GPU the gain is smaller, mostly from reading
  audio and finding speech while the GPU works on the other recording.

`/health` shows the batch size and workers in use.

## Privacy

- Uploads are held in a temporary file only during the request.
- Logs record counts and timings, never transcript text.
- If the client disconnects, the work stops.
- The server won't listen on the network without a strong API key.

## Settings

All are in `.env.example`. `WHISPER_COMPUTE_TYPE=int8_float16` uses less GPU memory.
For faster runs see [Speed](#speed).

## Development

```bash
uv sync
```

```bash
uv run pytest
```

Tests use generated audio and a stand-in model, so they need no GPU, model or recordings.
