# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.27"]
# ///
"""Send a folder of recordings to the Whisper API and save one JSON result per recording.

Safe to stop and re-run: recordings that already have a result are skipped. Recordings
are only ever opened for reading. The API key comes from WHISPER_API_KEY.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

RETRY_STATUSES = {500, 502, 503, 504}
FAILURES_FILE = "_failures.json"


class PermanentFailure(Exception):
    """Retrying will not help, for example a file that is not audio."""


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--pattern", default="*.wav")
    parser.add_argument("--workers", type=int, default=1, help="match the server's WHISPER_WORKERS")
    parser.add_argument("--limit", type=int, default=0, help="only process this many new recordings")
    parser.add_argument("--language", help="force a language instead of detecting it")
    parser.add_argument("--hotwords", help="replace the server's default terms")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-wait", type=float, default=15.0, help="seconds, doubles per retry")
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds to wait per recording")
    return parser.parse_args(argv)


def transcribe_one(client: httpx.Client, path: Path, args: argparse.Namespace) -> dict:
    form = {"response_format": "verbose_json"}
    if args.language:
        form["language"] = args.language
    if args.hotwords is not None:
        form["hotwords"] = args.hotwords

    for attempt in range(args.retries + 1):
        retry_reason = ""
        try:
            with path.open("rb") as audio:
                response = client.post(
                    f"{args.url.rstrip('/')}/v1/audio/transcriptions",
                    data=form,
                    files={"file": (path.name, audio, "application/octet-stream")},
                )
            if response.status_code == 200:
                return response.json()
            if response.status_code not in RETRY_STATUSES:
                raise PermanentFailure(f"HTTP {response.status_code}: {response.text[:200]}")
            retry_reason = f"HTTP {response.status_code}"
        except httpx.TransportError as exc:
            retry_reason = type(exc).__name__
        if attempt < args.retries:
            time.sleep(args.retry_wait * 2**attempt)
    raise RuntimeError(f"Gave up after {args.retries + 1} tries: {retry_reason}")


def save_json(path: Path, data: object) -> None:
    """Write in one step, so an interrupted run never leaves a half-written result."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    recordings = sorted(p for p in args.input_dir.glob(args.pattern) if p.is_file())
    todo = [p for p in recordings if not (args.output_dir / f"{p.name}.json").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(recordings)} recordings, {len(recordings) - len(todo)} skipped, {len(todo)} to do")

    headers = {}
    if key := os.environ.get("WHISPER_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    timeout = httpx.Timeout(args.timeout, connect=30.0)

    failures: dict[str, str] = {}
    done = 0
    lock = threading.Lock()

    def work(path: Path) -> None:
        nonlocal done
        try:
            result = transcribe_one(client, path, args)
            save_json(args.output_dir / f"{path.name}.json", {"source_file": path.name, **result})
        except (PermanentFailure, RuntimeError, ValueError, OSError) as exc:
            with lock:
                failures[path.name] = str(exc)
        with lock:
            done += 1
            print(f"{done}/{len(todo)} done, {len(failures)} failed", flush=True)

    with httpx.Client(headers=headers, timeout=timeout) as client:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(work, p) for p in todo]):
                future.result()

    failures_path = args.output_dir / FAILURES_FILE
    if failures:
        save_json(failures_path, failures)
        print(f"{len(failures)} failed, see {failures_path}")
        return 1
    failures_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
