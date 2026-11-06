"""Shared test helpers. All audio here is generated; no real recordings are used."""

import io
import threading
import time
import wave
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import uvicorn

from whisper_api.config import Settings
from whisper_api.engine import Engine


def tone(frequency: float, seconds: float = 1.0, rate: int = 8000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * 32767 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)


def wav_bytes(*channels: np.ndarray, rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(len(channels))
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.stack(channels, axis=1).reshape(-1).tobytes())
    return buffer.getvalue()


MONO_WAV = wav_bytes(tone(440))
STEREO_WAV = wav_bytes(tone(440), tone(1000))


def segment(start: float, end: float, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        start=start, end=end, text=text, avg_logprob=-0.2, compression_ratio=1.1, no_speech_prob=0.01
    )


class FakeModel:
    """Stands in for WhisperModel so tests need no download and no GPU."""

    supported_languages = ["en", "ms", "zh"]

    def __init__(self, script=None, on_segment=None) -> None:
        self.script = script or [[segment(0.0, 1.0, " hello")]]
        self.on_segment = on_segment
        self.calls: list[tuple[np.ndarray, dict]] = []
        self.produced = 0

    def transcribe(self, audio, **kwargs):
        call_number = len(self.calls)
        self.calls.append((audio, kwargs))
        found = self.script[min(call_number, len(self.script) - 1)]
        info = SimpleNamespace(language="ms", language_probability=0.9)
        return self._generate(found), info

    def _generate(self, found):
        for item in found:
            if self.on_segment:
                self.on_segment()
            self.produced += 1
            yield item


def make_settings(**overrides) -> Settings:
    defaults = {"api_key": "test-key", "workers": 1, "max_queue": 4, "max_upload_mb": 1}
    return Settings(_env_file=None, **{**defaults, **overrides})


def fake_factory(model: FakeModel):
    return lambda settings: Engine(
        model,
        model_name="fake",
        device="cpu",
        compute_type="int8",
        beam_size=settings.beam_size,
        default_hotwords=settings.hotwords,
    )


@contextmanager
def live_server(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(10)


@pytest.fixture
def auth() -> dict:
    return {"Authorization": "Bearer test-key"}
