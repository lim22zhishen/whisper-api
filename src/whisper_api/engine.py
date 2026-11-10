"""Speech-to-text engine: runs faster-whisper on each channel and merges the result."""

import logging
import os
import site
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel

from .audio import ChannelMode, load_channels
from .config import Settings

logger = logging.getLogger("whisper_api")

GPU_HELP = (
    "Could not run the speech model on the GPU. Check that the NVIDIA driver works "
    "(nvidia-smi) and that the GPU libraries are installed: pip install -e .[gpu]. "
    "On Linux also add them to LD_LIBRARY_PATH (see README)."
)


class EngineStartupError(RuntimeError):
    """The model could not be loaded or run."""


class InvalidOption(ValueError):
    """A request option has an unusable value."""


class TranscriptionCancelled(Exception):
    """The caller went away, so the work was stopped early."""


@dataclass(frozen=True)
class TranscribeOptions:
    language: str | None = None
    prompt: str | None = None
    hotwords: str | None = None  # None uses the server default, "" turns hints off
    channels: ChannelMode = "auto"


class LimitedLanguages:
    """Wraps the CTranslate2 model so language detection only picks allowed languages.

    faster-whisper detects the language of each 30-second window and writes the whole
    window in it. Left open, it sometimes picks a language nobody spoke, such as
    Indonesian for Malay, and that window comes out in the wrong language.
    """

    def __init__(self, model: Any, languages: list[str]) -> None:
        self._model = model
        self._tokens = {f"<|{code}|>" for code in languages}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def detect_language(self, *args: Any, **kwargs: Any) -> list[list[tuple[str, float]]]:
        return [self._limit(ranked) for ranked in self._model.detect_language(*args, **kwargs)]

    def _limit(self, ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
        kept = [(token, p) for token, p in ranked if token in self._tokens]
        total = sum(p for _, p in kept)
        if total <= 0:
            return kept or ranked
        # Rescale so the allowed languages' chances add up to 1.
        return [(token, p / total) for token, p in kept]


def parse_languages(value: str) -> list[str]:
    """Turn "ms, EN,zh" into ["ms", "en", "zh"]."""
    codes = (code.strip().lower() for code in value.split(","))
    return list(dict.fromkeys(code for code in codes if code))


def limit_languages(model: Any, languages: list[str]) -> None:
    """Make detection choose only from `languages`. An empty list leaves it open."""
    unknown = [code for code in languages if code not in model.supported_languages]
    if unknown:
        raise EngineStartupError(
            f"Unknown language code in WHISPER_LANGUAGES: {', '.join(unknown)}"
        )
    if languages:
        model.model = LimitedLanguages(model.model, languages)


class Engine:
    def __init__(
        self,
        model: Any,
        *,
        model_name: str = "",
        device: str = "",
        compute_type: str = "",
        beam_size: int = 5,
        default_hotwords: str = "",
        languages: list[str] | None = None,
        line_gap_seconds: float = 0.0,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.default_hotwords = default_hotwords
        self.languages = list(languages or [])  # only reported; limit_languages applies it
        self.line_gap_seconds = line_gap_seconds  # 0 keeps the model's own lines

    def transcribe(
        self,
        source: BinaryIO,
        options: TranscribeOptions,
        cancel: threading.Event | None = None,
    ) -> dict[str, Any]:
        cancel = cancel or threading.Event()
        language = options.language.lower() if options.language else None
        if language and language not in self.model.supported_languages:
            raise InvalidOption(f"Unknown language code: {options.language}")

        hotwords = self.default_hotwords if options.hotwords is None else options.hotwords
        decode_options = {
            "task": "transcribe",  # keep the original language, never translate
            "language": language,
            "multilingual": language is None,  # audio can switch language mid-way
            "initial_prompt": options.prompt or None,
            "hotwords": hotwords or None,
            "beam_size": self.beam_size,
            "vad_filter": True,  # skip silence
            "vad_parameters": {"min_silence_duration_ms": 500},
            # Feeding earlier text back in can cause repeated-phrase loops.
            "condition_on_previous_text": False,
            # Word times are only needed to split lines at pauses, and they cost speed.
            "word_timestamps": self.line_gap_seconds > 0,
        }

        duration = 0.0
        segments: list[dict[str, Any]] = []
        channels: list[dict[str, Any]] = []
        self._check(cancel)
        for name, samples in load_channels(source, options.channels):
            self._check(cancel)
            duration = max(duration, samples.size / 16000)
            channel_segments, info = self._run_channel(name, samples, decode_options, cancel)
            segments.extend(channel_segments)
            channels.append(
                {
                    "name": name,
                    "language": info.language if channel_segments and info else None,
                    "language_probability": (
                        info.language_probability if channel_segments and info else None
                    ),
                    "speech_seconds": round(sum(s["end"] - s["start"] for s in channel_segments), 2),
                }
            )

        segments.sort(key=lambda s: (s["start"], s["end"]))
        for index, segment in enumerate(segments):
            segment["id"] = index

        return {
            "language": _main_language(channels),
            "duration": round(duration, 2),
            "text": " ".join(s["text"] for s in segments),
            "channels": channels,
            "segments": segments,
        }

    def _run_channel(
        self,
        name: str,
        samples: np.ndarray,
        decode_options: dict[str, Any],
        cancel: threading.Event,
    ) -> tuple[list[dict[str, Any]], Any]:
        if samples.size == 0:
            return [], None

        found, info = self.model.transcribe(samples, **decode_options)
        results = []
        for segment in found:  # each step of this loop does the real work
            self._check(cancel)
            for start, end, text in self._lines(segment):
                results.append(
                    {
                        "channel": name,
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "text": text,
                        # The model scores its own line; the parts of a split line share it.
                        "avg_logprob": round(segment.avg_logprob, 4),
                        "compression_ratio": round(segment.compression_ratio, 4),
                        "no_speech_prob": round(segment.no_speech_prob, 4),
                    }
                )
        return results, info

    def _lines(self, segment: Any) -> list[tuple[float, float, str]]:
        """Split one model line wherever the speaker paused.

        With silence skipped, the model writes one line over speech that was far apart,
        such as short replies between the other side's turns. Word times point back to
        the real audio, so a long gap between two words is a real pause.
        """
        words = getattr(segment, "words", None) if self.line_gap_seconds > 0 else None
        if not words:
            return [(segment.start, segment.end, segment.text.strip())]

        groups = [[words[0]]]
        for word in words[1:]:
            if word.start - groups[-1][-1].end >= self.line_gap_seconds:
                groups.append([word])
            else:
                groups[-1].append(word)

        lines = []
        for group in groups:
            text = "".join(word.word for word in group).strip()
            if text:
                lines.append((group[0].start, group[-1].end, text))
        return lines or [(segment.start, segment.end, segment.text.strip())]

    @staticmethod
    def _check(cancel: threading.Event) -> None:
        if cancel.is_set():
            raise TranscriptionCancelled


def _main_language(channels: list[dict[str, Any]]) -> str | None:
    spoken = [c for c in channels if c["language"]]
    return max(spoken, key=lambda c: c["speech_seconds"])["language"] if spoken else None


def load_engine(settings: Settings) -> Engine:
    """Load the model and prove it can run, so a bad GPU setup fails at startup."""
    device = settings.device
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute_type = settings.compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    # Half the logical CPUs is about the number of physical cores.
    cpu_threads = settings.cpu_threads or max(1, (os.cpu_count() or 2) // 2 // settings.workers)

    if device == "cuda":
        _expose_pip_gpu_libraries()

    logger.info("Loading model %s on %s (%s)", settings.model, device, compute_type)
    try:
        model = WhisperModel(
            settings.model,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=settings.workers,
            download_root=settings.model_dir or None,
        )
        # GPU libraries load on first use, not when the model is created.
        silence = np.zeros(16000, dtype=np.float32)
        list(model.transcribe(silence, language="en", beam_size=1)[0])
    except (RuntimeError, ValueError, OSError) as exc:
        if device == "cuda":
            raise EngineStartupError(f"{GPU_HELP} Original error: {exc}") from exc
        raise EngineStartupError(f"Could not load the speech model: {exc}") from exc

    languages = parse_languages(settings.languages)
    limit_languages(model, languages)
    if languages:
        logger.info("Language detection limited to %s", ", ".join(languages))

    logger.info("Model ready")
    return Engine(
        model,
        model_name=settings.model,
        device=device,
        compute_type=compute_type,
        beam_size=settings.beam_size,
        default_hotwords=settings.hotwords,
        languages=languages,
        line_gap_seconds=settings.line_gap_seconds,
    )


def _expose_pip_gpu_libraries() -> None:
    """Let Windows find the DLLs from the `gpu` extra. Linux needs LD_LIBRARY_PATH."""
    if os.name != "nt":
        return
    for base in site.getsitepackages():
        for package in ("cublas", "cudnn"):
            folder = Path(base) / "nvidia" / package / "bin"
            if folder.is_dir():
                os.environ["PATH"] = f"{folder}{os.pathsep}{os.environ.get('PATH', '')}"
