import io
import threading

import pytest
from conftest import MONO_WAV, STEREO_WAV, FakeModel, segment, wav_bytes
import numpy as np

import whisper_api.engine as engine_module
from whisper_api.config import Settings
from whisper_api.engine import (
    Engine,
    EngineStartupError,
    InvalidOption,
    LimitedLanguages,
    TranscribeOptions,
    TranscriptionCancelled,
)


def run(model, wav=STEREO_WAV, options=TranscribeOptions(), cancel=None, **engine_args):
    return Engine(model, **engine_args).transcribe(io.BytesIO(wav), options, cancel)


def test_stereo_segments_are_merged_in_time_order_with_their_channel():
    model = FakeModel([
        [segment(0.0, 1.0, " left one"), segment(4.0, 5.0, " left two")],
        [segment(2.0, 3.0, " right one")],
    ])
    result = run(model)
    assert [(s["id"], s["channel"], s["text"]) for s in result["segments"]] == [
        (0, "left", "left one"),
        (1, "right", "right one"),
        (2, "left", "left two"),
    ]
    assert result["text"] == "left one right one left two"
    assert [c["name"] for c in result["channels"]] == ["left", "right"]


def test_each_channel_gets_its_own_audio():
    model = FakeModel()
    run(model)
    (left, _), (right, _) = model.calls
    assert not np.allclose(left, right, atol=1e-3)


def test_default_settings_reach_the_model():
    model = FakeModel()
    run(model, wav=MONO_WAV, default_hotwords="Kubernetes, PostgreSQL", beam_size=3)
    _, kwargs = model.calls[0]
    assert kwargs["task"] == "transcribe"
    assert kwargs["language"] is None
    assert kwargs["multilingual"] is True
    assert kwargs["vad_filter"] is True
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["hotwords"] == "Kubernetes, PostgreSQL"
    assert kwargs["beam_size"] == 3


def test_forcing_a_language_turns_off_detection():
    model = FakeModel()
    run(model, wav=MONO_WAV, options=TranscribeOptions(language="MS"))
    _, kwargs = model.calls[0]
    assert kwargs["language"] == "ms"
    assert kwargs["multilingual"] is False


@pytest.mark.parametrize("requested, expected", [("PostgreSQL", "PostgreSQL"), ("", None)])
def test_request_hotwords_replace_the_default(requested, expected):
    model = FakeModel()
    run(model, wav=MONO_WAV, options=TranscribeOptions(hotwords=requested), default_hotwords="Kubernetes")
    assert model.calls[0][1]["hotwords"] == expected


def test_unknown_language_is_rejected_before_any_work():
    model = FakeModel()
    with pytest.raises(InvalidOption):
        run(model, wav=MONO_WAV, options=TranscribeOptions(language="xx"))
    assert model.calls == []


def test_cancel_stops_between_segments():
    cancel = threading.Event()
    model = FakeModel([[segment(float(i), i + 1.0, " x") for i in range(100)]], on_segment=None)
    model.on_segment = lambda: cancel.set() if model.produced == 3 else None
    with pytest.raises(TranscriptionCancelled):
        run(model, wav=MONO_WAV, cancel=cancel)
    assert model.produced < 100


def test_channel_with_no_speech_reports_no_language():
    model = FakeModel([[segment(0.0, 1.0, " hi")], []])
    result = run(model)
    assert result["channels"][1]["language"] is None
    assert result["language"] == "ms"


@pytest.mark.parametrize("model_dir, expected", [("", None), (None, None), ("models", "models")])
def test_empty_model_folder_uses_the_default_cache(monkeypatch, model_dir, expected):
    seen = {}

    def fake_whisper_model(name, **kwargs):
        seen.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(engine_module, "WhisperModel", fake_whisper_model)
    settings = Settings(_env_file=None, device="cpu", model_dir=model_dir)
    engine_module.load_engine(settings)
    assert seen["download_root"] == expected


def test_file_with_no_samples_gives_an_empty_result():
    model = FakeModel()
    result = run(model, wav=wav_bytes(np.zeros(0, dtype=np.int16)))
    assert result["segments"] == []
    assert result["language"] is None
    assert model.calls == []


class FakeTranslator:
    """Stands in for the CTranslate2 model that WhisperModel keeps in `.model`."""

    is_multilingual = True

    def __init__(self, ranked=(("<|id|>", 0.6), ("<|ms|>", 0.3), ("<|en|>", 0.1))):
        self.ranked = list(ranked)

    def detect_language(self, features):
        return [self.ranked]


def test_limited_detection_drops_other_languages_and_rescales():
    limited = LimitedLanguages(FakeTranslator(), ["ms", "en"])
    [ranked] = limited.detect_language(None)
    assert [token for token, _ in ranked] == ["<|ms|>", "<|en|>"]
    assert [p for _, p in ranked] == pytest.approx([0.75, 0.25])
    assert limited.is_multilingual is True  # everything else passes through


def test_limited_detection_survives_zero_chances():
    limited = LimitedLanguages(FakeTranslator([("<|id|>", 1.0), ("<|ms|>", 0.0)]), ["ms"])
    assert limited.detect_language(None) == [[("<|ms|>", 0.0)]]


def load_with_languages(monkeypatch, languages):
    model = FakeModel()
    model.model = FakeTranslator()
    monkeypatch.setattr(engine_module, "WhisperModel", lambda name, **kwargs: model)
    settings = Settings(_env_file=None, device="cpu", languages=languages)
    return engine_module.load_engine(settings), model


def test_language_limit_is_applied_at_startup(monkeypatch):
    engine, model = load_with_languages(monkeypatch, " MS, en,,ms")
    assert engine.languages == ["ms", "en"]
    [ranked] = model.model.detect_language(None)
    assert ranked[0][0] == "<|ms|>"


def test_no_language_limit_leaves_detection_open(monkeypatch):
    engine, model = load_with_languages(monkeypatch, "")
    assert engine.languages == []
    assert isinstance(model.model, FakeTranslator)


def test_unknown_code_in_the_language_limit_stops_startup(monkeypatch):
    with pytest.raises(EngineStartupError, match="xx"):
        load_with_languages(monkeypatch, "ms,xx")
