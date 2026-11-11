import inspect
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


REPLIES = [(0.0, 0.4, " yes"), (0.5, 0.9, " okay"), (9.0, 9.5, " no"), (20.0, 20.4, " 好"), (20.4, 20.8, "的")]


def test_a_line_is_split_where_the_speaker_paused():
    model = FakeModel([[segment(0.0, 20.8, " yes okay no 好的", words=REPLIES)]])
    result = run(model, wav=MONO_WAV, line_gap_seconds=1.0)
    assert [(s["id"], s["start"], s["end"], s["text"]) for s in result["segments"]] == [
        (0, 0.0, 0.9, "yes okay"),
        (1, 9.0, 9.5, "no"),
        (2, 20.0, 20.8, "好的"),
    ]
    assert {s["avg_logprob"] for s in result["segments"]} == {-0.2}
    assert result["channels"][0]["speech_seconds"] == 2.2
    assert model.calls[0][1]["word_timestamps"] is True


def test_split_replies_slot_between_the_other_sides_lines():
    model = FakeModel([
        [segment(0.0, 9.5, " yes okay no", words=REPLIES[:3])],
        [segment(2.0, 8.0, " a long question")],
    ])
    result = run(model, line_gap_seconds=1.0)
    assert [(s["channel"], s["text"]) for s in result["segments"]] == [
        ("left", "yes okay"),
        ("right", "a long question"),
        ("left", "no"),
    ]


def test_no_line_gap_keeps_the_models_own_lines():
    model = FakeModel([[segment(0.0, 20.8, " yes okay no", words=REPLIES[:3])]])
    result = run(model, wav=MONO_WAV)
    assert [s["text"] for s in result["segments"]] == ["yes okay no"]
    assert model.calls[0][1]["word_timestamps"] is False


@pytest.mark.parametrize(
    "words, expected",
    [
        # Unplaced word far after a reply: it joins the reply, keeping the reply's times.
        ([(0.0, 0.4, " yes"), (0.5, 0.9, " okay"), (20.0, 20.0, " lah")], [(0.0, 0.9, "yes okay lah")]),
        # Unplaced word right before a later reply joins that reply instead.
        (
            [(0.0, 0.4, " yes"), (8.5, 8.5, " no"), (9.6, 10.0, " problem")],
            [(0.0, 0.4, "yes"), (9.6, 10.0, "no problem")],
        ),
        # Unplaced word first: it joins the line after it.
        ([(3.0, 3.0, " so"), (8.0, 8.4, " okay")], [(8.0, 8.4, "so okay")]),
    ],
)
def test_a_word_with_no_length_never_becomes_its_own_line(words, expected):
    model = FakeModel([[segment(0.0, 20.0, " text", words=words)]])
    result = run(model, wav=MONO_WAV, line_gap_seconds=1.0)
    assert [(s["start"], s["end"], s["text"]) for s in result["segments"]] == expected


def test_a_line_whose_words_all_have_no_length_keeps_the_models_times():
    model = FakeModel([[segment(4.0, 6.0, " hello there", words=[(5.0, 5.0, " hello"), (5.0, 5.0, " there")])]])
    result = run(model, wav=MONO_WAV, line_gap_seconds=1.0)
    assert [(s["start"], s["end"], s["text"]) for s in result["segments"]] == [(4.0, 6.0, "hello there")]


def test_a_line_without_word_times_is_kept_whole():
    model = FakeModel([[segment(0.0, 5.0, " hello there")]])
    result = run(model, wav=MONO_WAV, line_gap_seconds=1.0)
    assert [(s["start"], s["end"], s["text"]) for s in result["segments"]] == [(0.0, 5.0, "hello there")]


@pytest.fixture
def batch_pipelines(monkeypatch):
    """Replaces faster-whisper's batch mode; lists the model each pipeline was made for."""
    made = []

    class FakePipeline:
        def __init__(self, model):
            made.append(model)
            self.model = model

        def transcribe(self, audio, **kwargs):
            return self.model.transcribe(audio, **kwargs)

    monkeypatch.setattr(engine_module, "BatchedInferencePipeline", FakePipeline)
    return made


def test_batch_mode_sends_the_usual_options_and_the_batch_size(batch_pipelines):
    model = FakeModel([[segment(0.0, 20.8, " yes okay no 好的", words=REPLIES)]])
    result = run(model, default_hotwords="Kubernetes", line_gap_seconds=1.0, batch_size=8)
    assert batch_pipelines == [model, model]  # a new pipeline for each channel
    _, kwargs = model.calls[0]
    assert kwargs["batch_size"] == 8
    assert kwargs["without_timestamps"] is False
    assert kwargs["task"] == "transcribe"
    assert kwargs["multilingual"] is True
    assert kwargs["hotwords"] == "Kubernetes"
    assert kwargs["word_timestamps"] is True
    assert [s["text"] for s in result["segments"] if s["channel"] == "left"] == ["yes okay", "no", "好的"]


def test_batch_mode_in_the_library_takes_every_option_sent(batch_pipelines):
    from faster_whisper import BatchedInferencePipeline

    model = FakeModel()
    run(model, wav=MONO_WAV, line_gap_seconds=1.0, batch_size=4)
    accepted = inspect.signature(BatchedInferencePipeline.transcribe).parameters
    assert set(model.calls[0][1]) <= set(accepted)


def test_batch_size_zero_runs_pieces_one_at_a_time(batch_pipelines):
    model = FakeModel()
    run(model, wav=MONO_WAV)
    assert batch_pipelines == []
    assert "batch_size" not in model.calls[0][1]


def test_batch_size_setting_reaches_the_engine(monkeypatch):
    monkeypatch.setattr(engine_module, "WhisperModel", lambda name, **kwargs: FakeModel())
    engine = engine_module.load_engine(Settings(_env_file=None, device="cpu", batch_size=8))
    assert engine.batch_size == 8


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
