import io
import threading

import pytest
from conftest import MONO_WAV, STEREO_WAV, FakeModel, segment, wav_bytes
import numpy as np

from whisper_api.engine import Engine, InvalidOption, TranscribeOptions, TranscriptionCancelled


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


def test_file_with_no_samples_gives_an_empty_result():
    model = FakeModel()
    result = run(model, wav=wav_bytes(np.zeros(0, dtype=np.int16)))
    assert result["segments"] == []
    assert result["language"] is None
    assert model.calls == []
