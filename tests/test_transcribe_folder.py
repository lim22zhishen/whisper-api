import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from conftest import MONO_WAV, STEREO_WAV, FakeModel, fake_factory, live_server, make_settings

from whisper_api.app import create_app

_spec = importlib.util.spec_from_file_location(
    "transcribe_folder", Path(__file__).parents[1] / "scripts" / "transcribe_folder.py"
)
transcribe_folder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(transcribe_folder)


@pytest.fixture
def recordings(tmp_path) -> Path:
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "a.wav").write_bytes(MONO_WAV)
    (folder / "b.wav").write_bytes(STEREO_WAV)
    return folder


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("WHISPER_API_KEY", "test-key")
    model = FakeModel()
    with live_server(create_app(make_settings(), fake_factory(model))) as url:
        yield url, model


def run(url, recordings, out, *extra):
    return transcribe_folder.main(
        [str(recordings), str(out), "--url", url, "--retries", "1", "--retry-wait", "0", *extra]
    )


def test_saves_one_result_per_recording_and_leaves_sources_untouched(server, recordings, tmp_path):
    url, _ = server
    out = tmp_path / "out"
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in recordings.iterdir()}

    assert run(url, recordings, out) == 0

    assert sorted(p.name for p in out.iterdir()) == ["a.wav.json", "b.wav.json"]
    saved = json.loads((out / "b.wav.json").read_text(encoding="utf-8"))
    assert saved["source_file"] == "b.wav"
    assert [c["name"] for c in saved["channels"]] == ["left", "right"]
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in recordings.iterdir()}
    assert after == before


def test_rerun_does_not_repeat_finished_work(server, recordings, tmp_path):
    url, model = server
    out = tmp_path / "out"
    assert run(url, recordings, out) == 0
    calls_after_first_run = len(model.calls)

    assert run(url, recordings, out) == 0
    assert len(model.calls) == calls_after_first_run


def test_interrupted_run_picks_up_the_rest(server, recordings, tmp_path):
    url, model = server
    out = tmp_path / "out"
    assert run(url, recordings, out, "--limit", "1") == 0
    assert [p.name for p in out.iterdir()] == ["a.wav.json"]

    assert run(url, recordings, out) == 0
    assert sorted(p.name for p in out.iterdir()) == ["a.wav.json", "b.wav.json"]
    assert len(model.calls) == 3  # one mono call, then two stereo channels


def test_bad_file_is_recorded_and_does_not_stop_the_others(server, recordings, tmp_path):
    url, _ = server
    out = tmp_path / "out"
    (recordings / "bad.wav").write_bytes(b"not audio" * 50)

    assert run(url, recordings, out) == 1

    assert (out / "a.wav.json").exists() and (out / "b.wav.json").exists()
    assert not (out / "bad.wav.json").exists()
    failures = json.loads((out / "_failures.json").read_text(encoding="utf-8"))
    assert list(failures) == ["bad.wav"]

    # Once the file is fixed, a re-run succeeds and clears the failure list.
    (recordings / "bad.wav").write_bytes(MONO_WAV)
    assert run(url, recordings, out) == 0
    assert not (out / "_failures.json").exists()


def test_wrong_key_fails_without_saving_anything(server, recordings, tmp_path, monkeypatch):
    url, model = server
    monkeypatch.setenv("WHISPER_API_KEY", "wrong")
    out = tmp_path / "out"
    assert run(url, recordings, out) == 1
    assert model.calls == []
    assert [p.name for p in out.iterdir()] == ["_failures.json"]


def test_server_down_is_retried_then_reported(recordings, tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_API_KEY", "test-key")
    out = tmp_path / "out"
    assert run("http://127.0.0.1:9", recordings, out) == 1
    assert "Gave up" in json.loads((out / "_failures.json").read_text(encoding="utf-8"))["a.wav"]
