import logging
import threading
import time

import httpx
import pytest
from conftest import MONO_WAV, STEREO_WAV, FakeModel, fake_factory, live_server, make_settings, segment
from fastapi.testclient import TestClient

from whisper_api.__main__ import startup_problem
from whisper_api.app import create_app

URL = "/v1/audio/transcriptions"


def client_for(model=None, **settings):
    app = create_app(make_settings(**settings), fake_factory(model or FakeModel()))
    return TestClient(app)


def upload(data=MONO_WAV):
    return {"file": ("audio.wav", data, "audio/wav")}


def test_health_needs_no_key_and_reveals_no_secret():
    with client_for() as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "test-key" not in response.text


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Bearer tést".encode()}])
def test_transcription_rejects_a_missing_or_wrong_key(headers):
    with client_for() as client:
        assert client.post(URL, files=upload(), headers=headers).status_code == 401


def test_oversized_upload_is_refused(auth):
    with client_for(max_upload_mb=1) as client:
        big = upload(b"x" * (1024 * 1024 + 1))
        assert client.post(URL, files=big, headers=auth).status_code == 413


def test_upload_without_a_length_is_refused(auth):
    with client_for() as client:
        chunks = iter([b"x"])
        assert client.post(URL, content=chunks, headers=auth).status_code == 411


def test_json_response_has_just_the_text(auth):
    with client_for() as client:
        response = client.post(URL, files=upload(), headers=auth)
    assert response.status_code == 200
    assert response.json() == {"text": "hello"}


def test_text_response(auth):
    with client_for() as client:
        response = client.post(URL, files=upload(), data={"response_format": "text"}, headers=auth)
    assert response.text == "hello"


def test_verbose_response_splits_stereo(auth):
    model = FakeModel([[segment(0.0, 1.0, " speaker one")], [segment(1.0, 2.0, " speaker two")]])
    with client_for(model) as client:
        response = client.post(
            URL, files=upload(STEREO_WAV), data={"response_format": "verbose_json"}, headers=auth
        )
    body = response.json()
    assert body["task"] == "transcribe"
    assert body["model"] == "fake"
    assert body["duration"] == pytest.approx(1.0, abs=0.05)
    assert [(s["channel"], s["text"]) for s in body["segments"]] == [
        ("left", "speaker one"),
        ("right", "speaker two"),
    ]


def test_openai_style_model_field_is_accepted(auth):
    with client_for() as client:
        response = client.post(URL, files=upload(), data={"model": "whisper-1"}, headers=auth)
    assert response.status_code == 200


@pytest.mark.parametrize(
    "files, data, status",
    [
        ({"file": ("x.wav", b"not audio" * 50, "audio/wav")}, {}, 400),
        (upload(), {"language": "xx"}, 400),
        (upload(), {"response_format": "srt"}, 422),
        (upload(), {"channels": "nonsense"}, 422),
    ],
)
def test_bad_requests_get_a_clear_status(auth, files, data, status):
    with client_for() as client:
        assert client.post(URL, files=files, data=data, headers=auth).status_code == status


def test_transcript_text_is_never_logged(auth, caplog):
    secret = "my ic number is nine nine zero"
    model = FakeModel([[segment(0.0, 1.0, f" {secret}")]])
    with caplog.at_level(logging.DEBUG), client_for(model) as client:
        assert client.post(URL, files=upload(), headers=auth).status_code == 200
    assert secret not in caplog.text
    assert "Transcribed" in caplog.text


def test_busy_server_answers_503_then_recovers(auth):
    release = threading.Event()
    entered = threading.Event()

    def block():
        entered.set()
        release.wait(10)

    model = FakeModel(on_segment=block)
    app = create_app(make_settings(workers=1, max_queue=0), fake_factory(model))
    with live_server(app) as url, httpx.Client(headers=auth, timeout=20) as client:
        results = []
        first = threading.Thread(
            target=lambda: results.append(client.post(url + URL, files=upload()).status_code)
        )
        first.start()
        assert entered.wait(10)

        busy = client.post(url + URL, files=upload())
        assert busy.status_code == 503
        assert busy.headers["Retry-After"]

        release.set()
        first.join(10)
        assert results == [200]
        assert client.post(url + URL, files=upload()).status_code == 200


def test_client_disconnect_stops_the_work(auth):
    model = FakeModel(
        [[segment(float(i), i + 1.0, " x") for i in range(200)]],
        on_segment=lambda: time.sleep(0.05),
    )
    app = create_app(make_settings(), fake_factory(model))
    with live_server(app) as url:
        with pytest.raises(httpx.TimeoutException):
            httpx.post(url + URL, files=upload(), headers=auth, timeout=0.5)

        time.sleep(2.5)  # enough for the 1 second disconnect check to fire
        stopped_at = model.produced
        time.sleep(0.5)
        assert model.produced == stopped_at
        assert stopped_at < 200

        assert httpx.get(url + "/health").json()["in_flight"] == 0


@pytest.mark.parametrize(
    "host, key, refused",
    [
        ("0.0.0.0", "", True),
        ("192.168.1.10", "", True),
        ("0.0.0.0", "secret", False),
        ("127.0.0.1", "", False),
        ("localhost", "", False),
    ],
)
def test_network_listening_requires_a_key(host, key, refused):
    assert bool(startup_problem(make_settings(host=host, api_key=key))) is refused
