import io

import numpy as np
import pytest
from conftest import MONO_WAV, STEREO_WAV, tone, wav_bytes

from whisper_api.audio import AudioDecodeError, load_channels


def dominant_frequency(samples: np.ndarray, rate: int = 16000) -> float:
    spectrum = np.abs(np.fft.rfft(samples))
    return float(np.fft.rfftfreq(samples.size, 1 / rate)[spectrum.argmax()])


def test_mono_stays_single_and_is_resampled_to_16k():
    [(name, samples)] = load_channels(io.BytesIO(MONO_WAV))
    assert name == "mono"
    assert abs(samples.size - 16000) < 100


def test_stereo_is_split_without_swapping_sides():
    [(left_name, left), (right_name, right)] = load_channels(io.BytesIO(STEREO_WAV))
    assert (left_name, right_name) == ("left", "right")
    assert dominant_frequency(left) == pytest.approx(440, abs=5)
    assert dominant_frequency(right) == pytest.approx(1000, abs=5)


def test_stereo_with_identical_channels_is_not_transcribed_twice():
    same = tone(440)
    assert [name for name, _ in load_channels(io.BytesIO(wav_bytes(same, same)))] == ["mono"]


def test_mix_mode_gives_one_signal_for_stereo():
    assert [name for name, _ in load_channels(io.BytesIO(STEREO_WAV), "mix")] == ["mono"]


def test_non_audio_is_rejected():
    with pytest.raises(AudioDecodeError):
        load_channels(io.BytesIO(b"this is not audio" * 100))
