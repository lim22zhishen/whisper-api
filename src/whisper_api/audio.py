"""Decode uploaded audio into the separate signals to transcribe."""

from typing import BinaryIO, Literal

import av
import numpy as np
from faster_whisper.audio import decode_audio

ChannelMode = Literal["auto", "mix"]

SAMPLE_RATE = 16000

# Stereo files whose channels differ by less than this are really mono.
_SAME_SIGNAL_TOLERANCE = 1e-3


class AudioDecodeError(ValueError):
    """The upload could not be read as audio."""


def load_channels(source: BinaryIO, mode: ChannelMode = "auto") -> list[tuple[str, np.ndarray]]:
    """Return (channel name, 16 kHz samples) pairs: left and right, or a single mono."""
    try:
        if mode == "auto" and _count_channels(source) == 2:
            source.seek(0)
            left, right = decode_audio(source, SAMPLE_RATE, split_stereo=True)
            if not np.allclose(left, right, atol=_SAME_SIGNAL_TOLERANCE):
                return [("left", left), ("right", right)]
            return [("mono", left)]

        source.seek(0)
        return [("mono", decode_audio(source, SAMPLE_RATE))]
    except av.error.FFmpegError as exc:
        raise AudioDecodeError("Could not read the file as audio") from exc


def _count_channels(source: BinaryIO) -> int:
    source.seek(0)
    with av.open(source, mode="r", metadata_errors="ignore") as container:
        if not container.streams.audio:
            raise AudioDecodeError("The file has no audio track")
        return container.streams.audio[0].codec_context.channels
