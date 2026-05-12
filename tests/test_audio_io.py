import numpy as np

from voice.audio_io import rms


def test_rms_returns_zero_for_silence() -> None:
    assert rms(np.zeros((160, 1), dtype="float32")) == 0.0


def test_rms_increases_with_signal_level() -> None:
    quiet = rms(np.full((160, 1), 0.01, dtype="float32"))
    loud = rms(np.full((160, 1), 0.1, dtype="float32"))

    assert loud > quiet
