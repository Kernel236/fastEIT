"""Tests for reconstruction.postprocess - low-pass of the reconstructed images."""

import numpy as np

from fasteit.reconstruction.postprocess import lowpass

FS = 50.0


def sine(freq_hz, n_frames=3000):
    t = np.arange(n_frames) / FS
    return np.sin(2 * np.pi * freq_hz * t)[:, None] * np.ones((1, 4))


class TestLowpass:
    def test_breathing_passes_and_heart_beat_is_removed(self):
        breath, heart = sine(0.25), 0.5 * sine(1.8)  # 15/min and 108/min
        out = lowpass(breath + heart, FS)
        middle = slice(500, -500)
        np.testing.assert_allclose(out[middle], breath[middle], atol=0.03)

    def test_zero_phase(self):
        """The Dräger images are not delayed: neither is the filtered reconstruction."""
        breath = sine(0.25)
        out = lowpass(breath, FS)
        peak_in = np.argmax(breath[1000:1300, 0])
        peak_out = np.argmax(out[1000:1300, 0])
        assert abs(peak_out - peak_in) <= 1
