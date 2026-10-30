"""Tests for the pressures of each breath, on a synthetic volume-control curve."""

import numpy as np
import pytest

from fasteit.config import SegmentationConfig
from fasteit.segmentation.breath_pressures import (
    inspiration_onsets,
    plateau_pressures,
    pressure_before,
)

CFG = SegmentationConfig()


def vcv_pressure(
    peeps, fs=50, period=4.0, inspiration=1.0, pause=0.8, peak=20.0, plateau=12.0
):
    """Volume-control airway pressure: ramp, end-inspiratory pause, passive expiration.

    A new PEEP is applied from the expiration of the breath before, as a ventilator
    does. Returns the pressure and the frame where each breath starts.
    """
    t = np.arange(0, period, 1 / fs)
    pauses = np.broadcast_to(pause, len(peeps))  # one pause, or one per breath
    breaths, starts = [], []
    for k, peep in enumerate(peeps):
        next_peep = peeps[k + 1] if k + 1 < len(peeps) else peep
        pause = pauses[k]
        p = np.full(len(t), float(peep))
        ramp = t < inspiration
        p[ramp] = peep + peak * t[ramp] / inspiration
        held = (t >= inspiration) & (t < inspiration + pause)
        p[held] = peep + plateau
        # The expiratory valve opens: the airway pressure falls within about 0.1 s,
        # from the plateau, or from the peak if there is no pause.
        top = peep + (peak if pause == 0 else plateau)
        out = t >= inspiration + pause
        p[out] = next_peep + (top - next_peep) * np.exp(
            -(t[out] - inspiration - pause) / 0.05
        )
        starts.append(len(t) * len(breaths))
        breaths.append(p)
    return np.concatenate(breaths), np.array(starts)


def test_inspiration_onsets_are_the_start_of_the_pressure_rise():
    paw, starts = vcv_pressure([10] * 6)
    eit_starts = starts + 3  # the EIT valley a few frames later
    onsets = inspiration_onsets(paw, 50, eit_starts)
    assert np.all(np.abs(onsets[1:] - starts[1:]) <= 2)


def test_peep_and_plateau_are_read_on_the_curve():
    paw, starts = vcv_pressure([12, 12, 12, 10, 10, 10])
    onsets = inspiration_onsets(paw, 50, starts)
    peep = pressure_before(paw, 50, onsets, CFG.peep_seconds)
    plateau = plateau_pressures(paw, 50, onsets, peep, CFG.plateau_window_seconds)
    np.testing.assert_allclose(peep[1:], [12, 12, 10, 10, 10], atol=0.05)
    # Plateau = PEEP of the breath + 12; the last breath has no next breath.
    np.testing.assert_allclose(plateau[1:-1], [24, 24, 22, 22], atol=0.05)
    assert np.isnan(plateau[-1])


def test_without_an_end_inspiratory_pause_there_is_no_plateau():
    paw, starts = vcv_pressure([10] * 5, pause=0.0)
    onsets = inspiration_onsets(paw, 50, starts)
    peep = pressure_before(paw, 50, onsets, CFG.peep_seconds)
    assert np.isnan(
        plateau_pressures(paw, 50, onsets, peep, CFG.plateau_window_seconds)
    ).all()


def test_missing_pressure_gives_nan_for_that_breath_only():
    paw, starts = vcv_pressure([10] * 5)
    paw[starts[2] - 10 : starts[2] + 10] = np.nan
    onsets = inspiration_onsets(paw, 50, starts)
    peep = pressure_before(paw, 50, onsets, CFG.peep_seconds)
    assert onsets[2] == -1 and np.isnan(peep[2])
    assert np.isfinite(peep[[1, 3, 4]]).all()


@pytest.mark.parametrize("fs", [20, 50])
def test_both_draeger_sampling_rates(fs):
    paw, starts = vcv_pressure([8] * 5, fs=fs)
    onsets = inspiration_onsets(paw, fs, starts)
    peep = pressure_before(paw, fs, onsets, CFG.peep_seconds)
    np.testing.assert_allclose(peep[1:], 8, atol=0.05)


def test_a_short_hold_gives_no_plateau():
    # A 0.3 s hold is too short: the reading would fall on the relaxation after the peak.
    paw, starts = vcv_pressure([10] * 5, pause=0.3)
    onsets = inspiration_onsets(paw, 50, starts)
    peep = pressure_before(paw, 50, onsets, CFG.peep_seconds)
    assert np.isnan(
        plateau_pressures(paw, 50, onsets, peep, CFG.plateau_window_seconds)
    ).all()


def test_missing_pressure_in_the_breath_gives_no_plateau():
    paw, starts = vcv_pressure([10] * 5)
    onsets = inspiration_onsets(paw, 50, starts)
    peep = pressure_before(paw, 50, onsets, CFG.peep_seconds)
    paw[starts[2] + 60] = np.nan  # during the hold of breath 2
    plateau = plateau_pressures(paw, 50, onsets, peep, CFG.plateau_window_seconds)
    assert np.isnan(plateau[2]) and np.isfinite(plateau[[1, 3]]).all()


def test_missing_pressure_before_the_rise_gives_no_onset():
    paw, starts = vcv_pressure([10] * 5)
    paw[starts[3] - 20] = np.nan  # in the 0.5 s lookback, not in the slope window
    onsets = inspiration_onsets(paw, 50, starts + 25)
    assert onsets[3] == -1


def test_the_first_breath_at_the_start_of_the_recording():
    paw, starts = vcv_pressure([10] * 4)
    onsets = inspiration_onsets(paw, 50, starts)
    assert onsets[0] == 0  # no fake rise from padding at the edge


def pcv_pressure(peeps, fs=50, period=4.0, inspiration=1.2, above_peep=15.0):
    """Pressure-control airway pressure: fast rise, flat inspiratory pressure, expiration."""
    t = np.arange(0, period, 1 / fs)
    noise = np.random.default_rng(1).normal(0, 0.1, (len(peeps), len(t)))
    breaths, starts = [], []
    for k, peep in enumerate(peeps):
        next_peep = peeps[k + 1] if k + 1 < len(peeps) else peep
        p = np.full(len(t), float(peep))
        rise = t < 0.1
        p[rise] = peep + above_peep * t[rise] / 0.1
        flat = (t >= 0.1) & (t < inspiration)
        p[flat] = peep + above_peep
        out = t >= inspiration
        top = peep + above_peep
        p[out] = next_peep + (top - next_peep) * np.exp(-(t[out] - inspiration) / 0.05)
        starts.append(len(t) * k)
        breaths.append(p + noise[k])
    return np.concatenate(breaths), np.array(starts)


def test_pressure_control_reads_the_flat_inspiratory_pressure():
    paw, starts = pcv_pressure([10, 10, 10, 8, 8, 8])
    onsets = inspiration_onsets(paw, 50, starts)
    peep = pressure_before(paw, 50, onsets, CFG.peep_seconds)
    plateau = plateau_pressures(paw, 50, onsets, peep, CFG.plateau_window_seconds)
    np.testing.assert_allclose(plateau[1:-1] - peep[1:-1], 15, atol=0.2)


def test_one_inspiratory_hold_among_breaths_without_pause():
    paw, starts = vcv_pressure([10] * 6, pause=[0, 0, 1.5, 0, 0, 0])
    onsets = inspiration_onsets(paw, 50, starts)
    peep = pressure_before(paw, 50, onsets, CFG.peep_seconds)
    plateau = plateau_pressures(paw, 50, onsets, peep, CFG.plateau_window_seconds)
    assert np.flatnonzero(np.isfinite(plateau)).tolist() == [2]
    assert plateau[2] == pytest.approx(22, abs=0.05)
