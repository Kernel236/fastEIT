"""Tests for the recruitment-to-inflation ratio, with answers computed by hand."""

import numpy as np
import pytest

from fasteit.features.recruitment import recruitment_to_inflation, ri_steps

PROTOCOL = [16, 14, 12, 10, 8, 6, 4, 2, 0]  # peep_set of a decremental trial


def test_the_steps_at_the_two_levels_of_the_protocol():
    assert ri_steps(np.array(PROTOCOL, dtype=float), 14, 4) == (1, 6)
    assert ri_steps(np.array(PROTOCOL, dtype=float), 16, 6) == (0, 5)


def test_a_level_that_is_not_one_step_of_the_protocol_gives_no_pair():
    protocol = np.array(PROTOCOL, dtype=float)
    # 15 and 5 are not levels: no step is taken for being the nearest.
    assert ri_steps(protocol, 15, 5) is None
    assert ri_steps(np.array([16, 14, 12, 10, 8, 6.0]), 14, 4) is None
    assert ri_steps(np.array([14, 14, 4.0]), 14, 4) is None  # two steps at 14
    assert ri_steps(np.full(9, np.nan), 14, 4) is None  # no levels of the protocol


def test_ri_and_recruited_volume_with_a_known_answer():
    # Low step: TIV 20, driving 10 -> 2 units/cmH2O; a drop of 10 cmH2O predicts an EELI
    # drop of 20; the EELI drops by 30, so 10 units (50 %) were recruited: R/I 0.5.
    # With 400 mL at the low step, 1 unit = 400 / 20 = 20 mL: recruited 200 mL.
    ri, recruited = recruitment_to_inflation(
        eeli_high=130, eeli_low=100, tiv_low=20, driving_low=10, peep_drop=10,
        tidal_volume_low=400,
    )  # fmt: skip
    assert ri == pytest.approx(0.5)
    assert recruited == pytest.approx(200)
    # The same with the original formula: C_low = 400 / 10 = 40 mL/cmH2O,
    # dEELV = 30 x 20 = 600 mL, recruited 600 - 40 x 10 = 200, C_rec = 20, R/I = 0.5.


def test_ri_needs_no_tidal_volume():
    ri, recruited = recruitment_to_inflation(130, 100, 20, 10, 10, np.nan)
    assert ri == pytest.approx(0.5) and np.isnan(recruited)


@pytest.mark.parametrize(
    "driving_low, peep_drop", [(np.nan, 10), (0, 10), (10, 0), (10, -2)]
)
def test_without_compliance_or_drop_there_is_no_ri(driving_low, peep_drop):
    ri, recruited = recruitment_to_inflation(130, 100, 20, driving_low, peep_drop, 400)
    assert np.isnan(ri) and np.isnan(recruited)
