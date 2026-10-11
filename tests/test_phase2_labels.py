import numpy as np
import pytest
from phase2_helpers import settings, stream

from lob_research.phase2.labels import build_labels


def test_hand_wall_clock_100ms_and_1s():
    s = stream(mids=[100, 100.01, 100.02, 100.03, 100.05])
    v, _ = build_labels(s, settings())
    assert v["target_state_source_order_100ms"][0][0] == 101
    assert v["MID_MOVE_100MS"][0] == 0.02
    assert v["FWD_MID_RET_100MS"][0] == pytest.approx(0.0002)
    assert v["target_age_ms_100ms"][0] == 10
    assert v["MID_MOVE_1S"][0] == 0.03
    assert v["DIRECTION_100MS"][0] == "UP"
    assert np.isnan(v["MID_MOVE_TICKS_1S"]).all()


def test_no_event_target_available_flat():
    s = stream((0, 2000), [100, 105])
    v, _ = build_labels(s, settings())
    assert v["target_available_1s"][0]
    assert v["target_midprice_1s"][0] == 100
    assert v["target_age_ms_1s"][0] == 1000
    assert v["DIRECTION_1S"][0] == "FLAT"


def test_session_end_equal_and_beyond_no_overnight():
    s = stream((0, 900), end_ms=1000)
    v, _ = build_labels(s, settings())
    assert not v["target_available_1s"].any()
    assert not v["target_available_100ms"][1]  # exact close is excluded
    assert not v["target_available_10s"].any()
    assert v["DIRECTION_1S"][0] is None
    s.time[-1] = 1000000000
    with pytest.raises(ValueError, match="cross-session"):
        build_labels(s, settings())


def test_target_tie_uses_last_source_order():
    s = stream((0, 100, 100, 110), [100, 101, 102, 103])
    v, _ = build_labels(s, settings())
    assert v["target_midprice_100ms"][0] == 102
    assert v["target_state_source_order_100ms"][0][0] == 101
    assert v["target_age_ms_100ms"][0] == 0


def test_invalid_interval_not_silently_forward_filled():
    s = stream((0, 300, 500, 2000))
    s.eligible[1] = False
    v, _ = build_labels(s, settings())
    assert not v["target_available_1s"][0]
    assert v["target_unavailable_reason_1s"][0] == "invalid_book_interval"
    assert v["target_available_1s"][2]
    s.eligible[2] = False
    v, _ = build_labels(s, settings())
    assert not v["target_available_1s"][0]


def test_exact_direction_half_encoding_unit_and_synthetic_ticks():
    s = stream((0, 100, 200))
    s.ask[1, 0] += 1
    s.ask[2, 0] -= 1
    v, _ = build_labels(s, settings(tick=0.0001))
    assert v["MID_MOVE_100MS"][0] == 0.00005
    assert v["MID_MOVE_TICKS_100MS"][0] == 0.5
    assert list(v["DIRECTION_100MS"][:2]) == ["UP", "DOWN"]
