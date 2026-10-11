import numpy as np
import pytest
from phase2_helpers import settings, stream

from lob_research.phase2.features import build_features
from lob_research.phase2.order_flow import ofi_event


@pytest.mark.parametrize("bid,ask,expected", [(1500, 500, 0.5), (1000, 1000, 0), (500, 1500, -0.5)])
def test_hand_obi_all_depths(bid, ask, expected):
    s = stream()
    s.bsize[:] = bid
    s.asize[:] = ask
    f, _ = build_features(s, settings())
    for k in (1, 3, 5, 10):
        assert np.all(f[f"OBI_L{k}"] == expected)
        assert np.all(f[f"BID_DEPTH_L{k}"] == k * bid)
        assert np.all(f[f"DEPTH_RATIO_L{k}"] == bid / ask)
    np.testing.assert_allclose(f["WOBI_L10"], expected, atol=1e-15)


def test_zero_denominator_missing_depth_and_no_infinity():
    s = stream()
    s.bsize[0] = 0
    s.asize[0] = 0
    s.bid[1, 9] = np.nan
    f, _ = build_features(s, settings())
    assert np.isnan(f["OBI_L1"][0])
    assert np.isnan(f["OBI_L10"][1]) and f["OBI_L1"][1] == 0.5
    assert not f["feature_static_available"][1]
    assert not any(np.isinf(v).any() for v in f.values())


@pytest.mark.parametrize("qb,qa,expected", [(100, 100, 101), (300, 100, 101.5), (100, 300, 100.5)])
def test_hand_microprice_and_verified_synthetic_tick(qb, qa, expected):
    s = stream()
    s.bid[:] = 1000000 - np.arange(10) * 100
    s.ask[:] = 1020000 + np.arange(10) * 100
    s.bsize[:] = qb
    s.asize[:] = qa
    f, _ = build_features(s, settings(tick=0.5))
    assert np.all(f["MICROPRICE"] == expected)
    assert np.all(f["MICROPRICE_DEVIATION_TICKS"] == (expected - 101) / 0.5)
    null, _ = build_features(s, settings())
    assert np.isnan(null["MICROPRICE_DEVIATION_TICKS"]).all()


@pytest.mark.parametrize(
    "side,price_delta,size,expected",
    [
        ("bid", 0, 150, 50),
        ("bid", 0, 60, -40),
        ("bid", 100, 150, 150),
        ("bid", -100, 150, -100),
        ("ask", 0, 150, -50),
        ("ask", 0, 60, 40),
        ("ask", -100, 150, -150),
        ("ask", 100, 150, 100),
    ],
)
def test_hand_ofi_signs(side, price_delta, size, expected):
    s = stream((0, 1))
    s.bsize[:] = 100
    s.asize[:] = 100
    getattr(s, side)[1, 0] += price_delta
    getattr(s, "bsize" if side == "bid" else "asize")[1, 0] = size
    actual = ofi_event(s, np.ones(2, bool))
    assert np.isnan(actual[0]) and actual[1] == expected


def test_window_left_exclusive_right_source_inclusive():
    s = stream((0, 100, 100, 200))
    s.bsize[:, 0] = [100, 120, 150, 160]
    f, _ = build_features(s, settings())
    np.testing.assert_equal(f["EVENT_COUNT_100MS"], [np.nan, 1, 2, 1])
    np.testing.assert_equal(f["OFI_100MS"], [np.nan, 20, 50, 10])
    assert f["EVENT_RATE_100MS"][2] == 20
    assert not f["feature_1s_available"].any()


def test_historical_asof_last_before_not_first_after():
    s = stream((0, 50, 90, 130, 200), [100, 100.01, 100.02, 100.03, 101])
    f, _ = build_features(s, settings())
    assert f["MID_RETURN_100MS"][-1] == pytest.approx(101 / 100.02 - 1)


def test_activity_resting_side_and_previous_times():
    s = stream((0, 100, 200, 300, 400, 500))
    s.kind[:] = ["A", "E", "P", "X", "A", "E"]
    s.side[:] = ["B", "B", "S", "S", "B", "S"]
    s.bid[2:] += 10
    s.ask[2:] += 10
    f, _ = build_features(s, settings())
    assert f["EXECUTE_ASK_SIDE_COUNT_100MS"][2] == 1
    assert f["HIDDEN_EXECUTE_COUNT_100MS"][2] == 1
    assert np.isnan(f["TIME_SINCE_PREVIOUS_TRADE"][1])
    assert f["TIME_SINCE_PREVIOUS_TRADE"][2] == 100
    assert np.isnan(f["TIME_SINCE_PREVIOUS_BOOK_PRICE_CHANGE"][2])
    assert f["TIME_SINCE_PREVIOUS_BOOK_PRICE_CHANGE"][3] == 100
    assert all("NEXT" not in k for k in f)


def test_invalid_history_resets_and_unavailable_is_not_zero():
    s = stream((0, 100, 200, 300, 400))
    s.eligible[1] = False
    f, _ = build_features(s, settings())
    assert np.isnan(f["OFI_EVENT"][1:3]).all()
    assert np.isnan(f["EVENT_COUNT_100MS"][2])
    assert f["EVENT_COUNT_100MS"][3] == 1
    assert np.isnan(f["TIME_SINCE_PREVIOUS_EVENT"][2])
