import numpy as np
import pytest
from scipy.stats import false_discovery_control, rankdata, spearmanr

from lob_research.phase3.grid import build_grid, time_blocks
from lob_research.phase3.inference import (
    WeightedSpearman,
    block_indices,
    bootstrap,
    circular_offsets,
    empirical_p,
    permutation,
)
from lob_research.phase3.statistics import (
    assign_bins,
    empirical_edges,
    fdr,
    lag_feature,
    random_feature,
    spearman,
)


def source(times=(0, 900, 1100), mids=(100, 100.01, 100.03)):
    n = len(times)
    s = {
        "timestamp": np.array(times, dtype=np.int64) * 1000000,
        "source_order": np.arange(100, 100 + n),
        "event_index": np.arange(n),
        "book_valid": np.ones(n, bool),
        "mid_sum_price_units": np.rint(np.array(mids) * 20000).astype(np.int64),
        "midprice": np.array(mids),
        "X": np.arange(n, dtype=float),
        "FWD_MID_RET_1S": np.full(n, 999.0),
    }
    for name in ["best_bid_price", "best_ask_price", "best_bid_size", "best_ask_size", "spread"]:
        s[name] = np.ones(n)
    return s


def test_phase3_hand_grid_target_last_before():
    g = build_grid(source(), ["X"], 0, 3000000000)
    assert list(g["grid_timestamp"]) == [0, 1000000000, 2000000000]
    assert g["target_state_order_1S"][0] == 101
    assert g["GRID_FWD_MID_RET_1S"][0] == pytest.approx(0.0001)
    assert g["source_state_order"][1] == 101 and g["X"][1] == 1
    assert g["grid_state_age_ms"][1] == 100


def test_phase3_no_event_flat_and_boundary():
    g = build_grid(source((0, 2000), (100, 105)), ["X"], 0, 3000000000)
    assert g["target_available_1S"][0] and g["GRID_DIRECTION_1S"][0] == "FLAT"
    assert not g["target_available_1S"][2]
    assert g["target_reason_1S"][2] == "session_boundary"


def test_phase3_same_timestamp_last_source_and_future_feature_mutation():
    s = source((0, 1000, 1000, 1100), (100, 101, 102, 103))
    g = build_grid(s, ["X"], 0, 3000000000)
    assert g["source_state_order"][1] == 102
    assert g["target_state_order_1S"][0] == 102
    s["X"][-1] = 9999
    changed = build_grid(s, ["X"], 0, 3000000000)
    np.testing.assert_array_equal(g["X"][:2], changed["X"][:2])


def test_phase3_invalid_interval_not_skipped():
    s = source((0, 400, 500, 1000), (100, 101, 102, 103))
    s["book_valid"][1] = False
    g = build_grid(s, ["X"], 0, 3000000000)
    assert not g["target_available_1S"][0]
    assert g["target_reason_1S"][0] == "invalid_book_interval"
    assert g["target_available_1S"][1]
    s["book_valid"][-1] = False
    g = build_grid(s, ["X"], 0, 3000000000)
    assert not g["grid_state_available"][1] and np.isnan(g["X"][1])


def test_phase3_initial_missing_grid_and_no_cross_session():
    s = source((60, 900, 1100))
    g = build_grid(s, ["X"], 0, 3000000000)
    assert not g["grid_state_available"][0]
    assert g["source_event_index"][0] == -1 and np.isnan(g["grid_state_age_ms"][0])
    with pytest.raises(ValueError, match="cross-session"):
        build_grid(s, ["X"], 100000000, 3000000000)


@pytest.mark.parametrize("key", ["timestamp", "source_order"])
def test_phase3_unordered_rejected(key):
    s = source()
    s[key][1] = s[key][2] + 1
    with pytest.raises(ValueError, match="source order"):
        build_grid(s, ["X"], 0, 3000000000)


def test_phase3_grid_target_is_not_source_event_target():
    s = source((0, 900, 1100, 1900, 2100), (100, 101, 103, 104, 109))
    g = build_grid(s, ["X"], 0, 4000000000)
    assert g["GRID_FWD_MID_RET_1S"][1] == pytest.approx(104 / 101 - 1)
    assert g["GRID_FWD_MID_RET_1S"][1] != s["FWD_MID_RET_1S"][1]
    assert len(g["X"]) != len(s["X"])


def test_phase3_quintiles_ties_lower_and_empty_bins():
    x = np.array([0.0, 0, 0, 0, 1, 2, 3, 4, 5, 6])
    edges = empirical_edges(x)
    q = assign_bins(x, edges)
    assert len(q) == 10 and np.all(q[:4] == 1)
    assert assign_bins(np.array([edges[0], np.nan]), edges).tolist() == [1, 0]
    assert np.all(assign_bins(np.ones(10), empirical_edges(np.ones(10))) == 1)


def test_phase3_spearman_hand_ties_and_missing():
    x = np.array([1.0, 2, 2, 4, np.nan])
    y = np.array([4.0, 1, 2, 3, 99])
    expected = np.corrcoef([1, 2.5, 2.5, 4], [4, 1, 2, 3])[0, 1]
    assert spearman(x, y) == pytest.approx(expected)
    assert np.isnan(spearman(np.ones(5), np.arange(5)))


@pytest.mark.parametrize("length", [3, 10, 30])
def test_phase3_clock_blocks_preserve_adjacency_and_reproducibility(length):
    a = block_indices(100, length, np.random.default_rng(14))
    b = block_indices(100, length, np.random.default_rng(14))
    np.testing.assert_array_equal(a, b)
    for start in range(0, 100, length):
        assert np.all(np.diff(a[start : start + length]) % 100 == 1)
    assert len(a) == 100


def test_phase3_weighted_reranking_equals_literal_bootstrap():
    x = np.array([0.0, 1, 1, 3, 4, np.nan, 6, 7, 8, 8])
    y = np.array([1.0, 0, 0, 2, 4, 9, np.nan, 0, 1, 3])
    ref = WeightedSpearman(x, y)
    rng = np.random.default_rng(42)
    for _ in range(30):
        indices = block_indices(len(x), 3, rng)
        actual = ref.evaluate(np.bincount(indices, minlength=len(x)))
        expected = spearman(x[indices], y[indices])
        assert (
            actual == pytest.approx(expected, abs=1e-12) or np.isnan(actual) and np.isnan(expected)
        )


def test_phase3_bootstrap_fixed_seed_and_fixed_bin_spread():
    x = np.arange(60, dtype=float)
    y = x / 100
    edges = {"X": empirical_edges(x)}
    a = bootstrap({"X": x}, y, edges, 10, 10, 123)
    b = bootstrap({"X": x}, y, edges, 10, 10, 123)
    np.testing.assert_array_equal(a["X"], b["X"])
    np.testing.assert_allclose(a["X"][:, 0], 1)


def test_phase3_circular_offsets_exclusion_missingness_and_null_p():
    shifts = circular_offsets(100, 20, 30, 123)
    assert np.all((shifts >= 20) & (shifts <= 80))
    np.testing.assert_array_equal(shifts, circular_offsets(100, 20, 30, 123))
    x = np.arange(100, dtype=float)
    y = x.copy()
    y[0] = np.nan
    draws = permutation({"X": x}, y, shifts)
    for j, k in enumerate(shifts):
        assert draws["X"][j] == pytest.approx(spearman(x, np.roll(y, k)))
        assert np.isnan(np.roll(y, k)).sum() == 1
    assert empirical_p(0.8, np.array([0.1, 0.9, -0.8])) == 0.75
    assert empirical_p(1, np.array([0.1, 0.2])) == pytest.approx(1 / 3)


def test_phase3_fdr_known_values_and_scipy():
    p = np.array([0.01, 0.02, 0.03, 0.2])
    np.testing.assert_allclose(fdr(p), [0.04, 0.04, 0.04, 0.2])
    np.testing.assert_allclose(fdr(p, by=True), false_discovery_control(p, method="by"))
    with pytest.raises(ValueError):
        fdr([np.nan])


def test_phase3_time_blocks_complete_and_half_open():
    t = np.arange(23400, dtype=np.int64) * 1000000000
    blocks = time_blocks(t, 0, 23400 * 1000000000)
    np.testing.assert_array_equal(np.bincount(blocks), np.full(5, 4680))
    assert blocks[4679] == 0 and blocks[4680] == 1
    with pytest.raises(ValueError):
        time_blocks(np.array([23400]), 0, 23400)


def test_phase3_random_control_single_seed_no_market_input():
    x = random_feature(100, 123)
    np.testing.assert_array_equal(x, random_feature(100, 123))
    np.testing.assert_allclose(
        x[:3], [-0.9891213503478509, -0.3677866514678832, 1.2879252612892487]
    )


def test_phase3_lag_no_wrap_same_pair_mask_and_sign_inversion():
    x = np.array([1.0, 2, 2, 4, 0, 3])
    y = np.array([2.0, 3, 1, 4, 5, 0])
    lag = lag_feature(x, 2)
    assert np.isnan(lag[:2]).all()
    np.testing.assert_array_equal(lag[2:], x[:-2])
    assert spearman(-x, y) == pytest.approx(-spearman(x, y), abs=1e-12)
    np.testing.assert_allclose(rankdata(-x), len(x) + 1 - rankdata(x))
    assert spearman(x, y) == pytest.approx(spearmanr(x, y).statistic)
