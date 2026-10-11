import numpy as np
import pytest
from phase2_helpers import settings, stream

from lob_research.phase2.features import build_features
from lob_research.phase2.labels import build_labels


def mutate_after(s, k):
    s = s.copy()
    s.bid[k + 1 :] += 10000
    s.ask[k + 1 :] += 20000
    s.bsize[k + 1 :] *= 3
    s.asize[k + 1 :] *= 2
    s.kind[k + 1 :] = "P"
    s.side[k + 1 :] = "S"
    s.size[k + 1 :] *= 7
    return s


def test_future_mutation_every_feature_and_settled_label():
    s = stream(np.arange(0, 15001, 100), end_ms=30000)
    k = 100
    changed = mutate_after(s, k)
    a, _ = build_features(s, settings())
    b, _ = build_features(changed, settings())
    for name in a:
        np.testing.assert_array_equal(a[name][: k + 1], b[name][: k + 1], err_msg=name)
    la, _ = build_labels(s, settings())
    lb, _ = build_labels(changed, settings())
    for tag in ("100ms", "1s", "5s", "10s"):
        settled = la[f"target_time_requested_{tag}"] <= s.time[k]
        for name in la:
            if name.lower().endswith("_" + tag):
                va, vb = la[name], lb[name]
                if isinstance(va, tuple):
                    for x, y in zip(va, vb):
                        np.testing.assert_array_equal(x[settled], y[settled])
                else:
                    np.testing.assert_array_equal(va[settled], vb[settled])


def test_same_timestamp_future_mutation_and_prefix_equivalence():
    s = stream((0, 123, 123, 200, 1000, 1200), end_ms=20000)
    assert s.order[1] == 100 and s.order[2] == 101
    a, _ = build_features(s, settings())
    b, _ = build_features(mutate_after(s, 1), settings())
    prefix, _ = build_features(s.take(slice(0, 2)), settings())
    for name in a:
        np.testing.assert_array_equal(a[name][:2], b[name][:2], err_msg=name)
        np.testing.assert_array_equal(a[name][:2], prefix[name], err_msg=name)
    assert a["EVENT_COUNT_100MS"][1] == 1 and a["EVENT_COUNT_100MS"][2] == 2


def test_unsettled_target_tie_distinction():
    s = stream((0, 100, 100, 200))
    a, _ = build_labels(s, settings())
    b, _ = build_labels(mutate_after(s, 1), settings())
    assert a["FWD_MID_RET_100MS"][0] != b["FWD_MID_RET_100MS"][0]
    # Target ends at end-of-tie key, after cutoff (100 ms, source order 100).
    assert a["target_state_source_order_100ms"][0][0] > s.order[1]


@pytest.mark.parametrize("indices", [[0, 2, 1, 3], [1, 0, 2, 3], [0, 1, 1, 3]])
def test_row_order_rejected(indices):
    s = stream((0, 100, 100, 200)).take(indices)
    with pytest.raises(ValueError, match="unordered"):
        build_features(s, settings())
    with pytest.raises(ValueError, match="unordered"):
        build_labels(s, settings())


def test_label_availability_cannot_change_features():
    s = stream((0, 100, 1000), end_ms=2000)
    other = s.copy()
    other.end = 20000000000
    a, _ = build_features(s, settings())
    b, _ = build_features(other, settings())
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    la, _ = build_labels(s, settings())
    lb, _ = build_labels(other, settings())
    assert not np.array_equal(la["target_available_10s"], lb["target_available_10s"])
