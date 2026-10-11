import numpy as np

from lob_research.phase2.common import Stream


def settings(tick=None):
    return {
        "primary_horizon_ms": 1000,
        "secondary_horizons_ms": [100, 5000, 10000],
        "history_windows_ms": [100, 1000, 5000, 10000],
        "book_depths": [1, 3, 5, 10],
        "tick_size_usd": tick,
        "weighting": "reciprocal_rank",
    }


def stream(times=(0, 50, 90, 130, 1200), mids=None, end_ms=20000):
    n = len(times)
    mid = np.array(mids if mids is not None else [100] * n)
    b = np.rint(mid * 10000 - 100).astype(float)
    a = np.rint(mid * 10000 + 100).astype(float)
    return Stream(
        np.array(times, dtype=np.int64) * 1000000,
        np.arange(99, 99 + n, dtype=np.int64),
        b[:, None] - np.arange(10) * 100,
        a[:, None] + np.arange(10) * 100,
        np.full((n, 10), 1500.0),
        np.full((n, 10), 500.0),
        np.array(["A"] * n),
        np.array(["B"] * n),
        np.full(n, 100.0),
        np.ones(n, bool),
        0,
        end_ms * 1000000,
        "2003-01-03",
        "TEST",
    )
