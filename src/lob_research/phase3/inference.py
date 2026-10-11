"""Paired clock-block resampling and structure-preserving target shifts."""

import numpy as np

from .statistics import assign_bins, spearman


def block_indices(n, length, rng):
    if length <= 0 or length > n:
        raise ValueError("invalid block length")
    starts = rng.integers(0, n, size=(n + length - 1) // length)
    return ((starts[:, None] + np.arange(length)) % n).ravel()[:n]


class WeightedSpearman:
    """Exact average ranks of repeated observations, not ranks fixed before resampling."""

    def __init__(self, x, y):
        self.good = np.isfinite(x) & np.isfinite(y)
        self.x = np.asarray(x)[self.good]
        self.y = np.asarray(y)[self.good]
        _, self.ix = np.unique(self.x, return_inverse=True)
        _, self.iy = np.unique(self.y, return_inverse=True)

    def evaluate(self, weights):
        w = np.asarray(weights)[self.good]
        total = w.sum()
        if total < 3:
            return np.nan
        cx = np.bincount(self.ix, weights=w)
        cy = np.bincount(self.iy, weights=w)
        rx = (np.cumsum(cx) - cx / 2 + 0.5)[self.ix] - (total + 1) / 2
        ry = (np.cumsum(cy) - cy / 2 + 0.5)[self.iy] - (total + 1) / 2
        denom = np.sqrt(np.sum(w * rx * rx) * np.sum(w * ry * ry))
        return float(np.sum(w * rx * ry) / denom) if denom > 0 else np.nan


def bootstrap(features, y, edges, length, replicates, seed, progress=None):
    n = len(y)
    rng = np.random.default_rng(seed)
    refs = {name: WeightedSpearman(x, y) for name, x in features.items()}
    bins = {name: assign_bins(x, edges[name]) for name, x in features.items()}
    output = {name: np.empty((replicates, 2)) for name in features}
    for r in range(replicates):
        weights = np.bincount(block_indices(n, length, rng), minlength=n)
        for name, ref in refs.items():
            output[name][r, 0] = ref.evaluate(weights)
            good = ref.good
            q = bins[name]
            means = []
            for level in (1, 5):
                mask = good & (q == level)
                w = weights[mask]
                means.append(np.sum(w * y[mask]) / w.sum() if w.sum() else np.nan)
            output[name][r, 1] = means[1] - means[0]
        if progress and (r + 1) % 200 == 0:
            progress(f"Bootstrap {length}s: {r + 1}/{replicates}")
    return output


def circular_offsets(n, minimum, replicates, seed):
    if minimum <= 0 or n < 2 * minimum:
        raise ValueError("no admissible large shifts")
    return np.random.default_rng(seed).integers(minimum, n - minimum + 1, size=replicates)


def permutation(features, y, offsets, progress=None):
    draws = {name: np.empty(len(offsets)) for name in features}
    for r, offset in enumerate(offsets):
        shifted = np.roll(y, int(offset))
        for name, x in features.items():
            draws[name][r] = spearman(x, shifted)
        if progress and (r + 1) % 200 == 0:
            progress(f"Circular shifts: {r + 1}/{len(offsets)}")
    return draws


def empirical_p(observed, null):
    null = np.asarray(null)
    if not np.isfinite(observed) or np.any(~np.isfinite(null)):
        return np.nan
    return float((1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1))
