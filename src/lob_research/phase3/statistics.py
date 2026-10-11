"""Transparent univariate statistics, tie-preserving bins, and FDR."""

import numpy as np
from scipy.stats import rankdata


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    a, b = x[good], y[good]
    if len(a) < 3:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denominator) if denominator > 0 else np.nan


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    return pearson(rankdata(x[good], method="average"), rankdata(y[good], method="average"))


def association(x, y):
    return {
        "n": int(np.sum(np.isfinite(x) & np.isfinite(y))),
        "spearman_rho": spearman(x, y),
        "pearson_r": pearson(x, y),
    }


def empirical_edges(x, groups=5):
    x = np.asarray(x, float)
    finite = x[np.isfinite(x)]
    if not len(finite):
        raise ValueError("no finite bin population")
    return np.quantile(finite, np.arange(1, groups) / groups, method="linear")


def assign_bins(x, edges):
    x = np.asarray(x, float)
    return np.where(np.isfinite(x), np.searchsorted(edges, x, side="left") + 1, 0)


def bin_diagnostics(x, y, direction, move, edges):
    bins = assign_bins(x, edges)
    good = np.isfinite(x) & np.isfinite(y)
    rows = []
    for q in range(1, len(edges) + 2):
        selected = good & (bins == q)
        n = int(selected.sum())
        counts = {
            label: int(np.sum(direction[selected] == label)) for label in ["UP", "FLAT", "DOWN"]
        }
        rows.append(
            {
                "quintile": q,
                "n": n,
                "mean_return": float(y[selected].mean()) if n else np.nan,
                "median_return": float(np.median(y[selected])) if n else np.nan,
                "mean_move_usd": float(move[selected].mean()) if n else np.nan,
                **{label.lower() + "_count": c for label, c in counts.items()},
                **{"p_" + label.lower(): c / n if n else np.nan for label, c in counts.items()},
                "direction_edge": (counts["UP"] - counts["DOWN"]) / n if n else np.nan,
            }
        )
    return rows


def monotonicity(rows, key):
    if any(row["n"] == 0 for row in rows):
        return np.nan
    return spearman(np.arange(1, len(rows) + 1), np.array([row[key] for row in rows]))


def fdr(pvalues, by=False):
    p = np.asarray(pvalues, float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("FDR requires finite p-values in [0,1]")
    order = np.argsort(p, kind="stable")
    m = len(p)
    factor = np.sum(1 / np.arange(1, m + 1)) if by else 1
    adjusted = np.minimum.accumulate((p[order] * m * factor / np.arange(1, m + 1))[::-1])[::-1]
    result = np.empty(m)
    result[order] = np.minimum(adjusted, 1)
    return result


def random_feature(n, seed):
    return np.random.default_rng(seed).standard_normal(n)


def lag_feature(x, lag):
    if lag <= 0 or lag >= len(x):
        raise ValueError("positive interior lag required")
    result = np.full(len(x), np.nan)
    result[lag:] = x[:-lag]
    return result
