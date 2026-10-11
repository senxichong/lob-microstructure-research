"""Predeclared primary analysis followed by bounded secondary diagnostics."""

import numpy as np

from lob_research.phase2.common import suffix

from .grid import time_blocks
from .inference import bootstrap, circular_offsets, empirical_p, permutation
from .statistics import (
    assign_bins,
    association,
    bin_diagnostics,
    empirical_edges,
    fdr,
    lag_feature,
    monotonicity,
    random_feature,
    spearman,
)


def run_study(grid, source, cfg, start, end, progress=print):
    names = cfg["features"]
    directional = cfg["directional_features"]
    y = grid["GRID_FWD_MID_RET_1S"]
    n = len(y)
    grid["RANDOM_CONTROL"] = random_feature(n, cfg["random_feature_seed"])
    features = {name: grid[name] for name in [*names, "RANDOM_CONTROL"]}
    edges = {
        name: empirical_edges(x[np.isfinite(x) & np.isfinite(y)]) for name, x in features.items()
    }
    tables = {}
    primary = []
    quintiles = []
    directions = []
    qrows = {}
    for name in names:
        x = features[name]
        rows = bin_diagnostics(
            x, y, grid["GRID_DIRECTION_1S"], grid["GRID_MID_MOVE_1S"], edges[name]
        )
        qrows[name] = rows
        quintiles.extend({"feature": name, **row} for row in rows)
        primary.append(
            {
                "feature": name,
                **association(x, y),
                "q5_minus_q1_mean_return": rows[-1]["mean_return"] - rows[0]["mean_return"],
                "q5_minus_q1_mean_move_usd": rows[-1]["mean_move_usd"] - rows[0]["mean_move_usd"],
                "q5_minus_q1_direction_edge": rows[-1]["direction_edge"]
                - rows[0]["direction_edge"],
                "mean_return_monotonicity": monotonicity(rows, "mean_return"),
                "direction_edge_monotonicity": monotonicity(rows, "direction_edge"),
                "nonempty_quintiles": sum(row["n"] > 0 for row in rows),
            }
        )
        for direction in ["UP", "FLAT", "DOWN"]:
            a = x[(grid["GRID_DIRECTION_1S"] == direction) & np.isfinite(x) & np.isfinite(y)]
            q = np.quantile(a, [0.05, 0.25, 0.5, 0.75, 0.95]) if len(a) else [np.nan] * 5
            directions.append(
                {
                    "feature": name,
                    "direction": direction,
                    "n": len(a),
                    **dict(zip(["p05", "p25", "median", "p75", "p95"], q)),
                    "iqr": q[3] - q[1],
                }
            )
    intervals = []
    boot_draws = []
    for length in [
        cfg["bootstrap_primary_block_seconds"],
        *cfg["bootstrap_robustness_block_seconds"],
    ]:
        seed = cfg["bootstrap_seed"] + length
        draws = bootstrap(features, y, edges, length, cfg["bootstrap_replicates"], seed, progress)
        for name, a in draws.items():
            for metric, col in [("spearman_rho", 0), ("q5_minus_q1_mean_return", 1)]:
                finite = a[np.isfinite(a[:, col]), col]
                lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
                intervals.append(
                    {
                        "feature": name,
                        "block_seconds": length,
                        "primary": length == cfg["bootstrap_primary_block_seconds"],
                        "metric": metric,
                        "ci_low": lo,
                        "ci_high": hi,
                        "finite_replicates": len(finite),
                        "replicates": cfg["bootstrap_replicates"],
                        "seed": seed,
                        "method": "paired circular clock blocks; percentile; exact re-ranking",
                    }
                )
            boot_draws.extend(
                {
                    "feature": name,
                    "block_seconds": length,
                    "replicate": i,
                    "spearman_rho": a[i, 0],
                    "q5_minus_q1_mean_return": a[i, 1],
                }
                for i in range(len(a))
            )
    shifts = circular_offsets(
        n,
        cfg["minimum_circular_shift_seconds"],
        cfg["permutation_replicates"],
        cfg["permutation_seed"],
    )
    null = permutation(features, y, shifts, progress)
    permutations = []
    permutation_draws = []
    for name, x in features.items():
        observed = spearman(x, y)
        a = null[name]
        permutations.append(
            {
                "feature": name,
                "observed_rho": observed,
                "empirical_p": empirical_p(observed, a),
                "null_mean": float(a.mean()),
                "null_p025": float(np.quantile(a, 0.025)),
                "null_p975": float(np.quantile(a, 0.975)),
                "replicates": len(shifts),
                "seed": cfg["permutation_seed"],
                "minimum_shift_seconds": cfg["minimum_circular_shift_seconds"],
                "null": "approximate circular target-shift; full clock and missingness retained",
            }
        )
        permutation_draws.extend(
            {
                "feature": name,
                "replicate": i,
                "offset_seconds": int(shifts[i]),
                "spearman_rho": a[i],
            }
            for i in range(len(a))
        )
    raw = np.array([row["empirical_p"] for row in permutations[: len(names)]])
    bh, by = fdr(raw), fdr(raw, by=True)
    multiple = []
    for i, row in enumerate(primary):
        ci = next(
            r
            for r in intervals
            if r["feature"] == row["feature"] and r["primary"] and r["metric"] == "spearman_rho"
        )
        row.update(
            bootstrap_ci_low=ci["ci_low"],
            bootstrap_ci_high=ci["ci_high"],
            permutation_p=raw[i],
            fdr_bh_q=bh[i],
            fdr_by_q=by[i],
        )
        multiple.append(
            {
                "feature": row["feature"],
                "empirical_p": raw[i],
                "bh_q": bh[i],
                "by_q": by[i],
                "family_size": len(names),
                "bh_reject_005": bh[i] <= cfg["alpha"],
                "by_reject_005": by[i] <= cfg["alpha"],
            }
        )
    random_perm = permutations[-1]
    random_ci = next(
        r
        for r in intervals
        if r["feature"] == "RANDOM_CONTROL" and r["primary"] and r["metric"] == "spearman_rho"
    )
    random_pass = (
        random_perm["empirical_p"] > 0.05 and random_ci["ci_low"] <= 0 <= random_ci["ci_high"]
    )
    tables.update(
        primary_feature_statistics=primary,
        primary_feature_quintiles=quintiles,
        primary_direction_statistics=directions,
        bootstrap_intervals=intervals,
        bootstrap_draws=boot_draws,
        permutation_results=permutations,
        permutation_draws=permutation_draws,
        multiple_testing=multiple,
        quintile_edges=[
            {"feature": name, **{f"edge_{i + 1}": v for i, v in enumerate(e)}}
            for name, e in edges.items()
        ],
        random_control=[
            {
                "feature": "RANDOM_CONTROL",
                "seed": cfg["random_feature_seed"],
                **association(features["RANDOM_CONTROL"], y),
                "bootstrap_ci_low": random_ci["ci_low"],
                "bootstrap_ci_high": random_ci["ci_high"],
                "permutation_p": random_perm["empirical_p"],
                "passed": random_pass,
                "redraws": 0,
            }
        ],
    )
    progress(
        "Primary 1s associations, bootstrap and permutation complete. Starting predefined secondary diagnostics."
    )

    # All following estimates are descriptive, with no secondary significance search.
    horizon = []
    horizon_quintiles = []
    for h in [100, 1000, 5000, 10000]:
        tag = suffix(h)
        target = grid[f"GRID_FWD_MID_RET_{tag}"]
        for name in names:
            rows = bin_diagnostics(
                features[name],
                target,
                grid[f"GRID_DIRECTION_{tag}"],
                grid[f"GRID_MID_MOVE_{tag}"],
                edges[name],
            )
            horizon.append(
                {
                    "feature": name,
                    "horizon_ms": h,
                    **association(features[name], target),
                    "q5_minus_q1_direction_edge": rows[-1]["direction_edge"]
                    - rows[0]["direction_edge"],
                    "q5_minus_q1_mean_return": rows[-1]["mean_return"] - rows[0]["mean_return"],
                }
            )
            horizon_quintiles.extend({"feature": name, "horizon_ms": h, **row} for row in rows)
    blocks = time_blocks(grid["grid_timestamp"], start, end, cfg["time_blocks"])
    tod = []
    interior = []
    placebo = []
    signs = []
    interior_mask = (
        grid["grid_timestamp"] >= start + cfg["interior_exclusion_seconds"] * 1000000000
    ) & (grid["grid_timestamp"] < end - cfg["interior_exclusion_seconds"] * 1000000000)
    for name, x in features.items():
        if name == "RANDOM_CONTROL":
            continue
        for b in range(cfg["time_blocks"]):
            mask = blocks == b
            r = association(x[mask], y[mask])
            tod.append(
                {
                    "feature": name,
                    "block": chr(65 + b),
                    "grid_points": int(mask.sum()),
                    "start_offset_seconds": b * (end - start) // cfg["time_blocks"] // 1000000000,
                    "end_offset_seconds": (b + 1)
                    * (end - start)
                    // cfg["time_blocks"]
                    // 1000000000,
                    **r,
                    "sign": int(np.sign(r["spearman_rho"]))
                    if np.isfinite(r["spearman_rho"])
                    else 0,
                }
            )
        for scope, mask in [("full", np.ones(n, bool)), ("interior", interior_mask)]:
            interior.append({"feature": name, "sample": scope, **association(x[mask], y[mask])})
        lag = lag_feature(x, cfg["placebo_lag_seconds"])
        mask = np.isfinite(x) & np.isfinite(lag) & np.isfinite(y)
        aligned = spearman(x[mask], y[mask])
        shifted = spearman(lag[mask], y[mask])
        placebo.append(
            {
                "feature": name,
                "lag_seconds": cfg["placebo_lag_seconds"],
                "n_same_pairs": int(mask.sum()),
                "aligned_rho_same_pairs": aligned,
                "placebo_rho": shifted,
                "absolute_weakened": abs(shifted) < abs(aligned),
            }
        )
        if name in directional:
            rho = spearman(x, y)
            inverse = spearman(-x, y)
            signs.append(
                {
                    "feature": name,
                    "rho": rho,
                    "inverted_rho": inverse,
                    "sum_error": rho + inverse,
                    "passed": abs(rho + inverse) <= 1e-12,
                }
            )
    conditioning_edges = []
    for regime, column in [("activity", "EVENT_RATE_1S"), ("spread", "RELATIVE_SPREAD")]:
        values = grid[column]
        e = empirical_edges(values[np.isfinite(y) & np.isfinite(values)], 3)
        levels = assign_bins(values, e)
        rows = []
        conditioning_edges.append(
            {"regime": regime, "feature": column, "edge_1": e[0], "edge_2": e[1]}
        )
        for name in directional:
            for k, label in enumerate(["low", "medium", "high"], 1):
                mask = levels == k
                rows.append(
                    {
                        "feature": name,
                        "regime": label,
                        "analysis": "EX-POST DESCRIPTIVE REGIME ANALYSIS",
                        **association(features[name][mask], y[mask]),
                    }
                )
        tables[regime + "_conditioning"] = rows
    correlations = []
    for a in names:
        for b in names:
            correlations.append(
                {"feature_a": a, "feature_b": b, **association(features[a], features[b])}
            )
    event = []
    for row in primary:
        name = row["feature"]
        event.append(
            {
                "feature": name,
                **association(source[name], source["FWD_MID_RET_1S"]),
                "grid_rho": row["spearman_rho"],
                "grid_n": row["n"],
                "status": "HIGH-FREQUENCY DESCRIPTIVE ESTIMATE; no event-level inference",
            }
        )
    extreme = []
    for name in ["OBI_L1", "OBI_L5", "OBI_L10", "WOBI_L10"]:
        x = features[name]
        level = np.select(
            [x < -0.6, x < -0.2, x <= 0.2, x <= 0.6, x <= 1], [1, 2, 3, 4, 5], default=0
        )
        for k, label in enumerate(["[-1,-.6)", "[-.6,-.2)", "[-.2,.2]", "(.2,.6]", "(.6,1]"], 1):
            good = (level == k) & np.isfinite(y)
            extreme.append(
                {
                    "feature": name,
                    "fixed_bin": label,
                    "n": int(good.sum()),
                    "mean_return": float(y[good].mean()) if good.any() else np.nan,
                    "direction_edge": float(np.mean(np.sign(grid["GRID_MID_MOVE_1S"][good])))
                    if good.any()
                    else np.nan,
                }
            )
    tables.update(
        horizon_decay=horizon,
        horizon_quintiles=horizon_quintiles,
        time_of_day_stability=tod,
        interior_session_robustness=interior,
        placebo_results=placebo,
        sign_inversion=signs,
        conditioning_edges=conditioning_edges,
        feature_correlation=correlations,
        event_level_secondary=event,
        extreme_imbalance=extreme,
    )
    return tables
