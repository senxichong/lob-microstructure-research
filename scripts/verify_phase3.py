"""Independent saved-data audit: bisect alignment, SciPy statistics, scalar ranks."""

import argparse
import bisect
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import false_discovery_control, rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lob_research.download import sha256
from lob_research.phase2.artifacts import write_csv
from lob_research.phase3.artifacts import check_frozen, code_hash, load_phase2, read_grid
from lob_research.phase3.grid import epoch_ns
from lob_research.phase3.statistics import assign_bins, spearman
from lob_research.timestamps import iso_utc


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def scalar_ranks(values):
    values = list(values)
    return np.array(
        [1 + sum(b < a for b in values) + (sum(b == a for b in values) - 1) / 2 for a in values]
    )


def scalar_rho(x, y):
    rx, ry = scalar_ranks(x), scalar_ranks(y)
    a, b = rx - rx.mean(), ry - ry.mean()
    return float(sum(a * b) / (sum(a * a) * sum(b * b)) ** 0.5)


def primary_table_line(row):
    def f(key):
        return float(row[key])

    return f"| {row['feature']} | {int(row['n']):,} | {f('spearman_rho'):.6f} | [{f('bootstrap_ci_low'):.6f}, {f('bootstrap_ci_high'):.6f}] | {f('permutation_p'):.6f} | {f('fdr_bh_q'):.6f} | {f('fdr_by_q'):.6f} | {f('q5_minus_q1_mean_return') * 10000:.6f} | {f('q5_minus_q1_direction_edge'):.6f} | {f('mean_return_monotonicity'):.3f} |"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = check_frozen()
    directory = Path(cfg["reports_dir"])
    manifest = json.loads(Path("PHASE3_MANIFEST.json").read_text(encoding="utf-8"))
    checks = []

    def check(name, violation, detail=""):
        count = (
            int(np.count_nonzero(violation))
            if isinstance(violation, np.ndarray)
            else int(violation)
        )
        checks.append({"check": name, "passed": count == 0, "violations": count, "detail": detail})

    def close(name, a, b, detail=""):
        check(
            name,
            ~np.isclose(
                np.asarray(a, dtype=float),
                np.asarray(b, dtype=float),
                atol=1e-12,
                rtol=1e-12,
                equal_nan=True,
            ),
            detail,
        )

    primary = csv_rows(directory / "primary_feature_statistics.csv")

    def report_checks():
        report = Path("PHASE3_REPORT.md").read_text(encoding="utf-8")
        for row in primary:
            check("report_primary_" + row["feature"], primary_table_line(row) not in report)
        check("report_source_hash", cfg["input_sha256"] not in report)
        check("report_grid_count", f"**{manifest['grid_rows']:,} grid points**" not in report)
        check(
            "report_target_count",
            f"**{manifest['valid_primary_targets']:,} valid 1s targets**" not in report,
        )

        def table_row(values):
            return "| " + " | ".join(map(str, values)) + " |"

        horizon_rows = csv_rows(directory / "horizon_decay.csv")
        time_rows = csv_rows(directory / "time_of_day_stability.csv")
        interior_rows = csv_rows(directory / "interior_session_robustness.csv")
        activity_rows = csv_rows(directory / "activity_conditioning.csv")
        spread_rows = csv_rows(directory / "spread_conditioning.csv")
        for name in cfg["features"]:
            expected = table_row(
                [
                    name,
                    *[
                        f"{float(next(r for r in horizon_rows if r['feature'] == name and int(r['horizon_ms']) == h)['spearman_rho']):.6f}"
                        for h in (100, 1000, 5000, 10000)
                    ],
                ]
            )
            check("report_horizons_" + name, expected not in report)
            expected = table_row(
                [
                    name,
                    *[
                        f"{float(next(r for r in time_rows if r['feature'] == name and r['block'] == b)['spearman_rho']):.6f}"
                        for b in "ABCDE"
                    ],
                    f"{float(next(r for r in interior_rows if r['feature'] == name and r['sample'] == 'interior')['spearman_rho']):.6f}",
                ]
            )
            check("report_time_interior_" + name, expected not in report)
            if name in cfg["directional_features"]:
                expected = table_row(
                    [
                        name,
                        *[
                            f"{float(next(r for r in rows if r['feature'] == name and r['regime'] == regime)['spearman_rho']):.6f}"
                            for rows in (activity_rows, spread_rows)
                            for regime in ("low", "medium", "high")
                        ],
                    ]
                )
                check("report_regimes_" + name, expected not in report)
        for row in csv_rows(directory / "event_level_secondary.csv"):
            expected = table_row(
                [
                    row["feature"],
                    row["n"],
                    f"{float(row['spearman_rho']):.6f}",
                    row["grid_n"],
                    f"{float(row['grid_rho']):.6f}",
                ]
            )
            check("report_event_" + row["feature"], expected not in report)
        for row in csv_rows(directory / "placebo_results.csv"):
            expected = table_row(
                [
                    row["feature"],
                    f"{float(row['aligned_rho_same_pairs']):.6f}",
                    f"{float(row['placebo_rho']):.6f}",
                    row["absolute_weakened"],
                ]
            )
            check("report_placebo_" + row["feature"], expected not in report)
        for row in csv_rows(directory / "hypothesis_results.csv"):
            check(
                "report_hypothesis_" + row["hypothesis"],
                table_row([row["hypothesis"], row["verdict"], row["evidence"]]) not in report,
            )
        for row in csv_rows(directory / "family_verdicts.csv"):
            check(
                "report_family_" + row["family"],
                table_row([row["family"], row["verdict"]]) not in report,
            )
        control = csv_rows(directory / "random_control.csv")[0]
        check("report_random_rho", f"rho {float(control['spearman_rho']):.9f}" not in report)
        check("report_random_p", f"p={float(control['permutation_p']):.9f}" not in report)

    check(
        "frozen_phase2_and_upstream",
        0,
        "all 88 frozen baseline files checked, except documented README/gitignore append permission",
    )
    amendment = json.loads((directory / "console_fix_audit.json").read_text(encoding="utf-8"))
    check("current_production_code", code_hash() != amendment["current_code_sha256"])
    check(
        "archived_execution_runner",
        sha256(amendment["original_runner_path"]) != amendment["original_runner_sha256"],
    )
    h = hashlib.sha256()
    for path in sorted(Path("src/lob_research/phase3").glob("*.py")) + [
        Path("scripts/run_phase3.py")
    ]:
        content = (
            Path(amendment["original_runner_path"]).read_bytes()
            if path.name == "run_phase3.py"
            else path.read_bytes()
        )
        h.update(path.as_posix().encode() + b"\0" + content)
    check("frozen_execution_code", h.hexdigest() != manifest["code_sha256"])
    for path, digest in amendment["unchanged_analytic_modules"].items():
        check("unchanged_analytic_" + Path(path).name, sha256(path) != digest)
    original = Path(amendment["original_runner_path"]).read_text(encoding="utf-8")
    amended = Path("scripts/run_phase3.py").read_text(encoding="utf-8")
    check(
        "console_only_change",
        amended.replace("            default=lambda value: value.item(),\n", "") != original,
    )
    check("phase3_grid_hash", sha256(manifest["output_path"]) != manifest["output_sha256"])
    for path, digest in manifest["csv_sha256"].items():
        check("csv_hash_" + Path(path).name, sha256(path) != digest)
    check(
        "primary_horizon",
        cfg["primary_horizon_ms"] != 1000 or manifest["primary_horizon_ms"] != 1000,
    )
    check(
        "frozen_feature_set",
        manifest["features"] != cfg["features"]
        or [r["feature"] for r in primary] != cfg["features"],
    )
    status = json.loads((directory / "run_status.json").read_text(encoding="utf-8"))
    check(
        "run_status",
        status["status"] != "complete"
        or status["manifest_sha256"] != sha256("PHASE3_MANIFEST.json"),
    )
    if args.report_only:
        prior = json.loads(
            (directory / "independent_verification.json").read_text(encoding="utf-8")
        )
        check("core_verifier_unchanged", prior["verifier_sha256"] != sha256(__file__))
        check("core_audit_passed", prior["critical_failures"] != 0)
        old = csv_rows(directory / "invariant_checks.csv")
        fresh_names = {r["check"] for r in checks}
        checks[:0] = [
            {
                "check": r["check"],
                "passed": r["passed"] == "True",
                "violations": int(r["violations"]),
                "detail": r["detail"],
            }
            for r in old
            if r["check"] not in fresh_names and not r["check"].startswith("report_")
        ]
        report_checks()
        result = {
            **prior,
            "status": "PASS" if all(r["passed"] for r in checks) else "FAIL",
            "check_count": len(checks),
            "critical_failures": sum(not r["passed"] for r in checks),
            "report_verified": True,
            "report_sha256": sha256("PHASE3_REPORT.md"),
        }
        write_csv(directory / "invariant_checks.csv", checks)
        (directory / "independent_verification.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 1

    s = load_phase2(cfg)
    g = read_grid(manifest["output_path"])
    t = s["timestamp"].tolist()
    start, end = epoch_ns(cfg["session_start_utc"]), epoch_ns(cfg["session_end_utc"])
    clock = np.arange(start, end, 1000000000, dtype=np.int64)
    n = len(clock)
    check(
        "grid_row_count", n != 23400 or len(g["grid_timestamp"]) != n or manifest["grid_rows"] != n
    )
    check(
        "grid_unique_regular",
        not np.array_equal(g["grid_timestamp"], clock) or len(set(clock)) != n,
    )
    source_index = np.array([bisect.bisect_right(t, int(v)) - 1 for v in clock])
    safe = np.maximum(source_index, 0)
    current = (source_index >= 0) & s["book_valid"][safe]
    check("grid_source_asof", g["source_event_index"] != np.where(current, source_index, -1))
    check(
        "grid_source_order",
        g["source_state_order"] != np.where(current, s["source_order"][safe], -1),
    )
    check(
        "grid_source_time",
        g["source_state_timestamp"] != np.where(current, s["timestamp"][safe], 0),
    )
    check("grid_availability", g["grid_state_available"] != current)
    close(
        "grid_age",
        g["grid_state_age_ms"],
        np.where(current, (clock - s["timestamp"][safe]) / 1000000, np.nan),
    )
    check("grid_age_nonnegative", g["grid_state_age_ms"] < 0)
    for name in cfg["features"] + [
        "midprice",
        "best_bid_price",
        "best_ask_price",
        "best_bid_size",
        "best_ask_size",
        "spread",
    ]:
        close("copied_source_" + name, g[name], np.where(current, s[name][safe], np.nan))
    bad = np.r_[0, np.cumsum(~s["book_valid"])]
    sums = s["mid_sum_price_units"]
    expected_targets = {}
    for h, tag in [(100, "100MS"), (1000, "1S"), (5000, "5S"), (10000, "10S")]:
        tau = clock + h * 1000000
        j = np.array([bisect.bisect_right(t, int(v)) - 1 for v in tau])
        js = np.maximum(j, 0)
        uninterrupted = bad[js + 1] == bad[safe]
        good = current & (tau < end) & uninterrupted & (j >= source_index)
        ret = np.where(good, sums[js] / sums[safe] - 1, np.nan)
        expected_targets[tag] = ret
        check("target_available_" + tag, g["target_available_" + tag] != good)
        check("requested_target_" + tag, g["target_timestamp_" + tag] != tau)
        check(
            "resolved_target_index_" + tag, g["target_event_index_" + tag] != np.where(good, j, -1)
        )
        check(
            "resolved_target_order_" + tag,
            g["target_state_order_" + tag] != np.where(good, s["source_order"][js], -1),
        )
        check(
            "resolved_target_timestamp_" + tag,
            g["target_state_timestamp_" + tag] != np.where(good, s["timestamp"][js], 0),
        )
        close("target_return_" + tag, g["GRID_FWD_MID_RET_" + tag], ret)
        close(
            "target_move_" + tag,
            g["GRID_MID_MOVE_" + tag],
            np.where(good, (sums[js] - sums[safe]) / 20000, np.nan),
        )
        close(
            "target_age_" + tag,
            g["target_age_ms_" + tag],
            np.where(good, (tau - s["timestamp"][js]) / 1000000, np.nan),
        )
        direction = np.where(
            sums[js] > sums[safe], "UP", np.where(sums[js] < sums[safe], "DOWN", "FLAT")
        ).astype(object)
        direction[~good] = None
        check("target_direction_" + tag, g["GRID_DIRECTION_" + tag] != direction)
    check(
        "valid_primary_target_count",
        np.isfinite(expected_targets["1S"]).sum() != manifest["valid_primary_targets"],
    )
    y = expected_targets["1S"]
    names = cfg["features"]
    allnames = names + ["RANDOM_CONTROL"]
    close(
        "random_seed_generation",
        g["RANDOM_CONTROL"],
        np.random.default_rng(cfg["random_feature_seed"]).standard_normal(n),
    )
    qtable = csv_rows(directory / "primary_feature_quintiles.csv")
    edge_table = csv_rows(directory / "quintile_edges.csv")
    edges = {
        r["feature"]: np.array([float(r[f"edge_{k}"]) for k in range(1, 5)]) for r in edge_table
    }
    quintile_assignments = {}
    for row in primary:
        name = row["feature"]
        x = g[name]
        mask = np.isfinite(x) & np.isfinite(y)
        result = spearmanr(x[mask], y[mask]).statistic
        close("primary_scipy_rho_" + name, result, float(row["spearman_rho"]))
        check("primary_n_" + name, int(row["n"]) != mask.sum())
        close(
            "primary_pearson_" + name, np.corrcoef(x[mask], y[mask])[0, 1], float(row["pearson_r"])
        )
        # Independently interpolate sorted values for linear empirical quantiles.
        ordered = sorted(x[mask].tolist())
        cuts = []
        for fraction in (0.2, 0.4, 0.6, 0.8):
            pos = (len(ordered) - 1) * fraction
            k = int(pos)
            weight = pos - k
            cuts.append(ordered[k] * (1 - weight) + ordered[min(k + 1, len(ordered) - 1)] * weight)
        close("quintile_edges_" + name, cuts, edges[name])
        bins = np.array(
            [bisect.bisect_left(cuts, float(v)) + 1 if np.isfinite(v) else 0 for v in x]
        )
        quintile_assignments[name] = bins
        rows = [r for r in qtable if r["feature"] == name]
        check("quintile_total_" + name, sum(int(r["n"]) for r in rows) != int(row["n"]))
        for r in rows:
            q = int(r["quintile"])
            take = mask & (bins == q)
            count = int(take.sum())
            check(f"quintile_n_{name}_{q}", count != int(r["n"]))
            close(
                f"quintile_mean_{name}_{q}",
                y[take].mean() if count else np.nan,
                float(r["mean_return"]),
            )
            close(
                f"quintile_median_{name}_{q}",
                np.median(y[take]) if count else np.nan,
                float(r["median_return"]),
            )
            close(
                f"quintile_edge_{name}_{q}",
                np.sign(g["GRID_MID_MOVE_1S"][take]).mean() if count else np.nan,
                float(r["direction_edge"]),
            )
            check(
                f"quintile_classes_{name}_{q}",
                sum(int(r[k + "_count"]) for k in ["up", "flat", "down"]) != count,
            )
    perm = csv_rows(directory / "permutation_results.csv")
    draws = csv_rows(directory / "permutation_draws.csv")
    offsets = np.random.default_rng(cfg["permutation_seed"]).integers(
        600, n - 600 + 1, size=cfg["permutation_replicates"]
    )
    for name in allnames:
        rows = [r for r in draws if r["feature"] == name]
        check(
            "permutation_seed_offsets_" + name,
            not np.array_equal(np.array([int(r["offset_seconds"]) for r in rows]), offsets),
        )
        observed = spearmanr(
            g[name][np.isfinite(g[name]) & np.isfinite(y)], y[np.isfinite(g[name]) & np.isfinite(y)]
        ).statistic
        null = np.array([float(r["spearman_rho"]) for r in rows])
        # Fixed spread of 21 real null draws independently recalculated for every feature.
        for index in np.linspace(0, len(rows) - 1, 21).astype(int):
            shifted = np.concatenate((y[-offsets[index] :], y[: -offsets[index]]))
            mask = np.isfinite(g[name]) & np.isfinite(shifted)
            close(
                f"null_draw_{name}_{index}",
                spearmanr(g[name][mask], shifted[mask]).statistic,
                null[index],
            )
        pr = next(r for r in perm if r["feature"] == name)
        close(
            "permutation_empirical_p_" + name,
            (1 + np.sum(abs(null) >= abs(observed))) / (len(null) + 1),
            float(pr["empirical_p"]),
        )
    multiple = csv_rows(directory / "multiple_testing.csv")
    pvals = np.array([float(r["permutation_p"]) for r in primary])
    for method, key in [("bh", "fdr_bh_q"), ("by", "fdr_by_q")]:
        close(
            "fdr_scipy_" + method,
            false_discovery_control(pvals, method=method),
            np.array([float(r[key]) for r in primary]),
        )
        close(
            "fdr_table_" + method,
            np.array([float(r[method + "_q"]) for r in multiple]),
            np.array([float(r[key]) for r in primary]),
        )
    boots = csv_rows(directory / "bootstrap_draws.csv")
    intervals = csv_rows(directory / "bootstrap_intervals.csv")
    for length in [60, 30, 300]:
        rng = np.random.default_rng(cfg["bootstrap_seed"] + length)
        selected_indices = {}
        for r in range(cfg["bootstrap_replicates"]):
            starts = rng.integers(0, n, size=(n + length - 1) // length)
            if r in (0, 499, 999):
                selected_indices[r] = np.concatenate(
                    [np.arange(k, k + length) % n for k in starts]
                )[:n]
        for name in allnames:
            rows = [r for r in boots if r["feature"] == name and int(r["block_seconds"]) == length]
            check(f"bootstrap_count_{length}_{name}", len(rows) != 1000)
            for replicate, indices in selected_indices.items():
                xsample, ysample = g[name][indices], y[indices]
                good = np.isfinite(xsample) & np.isfinite(ysample)
                expected = spearmanr(xsample[good], ysample[good]).statistic
                close(
                    f"bootstrap_literal_{length}_{name}_{replicate}",
                    expected,
                    float(rows[replicate]["spearman_rho"]),
                )
                bins = np.array([bisect.bisect_left(edges[name], v) + 1 for v in xsample[good]])
                means = [
                    ysample[good][bins == q].mean() if np.any(bins == q) else np.nan for q in (1, 5)
                ]
                close(
                    f"bootstrap_qspread_{length}_{name}_{replicate}",
                    means[1] - means[0],
                    float(rows[replicate]["q5_minus_q1_mean_return"]),
                )
            for metric in ["spearman_rho", "q5_minus_q1_mean_return"]:
                values = np.array([float(r[metric]) for r in rows])
                finite = values[np.isfinite(values)]
                record = next(
                    r
                    for r in intervals
                    if r["feature"] == name
                    and int(r["block_seconds"]) == length
                    and r["metric"] == metric
                )
                close(
                    f"bootstrap_interval_{length}_{name}_{metric}",
                    np.quantile(finite, [0.025, 0.975]),
                    [float(record["ci_low"]), float(record["ci_high"])],
                )
                check(
                    f"bootstrap_seed_{length}_{name}_{metric}",
                    int(record["seed"]) != cfg["bootstrap_seed"] + length,
                )
    for r in csv_rows(directory / "time_of_day_stability.csv"):
        k = ord(r["block"]) - 65
        left = start + k * (end - start) // 5
        right = start + (k + 1) * (end - start) // 5
        mask = (clock >= left) & (clock < right) & np.isfinite(g[r["feature"]]) & np.isfinite(y)
        check(
            "tod_n_" + r["feature"] + r["block"],
            int(r["n"]) != mask.sum() or int(r["grid_points"]) != 4680,
        )
        close(
            "tod_rho_" + r["feature"] + r["block"],
            spearmanr(g[r["feature"]][mask], y[mask]).statistic,
            float(r["spearman_rho"]),
        )
    for r in csv_rows(directory / "event_level_secondary.csv"):
        x = s[r["feature"]]
        target = s["FWD_MID_RET_1S"]
        mask = np.isfinite(x) & np.isfinite(target)
        close(
            "event_descriptive_" + r["feature"],
            spearmanr(x[mask], target[mask]).statistic,
            float(r["spearman_rho"]),
        )
        check(
            "event_grid_separation_" + r["feature"],
            int(r["n"]) == int(r["grid_n"]) or "DESCRIPTIVE" not in r["status"],
        )
    for r in csv_rows(directory / "placebo_results.csv"):
        x = g[r["feature"]]
        lag = np.r_[np.full(600, np.nan), x[:-600]]
        mask = np.isfinite(x) & np.isfinite(lag) & np.isfinite(y)
        close(
            "placebo_" + r["feature"],
            spearmanr(lag[mask], y[mask]).statistic,
            float(r["placebo_rho"]),
        )
        close(
            "placebo_same_pairs_" + r["feature"],
            spearmanr(x[mask], y[mask]).statistic,
            float(r["aligned_rho_same_pairs"]),
        )
    for r in csv_rows(directory / "sign_inversion.csv"):
        check(
            "sign_inversion_" + r["feature"],
            abs(float(r["rho"]) + float(r["inverted_rho"])) > 1e-12,
        )
    control = csv_rows(directory / "random_control.csv")[0]
    control_pass = float(control["permutation_p"]) > 0.05 and float(
        control["bootstrap_ci_low"]
    ) <= 0 <= float(control["bootstrap_ci_high"])
    check(
        "random_control",
        not control_pass,
        "frozen seed; no redraw; p>.05 and 60s interval includes zero",
    )

    comparisons = []
    manual_stats = []
    for name in names:
        valid = np.flatnonzero(np.isfinite(g[name]) & np.isfinite(y))
        selected = valid[np.linspace(0, len(valid) - 1, 101).astype(int)]
        expected_x = []
        expected_y = []
        for i in selected:
            j = bisect.bisect_right(t, int(clock[i])) - 1
            k = bisect.bisect_right(t, int(clock[i]) + 1000000000) - 1
            expected_x.append(float(s[name][j]))
            expected_y.append(float(sums[k] / sums[j] - 1))
            for field, expected, actual in [
                ("source_index", j, g["source_event_index"][i]),
                ("source_order", s["source_order"][j], g["source_state_order"][i]),
                ("feature_value", s[name][j], g[name][i]),
                ("target_source_order", s["source_order"][k], g["target_state_order_1S"][i]),
                ("future_return", expected_y[-1], y[i]),
                (
                    "quintile",
                    bisect.bisect_left(edges[name], float(s[name][j])) + 1,
                    assign_bins(g[name][i : i + 1], edges[name])[0],
                ),
            ]:
                matched = bool(np.isclose(expected, actual, atol=1e-12, rtol=1e-12))
                comparisons.append(
                    {
                        "feature": name,
                        "grid_index": int(i),
                        "field": field,
                        "independent": expected,
                        "production": actual,
                        "passed": matched,
                    }
                )
        rx, ry = scalar_ranks(expected_x), scalar_ranks(expected_y)
        for rank_name, expected, actual in [
            ("feature_rank", rx, rankdata(g[name][selected])),
            ("target_rank", ry, rankdata(y[selected])),
        ]:
            for i, a, b in zip(selected, expected, actual):
                comparisons.append(
                    {
                        "feature": name,
                        "grid_index": int(i),
                        "field": rank_name,
                        "independent": a,
                        "production": b,
                        "passed": a == b,
                    }
                )
        expected = scalar_rho(expected_x, expected_y)
        actual = spearman(g[name][selected], y[selected])
        close("independent_small_sample_rho_" + name, expected, actual)
        manual_stats.append(
            {
                "feature": name,
                "sample_count": len(selected),
                "independent_scalar_rho": expected,
                "production_rho": actual,
                "passed": bool(np.isclose(expected, actual, atol=1e-12, rtol=1e-12)),
            }
        )
    write_csv(directory / "manual_verification.csv", comparisons)
    write_csv(directory / "manual_statistics.csv", manual_stats)
    check("manual_comparisons", sum(not row["passed"] for row in comparisons))

    valid = np.flatnonzero(
        np.isfinite(y) & np.all(np.column_stack([np.isfinite(g[name]) for name in names]), axis=1)
    )
    selected = set(valid[np.linspace(0, len(valid) - 1, 20).astype(int)].tolist())
    for name, condition in [
        ("OBI_L1", lambda a: a > 0.6),
        ("OBI_L1", lambda a: a < -0.6),
        ("OFI_1S", lambda a: a > 0),
        ("OFI_1S", lambda a: a < 0),
        ("MICROPRICE_MINUS_MID", lambda a: a > 0),
        ("MICROPRICE_MINUS_MID", lambda a: a < 0),
    ]:
        matches = valid[condition(g[name][valid])]
        if len(matches):
            selected.add(int(matches[0]))
    selected.add(int(valid[np.argmin(abs(g["OBI_L1"][valid]))]))
    selected.update(
        (
            int(valid[np.argmax(g["EVENT_RATE_1S"][valid])]),
            int(valid[np.argmin(g["EVENT_RATE_1S"][valid])]),
        )
    )
    for direction in ["UP", "FLAT", "DOWN"]:
        matches = valid[g["GRID_DIRECTION_1S"][valid] == direction]
        if len(matches):
            selected.add(int(matches[0]))
    traces = [
        "# Phase 3 representative grid traces",
        "",
        "MSFT — single historical trading day — 2003-01-03. Local-only real rows. Deterministic time coverage plus predeclared feature/context/outcome examples, never profitable-trade selection.",
        "",
    ]
    for i in sorted(selected):
        j = int(g["source_event_index"][i])
        k = int(g["target_event_index_1S"][i])
        traces.extend(
            [
                f"## Grid row {i}: {iso_utc(int(clock[i]))}",
                "",
                f"Source key ({iso_utc(int(s['timestamp'][j]))}, {s['source_order'][j]}), age {g['grid_state_age_ms'][i]:g}ms. BBO {g['best_bid_price'][i]:.4f} x {g['best_bid_size'][i]:g} / {g['best_ask_price'][i]:.4f} x {g['best_ask_size'][i]:g}; current mid {g['midprice'][i]:.5f}.",
                "",
                "| Feature | Frozen source-state value |",
                "|---|---:|",
                *[f"| {name} | {g[name][i]:.12g} |" for name in names],
                "",
                f"Requested 1s target {iso_utc(int(clock[i]) + 1000000000)}; last as-of key ({iso_utc(int(s['timestamp'][k]))}, {s['source_order'][k]}), age {g['target_age_ms_1S'][i]:g}ms; future mid {sums[k] / 20000:.5f}; direction **{g['GRID_DIRECTION_1S'][i]}**; return {y[i]:.12g}.",
                "",
            ]
        )
    Path("reports/phase3_signal_traces.md").write_text("\n".join(traces), encoding="utf-8")
    check("trace_count", len(selected) < 20, str(len(selected)))
    model_files = []
    for directory_name, folders, files in os.walk("."):
        folders[:] = [
            name
            for name in folders
            if name not in {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
        ]
        model_files.extend(
            str(Path(directory_name) / name)
            for name in files
            if Path(name).suffix.lower()
            in {
                ".pkl",
                ".pickle",
                ".joblib",
                ".pt",
                ".pth",
                ".onnx",
                ".h5",
                ".keras",
                ".cbm",
                ".ubj",
                ".model",
            }
        )
    check("no_model_artifacts", len(model_files), repr(model_files))
    check(
        "no_model_or_execution_runs",
        manifest["model_fits"] != 0 or manifest["execution_simulations"] != 0,
    )
    if Path("PHASE3_REPORT.md").exists():
        report_checks()
    failed = [row for row in checks if not row["passed"]]
    result = {
        "status": ("PASS" if Path("PHASE3_REPORT.md").exists() else "CORE_PASS")
        if not failed
        else "FAIL",
        "check_count": len(checks),
        "critical_failures": len(failed),
        "failed_checks": failed,
        "violation_count": sum(row["violations"] for row in checks),
        "manual_feature_analyses": len(manual_stats),
        "manual_comparisons": len(comparisons),
        "manual_failures": sum(not row["passed"] for row in comparisons),
        "trace_count": len(selected),
        "random_control_passed": control_pass,
        "source_sha256": cfg["input_sha256"],
        "verifier_sha256": sha256(__file__),
        "runtime_seconds": time.perf_counter() - started,
        "report_verified": Path("PHASE3_REPORT.md").exists(),
    }
    write_csv(directory / "invariant_checks.csv", checks)
    (directory / "independent_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
