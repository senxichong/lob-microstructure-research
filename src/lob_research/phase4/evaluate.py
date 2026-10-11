"""All reported metrics start by reloading persisted OOS predictions."""

import numpy as np

from lob_research.phase3.inference import (
    WeightedSpearman,
    block_indices,
    circular_offsets,
    empirical_p,
)
from lob_research.phase3.statistics import spearman

from .core import CLASSES, classification_metrics, cm_metrics, confusion, regression_metrics
from .storage import dump, prediction_groups, write_csv


def evaluate():
    groups = prediction_groups("regression")
    classes = prediction_groups("classification")
    aggregate, byfold, caggregate, cfold, matrices = [], [], [], [], []
    for (embargo, name), d in groups.items():
        tag = dict(model_name=name, embargo_seconds=embargo)
        aggregate.append(dict(**tag, **regression_metrics(d["target"], d["prediction"])))
        for fold in np.unique(d["fold_id"]):
            m = d["fold_id"] == fold
            byfold.append(
                dict(
                    **tag,
                    fold_id=int(fold),
                    **regression_metrics(d["target"][m], d["prediction"][m]),
                )
            )
    for (embargo, name), d in classes.items():
        p = np.column_stack([d[k] for k in ("p_down", "p_flat", "p_up")])
        tag = dict(model_name=name, embargo_seconds=embargo)
        caggregate.append(dict(**tag, **classification_metrics(d["true_class"], p)))
        for fold in np.unique(d["fold_id"]):
            m = d["fold_id"] == fold
            cfold.append(
                dict(**tag, fold_id=int(fold), **classification_metrics(d["true_class"][m], p[m]))
            )
        for fold in [0, *np.unique(d["fold_id"])]:
            mask = np.ones(len(p), bool) if fold == 0 else d["fold_id"] == fold
            cm = confusion(d["true_class"][mask], d["predicted_class"][mask])
            for i, c in enumerate(CLASSES):
                for j, pc in enumerate(CLASSES):
                    matrices.append(
                        dict(
                            **tag,
                            fold_id=int(fold),
                            true_class=c,
                            predicted_class=pc,
                            count=int(cm[i, j]),
                        )
                    )
    for name, rows in [
        ("regression_metrics", aggregate),
        ("regression_fold_metrics", byfold),
        ("classification_metrics", caggregate),
        ("classification_fold_metrics", cfold),
        ("confusion_matrix", matrices),
        (
            "feature_ablation",
            [
                r
                for r in aggregate
                if r["embargo_seconds"] == 0
                and (r["model_name"].startswith("Without") or r["model_name"] == "Ridge")
            ],
        ),
        (
            "static_vs_flow",
            [
                r
                for r in aggregate
                if r["embargo_seconds"] == 0 and r["model_name"] in ("Static", "Flow", "Ridge")
            ],
        ),
        (
            "time_of_day_model_stability",
            [dict(task="regression", **r) for r in byfold]
            + [dict(task="classification", **r) for r in cfold],
        ),
    ]:
        write_csv(name, rows)
    d = classes[(0, "Logistic")]
    calibration, extreme = [], []
    for category in ("DOWN", "UP"):
        p = d["p_" + category.lower()]
        bins = np.minimum((p * 10).astype(int), 9)
        for b in range(10):
            m = bins == b
            calibration.append(
                dict(
                    class_name=category,
                    lower=b / 10,
                    upper=(b + 1) / 10,
                    n=int(m.sum()),
                    mean_probability=float(p[m].mean()) if m.any() else np.nan,
                    observed_frequency=float(np.mean(d["true_class"][m] == category))
                    if m.any()
                    else np.nan,
                )
            )
        cutoff = float(np.quantile(p, 0.9))
        m = p >= cutoff
        extreme.append(
            dict(
                class_name=category,
                quantile=0.9,
                cutoff=cutoff,
                n=int(m.sum()),
                mean_probability=float(p[m].mean()),
                **{c + "_frequency": float(np.mean(d["true_class"][m] == c)) for c in CLASSES},
            )
        )
    write_csv("calibration", calibration)
    write_csv("extreme_probability", extreme)
    return aggregate, byfold, caggregate, cfold


def inference(progress=print):
    groups = prediction_groups("regression")
    cls = prediction_groups("classification")
    names = [
        "OLS",
        "Ridge",
        "Static",
        "Flow",
        "OBI_L1",
        "MICROPRICE_MINUS_MID",
        "OFI_1S",
        "RandomLabel",
        "RandomFeature",
    ]
    anchor = groups[(0, "Ridge")]
    start, end = anchor["test_start"].min(), anchor["test_end"].max()
    n = int((end - start) // 10**9)
    pos = ((anchor["grid_timestamp"] - start) // 10**9).astype(int)
    y = np.full(n, np.nan)
    y[pos] = anchor["target"]
    xs = {}
    for name in names:
        d = groups[(0, name)]
        assert np.array_equal(d["grid_timestamp"], anchor["grid_timestamp"])
        np.testing.assert_array_equal(d["target"], anchor["target"])
        x = np.full(n, np.nan)
        x[pos] = d["prediction"]
        xs[name] = x
    weighted = {name: WeightedSpearman(x, y) for name, x in xs.items()}
    rng = np.random.default_rng(404060)
    draws = {name: np.empty(1000) for name in names}
    f1_delta = np.empty(1000)
    lc, bc = cls[(0, "Logistic")], cls[(0, "AlwaysFLAT")]
    for r in range(1000):
        w = np.bincount(block_indices(n, 60, rng), minlength=n)
        for name, ref in weighted.items():
            draws[name][r] = ref.evaluate(w)
        f1_delta[r] = (
            cm_metrics(confusion(lc["true_class"], lc["predicted_class"], w[pos]))[0]["macro_f1"]
            - cm_metrics(confusion(bc["true_class"], bc["predicted_class"], w[pos]))[0]["macro_f1"]
        )
        if (r + 1) % 250 == 0:
            progress(f"Paired bootstrap {r + 1}/1000")
    offsets = circular_offsets(n, 600, 1000, 404600)
    null = {name: np.empty(1000) for name in names}
    for r, offset in enumerate(offsets):
        shifted = np.roll(y, int(offset))
        for name, x in xs.items():
            null[name][r] = spearman(x, shifted)
        if (r + 1) % 250 == 0:
            progress(f"Circular shifts {r + 1}/1000")
    rows = []
    observed = {name: spearman(x, y) for name, x in xs.items()}
    for name in names:
        low, high = np.quantile(draws[name], [0.025, 0.975])
        rows.append(
            dict(
                model_name=name,
                n=len(pos),
                clock_seconds=n,
                block_seconds=60,
                replicates=1000,
                spearman=observed[name],
                ci_lower=float(low),
                ci_upper=float(high),
                shift_p=empirical_p(observed[name], null[name]),
            )
        )
    write_csv("regression_inference", rows)
    write_csv(
        "bootstrap_draws",
        [
            dict(
                replicate=i,
                **{k: float(v[i]) for k, v in draws.items()},
                logistic_minus_flat_f1=float(f1_delta[i]),
            )
            for i in range(1000)
        ],
    )
    write_csv(
        "shift_draws",
        [
            dict(
                replicate=i,
                offset_seconds=int(offsets[i]),
                **{k: float(v[i]) for k, v in null.items()},
            )
            for i in range(1000)
        ],
    )
    singles = ["OBI_L1", "MICROPRICE_MINUS_MID", "OFI_1S"]
    best = max(singles, key=lambda x: observed[x])
    comparisons = []
    for name in singles + ["BEST_SINGLE", "Static", "RandomLabel", "RandomFeature"]:
        base = np.max([draws[k] for k in singles], axis=0) if name == "BEST_SINGLE" else draws[name]
        obs = observed[best] if name == "BEST_SINGLE" else observed[name]
        delta = draws["Ridge"] - base
        lo, hi = np.quantile(delta, [0.025, 0.975])
        comparisons.append(
            dict(
                comparison="Ridge minus " + name,
                baseline_name=best if name == "BEST_SINGLE" else name,
                baseline_spearman=obs,
                ridge_spearman=observed["Ridge"],
                difference=observed["Ridge"] - obs,
                ci_lower=float(lo),
                ci_upper=float(hi),
                paired_replicates=1000,
            )
        )
    write_csv("single_feature_comparison", comparisons)
    f1obs = (
        classification_metrics(
            lc["true_class"], np.column_stack([lc[k] for k in ("p_down", "p_flat", "p_up")])
        )["macro_f1"]
        - classification_metrics(
            bc["true_class"], np.column_stack([bc[k] for k in ("p_down", "p_flat", "p_up")])
        )["macro_f1"]
    )
    dump(
        "classification_inference.json",
        dict(
            difference=f1obs,
            ci_lower=float(np.quantile(f1_delta, 0.025)),
            ci_upper=float(np.quantile(f1_delta, 0.975)),
            block_seconds=60,
            replicates=1000,
        ),
    )
    for control in ("RandomLabel", "RandomFeature"):
        r = next(r for r in rows if r["model_name"] == control)
        pair = next(c for c in comparisons if c["baseline_name"] == control)
        row = dict(
            **r,
            full_ridge_spearman=observed["Ridge"],
            control_minus_full=-pair["difference"],
            difference_ci_lower=-pair["ci_upper"],
            difference_ci_upper=-pair["ci_lower"],
        )
        if control == "RandomLabel":
            row["investigation_required"] = bool(
                (observed[control] > 0 and observed[control] >= 0.8 * observed["Ridge"])
                or r["ci_lower"] > 0
            )
            row["training_label_shift"] = 600
        else:
            row["material_improvement"] = bool(
                row["difference_ci_lower"] > 0 and row["control_minus_full"] >= 0.01
            )
            row["seed"] = 161803
        write_csv(
            "random_label_control" if control == "RandomLabel" else "random_feature_control", [row]
        )
    return rows, comparisons
