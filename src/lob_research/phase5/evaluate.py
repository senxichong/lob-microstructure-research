import json

import numpy as np

from lob_research.phase3.inference import WeightedSpearman, block_indices
from lob_research.phase3.statistics import pearson, spearman
from lob_research.phase4.core import (
    CLASSES,
    classification_metrics,
    cm_metrics,
    confusion,
    regression_metrics,
)
from lob_research.phase4.storage import prediction_groups

from .storage import OUT, csvwrite, groups


def combined(task):
    result = groups(task)
    names = (
        ["Ridge", "OBI_L1", "MICROPRICE_MINUS_MID", "OFI_1S"]
        if task == "regression"
        else ["Logistic", "AlwaysFLAT"]
    )
    frozen = prediction_groups(task)
    anchor = result["LightGBM"]
    for n in names:
        result[n] = frozen[(0, n)]
    for name, d in result.items():
        np.testing.assert_array_equal(d["grid_timestamp"], anchor["grid_timestamp"], err_msg=name)
        np.testing.assert_array_equal(d["target"], anchor["target"], err_msg=name)
        if task == "classification":
            np.testing.assert_array_equal(d["true_class"], anchor["true_class"], err_msg=name)
    return result


def probabilities(d):
    return np.column_stack([d[k] for k in ["p_down", "p_flat", "p_up"]])


def metrics():
    allrows = {}
    foldrows = {}
    conf = []
    calibration = []
    for task in ["regression", "classification"]:
        result = combined(task)
        rows = []
        folds = []
        for name, d in result.items():
            metric = regression_metrics if task == "regression" else classification_metrics
            y = d["target"] if task == "regression" else d["true_class"]
            p = d["prediction"] if task == "regression" else probabilities(d)
            rows.append(dict(model_name=name, **metric(y, p)))
            for f in range(1, 6):
                mask = d["fold_id"] == f
                folds.append(dict(model_name=name, fold_id=f, **metric(y[mask], p[mask])))
            if task == "classification":
                for f in range(6):
                    mask = d["fold_id"] == f if f else np.ones(len(y), bool)
                    cm = confusion(y[mask], d["predicted_class"][mask])
                    for i, c in enumerate(CLASSES):
                        for j, pc in enumerate(CLASSES):
                            conf.append(
                                dict(
                                    model_name=name,
                                    fold_id=f,
                                    true_class=c,
                                    predicted_class=pc,
                                    count=int(cm[i, j]),
                                )
                            )
                for category in ["DOWN", "UP"]:
                    pp = d["p_" + category.lower()]
                    bins = np.minimum((pp * 10).astype(int), 9)
                    for b in range(10):
                        mask = bins == b
                        calibration.append(
                            dict(
                                model_name=name,
                                class_name=category,
                                lower=b / 10,
                                upper=(b + 1) / 10,
                                n=int(mask.sum()),
                                mean_probability=float(pp[mask].mean()) if mask.any() else np.nan,
                                observed_frequency=float(np.mean(y[mask] == category))
                                if mask.any()
                                else np.nan,
                            )
                        )
        csvwrite(task + "_metrics", rows)
        csvwrite(task + "_fold_metrics", folds)
        allrows[task] = rows
        foldrows[task] = folds
    csvwrite("confusion_matrix", conf)
    csvwrite("calibration", calibration)
    correlations = []
    reg = combined("regression")
    for a in ["Ridge", "RF", "LightGBM"]:
        for b in ["Ridge", "RF", "LightGBM"]:
            correlations.append(
                dict(
                    model_a=a,
                    model_b=b,
                    spearman=spearman(reg[a]["prediction"], reg[b]["prediction"]),
                    pearson=pearson(reg[a]["prediction"], reg[b]["prediction"]),
                )
            )
    csvwrite("prediction_correlation", correlations)
    gap = []
    for a in json.loads((OUT / "model_metadata.json").read_text()):
        task = a["task"]
        field = "spearman" if task == "regression" else "macro_f1"
        test = next(
            r[field]
            for r in foldrows[task]
            if r["model_name"] == a["model_name"] and r["fold_id"] == a["fold_id"]
        )
        gap.append(
            dict(
                task=task,
                model_name=a["model_name"],
                fold_id=a["fold_id"],
                selected_family=a["family"],
                train_metric=a["train_score"],
                validation_metric=a["validation_score"],
                test_metric=test,
                train_minus_test=a["train_score"] - test,
            )
        )
    csvwrite("train_validation_test_gap", gap)
    return allrows, foldrows


def bootstrap(progress=print):
    reg = combined("regression")
    cla = combined("classification")
    anchor = reg["Ridge"]
    start = anchor["test_start"].min()
    n = int((anchor["test_end"].max() - start) // 10**9)
    pos = ((anchor["grid_timestamp"] - start) // 10**9).astype(int)
    y = np.full(n, np.nan)
    y[pos] = anchor["target"]
    refs = {}
    for name, d in reg.items():
        x = np.full(n, np.nan)
        x[pos] = d["prediction"]
        refs[name] = WeightedSpearman(x, y)
    rng = np.random.default_rng(505060)
    rd = {name: np.empty(1000) for name in reg}
    cd = {name: np.empty(1000) for name in cla}
    for r in range(1000):
        w = np.bincount(block_indices(n, 60, rng), minlength=n)
        for name, ref in refs.items():
            rd[name][r] = ref.evaluate(w)
        for name, d in cla.items():
            cd[name][r] = cm_metrics(confusion(d["true_class"], d["predicted_class"], w[pos]))[0][
                "macro_f1"
            ]
        if (r + 1) % 250 == 0:
            progress(f"Paired temporal bootstrap {r + 1}/1000")
    csvwrite(
        "bootstrap_draws",
        [
            dict(
                replicate=i,
                **{"reg_" + k: float(v[i]) for k, v in rd.items()},
                **{"class_" + k: float(v[i]) for k, v in cd.items()},
            )
            for i in range(1000)
        ],
    )
    outputs = {}
    for task, data, draws, baseline in [
        ("regression", reg, rd, "Ridge"),
        ("classification", cla, cd, "Logistic"),
    ]:
        observed = {
            k: (
                spearman(d["prediction"], d["target"])
                if task == "regression"
                else classification_metrics(d["true_class"], probabilities(d))["macro_f1"]
            )
            for k, d in data.items()
        }
        rows = []
        for name in ["RF", "LightGBM", "ValidationSelected"]:
            delta = draws[name] - draws[baseline]
            lo, hi = np.quantile(delta, [0.025, 0.975])
            rlo, rhi = np.quantile(draws[name], [0.025, 0.975])
            wins = 0
            d = data[name]
            b = data[baseline]
            for fold in range(1, 6):
                mask = d["fold_id"] == fold
                if task == "regression":
                    a_value = spearman(d["prediction"][mask], d["target"][mask])
                    b_value = spearman(b["prediction"][mask], b["target"][mask])
                else:
                    a_value = classification_metrics(d["true_class"][mask], probabilities(d)[mask])[
                        "macro_f1"
                    ]
                    b_value = classification_metrics(b["true_class"][mask], probabilities(b)[mask])[
                        "macro_f1"
                    ]
                wins += int(a_value > b_value)
            rows.append(
                dict(
                    model_name=name,
                    benchmark=baseline,
                    metric=observed[name],
                    benchmark_metric=observed[baseline],
                    observed_delta=observed[name] - observed[baseline],
                    bootstrap_mean_delta=float(delta.mean()),
                    ci_lower=float(lo),
                    ci_upper=float(hi),
                    model_ci_lower=float(rlo),
                    model_ci_upper=float(rhi),
                    fold_wins=wins,
                    block_seconds=60,
                    replicates=1000,
                )
            )
        csvwrite(
            "nonlinear_vs_ridge_bootstrap"
            if task == "regression"
            else "nonlinear_vs_logistic_bootstrap",
            rows,
        )
        outputs[task] = rows
    actual = spearman(reg["LightGBM"]["prediction"], reg["LightGBM"]["target"])
    for name, filename in [
        ("RandomLabel", "random_label_control"),
        ("TemporalPlacebo", "temporal_placebo"),
        ("RandomFeature", "random_feature_control"),
    ]:
        observed = spearman(reg[name]["prediction"], reg[name]["target"])
        lo, hi = np.quantile(rd[name], [0.025, 0.975])
        delta = rd[name] - rd["LightGBM"]
        dl, dh = np.quantile(delta, [0.025, 0.975])
        row = dict(
            model_name=name,
            spearman=observed,
            ci_lower=float(lo),
            ci_upper=float(hi),
            real_lightgbm=actual,
            delta=observed - actual,
            delta_ci_lower=float(dl),
            delta_ci_upper=float(dh),
        )
        if name != "RandomFeature":
            row["investigation_required"] = bool(
                lo > 0 or (observed > 0 and observed >= max(0.01, 0.8 * actual))
            )
        else:
            row["material_test_gain"] = bool(dl > 0 and observed - actual >= 0.01)
        csvwrite(filename, [row])
    rows = []
    for name in ["Static", "Flow", "LightGBM"]:
        obs = spearman(reg[name]["prediction"], reg[name]["target"])
        lo, hi = np.quantile(rd[name], [0.025, 0.975])
        delta = rd["LightGBM"] - rd[name]
        dl, dh = np.quantile(delta, [0.025, 0.975])
        rows.append(
            dict(
                model_name=name,
                spearman=obs,
                ci_lower=float(lo),
                ci_upper=float(hi),
                full_minus_model=actual - obs,
                difference_ci_lower=float(dl),
                difference_ci_upper=float(dh),
            )
        )
    csvwrite("static_flow_full", rows)
    return outputs
