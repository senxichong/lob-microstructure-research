"""Independent arithmetic audit: no production fitting/evaluation helpers."""

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "reports/phase4"
CLASSES = ["DOWN", "FLAT", "UP"]


def records(name):
    with (OUT / (name + ".csv")).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def arrays(path):
    t = pq.read_table(path)
    return {
        k: t[k].cast(pa.int64()).fill_null(0).to_numpy()
        if pa.types.is_timestamp(t[k].type)
        else t[k].to_numpy()
        for k in t.column_names
    }


def ns(s):
    return int(datetime.fromisoformat(s).timestamp()) * 10**9


def numeric(task, d):
    if task == "regression":
        y, p = d["target"], d["prediction"]
        variable = np.ptp(p) > 0 and np.ptp(y) > 0
        return dict(
            n=len(y),
            spearman=float(spearmanr(p, y).statistic) if variable else np.nan,
            pearson=float(pearsonr(p, y).statistic) if variable else np.nan,
            mae=float(mean_absolute_error(y, p)),
            mse=float(mean_squared_error(y, p)),
            r2=float(r2_score(y, p)),
        )
    y, pred = d["true_class"], d["predicted_class"]
    probabilities = np.column_stack([d["p_down"], d["p_flat"], d["p_up"]])
    assert np.array_equal(np.array(CLASSES)[probabilities.argmax(1)], pred)
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-12)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    q = np.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)
    q /= q.sum(1)[:, None]
    loss = float(-np.mean([np.log(q[i, CLASSES.index(c)]) for i, c in enumerate(y)]))
    m = dict(
        n=len(y),
        macro_f1=float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        accuracy=float(accuracy_score(y, pred)),
        log_loss=loss,
    )
    pr, re, f1, _ = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    for i, c in enumerate(CLASSES):
        m.update(
            {c + "_precision": float(pr[i]), c + "_recall": float(re[i]), c + "_f1": float(f1[i])}
        )
    return m


def main():
    checks = []

    def check(name, truth):
        checks.append(dict(check=name, passed=bool(truth)))
        if not truth:
            raise AssertionError(name)

    def close(name, a, b):
        check(name, np.allclose(a, b, atol=1e-12, rtol=1e-10, equal_nan=True))

    def h(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    frozen = json.loads((OUT / "upstream_freeze.json").read_text())
    for p, value in frozen["files"].items():
        if p == ".gitignore":
            continue
        b = (ROOT / p).read_bytes()
        if p == "RESEARCH_LOG.md":
            b = b[: frozen["log_bytes"]]
        check("upstream " + p, hashlib.sha256(b).hexdigest() == value)
    check(
        "phase2 hash",
        h(ROOT / "data/processed/phase2/msft_20030103_phase2_v2.0.0.parquet")
        == frozen["phase2_hash"],
    )
    grid_path = ROOT / "data/processed/phase3/msft_20030103_grid_v3.0.0.parquet"
    check("phase3 grid hash", h(grid_path) == frozen["grid_hash"])
    prereg = json.loads((OUT / "protocol_freeze.json").read_text())
    for p, value in prereg["documents"].items():
        check("preregistered " + p, h(ROOT / p) == value)
    check(
        "preregistered hypotheses",
        hashlib.sha256((ROOT / "RESEARCH_LOG.md").read_bytes()[: prereg["log_bytes"]]).hexdigest()
        == prereg["log_sha256"],
    )
    execution = json.loads((OUT / "execution_manifest.json").read_text())
    correction = json.loads((OUT / "metric_correction_audit.json").read_text())
    for p, value in execution["code_sha256"].items():
        original = (
            ROOT / correction["archived_original"]
            if p == "src/lob_research/phase4/core.py"
            else ROOT / p
        )
        check("executed code " + p, h(original) == value)
    original = (ROOT / correction["archived_original"]).read_text(encoding="utf-8")
    current = (ROOT / "src/lob_research/phase4/core.py").read_text(encoding="utf-8")
    check(
        "only metric guard changed",
        current
        == original.replace(
            "pearson=pearson(p, y),",
            "pearson=pearson(p, y) if np.ptp(p) > 0 and np.ptp(y) > 0 else np.nan,",
        ),
    )
    check(
        "corrected code digest",
        h(ROOT / "src/lob_research/phase4/core.py") == correction["current_code_sha256"],
    )
    for p, value in correction["after_hashes"].items():
        check("correction audit artifact " + p, h(ROOT / p) == value)
        if p.endswith((".parquet", "fitted_models.json")):
            check("fitted artifacts unchanged " + p, value == correction["before_hashes"][p])
    grid = arrays(grid_path)
    manifest = {
        int(r["fold_id"]): {k: ns(v) for k, v in r.items() if k.endswith(("_start", "_end"))}
        for r in records("fold_manifest")
    }
    fitted = json.loads((OUT / "fitted_models.json").read_text())
    fitmap = {(a["embargo_seconds"], a["model_name"], a["fold_id"]): a for a in fitted}
    gt = grid["grid_timestamp"]
    tau = grid["target_timestamp_1S"]
    for a in fitted:
        f = manifest[a["fold_id"]]
        gap = a["embargo_seconds"] * 10**9
        tr = (
            (gt >= f["train_start"])
            & (gt < f["train_end"])
            & grid["target_available_1S"]
            & (tau < f["train_end"] - gap)
        )
        va = (
            (gt >= f["validation_start"])
            & (gt < f["validation_end"])
            & grid["target_available_1S"]
            & (tau < f["validation_end"] - gap)
        )
        x = np.column_stack([grid[name] for name in a["features"]])
        median = np.nanmedian(x[tr], axis=0)
        z = np.where(np.isfinite(x[tr]), x[tr], median)
        mean = z.mean(0)
        scale = z.std(0)
        scale[scale == 0] = 1
        tag = str((a["embargo_seconds"], a["model_name"], a["fold_id"]))
        close("train-only median " + tag, median, a["median"])
        close("train-only mean " + tag, mean, a["mean"])
        close("train-only scale " + tag, scale, a["scale"])
        candidates = a["validation_scores"]
        best = max(c["score"] for c in candidates)
        ties = [c["parameter"] for c in candidates if c["score"] >= best - 1e-12]
        chosen = min(ties) if a["model_name"] == "Logistic" else max(ties)
        check("validation-only selection " + tag, a["parameter"] == chosen)
        xv = (np.where(np.isfinite(x[va]), x[va], median) - mean) / scale
        beta = np.asarray(a["coefficient"])
        inter = np.asarray(a["intercept"])
        val = xv @ beta.T + inter
        if a["model_name"] == "Logistic":
            score = f1_score(
                grid["GRID_DIRECTION_1S"][va],
                np.array(CLASSES)[val.argmax(1)],
                labels=CLASSES,
                average="macro",
                zero_division=0,
            )
        else:
            score = spearmanr(val, grid["GRID_FWD_MID_RET_1S"][va]).statistic
        close(
            "selected validation score " + tag,
            score,
            next(c["score"] for c in candidates if c["selected"]),
        )
        check("purged training label " + tag, tau[tr].max() < f["validation_start"] - gap)
        check("purged validation label " + tag, tau[va].max() < f["test_start"] - gap)
    all_groups = {}
    recomputed = {}
    for task in ("regression", "classification"):
        data = arrays(OUT / f"{task}_oos_predictions.parquet")
        metrics = records(task + "_metrics")
        foldmetrics = records(task + "_fold_metrics")
        for row in metrics:
            embargo = int(row["embargo_seconds"])
            name = row["model_name"]
            m = (data["embargo_seconds"] == embargo) & (data["model_name"] == name)
            indices = np.flatnonzero(m)
            indices = indices[np.argsort(data["grid_timestamp"][indices])]
            d = {k: v[indices] for k, v in data.items()}
            all_groups[(task, embargo, name)] = d
            calc = numeric(task, d)
            recomputed[(task, embargo, name)] = calc
            for metric, v in calc.items():
                close(f"{task} {embargo} {name} {metric}", float(row[metric]), v)
            check(
                "unique OOS times " + str((task, embargo, name)),
                len(np.unique(d["grid_timestamp"])) == len(indices),
            )
            for fold in range(1, 6):
                f = manifest[fold]
                fm = d["fold_id"] == fold
                sub = {k: v[fm] for k, v in d.items()}
                ts = sub["grid_timestamp"]
                expected = gt[
                    (gt >= f["test_start"])
                    & (gt < f["test_end"])
                    & grid["target_available_1S"]
                    & (tau < f["test_end"])
                ]
                check("OOS only " + str((task, embargo, name, fold)), np.array_equal(ts, expected))
                ix = np.searchsorted(gt, ts)
                close("actual targets", sub["target"], grid["GRID_FWD_MID_RET_1S"][ix])
                close(
                    "source time", sub["source_state_timestamp"], grid["source_state_timestamp"][ix]
                )
                check("causal source state", np.all(sub["source_state_timestamp"] <= ts))
                check("target horizon", np.array_equal(sub["target_timestamp"], ts + 10**9))
                for field, v in f.items():
                    check("partition metadata " + field, np.all(sub[field] == v))
                check("version tags", set(sub["feature_version"]) == {"4.0.0"})
                values = numeric(task, sub)
                fr = next(
                    r
                    for r in foldmetrics
                    if r["model_name"] == name
                    and int(r["embargo_seconds"]) == embargo
                    and int(r["fold_id"]) == fold
                )
                for metric, v in values.items():
                    close(
                        "fold metric " + str((task, embargo, name, fold, metric)),
                        float(fr[metric]),
                        v,
                    )
                artifact = fitmap.get((embargo, name, fold))
                if artifact:
                    a = artifact
                    x = np.column_stack([grid[c][ix] for c in a["features"]])
                    x = (np.where(np.isfinite(x), x, np.array(a["median"])) - a["mean"]) / a[
                        "scale"
                    ]
                    prediction = x @ np.asarray(a["coefficient"]).T + np.asarray(a["intercept"])
                    if task == "classification":
                        prediction = softmax(prediction, axis=1)
                        saved = np.column_stack([sub[k] for k in ("p_down", "p_flat", "p_up")])
                    else:
                        saved = sub["prediction"]
                    close(
                        "coefficient prediction reconstruction " + str((embargo, name, fold)),
                        prediction,
                        saved,
                    )
                    close("hyperparameter row tags", sub["selected_hyperparameter"], a["parameter"])
                if task == "classification":
                    check(
                        "true class labels",
                        np.array_equal(sub["true_class"], grid["GRID_DIRECTION_1S"][ix]),
                    )
                train = (
                    (gt >= f["train_start"])
                    & (gt < f["train_end"])
                    & grid["target_available_1S"]
                    & (tau < f["train_end"] - embargo * 10**9)
                )
                if name == "ZeroReturn":
                    close("zero baseline predictions", sub["prediction"], 0)
                elif name == "TrainingMean":
                    close(
                        "training mean baseline",
                        sub["prediction"],
                        grid["GRID_FWD_MID_RET_1S"][train].mean(),
                    )
                elif name in ("OBI_L1", "MICROPRICE_MINUS_MID", "OFI_1S"):
                    close("raw single feature scores " + name, sub["prediction"], grid[name][ix])
                elif name in ("AlwaysFLAT", "TrainingPrior"):
                    expected_prior = (
                        [0, 1, 0]
                        if name == "AlwaysFLAT"
                        else [np.mean(grid["GRID_DIRECTION_1S"][train] == c) for c in CLASSES]
                    )
                    close(
                        "classification baseline " + name,
                        np.column_stack([sub[k] for k in ["p_down", "p_flat", "p_up"]]),
                        expected_prior,
                    )
                for partition in ("train", "validation", "test"):
                    candidates = (gt >= f[partition + "_start"]) & (gt < f[partition + "_end"])
                    valid = candidates & grid["target_available_1S"]
                    gap = embargo * 10**9 if partition != "test" else 0
                    keep = valid & (tau < f[partition + "_end"] - gap)
                    purged = int((valid & ~keep).sum())
                    check(
                        "prediction purge tag " + partition,
                        np.all(sub[partition + "_purged"] == purged),
                    )
        if task == "classification":
            for row in records("confusion_matrix"):
                d = all_groups[(task, int(row["embargo_seconds"]), row["model_name"])]
                fold = int(row["fold_id"])
                m = np.ones(len(d["true_class"]), bool) if fold == 0 else d["fold_id"] == fold
                cm = confusion_matrix(d["true_class"][m], d["predicted_class"][m], labels=CLASSES)
                check(
                    "confusion cell",
                    int(row["count"])
                    == cm[CLASSES.index(row["true_class"]), CLASSES.index(row["predicted_class"])],
                )
    boots = records("bootstrap_draws")
    shifts = records("shift_draws")
    for row in records("purge_summary"):
        fold = manifest[int(row["fold_id"])]
        partition = row["partition"]
        gap = int(row["embargo_seconds"]) * 10**9 if partition != "test" else 0
        mask = (gt >= fold[partition + "_start"]) & (gt < fold[partition + "_end"])
        available = mask & grid["target_available_1S"]
        retained = available & (tau < fold[partition + "_end"] - gap)
        for field, value in dict(
            clock_rows=mask.sum(),
            unavailable=(mask & ~available).sum(),
            purged=(available & ~retained).sum(),
            retained=retained.sum(),
            max_target_timestamp=tau[retained].max(),
        ).items():
            check("purge summary " + field, int(row[field]) == int(value))
    for row in records("calibration"):
        d = all_groups[("classification", 0, "Logistic")]
        p = d["p_" + row["class_name"].lower()]
        low, high = float(row["lower"]), float(row["upper"])
        mask = (p >= low) & ((p < high) if high < 1 else (p <= high))
        check("calibration bin count", int(row["n"]) == mask.sum())
        close(
            "calibration mean",
            float(row["mean_probability"]),
            p[mask].mean() if mask.any() else np.nan,
        )
        close(
            "calibration frequency",
            float(row["observed_frequency"]),
            np.mean(d["true_class"][mask] == row["class_name"]) if mask.any() else np.nan,
        )
    for row in records("extreme_probability"):
        d = all_groups[("classification", 0, "Logistic")]
        p = d["p_" + row["class_name"].lower()]
        cutoff = np.quantile(p, 0.9)
        mask = p >= cutoff
        close("extreme quantile", float(row["cutoff"]), cutoff)
        check("extreme count", int(row["n"]) == mask.sum())
        for c in CLASSES:
            close(
                "extreme outcome " + c,
                float(row[c + "_frequency"]),
                np.mean(d["true_class"][mask] == c),
            )
    for row in records("model_coefficients"):
        a = fitmap[(int(row["embargo_seconds"]), row["model_name"], int(row["fold_id"]))]
        j = a["features"].index(row["feature"])
        i = CLASSES.index(row["class_name"]) if a["kind"] == "Logistic" else 0
        close(
            "coefficient csv",
            float(row["standardized_coefficient"]),
            np.atleast_2d(a["coefficient"])[i, j],
        )
        close("intercept csv", float(row["intercept"]), np.atleast_1d(a["intercept"])[i])
    for filename in [
        "feature_ablation",
        "static_vs_flow",
        "random_label_control",
        "random_feature_control",
    ]:
        for row in records(filename):
            close(
                filename + " score",
                float(row["spearman"]),
                recomputed[("regression", 0, row["model_name"])]["spearman"],
            )
    ci = json.loads((OUT / "classification_inference.json").read_text())
    f1_draws = [float(r["logistic_minus_flat_f1"]) for r in boots]
    close(
        "classification CI", np.quantile(f1_draws, [0.025, 0.975]), [ci["ci_lower"], ci["ci_upper"]]
    )
    close(
        "classification point difference",
        ci["difference"],
        recomputed[("classification", 0, "Logistic")]["macro_f1"]
        - recomputed[("classification", 0, "AlwaysFLAT")]["macro_f1"],
    )
    for row in records("regression_inference"):
        name = row["model_name"]
        b = np.array([float(r[name]) for r in boots])
        null = np.array([float(r[name]) for r in shifts])
        lo, hi = np.quantile(b, [0.025, 0.975])
        close("CI " + name, [lo, hi], [float(row["ci_lower"]), float(row["ci_upper"])])
        obs = recomputed[("regression", 0, name)]["spearman"]
        close("inference observed " + name, float(row["spearman"]), obs)
        close("shift p " + name, float(row["shift_p"]), (1 + sum(abs(null) >= abs(obs))) / 1001)
    for row in records("single_feature_comparison"):
        name = row["comparison"].removeprefix("Ridge minus ")
        b = np.array([float(r["Ridge"]) for r in boots])
        baseline = (
            np.max(
                [
                    [float(r[n]) for r in boots]
                    for n in ["OBI_L1", "MICROPRICE_MINUS_MID", "OFI_1S"]
                ],
                axis=0,
            )
            if name == "BEST_SINGLE"
            else np.array([float(r[name]) for r in boots])
        )
        close(
            "paired CI " + name,
            np.quantile(b - baseline, [0.025, 0.975]),
            [float(row["ci_lower"]), float(row["ci_upper"])],
        )
        close(
            "paired point " + name,
            float(row["difference"]),
            float(row["ridge_spearman"]) - float(row["baseline_spearman"]),
        )
    # Literal independent resampling and scipy ranks for three recorded draws.
    anchor = all_groups[("regression", 0, "Ridge")]
    start = anchor["test_start"].min()
    end = anchor["test_end"].max()
    n = int((end - start) // 10**9)
    pos = ((anchor["grid_timestamp"] - start) // 10**9).astype(int)
    y = np.full(n, np.nan)
    y[pos] = anchor["target"]
    rng = np.random.default_rng(404060)
    for r in range(3):
        starts = rng.integers(0, n, size=(n + 59) // 60)
        sample = ((starts[:, None] + np.arange(60)) % n).ravel()[:n]
        for name in [
            "Ridge",
            "OLS",
            "OBI_L1",
            "MICROPRICE_MINUS_MID",
            "OFI_1S",
            "Static",
            "Flow",
            "RandomLabel",
            "RandomFeature",
        ]:
            x = np.full(n, np.nan)
            x[pos] = all_groups[("regression", 0, name)]["prediction"]
            mask = np.isfinite(x[sample]) & np.isfinite(y[sample])
            close(
                "literal bootstrap " + str((r, name)),
                spearmanr(x[sample][mask], y[sample][mask]).statistic,
                float(boots[r][name]),
            )
            offset = int(shifts[r]["offset_seconds"])
            sy = np.roll(y, offset)
            good = np.isfinite(x) & np.isfinite(sy)
            close(
                "independent shift " + str((r, name)),
                spearmanr(x[good], sy[good]).statistic,
                float(shifts[r][name]),
            )
    report = (ROOT / "PHASE4_REPORT.md").read_text(encoding="utf-8")
    # Match visible table rows, not hidden metadata or a report-generated checksum.
    for (task, embargo, name), m in recomputed.items():
        if embargo or name not in (
            ["OLS", "Ridge", "Static", "Flow", "ZeroReturn", "TrainingMean"]
            if task == "regression"
            else ["Logistic", "AlwaysFLAT", "TrainingPrior"]
        ):
            continue
        fields = (
            ["spearman", "pearson", "mae", "mse", "r2"]
            if task == "regression"
            else ["macro_f1", "balanced_accuracy", "accuracy", "log_loss"]
        )
        expected = "| " + name + " | " + " | ".join(f"{m[k]:.9g}" for k in fields) + " |"
        check("visible report metrics " + task + " " + name, expected in report)
    for name in ["OBI_L1", "MICROPRICE_MINUS_MID", "OFI_1S", "Ridge"]:
        m = recomputed[("regression", 0, name)]
        expected = "| " + name + " | " + f"{m['spearman']:.9g} | {m['pearson']:.9g} |"
        check("visible single-feature correlation " + name, expected in report)
    for row in records("regression_inference"):
        if row["model_name"] in ("OLS", "Ridge"):
            line = (
                "| "
                + row["model_name"]
                + " | "
                + " | ".join(f"{float(row[k]):.9g}" for k in ["ci_lower", "ci_upper", "shift_p"])
                + " |"
            )
            check("visible report inference " + row["model_name"], line in report)
    for name in ("reproducibility_audit.json", "isolation_audit.json"):
        audit = json.loads((OUT / name).read_text())
        check(name, audit["status"] == "PASS")
    result = dict(
        status="PASS",
        checks=len(checks),
        critical_failures=0,
        report_verified=True,
        report_sha256=h(ROOT / "PHASE4_REPORT.md"),
        verifier_sha256=h(__file__),
        prediction_hashes={p.name: h(p) for p in OUT.glob("*_oos_predictions.parquet")},
    )
    (OUT / "independent_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (OUT / "verification_checks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["check", "passed"])
        w.writeheader()
        w.writerows(checks)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
