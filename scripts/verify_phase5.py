"""Independent saved-prediction, trained-model and temporal arithmetic audit."""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, f1_score
from verify_phase4 import arrays, numeric

OUT = Path("reports/phase5")
FEATURES = [
    "OBI_L1",
    "MICROPRICE_MINUS_MID",
    "OFI_100MS",
    "OFI_1S",
    "RELATIVE_SPREAD",
    "EVENT_RATE_1S",
    "TIME_SINCE_PREVIOUS_TRADE",
]
CLASSES = np.array(["DOWN", "FLAT", "UP"])


def read(name):
    with (OUT / (name + ".csv")).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks = []

    def ck(label, truth):
        checks.append(dict(check=label, passed=bool(truth)))
        if not truth:
            raise AssertionError(label)

    def close(label, a, b):
        ck(label, np.allclose(a, b, atol=1e-12, rtol=1e-10, equal_nan=True))

    frozen = json.loads((OUT / "upstream_freeze.json").read_text())
    for p, h in frozen["files"].items():
        b = Path(p).read_bytes()
        b = b[: frozen["log_bytes"]] if p == "RESEARCH_LOG.md" else b
        ck("upstream " + p, hashlib.sha256(b).hexdigest() == h)
    ck(
        "phase2 exact hash",
        sha(frozen["phase2_path"])
        == "a5a493636d9f6d592e2d229921533940ec7cfffcdc5da8f1c9fb636445d34e96",
    )
    proto = json.loads((OUT / "protocol_freeze.json").read_text())
    for p, h in proto["documents"].items():
        ck("protocol " + p, sha(p) == h)
    ck(
        "hypotheses prefix",
        hashlib.sha256(Path("RESEARCH_LOG.md").read_bytes()[: proto["log_bytes"]]).hexdigest()
        == proto["log_hash"],
    )
    for filename, key in [
        ("execution_manifest.json", "code_sha256"),
        ("evaluation_code.json", None),
    ]:
        data = json.loads((OUT / filename).read_text())
        data = data[key] if key else data
        for p, h in data.items():
            ck("executed code " + p, sha(p) == h)
    p3 = json.loads(Path("PHASE3_MANIFEST.json").read_text())
    g = arrays(p3["output_path"])
    gt = g["grid_timestamp"]
    tau = g["target_timestamp_1S"]
    with Path("reports/phase4/fold_manifest.csv").open(newline="", encoding="utf-8") as f:
        fr = list(csv.DictReader(f))
    folds = {
        int(r["fold_id"]): {
            k: int(datetime.fromisoformat(v).timestamp()) * 10**9
            for k, v in r.items()
            if k.endswith(("_start", "_end"))
        }
        for r in fr
    }
    metadata = json.loads((OUT / "model_metadata.json").read_text())
    models = json.loads((OUT / "model_artifacts.json").read_text())
    meta = {(a["task"], a["model_name"], a["fold_id"]): a for a in metadata}
    selections = read("model_selection")
    allgroups = {}
    calculated = {}
    p4 = json.loads(Path("PHASE4_MANIFEST.json").read_text())["evidence_sha256"]
    for task in ["regression", "classification"]:
        t = arrays(OUT / f"{task}_oos_predictions.parquet")
        benchmark_path = next(p for p in p4 if p.endswith(f"{task}_oos_predictions.parquet"))
        b = arrays(benchmark_path)
        names = (
            ["Ridge", "OBI_L1", "MICROPRICE_MINUS_MID", "OFI_1S"]
            if task == "regression"
            else ["Logistic", "AlwaysFLAT"]
        )
        datasets = {}
        for name in np.unique(t["model_name"]):
            ids = np.flatnonzero(t["model_name"] == name)
            ids = ids[np.argsort(t["grid_timestamp"][ids])]
            datasets[str(name)] = {k: v[ids] for k, v in t.items()}
        for name in names:
            ids = np.flatnonzero((b["model_name"] == name) & (b["embargo_seconds"] == 0))
            ids = ids[np.argsort(b["grid_timestamp"][ids])]
            datasets[name] = {k: v[ids] for k, v in b.items()}
        for name, d in datasets.items():
            tag = (task, name)
            allgroups[tag] = d
            expected = datasets[names[0]]["grid_timestamp"]
            ck(
                "identical OOS " + str(tag),
                np.array_equal(d["grid_timestamp"], expected) and len(expected) == 11695,
            )
            ck("unique OOS " + str(tag), len(np.unique(d["grid_timestamp"])) == 11695)
            m = numeric(task, d)
            calculated[tag] = m
            row = next(r for r in read(task + "_metrics") if r["model_name"] == name)
            for metric, value in m.items():
                close("aggregate " + str((task, name, metric)), value, float(row[metric]))
            for fold, f in folds.items():
                mask = d["fold_id"] == fold
                sub = {k: v[mask] for k, v in d.items()}
                ts = sub["grid_timestamp"]
                test = (
                    (gt >= f["test_start"])
                    & (gt < f["test_end"])
                    & g["target_available_1S"]
                    & (tau < f["test_end"])
                )
                ck("test-only membership " + str((task, name, fold)), np.array_equal(ts, gt[test]))
                for field, v in f.items():
                    ck("metadata " + field, np.all(sub[field] == v))
                close("labels", sub["target"], g["GRID_FWD_MID_RET_1S"][test])
                ck("target timestamp", np.array_equal(sub["target_timestamp"], tau[test]))
                ck(
                    "source state",
                    np.array_equal(
                        sub["source_state_timestamp"], g["source_state_timestamp"][test]
                    ),
                )
                ck("causal feature timestamp", np.all(sub["source_state_timestamp"] <= ts))
                if task == "classification":
                    ck(
                        "true classes",
                        np.array_equal(sub["true_class"], g["GRID_DIRECTION_1S"][test]),
                    )
                values = numeric(task, sub)
                r = next(
                    r
                    for r in read(task + "_fold_metrics")
                    if r["model_name"] == name and int(r["fold_id"]) == fold
                )
                for metric, v in values.items():
                    close("fold metrics " + str((task, name, fold, metric)), v, float(r[metric]))
                if (task, name, fold) not in meta:
                    continue
                a = meta[(task, name, fold)]
                ck("feature version", set(sub["feature_version"]) == {"5.0.0"})
                expected_features = (
                    FEATURES
                    if name not in ["Static", "Flow", "RandomFeature"]
                    else [FEATURES[i] for i in ([0, 1, 4] if name == "Static" else [2, 3, 5, 6])]
                    if name != "RandomFeature"
                    else FEATURES + ["RANDOM_CONTROL"]
                )
                ck("whitelist " + str((task, name, fold)), a["features"] == expected_features)
                tr = (
                    (gt >= f["train_start"])
                    & (gt < f["train_end"])
                    & g["target_available_1S"]
                    & (tau < f["train_end"])
                )
                va = (
                    (gt >= f["validation_start"])
                    & (gt < f["validation_end"])
                    & g["target_available_1S"]
                    & (tau < f["validation_end"])
                )
                ck(
                    "train validation target purge",
                    tau[tr].max() < f["validation_start"] and tau[va].max() < f["test_start"],
                )
                ck(
                    "disjoint partitions",
                    not np.any(tr & va) and not np.any(tr & test) and not np.any(va & test),
                )
                x = np.column_stack([g[k] for k in a["features"]])
                median = np.nanmedian(x[tr], axis=0)
                close("training-only imputation", median, a["median"])
                x = np.where(np.isfinite(x), x, median)
                path = next(
                    p
                    for p in models
                    if p["task"] == task and p["model_name"] == name and p["fold_id"] == fold
                )
                ck("serialized model hash", sha(path["path"]) == path["sha256"])
                bundle = joblib.load(path["path"])
                model = bundle["model"]
                ck("serialized metadata", bundle["metadata"] == a)

                def predict(xx):
                    if hasattr(model, "booster_"):
                        return model.booster_.predict(xx, num_threads=1)
                    return model.predict(xx) if task == "regression" else model.predict_proba(xx)

                pred = predict(x[test])
                vp = predict(x[va])
                tp = predict(x[tr])
                saved = (
                    sub["prediction"]
                    if task == "regression"
                    else np.column_stack([sub[k] for k in ["p_down", "p_flat", "p_up"]])
                )
                close("saved model predicts saved OOS " + str((task, name, fold)), pred, saved)
                y = g["GRID_FWD_MID_RET_1S" if task == "regression" else "GRID_DIRECTION_1S"]
                yt = y[tr].copy()
                if name == "RandomLabel":
                    chunks = [yt[i : i + 60] for i in range(0, len(yt), 60)]
                    order = np.random.default_rng(505028 + fold).permutation(len(chunks))
                    yt = np.concatenate([chunks[i] for i in order])
                elif name == "TemporalPlacebo":
                    yt = np.roll(yt, 600)
                if task == "regression":
                    vscore = spearmanr(vp, y[va]).statistic
                    tscore = spearmanr(tp, yt).statistic
                else:
                    ck("probability ordering", np.array_equal(model.classes_, CLASSES))
                    vscore = f1_score(y[va], CLASSES[vp.argmax(1)], labels=CLASSES, average="macro")
                    tscore = f1_score(yt, CLASSES[tp.argmax(1)], labels=CLASSES, average="macro")
                close("validation score", vscore, a["validation_score"])
                close("train score", tscore, a["train_score"])
                candidates = [
                    r
                    for r in selections
                    if r["task"] == task and r["model_name"] == name and int(r["fold_id"]) == fold
                ]
                best = max(float(r["validation_score"]) for r in candidates)
                chosen = next(r for r in candidates if float(r["validation_score"]) >= best - 1e-12)
                ck(
                    "validation-selected candidate",
                    int(chosen["candidate_id"]) == a["candidate_id"],
                )
                ck("candidate tags", np.all(sub["selected_candidate"] == a["candidate_id"]))
                if name == "ValidationSelected":
                    rf = meta[(task, "RF", fold)]
                    lgb = meta[(task, "LightGBM", fold)]
                    family = (
                        "RF"
                        if rf["validation_score"] >= lgb["validation_score"] - 1e-12
                        else "LightGBM"
                    )
                    ck("validation-only family selection", a["family"] == family)
                    other = datasets[family]
                    close(
                        "selected family OOS equality",
                        saved,
                        other["prediction"][mask]
                        if task == "regression"
                        else np.column_stack(
                            [other[k][mask] for k in ["p_down", "p_flat", "p_up"]]
                        ),
                    )
    for r in read("confusion_matrix"):
        d = allgroups[("classification", r["model_name"])]
        fold = int(r["fold_id"])
        mask = d["fold_id"] == fold if fold else np.ones(len(d["true_class"]), bool)
        cm = confusion_matrix(d["true_class"][mask], d["predicted_class"][mask], labels=CLASSES)
        ck(
            "confusion cell",
            int(r["count"])
            == cm[list(CLASSES).index(r["true_class"]), list(CLASSES).index(r["predicted_class"])],
        )
    for r in read("prediction_correlation"):
        x = allgroups[("regression", r["model_a"])]["prediction"]
        y = allgroups[("regression", r["model_b"])]["prediction"]
        close("prediction rank correlation", float(r["spearman"]), spearmanr(x, y).statistic)
        close("prediction Pearson", float(r["pearson"]), pearsonr(x, y).statistic)
    draws = read("bootstrap_draws")
    for task, filename, base, metric in [
        ("regression", "nonlinear_vs_ridge_bootstrap", "Ridge", "spearman"),
        ("classification", "nonlinear_vs_logistic_bootstrap", "Logistic", "macro_f1"),
    ]:
        prefix = "reg_" if task == "regression" else "class_"
        for row in read(filename):
            name = row["model_name"]
            delta = np.array([float(r[prefix + name]) - float(r[prefix + base]) for r in draws])
            close(
                "paired delta observed",
                float(row["observed_delta"]),
                calculated[(task, name)][metric] - calculated[(task, base)][metric],
            )
            close("paired delta mean", float(row["bootstrap_mean_delta"]), delta.mean())
            close(
                "paired interval",
                np.quantile(delta, [0.025, 0.975]),
                [float(row["ci_lower"]), float(row["ci_upper"])],
            )
    # Three literal bootstrap replicates, independently recomputed with scipy/sklearn.
    anchor = allgroups[("regression", "Ridge")]
    start = anchor["test_start"].min()
    n = int((anchor["test_end"].max() - start) // 10**9)
    pos = ((anchor["grid_timestamp"] - start) // 10**9).astype(int)
    mapping = np.full(n, -1, int)
    mapping[pos] = np.arange(len(pos))
    rng = np.random.default_rng(505060)
    for r in range(3):
        starts = rng.integers(0, n, size=(n + 59) // 60)
        clock = ((starts[:, None] + np.arange(60)) % n).ravel()[:n]
        ids = mapping[clock]
        ids = ids[ids >= 0]
        for (task, name), d in allgroups.items():
            actual = (
                spearmanr(d["prediction"][ids], d["target"][ids]).statistic
                if task == "regression"
                else f1_score(
                    d["true_class"][ids], d["predicted_class"][ids], labels=CLASSES, average="macro"
                )
            )
            close(
                "literal paired bootstrap " + str((r, task, name)),
                actual,
                float(draws[r][("reg_" if task == "regression" else "class_") + name]),
            )
    report = Path("PHASE5_REPORT.md").read_text(encoding="utf-8")
    for task, names, fields in [
        (
            "regression",
            ["Ridge", "RF", "LightGBM", "ValidationSelected"],
            ["spearman", "pearson", "mae", "mse", "r2"],
        ),
        (
            "classification",
            ["Logistic", "RF", "LightGBM", "ValidationSelected", "AlwaysFLAT"],
            ["macro_f1", "balanced_accuracy", "accuracy", "log_loss"],
        ),
    ]:
        for name in names:
            m = calculated[(task, name)]
            line = "| " + name + " | " + " | ".join(f"{m[k]:.9g}" for k in fields) + " |"
            ck("visible report metrics " + str((task, name)), line in report)
    for filename in ["nonlinear_vs_ridge_bootstrap", "nonlinear_vs_logistic_bootstrap"]:
        for row in read(filename):
            line = (
                "| "
                + row["model_name"]
                + " | "
                + " | ".join(
                    f"{float(row[k]):.9g}" for k in ["observed_delta", "ci_lower", "ci_upper"]
                )
                + " | "
                + row["fold_wins"]
                + " |"
            )
            ck("visible paired report " + filename + " " + row["model_name"], line in report)
    for filename in ["isolation_audit.json", "reproducibility_audit.json"]:
        ck(filename, json.loads((OUT / filename).read_text())["status"] == "PASS")
    result = dict(
        status="PASS",
        checks=len(checks),
        critical_failures=0,
        report_verified=True,
        report_sha256=sha("PHASE5_REPORT.md"),
        verifier_sha256=sha(__file__),
    )
    (OUT / "independent_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (OUT / "verification_checks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "passed"])
        writer.writeheader()
        writer.writerows(checks)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
