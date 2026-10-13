"""Literal resampling and metrics, independent of Phase 3–5 statistics code."""

from datetime import datetime

import joblib
import numpy as np
from scipy.special import softmax

from .common import (
    CHECKS,
    arrays,
    blocks,
    close,
    corr,
    dump,
    js,
    predictions,
    readcsv,
    record,
    write,
)

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


def class_metrics(y, p):
    cm = np.zeros((3, 3), int)
    for i, a in enumerate(CLASSES):
        for j, b in enumerate(CLASSES):
            cm[i, j] = np.sum((y == a) & (p == b))
    tp = np.diag(cm)
    den = cm.sum(0) + cm.sum(1)
    return dict(
        macro_f1=float(np.divide(2 * tp, den, out=np.zeros(3), where=den > 0).mean()),
        balanced_accuracy=float(
            np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0).mean()
        ),
        accuracy=float(tp.sum() / cm.sum()),
    )


def adjusted(p, by=False):
    p = np.asarray(p)
    order = np.argsort(p)
    m = len(p)
    values = p[order] * m / np.arange(1, m + 1) * (sum(1 / np.arange(1, m + 1)) if by else 1)
    out = np.empty(m)
    out[order] = np.minimum(1, np.minimum.accumulate(values[::-1])[::-1])
    return out


def phase3(g, cfg):
    saved = {r["feature"]: r for r in readcsv("reports/phase3/primary_feature_statistics.csv")}
    quintiles = readcsv("reports/phase3/primary_feature_quintiles.csv")
    features = cfg["features"]
    y = g["GRID_FWD_MID_RET_1S"]
    n = len(y)
    observed = [corr(g[f], y) for f in features]
    draw = np.empty((1000, len(features)))
    null = np.empty_like(draw)
    rng = np.random.default_rng(314219)
    edges = {
        f: np.quantile(g[f][np.isfinite(g[f]) & np.isfinite(y)], [0.2, 0.4, 0.6, 0.8])
        for f in features
    }
    qdraw = np.empty_like(draw)
    for r in range(1000):
        ix = blocks(n, 60, rng)
        yy = y[ix]
        for j, f in enumerate(features):
            xx = g[f][ix]
            draw[r, j] = corr(xx, yy)
            q = np.searchsorted(edges[f], xx, side="left") + 1
            ok = np.isfinite(xx) & np.isfinite(yy)
            qdraw[r, j] = yy[ok & (q == 5)].mean() - yy[ok & (q == 1)].mean()
    shifts = np.random.default_rng(271828).integers(600, n - 600 + 1, 1000)
    for r, s in enumerate(shifts):
        yy = np.roll(y, int(s))
        for j, f in enumerate(features):
            null[r, j] = corr(g[f], yy)
    p = (1 + (np.abs(null) >= np.abs(observed)).sum(0)) / 1001
    bh, by = adjusted(p), adjusted(p, True)
    rows = []
    intervals = readcsv("reports/phase3/bootstrap_intervals.csv")
    for j, f in enumerate(features):
        lo, hi = np.quantile(draw[:, j], [0.025, 0.975])
        s = saved[f]
        vals = dict(
            spearman_rho=observed[j],
            bootstrap_ci_low=lo,
            bootstrap_ci_high=hi,
            permutation_p=p[j],
            fdr_bh_q=bh[j],
            fdr_by_q=by[j],
        )
        for k, v in vals.items():
            close("phase3", f + " " + k, v, float(s[k]))
        for q in range(1, 6):
            m = (
                np.isfinite(g[f])
                & np.isfinite(y)
                & (np.searchsorted(edges[f], g[f], side="left") + 1 == q)
            )
            row = next(r for r in quintiles if r["feature"] == f and int(r["quintile"]) == q)
            close("phase3", f + f" q{q} count", int(m.sum()), int(row["n"]), 0)
            if m.any():
                close("phase3", f + f" q{q} mean", y[m].mean(), float(row["mean_return"]))
        qi = next(
            r
            for r in intervals
            if r["feature"] == f
            and r["block_seconds"] == "60"
            and r["metric"] == "q5_minus_q1_mean_return"
        )
        close(
            "phase3",
            f + " quintile difference CI",
            np.quantile(qdraw[:, j], [0.025, 0.975]),
            [float(qi["ci_low"]), float(qi["ci_high"])],
        )
        rows.append(dict(phase=3, feature=f, **vals))
    record(
        "phase3",
        "shifts exclude nearby alignment",
        np.all((shifts >= 600) & (shifts <= n - 600)),
        detail="Full-session circular wrap is artificial null seam, not information fed to fitting.",
    )
    write("phase3_independent_statistics", rows)
    return rows


def folds(g):
    result = []
    for r in readcsv("reports/phase4/fold_manifest.csv"):
        f = {
            k: int(datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()) * 10**9
            for k, v in r.items()
            if k.endswith(("_start", "_end"))
        }
        f["fold_id"] = int(r["fold_id"])
        result.append(f)
    return result


def partitions(g, f, embargo=0):
    t = g["grid_timestamp"]
    tau = t + 10**9
    out = {}
    for name in ["train", "validation", "test"]:
        end = f[name + "_end"] - (embargo * 10**9 if name != "test" else 0)
        out[name] = np.flatnonzero(
            (t >= f[name + "_start"])
            & (t < f[name + "_end"])
            & (tau < end)
            & g["target_available_1S"]
            & g["grid_state_available"]
        )
    return out


def provenance(g):
    fs = folds(g)
    rows = []
    saved = js("reports/phase4/fitted_models.json")
    validation6 = arrays("reports/phase6/validation_predictions.parquet")
    selection5 = readcsv("reports/phase5/model_selection.csv")
    for f in fs:
        for gap in [0, 10]:
            parts = partitions(g, f, gap)
            for name, idx in parts.items():
                bound = f[name + "_end"] - (gap * 10**9 if name != "test" else 0)
                actual = g["target_state_timestamp_1S"][idx]
                record(
                    "folds",
                    f"fold{f['fold_id']} gap{gap} {name} purge",
                    bool(
                        np.all(actual <= g["grid_timestamp"][idx] + 10**9)
                        and np.all(actual < bound)
                    ),
                )
                rows.append(
                    dict(
                        phase=4,
                        fold=f["fold_id"],
                        embargo=gap,
                        partition=name,
                        rows=len(idx),
                        max_actual_target=int(actual.max()),
                        boundary=bound,
                    )
                )
            for meta in [
                r for r in saved if r["fold_id"] == f["fold_id"] and r["embargo_seconds"] == gap
            ]:
                names = meta["features"]
                x = np.column_stack([g[k] for k in names])
                tr = parts["train"]
                xt = x[tr]
                med = np.nanmedian(xt, axis=0)
                z = np.where(np.isfinite(xt), xt, med)
                sd = z.std(0)
                sd[sd == 0] = 1
                for k, a in [("median", med), ("mean", z.mean(0)), ("scale", sd)]:
                    close(
                        "provenance4",
                        f"{meta['model_name']} fold{f['fold_id']} gap{gap} {k}",
                        a,
                        meta[k],
                    )
                scores = meta["validation_scores"]
                top = max(r["score"] for r in scores)
                tied = [r for r in scores if r["score"] >= top - 1e-12]
                chosen = (
                    max(tied, key=lambda r: r["parameter"])
                    if meta["kind"] != "Logistic"
                    else min(tied, key=lambda r: r["parameter"])
                )
                close(
                    "provenance4",
                    f"{meta['model_name']} validation selection",
                    meta["parameter"],
                    chosen["parameter"],
                )
                # Replay selected model algebra; no fitting or use of test labels.
                for partition in ["validation", "test"]:
                    idx = parts[partition]
                    xx = (np.where(np.isfinite(x[idx]), x[idx], med) - z.mean(0)) / sd
                    raw = xx @ np.asarray(meta["coefficient"]).T + np.asarray(meta["intercept"])
                    if meta["kind"] == "Logistic":
                        pp = softmax(raw, axis=1)
                        value = class_metrics(g["GRID_DIRECTION_1S"][idx], CLASSES[pp.argmax(1)])[
                            "macro_f1"
                        ]
                        task = "classification"
                    else:
                        pp = raw
                        value = corr(pp, g["GRID_FWD_MID_RET_1S"][idx])
                        task = "regression"
                    if partition == "validation":
                        selected = next(r for r in scores if r["selected"])
                        close(
                            "provenance4",
                            meta["model_name"] + " selected validation score",
                            value,
                            selected["score"],
                        )
                        if meta["model_name"] == "Ridge" and gap == 0:
                            m = (validation6["model_name"] == "Ridge") & (
                                validation6["fold_id"] == f["fold_id"]
                            )
                            close(
                                "thresholds",
                                "Ridge validation reconstruction",
                                pp,
                                validation6["prediction"][m],
                            )
                    else:
                        a = predictions(4, task, meta["model_name"], gap)
                        mask = a["fold_id"] == f["fold_id"]
                        target = (
                            np.column_stack([a[k][mask] for k in ["p_down", "p_flat", "p_up"]])
                            if task == "classification"
                            else a["prediction"][mask]
                        )
                        close(
                            "provenance4", meta["model_name"] + " saved OOS prediction", pp, target
                        )
    # Deterministic random feature is an explicitly registered diagnostic only.
    gg = dict(g)
    for artifact in js("reports/phase5/model_artifacts.json"):
        bundle = joblib.load(artifact["path"])
        meta = bundle["metadata"]
        model = bundle["model"]
        f = fs[artifact["fold_id"] - 1]
        part = partitions(g, f)
        names = meta["features"]
        allowed = [
            FEATURES,
            [FEATURES[k] for k in [0, 1, 4]],
            [FEATURES[k] for k in [2, 3, 5, 6]],
            FEATURES + ["RANDOM_CONTROL"],
        ]
        record(
            "provenance5",
            artifact["path"] + " feature whitelist",
            names in allowed and not any("TICK" in x or "FWD" in x for x in names),
        )
        x = np.column_stack([gg[k] for k in names])
        median = np.nanmedian(x[part["train"]], axis=0)
        close("provenance5", artifact["path"] + " train medians", median, meta["median"])
        record(
            "provenance5",
            artifact["path"] + " fit sizes and seed",
            meta["training_rows"] == len(part["train"])
            and meta["validation_rows"] == len(part["validation"])
            and meta["seed"] == 505,
        )
        candidates = [
            dict(candidate_id=int(r["candidate_id"]), validation_score=float(r["validation_score"]))
            for r in selection5
            if r["model_name"] == artifact["model_name"]
            and r["task"] == artifact["task"]
            and int(r["fold_id"]) == artifact["fold_id"]
        ]
        if candidates:
            eligible = [
                r
                for r in candidates
                if r["validation_score"] >= max(c["validation_score"] for c in candidates) - 1e-12
            ]
            close(
                "provenance5",
                artifact["path"] + " selected candidate",
                eligible[0]["candidate_id"],
                meta["candidate_id"],
            )
        for partition in ["validation", "test"]:
            idx = part[partition]
            xx = np.where(np.isfinite(x[idx]), x[idx], median)
            task = artifact["task"]
            p = (
                model.booster_.predict(xx, num_threads=1)
                if hasattr(model, "booster_")
                else model.predict(xx)
                if task == "regression"
                else model.predict_proba(xx)
            )
            if partition == "validation":
                value = (
                    corr(p, g["GRID_FWD_MID_RET_1S"][idx])
                    if task == "regression"
                    else class_metrics(g["GRID_DIRECTION_1S"][idx], CLASSES[p.argmax(1)])[
                        "macro_f1"
                    ]
                )
                close(
                    "provenance5",
                    artifact["path"] + " validation metric",
                    value,
                    meta["validation_score"],
                )
                if task == "regression" and artifact["model_name"] in ["RF", "LightGBM"]:
                    m = (validation6["model_name"] == artifact["model_name"]) & (
                        validation6["fold_id"] == f["fold_id"]
                    )
                    close(
                        "thresholds",
                        artifact["path"] + " validation predictions",
                        p,
                        validation6["prediction"][m],
                    )
            else:
                a = predictions(5, task, artifact["model_name"])
                m = a["fold_id"] == artifact["fold_id"]
                close(
                    "provenance5",
                    artifact["path"] + " OOS clock",
                    g["grid_timestamp"][idx],
                    a["grid_timestamp"][m],
                    0,
                )
                expected = (
                    a["prediction"][m]
                    if task == "regression"
                    else np.column_stack([a[k][m] for k in ["p_down", "p_flat", "p_up"]])
                )
                close("provenance5", artifact["path"] + " OOS predictions", p, expected)
        params = model.get_params()
        record(
            "provenance5",
            artifact["path"] + " no early stop",
            not any(k.startswith("early_stopping") and v for k, v in params.items()),
        )
        rows.append(
            dict(
                phase=5,
                fold=f["fold_id"],
                model=artifact["model_name"],
                task=artifact["task"],
                features="|".join(names),
                candidate=meta["candidate_id"],
                seed=meta["seed"],
                artifact=artifact["path"],
                rows=len(part["test"]),
            )
        )
    write("model_provenance_audit", rows)


def metrics_and_stress():
    results = {}
    allrows = []
    for phase in [4, 5]:
        for task, names in [
            (
                "regression",
                ["OBI_L1", "OLS", "Ridge"] if phase == 4 else ["Ridge", "RF", "LightGBM"],
            ),
            ("classification", ["Logistic"] if phase == 4 else ["Logistic", "RF", "LightGBM"]),
        ]:
            saved = readcsv(f"reports/phase{phase}/{task}_metrics.csv")
            for name in names:
                a = predictions(phase, task, name)
                s = next(
                    r
                    for r in saved
                    if r["model_name"] == name and int(r.get("embargo_seconds", 0)) == 0
                )
                values = (
                    {"spearman": corr(a["prediction"], a["target"])}
                    if task == "regression"
                    else class_metrics(a["true_class"], a["predicted_class"])
                )
                for k, v in values.items():
                    close("metrics", f"phase{phase} {name} {k}", v, float(s[k]))
                record(
                    "metrics", f"phase{phase} {name} OOS rows", len(a["grid_timestamp"]) == 11695
                )
                results[f"phase{phase}_{task}_{name}"] = values
                allrows.append(dict(phase=phase, task=task, model=name, **values))
    # Preserve the full 11700-second clock, including purged holes.
    reg = {n: predictions(5, "regression", n) for n in ["Ridge", "RF", "LightGBM"]}
    cls = {n: predictions(5, "classification", n) for n in ["Logistic", "RF", "LightGBM"]}
    a = reg["Ridge"]
    n = 11700
    where = ((a["grid_timestamp"] - a["test_start"][0]) // 10**9).astype(int)
    lookup = np.full(n, -1)
    lookup[where] = np.arange(len(where))
    stress = []
    for length, seed in [(30, 707030), (60, 505060), (300, 707300)]:
        rng = np.random.default_rng(seed)
        draw = {
            f"{task}_{name}": []
            for task, names in [("regression", reg), ("classification", cls)]
            for name in names
        }
        for r in range(1000):
            idx = lookup[blocks(n, length, rng)]
            idx = idx[idx >= 0]
            for name, a in reg.items():
                draw["regression_" + name].append(corr(a["prediction"][idx], a["target"][idx]))
            for name, a in cls.items():
                draw["classification_" + name].append(
                    class_metrics(a["true_class"][idx], a["predicted_class"][idx])["macro_f1"]
                )
        if length == 60:
            saved = readcsv("reports/phase5/bootstrap_draws.csv")
            for k, v in draw.items():
                col = k.replace("regression_", "reg_").replace("classification_", "class_")
                close("bootstrap5", "literal draws " + col, v, [float(r[col]) for r in saved])
        for task, base, group in [
            ("regression", "Ridge", reg),
            ("classification", "Logistic", cls),
        ]:
            for name in ["RF", "LightGBM"]:
                delta = np.asarray(draw[task + "_" + name]) - draw[task + "_" + base]
                lo, hi = np.quantile(delta, [0.025, 0.975])
                wins = 0
                for f in range(1, 6):
                    m = group[name]["fold_id"] == f

                    def metric(a):
                        return (
                            corr(a["prediction"][m], a["target"][m])
                            if task == "regression"
                            else class_metrics(a["true_class"][m], a["predicted_class"][m])[
                                "macro_f1"
                            ]
                        )

                    wins += metric(group[name]) > metric(group[base])
                stress.append(
                    dict(
                        task=task,
                        model=name,
                        benchmark=base,
                        block_seconds=length,
                        seed=seed,
                        replicates=1000,
                        ci_lower=lo,
                        ci_upper=hi,
                        fold_wins=wins,
                    )
                )
                if length == 60:
                    file = (
                        "nonlinear_vs_ridge_bootstrap"
                        if task == "regression"
                        else "nonlinear_vs_logistic_bootstrap"
                    )
                    s = next(
                        r
                        for r in readcsv("reports/phase5/" + file + ".csv")
                        if r["model_name"] == name
                    )
                    close(
                        "bootstrap5",
                        name + " " + task + " interval",
                        [lo, hi],
                        [float(s["ci_lower"]), float(s["ci_upper"])],
                    )
                    close(
                        "bootstrap5", name + " " + task + " fold wins", wins, int(s["fold_wins"]), 0
                    )
    write("nonlinear_block_stress", stress)
    acf = []
    a = reg["LightGBM"]
    for name, v in [
        ("prediction", a["prediction"]),
        ("target", a["target"]),
        ("residual", a["target"] - a["prediction"]),
    ]:
        clock = np.full(n, np.nan)
        clock[where] = v
        rho = []
        for lag in range(1, 301):
            ok = np.isfinite(clock[:-lag]) & np.isfinite(clock[lag:])
            r = np.corrcoef(clock[:-lag][ok], clock[lag:][ok])[0, 1]
            rho.append(r)
            acf.append(dict(series=name, lag_seconds=lag, autocorrelation=r, pairs=int(ok.sum())))
        first_nonpositive = next((i for i, r in enumerate(rho) if r <= 0), len(rho))
        tau = 1 + 2 * sum(rho[:first_nonpositive])
        results["ess_" + name] = dict(
            observations=len(v),
            integrated_time=tau,
            approximate_ess=len(v) / tau,
            truncation_lag=first_nonpositive,
            meaning="mean-process diagnostic, not Spearman or model-comparison degrees of freedom",
        )
    write("autocorrelation", acf)
    write("independent_model_metrics", allrows)
    dump("statistical_summary.json", dict(metrics=results, stress=stress))
    return results


def run(g, cfg):
    phase3(g, cfg)
    dump("phase3_checks.json", CHECKS)
    provenance(g)
    dump("provenance_checks.json", CHECKS)
    metrics_and_stress()
    write(
        "statistical_reproduction",
        [
            r
            for r in CHECKS
            if r["section"]
            in ["phase3", "folds", "provenance4", "provenance5", "metrics", "bootstrap5"]
        ],
    )
    dump("statistical_checks.json", CHECKS)
