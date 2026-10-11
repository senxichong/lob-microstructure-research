import json

import joblib
import numpy as np

from lob_research.phase4.core import FEATURES, partitions
from lob_research.phase4.storage import prediction_groups

from .core import impute, predict, score
from .storage import OUT, csvwrite, groups


def diagnose(g, progress=print):
    importance = []
    perm = []
    complexity = []
    response = []
    interactions = []
    summary = []
    paths = json.loads((OUT / "model_artifacts.json").read_text())
    for item in paths:
        if item["model_name"] not in ["RF", "LightGBM", "RandomFeature"]:
            continue
        bundle = joblib.load(item["path"])
        model = bundle["model"]
        a = bundle["metadata"]
        task = a["task"]
        name = a["model_name"]
        fold = a["fold_id"]
        names = a["features"]
        part, _ = partitions(g, a, 0)
        tr, te = part["train"], part["test"]
        x = np.column_stack([g[n] for n in names])
        x = impute(x, a["median"])
        xt = x[te]
        y = g["GRID_FWD_MID_RET_1S" if task == "regression" else "GRID_DIRECTION_1S"][te]
        base = predict(model, xt, task)
        base_score = score(task, y, base)
        if name == "RF":
            native = model.feature_importances_
            splits = np.full(len(names), np.nan)
            depths = [m.tree_.max_depth for m in model.estimators_]
            leaves = [m.tree_.n_leaves for m in model.estimators_]
            comp = dict(
                number_trees=len(depths),
                depth_min=min(depths),
                depth_mean=float(np.mean(depths)),
                depth_max=max(depths),
                mean_leaves=float(np.mean(leaves)),
                boosting_rounds=None,
            )
        else:
            native = model.booster_.feature_importance("gain")
            splits = model.booster_.feature_importance("split")
            comp = dict(
                number_trees=model.booster_.num_trees(),
                boosting_rounds=model.booster_.current_iteration(),
                depth_min=None,
                depth_mean=None,
                depth_max=a["parameters"]["max_depth"],
                mean_leaves=None,
            )
        complexity.append(
            dict(
                task=task,
                model_name=name,
                fold_id=fold,
                training_rows=len(tr),
                **comp,
                **a["parameters"],
            )
        )
        total = native.sum()
        for j, f in enumerate(names):
            importance.append(
                dict(
                    task=task,
                    model_name=name,
                    fold_id=fold,
                    feature=f,
                    importance_type="impurity" if name == "RF" else "gain",
                    native_importance=float(native[j]),
                    normalized_importance=float(native[j] / total) if total else 0,
                    split_count=float(splits[j]),
                )
            )
            for offset in [300, 600, 900]:
                changed = xt.copy()
                changed[:, j] = np.roll(changed[:, j], offset)
                value = score(task, y, predict(model, changed, task))
                perm.append(
                    dict(
                        task=task,
                        model_name=name,
                        fold_id=fold,
                        feature=f,
                        offset_seconds=offset,
                        baseline_score=base_score,
                        permuted_score=value,
                        degradation=base_score - value,
                    )
                )
        if task == "regression" and name == "LightGBM":
            background = xt[np.linspace(0, len(xt) - 1, 200, dtype=int)].copy()
            for feature in ["OBI_L1", "MICROPRICE_MINUS_MID", "OFI_1S", "OFI_100MS"]:
                j = names.index(feature)
                values = np.unique(np.quantile(x[tr, j], np.arange(0.05, 1, 0.1)))
                for value in values:
                    b = background.copy()
                    b[:, j] = value
                    response.append(
                        dict(
                            fold_id=fold,
                            feature=feature,
                            value=float(value),
                            mean_prediction=float(predict(model, b, task).mean()),
                            background_rows=len(b),
                            grid_source="training quantiles",
                        )
                    )
            for f1, f2 in [
                ("OBI_L1", "OFI_1S"),
                ("OBI_L1", "RELATIVE_SPREAD"),
                ("OFI_1S", "EVENT_RATE_1S"),
                ("MICROPRICE_MINUS_MID", "OBI_L1"),
            ]:
                j, k = names.index(f1), names.index(f2)
                v1 = np.unique(np.quantile(x[tr, j], [0.1, 0.3, 0.5, 0.7, 0.9]))
                v2 = np.unique(np.quantile(x[tr, k], [0.1, 0.3, 0.5, 0.7, 0.9]))
                surface = np.empty((len(v1), len(v2)))
                for r, a1 in enumerate(v1):
                    for c, a2 in enumerate(v2):
                        b = background.copy()
                        b[:, j] = a1
                        b[:, k] = a2
                        surface[r, c] = predict(model, b, task).mean()
                        interactions.append(
                            dict(
                                fold_id=fold,
                                feature_a=f1,
                                feature_b=f2,
                                value_a=float(a1),
                                value_b=float(a2),
                                mean_prediction=float(surface[r, c]),
                            )
                        )
                residual = (
                    surface - surface.mean(0)[None, :] - surface.mean(1)[:, None] + surface.mean()
                )
                summary.append(
                    dict(
                        fold_id=fold,
                        feature_a=f1,
                        feature_b=f2,
                        nonadditive_rms=float(np.sqrt(np.mean(residual**2))),
                        surface_range=float(np.ptp(surface)),
                    )
                )
        progress(f"Interpretation {task} {name} fold {fold}")
    for n, rows in [
        ("feature_importance", importance),
        ("permutation_importance", perm),
        ("complexity_summary", complexity),
        ("response_shapes", response),
        ("interaction_surfaces", interactions),
        ("interaction_summary", summary),
    ]:
        csvwrite(n, rows)
    # The threshold is descriptive and never used for fitting or forecasting.
    d = groups("regression")["LightGBM"]
    ridge = prediction_groups("regression")[(0, "Ridge")]
    phase4 = json.loads((OUT.parent / "phase4/fitted_models.json").read_text())
    meta = json.loads((OUT / "model_metadata.json").read_text())
    distance = np.empty(len(d["target"]))
    for fold in range(1, 6):
        a = next(
            a
            for a in meta
            if a["task"] == "regression" and a["model_name"] == "LightGBM" and a["fold_id"] == fold
        )
        b = next(
            b
            for b in phase4
            if b["model_name"] == "Ridge" and b["fold_id"] == fold and b["embargo_seconds"] == 0
        )
        part, _ = partitions(g, a, 0)
        tr = part["train"]
        x = np.column_stack([g[n] for n in b["features"]])
        z = (np.where(np.isfinite(x[tr]), x[tr], b["median"]) - b["mean"]) / b["scale"]
        rp = z @ np.array(b["coefficient"]) + b["intercept"]
        mask = d["fold_id"] == fold
        rz = (ridge["prediction"][mask] - rp.mean()) / rp.std()
        lz = (d["prediction"][mask] - a["train_prediction_mean"]) / a["train_prediction_sd"]
        distance[mask] = abs(rz - lz)
    cutoff = np.quantile(distance, 0.9)
    high = distance >= cutoff
    ix = np.searchsorted(g["grid_timestamp"], d["grid_timestamp"])
    rows = []
    for fold in range(6):
        for group, selected in [("top_decile", high), ("remainder", ~high)]:
            mask = selected & ((d["fold_id"] == fold) if fold else True)
            rows.append(
                dict(
                    fold_id=fold,
                    group=group,
                    threshold=float(cutoff),
                    n=int(mask.sum()),
                    mean_absolute_standardized_difference=float(distance[mask].mean()),
                    mean_target=float(d["target"][mask].mean()),
                    mean_absolute_target=float(abs(d["target"][mask]).mean()),
                    ridge_spearman=score(
                        "regression", d["target"][mask], ridge["prediction"][mask]
                    ),
                    lightgbm_spearman=score("regression", d["target"][mask], d["prediction"][mask]),
                    **{f + "_mean": float(g[f][ix][mask].mean()) for f in FEATURES},
                )
            )
    csvwrite("disagreement_analysis", rows)
