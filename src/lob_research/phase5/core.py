import hashlib
import json

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from lob_research.phase3.statistics import spearman
from lob_research.phase4.core import (
    CLASSES,
    FEATURES,
    FLOW,
    STATIC,
    canonical,
    classification_metrics,
    partitions,
)

RF_GRID = [
    dict(max_depth=d, min_samples_leaf=m, max_features=f)
    for d, m, f in [(3, 100, "sqrt"), (6, 100, 1.0), (6, 20, "sqrt"), (None, 100, 1.0)]
]
LGB_GRID = [
    dict(num_leaves=leaves, max_depth=d, learning_rate=r, n_estimators=n, min_child_samples=m)
    for leaves, d, r, n, m in [
        (7, 3, 0.03, 100, 200),
        (7, 3, 0.08, 200, 100),
        (15, 4, 0.03, 200, 100),
        (15, 4, 0.08, 100, 200),
    ]
]


def whitelist(names, random=False):
    accepted = [FEATURES, STATIC, FLOW]
    if random:
        accepted.append(FEATURES + ["RANDOM_CONTROL"])
    if list(names) not in accepted:
        raise ValueError("Feature whitelist rejects unknown, future-derived or unregistered input")


def training_medians(x):
    median = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    if not np.isfinite(median).all():
        raise ValueError("entire training feature missing")
    return median


def impute(x, median):
    return np.where(np.isfinite(x), x, median)


def block_permuted(y, fold):
    chunks = [y[i : i + 60] for i in range(0, len(y), 60)]
    order = np.random.default_rng(505028 + fold).permutation(len(chunks))
    return np.concatenate([chunks[i] for i in order])


def make_model(family, task, params):
    if family == "RF":
        factory = RandomForestRegressor if task == "regression" else RandomForestClassifier
        return factory(n_estimators=150, bootstrap=True, n_jobs=1, random_state=505, **params)
    factory = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
    return factory(
        objective="regression" if task == "regression" else "multiclass",
        n_jobs=1,
        random_state=505,
        deterministic=True,
        force_col_wise=True,
        data_random_seed=505,
        feature_fraction_seed=505,
        bagging_seed=505,
        subsample=1,
        subsample_freq=0,
        colsample_bytree=1,
        max_bin=63,
        verbosity=-1,
        reg_alpha=0,
        reg_lambda=0,
        **params,
    )


def predict(model, x, task):
    if hasattr(model, "booster_"):
        return model.booster_.predict(x, num_threads=1)
    return model.predict(x) if task == "regression" else model.predict_proba(x)


def score(task, y, p):
    return spearman(p, y) if task == "regression" else classification_metrics(y, p)["macro_f1"]


def selection(values):
    s = np.where(np.isfinite(values), values, -np.inf)
    if not np.isfinite(s).any():
        raise ValueError("all validation scores undefined")
    return int(np.flatnonzero(s >= s.max() - 1e-12)[0])


def model_signature(model):
    h = hashlib.sha256()
    if hasattr(model, "booster_"):
        h.update(model.booster_.model_to_string().encode())
    else:
        for tree in model.estimators_:
            for name in ["children_left", "children_right", "feature", "threshold", "value"]:
                h.update(np.asarray(getattr(tree.tree_, name)).tobytes())
    return h.hexdigest()


def fit(xtrain, ytrain, xval, yval, names, family, task, fixed=None, random=False):
    whitelist(names, random)
    median = training_medians(xtrain)
    xt, xv = impute(xtrain, median), impute(xval, median)
    grid = RF_GRID if family == "RF" else LGB_GRID
    indices = range(len(grid)) if fixed is None else [fixed]
    models = []
    candidates = []
    for i in indices:
        m = make_model(family, task, grid[i])
        m.fit(xt, ytrain)
        if task == "classification":
            assert np.array_equal(m.classes_, CLASSES)
        value = score(task, yval, predict(m, xv, task))
        models.append(m)
        candidates.append(
            dict(
                candidate_id=i,
                validation_score=float(value),
                hyperparameters=json.dumps(grid[i], sort_keys=True),
            )
        )
    best = selection([r["validation_score"] for r in candidates])
    chosen = models[best]
    for j, row in enumerate(candidates):
        row["selected"] = j == best
    trainp = predict(chosen, xt, task)
    metadata = dict(
        family=family,
        task=task,
        features=list(names),
        median=median.tolist(),
        candidate_id=candidates[best]["candidate_id"],
        parameters=grid[candidates[best]["candidate_id"]],
        validation_score=candidates[best]["validation_score"],
        train_score=float(score(task, ytrain, trainp)),
        seed=505,
        feature_version="5.0.0",
        signature=model_signature(chosen),
        training_rows=len(ytrain),
        validation_rows=len(yval),
        train_prediction_mean=float(np.mean(trainp)) if task == "regression" else None,
        train_prediction_sd=float(np.std(trainp)) if task == "regression" else None,
    )
    return dict(model=chosen, metadata=metadata, candidates=candidates)


def run_fold(g, fold, diagnostics=True):
    g = canonical(g)
    part, purge = partitions(g, fold, 0)
    tr, va, te = [part[k] for k in ["train", "validation", "test"]]
    results = {}
    for task, target in [
        ("regression", "GRID_FWD_MID_RET_1S"),
        ("classification", "GRID_DIRECTION_1S"),
    ]:
        y = g[target]
        entries = {}
        for family in ["RF", "LightGBM"]:
            x = np.column_stack([g[f] for f in FEATURES])
            entry = fit(x[tr], y[tr], x[va], y[va], FEATURES, family, task)
            entry["prediction"] = predict(
                entry["model"], impute(x[te], entry["metadata"]["median"]), task
            )
            entries[family] = entry
        chosen = (
            "RF"
            if entries["RF"]["metadata"]["validation_score"]
            >= entries["LightGBM"]["metadata"]["validation_score"] - 1e-12
            else "LightGBM"
        )
        entries["ValidationSelected"] = entries[chosen]
        if task == "regression" and diagnostics:
            for name, names in [
                ("Static", STATIC),
                ("Flow", FLOW),
                ("RandomLabel", FEATURES),
                ("TemporalPlacebo", FEATURES),
                ("RandomFeature", FEATURES + ["RANDOM_CONTROL"]),
            ]:
                x = np.column_stack([g[f] for f in names])
                yt = (
                    block_permuted(y[tr], fold["fold_id"])
                    if name == "RandomLabel"
                    else np.roll(y[tr], 600)
                    if name == "TemporalPlacebo"
                    else y[tr]
                )
                fixed = (
                    None
                    if name in ["Static", "Flow"]
                    else entries["LightGBM"]["metadata"]["candidate_id"]
                )
                entry = fit(
                    x[tr],
                    yt,
                    x[va],
                    y[va],
                    names,
                    "LightGBM",
                    task,
                    fixed=fixed,
                    random=name == "RandomFeature",
                )
                entry["prediction"] = predict(
                    entry["model"], impute(x[te], entry["metadata"]["median"]), task
                )
                entries[name] = entry
        results[task] = entries
    return dict(fold=fold, partitions=part, purge=purge, results=results)


def compare(a, b, targets_a=None, targets_b=None):
    checks = 0
    for task in a["results"]:
        for name, left in a["results"][task].items():
            right = b["results"][task][name]
            assert left["metadata"] == right["metadata"]
            assert left["candidates"] == right["candidates"]
            np.testing.assert_allclose(left["prediction"], right["prediction"], rtol=0, atol=1e-12)
            if targets_a is not None:
                key = "GRID_FWD_MID_RET_1S" if task == "regression" else "GRID_DIRECTION_1S"
                np.testing.assert_allclose(
                    score(task, targets_a[key][a["partitions"]["test"]], left["prediction"]),
                    score(task, targets_b[key][b["partitions"]["test"]], right["prediction"]),
                    rtol=0,
                    atol=1e-12,
                )
            checks += 1
    return checks
