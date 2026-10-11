"""Partition-aware training: evaluation labels never enter fitting APIs."""

import csv
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from lob_research.phase3.grid import epoch_ns
from lob_research.phase3.statistics import pearson, spearman

FEATURES = [
    "OBI_L1",
    "MICROPRICE_MINUS_MID",
    "OFI_100MS",
    "OFI_1S",
    "RELATIVE_SPREAD",
    "EVENT_RATE_1S",
    "TIME_SINCE_PREVIOUS_TRADE",
]
STATIC = [FEATURES[i] for i in (0, 1, 4)]
FLOW = [FEATURES[i] for i in (2, 3, 5, 6)]
CLASSES = np.array(["DOWN", "FLAT", "UP"])
ALPHAS = [0, 0.01, 0.1, 1, 10, 100]
CS = [0.01, 0.1, 1, 10]
GRID = "data/processed/phase3/msft_20030103_grid_v3.0.0.parquet"
OUT = Path("reports/phase4")


def canonical(data, source=False):
    key = "timestamp" if source else "grid_timestamp"
    idx = np.lexsort((data["source_order"], data[key])) if source else np.argsort(data[key])
    out = {k: np.asarray(v)[idx].copy() for k, v in data.items()}
    if source:
        if np.any(np.diff(out["source_order"]) <= 0):
            raise ValueError("duplicate or nonauthoritative source order")
    elif np.any(np.diff(out[key]) <= 0):
        raise ValueError("duplicate grid timestamps")
    return out


def folds():
    with (OUT / "fold_manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            k: (int(v) if k == "fold_id" else epoch_ns(v))
            for k, v in r.items()
            if k == "fold_id" or k.endswith(("_start", "_end"))
        }
        for r in rows
    ]


def partitions(g, fold, embargo=0):
    t, tau = g["grid_timestamp"], g["target_timestamp_1S"]
    valid = g["target_available_1S"] & g["grid_state_available"]
    result, audit = {}, []
    for name in ("train", "validation", "test"):
        start, end = fold[name + "_start"], fold[name + "_end"]
        gap = embargo if name != "test" else 0
        candidate = (t >= start) & (t < end)
        available = candidate & valid
        kept = available & (tau < end - gap * 10**9)
        idx = np.flatnonzero(kept)
        if not len(idx):
            raise ValueError("empty partition")
        assert tau[idx].max() < end - gap * 10**9
        result[name] = idx
        audit.append(
            dict(
                fold_id=fold["fold_id"],
                embargo_seconds=embargo,
                partition=name,
                clock_rows=int(candidate.sum()),
                unavailable=int((candidate & ~valid).sum()),
                purged=int((available & ~kept).sum()),
                retained=len(idx),
                max_target_timestamp=int(tau[idx].max()),
                boundary_timestamp=end,
                information_cutoff=end - gap * 10**9,
            )
        )
    assert not set(result["train"]) & set(result["validation"])
    assert not set(result["train"]) & set(result["test"])
    assert not set(result["validation"]) & set(result["test"])
    return result, audit


class Preprocess:
    def fit(self, x):
        x = np.where(np.isfinite(x), x, np.nan)
        self.median = np.nanmedian(x, axis=0)
        if not np.isfinite(self.median).all():
            raise ValueError("all-missing training feature")
        z = np.where(np.isfinite(x), x, self.median)
        self.mean = z.mean(axis=0)
        self.scale = z.std(axis=0)
        self.scale[self.scale == 0] = 1
        return self

    def transform(self, x):
        return (np.where(np.isfinite(x), x, self.median) - self.mean) / self.scale


def regression_metrics(y, p):
    e = y - p
    denom = np.sum((y - y.mean()) ** 2)
    return dict(
        n=len(y),
        spearman=spearman(p, y),
        pearson=pearson(p, y),
        mae=float(np.mean(np.abs(e))),
        mse=float(np.mean(e**2)),
        r2=float(1 - np.sum(e**2) / denom) if denom else np.nan,
    )


def confusion(y, pred, weights=None):
    yi = np.searchsorted(CLASSES, y)
    pi = np.searchsorted(CLASSES, pred)
    return np.bincount(3 * yi + pi, weights=weights, minlength=9).reshape(3, 3)


def cm_metrics(cm):
    tp = np.diag(cm)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) != 0)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) != 0)
    f1 = np.divide(2 * tp, cm.sum(0) + cm.sum(1), out=np.zeros(3), where=cm.sum(0) + cm.sum(1) != 0)
    return (
        dict(
            macro_f1=float(f1.mean()),
            balanced_accuracy=float(recall.mean()),
            accuracy=float(tp.sum() / cm.sum()),
        ),
        precision,
        recall,
        f1,
    )


def classification_metrics(y, probs):
    pred = CLASSES[np.argmax(probs, axis=1)]
    cm = confusion(y, pred)
    m, prec, rec, f1 = cm_metrics(cm)
    p = np.clip(probs, np.finfo(float).eps, 1 - np.finfo(float).eps)
    p /= p.sum(1, keepdims=True)
    m.update(
        n=len(y), log_loss=float(-np.log(p[np.arange(len(y)), np.searchsorted(CLASSES, y)]).mean())
    )
    for i, c in enumerate(CLASSES):
        m.update(
            {
                f"{c}_precision": float(prec[i]),
                f"{c}_recall": float(rec[i]),
                f"{c}_f1": float(f1[i]),
            }
        )
    return m


def select(scores, parameters, prefer_large):
    a = np.asarray(scores)
    if not np.isfinite(a).all():
        raise ValueError("undefined validation metric")
    choices = np.flatnonzero(a >= a.max() - 1e-12)
    return (
        max(choices, key=lambda i: parameters[i])
        if prefer_large
        else min(choices, key=lambda i: parameters[i])
    )


def linear(alpha):
    return LinearRegression() if alpha == 0 else Ridge(alpha=alpha, solver="svd")


def fit_model(x_train, y_train, x_validation, y_validation, names, kind="Ridge", fixed=None):
    """No test data accepted by this function, including feature preprocessing."""
    prep = Preprocess().fit(x_train)
    xt, xv = prep.transform(x_train), prep.transform(x_validation)
    params = CS if kind == "Logistic" else ([0] if kind == "OLS" else ALPHAS)
    if fixed is not None:
        params = [fixed]
    models, scores = [], []
    for param in params:
        if kind == "Logistic":
            if set(y_train) != set(CLASSES):
                raise ValueError("require all training classes")
            model = LogisticRegression(
                C=param,
                solver="lbfgs",
                max_iter=2000,
                tol=1e-8,
                random_state=404,
                class_weight=None,
            )
        else:
            model = linear(param)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(xt, y_train)
        if kind == "Logistic":
            assert np.array_equal(model.classes_, CLASSES)
            score = classification_metrics(y_validation, model.predict_proba(xv))["macro_f1"]
        else:
            score = spearman(model.predict(xv), y_validation)
        models.append(model)
        scores.append(score)
    chosen = select(scores, params, prefer_large=kind != "Logistic")
    model = models[chosen]
    artifact = dict(
        features=list(names),
        kind=kind,
        parameter=float(params[chosen]),
        median=prep.median.tolist(),
        mean=prep.mean.tolist(),
        scale=prep.scale.tolist(),
        coefficient=np.asarray(model.coef_).tolist(),
        intercept=np.asarray(model.intercept_).tolist(),
        validation_scores=[
            dict(parameter=float(p), score=float(s), selected=i == chosen)
            for i, (p, s) in enumerate(zip(params, scores))
        ],
        iterations=np.asarray(getattr(model, "n_iter_", [])).tolist(),
    )
    return model, prep, artifact


def run_fold(g, fold, embargo=0, variants=True):
    g = canonical(g)
    part, audit = partitions(g, fold, embargo)
    tr, va, te = (part[k] for k in ("train", "validation", "test"))
    y, c = g["GRID_FWD_MID_RET_1S"], g["GRID_DIRECTION_1S"]
    definitions = {"OLS": FEATURES, "Ridge": FEATURES}
    if variants:
        definitions.update(
            Static=STATIC,
            Flow=FLOW,
            WithoutImbalance=FEATURES[1:],
            WithoutMicroprice=[f for f in FEATURES if f != FEATURES[1]],
            WithoutOFI=[f for f in FEATURES if f not in FEATURES[2:4]],
            WithoutContext=FEATURES[:4],
            RandomFeature=FEATURES + ["RANDOM_CONTROL"],
            RandomLabel=FEATURES,
        )
    reg, cls, artifacts = {}, {}, {}
    for name, names in definitions.items():
        x = np.column_stack([g[f] for f in names])
        yt = np.roll(y[tr], 600) if name == "RandomLabel" else y[tr]
        model, prep, artifact = fit_model(
            x[tr],
            yt,
            x[va],
            y[va],
            names,
            kind="OLS" if name == "OLS" else "Ridge",
            fixed=artifacts["Ridge"]["parameter"] if name == "RandomLabel" else None,
        )
        reg[name] = model.predict(prep.transform(x[te]))
        artifact["label_rotation_seconds"] = 600 if name == "RandomLabel" else 0
        artifacts[name] = artifact
    x = np.column_stack([g[f] for f in FEATURES])
    model, prep, artifact = fit_model(x[tr], c[tr], x[va], c[va], FEATURES, kind="Logistic")
    cls["Logistic"] = model.predict_proba(prep.transform(x[te]))
    artifacts["Logistic"] = artifact
    reg["ZeroReturn"] = np.zeros(len(te))
    reg["TrainingMean"] = np.full(len(te), y[tr].mean())
    for name in (FEATURES[0], FEATURES[1], FEATURES[3]):
        reg[name] = g[name][te]
    cls["AlwaysFLAT"] = np.tile([0.0, 1.0, 0.0], (len(te), 1))
    cls["TrainingPrior"] = np.tile([np.mean(c[tr] == v) for v in CLASSES], (len(te), 1))
    return dict(
        regression=reg,
        classification=cls,
        artifacts=artifacts,
        test_indices=te,
        purge=audit,
        fold=fold,
        embargo=embargo,
    )
