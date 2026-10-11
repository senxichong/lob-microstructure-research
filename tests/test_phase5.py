import numpy as np
import pytest
from test_phase4 import sample as source_sample
from threadpoolctl import threadpool_limits

from lob_research.phase3.inference import WeightedSpearman, block_indices
from lob_research.phase3.statistics import spearman
from lob_research.phase4.core import CLASSES, FEATURES, canonical, folds, partitions
from lob_research.phase5.core import (
    block_permuted,
    fit,
    impute,
    predict,
    selection,
    training_medians,
    whitelist,
)


@pytest.fixture
def sample():
    return source_sample.__wrapped__()


def test_phase5_same_fold_reuse(sample):
    g, f = sample
    p, _ = partitions(g, f)
    assert g["target_timestamp_1S"][p["test"]].max() < f["test_end"]
    assert len(folds()) == 5 and (folds()[0]["test_end"] - folds()[0]["test_start"]) == 2340 * 10**9


@pytest.mark.parametrize("family", ["RF", "LightGBM"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_phase5_reproducibility_test_isolation_and_row_order(sample, family, task):
    g, f = sample
    p, _ = partitions(g, f)
    tr, va, te = [p[k] for k in ["train", "validation", "test"]]
    x = np.column_stack([g[n] for n in FEATURES])
    key = "GRID_FWD_MID_RET_1S" if task == "regression" else "GRID_DIRECTION_1S"
    with threadpool_limits(limits=1):
        a = fit(x[tr], g[key][tr], x[va], g[key][va], FEATURES, family, task, fixed=0)
        g[key][te] = 0.99 if task == "regression" else "DOWN"
        # Future feature corruption starts strictly after the scored prefix.
        idx = np.random.default_rng(9).permutation(len(g["grid_timestamp"]))
        restored = canonical({k: v[idx] for k, v in g.items()})
        b = fit(x[tr], restored[key][tr], x[va], restored[key][va], FEATURES, family, task, fixed=0)
        assert a["metadata"] == b["metadata"]
        np.testing.assert_array_equal(
            predict(a["model"], impute(x[te], a["metadata"]["median"]), task),
            predict(b["model"], impute(x[te], b["metadata"]["median"]), task),
        )
    if task == "classification":
        assert np.array_equal(a["model"].classes_, CLASSES)
        probs = predict(a["model"], impute(x[te], a["metadata"]["median"]), task)
        np.testing.assert_allclose(probs.sum(1), 1, atol=1e-12)


def test_phase5_validation_only_selection():
    assert selection([0.1, 0.2, 0.2, 0.01]) == 1
    assert selection([np.nan, 0.1, 0.05]) == 1
    with pytest.raises(ValueError):
        selection([np.nan, np.nan])


def test_phase5_whitelist_future_trap(sample, monkeypatch):
    from lob_research.phase5 import core

    g, _ = sample
    invalid = np.column_stack([g[n] for n in FEATURES] + [g["GRID_FWD_MID_RET_1S"]])

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("A future-derived column reached model construction")

    monkeypatch.setattr(core, "make_model", forbidden_fit)
    with pytest.raises(ValueError, match="whitelist"):
        fit(
            invalid[:1200],
            g["GRID_FWD_MID_RET_1S"][:1200],
            invalid[1200:1800],
            g["GRID_FWD_MID_RET_1S"][1200:1800],
            FEATURES + ["FUTURE_RETURN_1S"],
            "LightGBM",
            "regression",
        )
    whitelist(FEATURES)
    with pytest.raises(ValueError, match="whitelist"):
        whitelist(FEATURES + ["FUTURE_RETURN_1S"])
    with pytest.raises(ValueError):
        whitelist(FEATURES + ["RANDOM_CONTROL"])
    whitelist(FEATURES + ["RANDOM_CONTROL"], random=True)


def test_phase5_training_only_median():
    p = training_medians(np.array([[1, np.nan], [3, 10], [5, 20]]))
    np.testing.assert_equal(p, [3, 15])
    np.testing.assert_equal(impute(np.array([[np.nan, 1e10]]), p), [[3, 1e10]])


def test_phase5_random_training_block_control():
    y = np.arange(601)
    actual = block_permuted(y, 1)
    np.testing.assert_array_equal(np.sort(actual), y)
    np.testing.assert_array_equal(actual, block_permuted(y, 1))
    assert not np.array_equal(actual, y)
    assert sum(np.diff(actual) == 1) >= 580


def test_phase5_paired_clock_bootstrap_with_holes():
    x = np.arange(121, dtype=float)
    y = np.sin(x)
    z = -x
    y[60] = np.nan
    ix = block_indices(len(x), 60, np.random.default_rng(505060))
    w = np.bincount(ix, minlength=len(x))
    delta = WeightedSpearman(x, y).evaluate(w) - WeightedSpearman(z, y).evaluate(w)
    assert delta == pytest.approx(spearman(x[ix], y[ix]) - spearman(z[ix], y[ix]))


def test_phase5_model_serialization(tmp_path, sample):
    import joblib

    g, _ = sample
    x = np.column_stack([g[n] for n in FEATURES])
    y = g["GRID_FWD_MID_RET_1S"]
    a = fit(
        x[:1200], y[:1200], x[1200:1800], y[1200:1800], FEATURES, "LightGBM", "regression", fixed=0
    )
    path = tmp_path / "model.joblib"
    joblib.dump(a, path)
    b = joblib.load(path)
    np.testing.assert_array_equal(
        predict(a["model"], x[1800:], "regression"), predict(b["model"], x[1800:], "regression")
    )


def test_phase5_prediction_serialization_and_future_invariance(tmp_path, monkeypatch, sample):
    from lob_research.phase5 import storage
    from lob_research.phase5.core import compare, run_fold

    g, f = sample
    f = dict(f, test_end=2100 * 10**9)
    with threadpool_limits(limits=1):
        result = run_fold(g, f, diagnostics=False)
        changed = {k: v.copy() for k, v in g.items()}
        for name in FEATURES:
            changed[name][2101:] = 1e15
        future = run_fold(changed, f, diagnostics=False)
        compare(result, future, g, changed)
    monkeypatch.setattr(storage, "OUT", tmp_path)
    monkeypatch.setattr(storage, "ART", tmp_path / "models")
    storage.persist(g, [result])
    for task in ["regression", "classification"]:
        for name, data in storage.groups(task).items():
            np.testing.assert_array_equal(
                data["grid_timestamp"], g["grid_timestamp"][result["partitions"]["test"]]
            )
            assert np.all(data["target_timestamp"] < f["test_end"])
            if task == "regression":
                np.testing.assert_array_equal(
                    data["prediction"], result["results"][task][name]["prediction"]
                )
