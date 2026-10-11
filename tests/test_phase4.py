import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from lob_research.phase4.audits import compare, mutate_test
from lob_research.phase4.core import (
    CLASSES,
    FEATURES,
    Preprocess,
    canonical,
    classification_metrics,
    fit_model,
    folds,
    partitions,
    regression_metrics,
    run_fold,
    select,
)


@pytest.fixture
def sample():
    rng = np.random.default_rng(907)
    n = 2400
    t = np.arange(n, dtype=np.int64) * 10**9
    g = {f: rng.normal(size=n) for f in FEATURES + ["RANDOM_CONTROL"]}
    y = np.where(np.abs(g["OBI_L1"]) > 0.6, np.sign(g["OBI_L1"]) * 0.001, 0.0)
    g.update(
        grid_timestamp=t,
        source_state_timestamp=t - 1000,
        target_timestamp_1S=t + 10**9,
        target_available_1S=np.ones(n, bool),
        grid_state_available=np.ones(n, bool),
        GRID_FWD_MID_RET_1S=y,
        GRID_DIRECTION_1S=np.where(y > 0, "UP", np.where(y < 0, "DOWN", "FLAT")),
    )
    f = dict(
        fold_id=1,
        train_start=0,
        train_end=1200 * 10**9,
        validation_start=1200 * 10**9,
        validation_end=1800 * 10**9,
        test_start=1800 * 10**9,
        test_end=2400 * 10**9,
    )
    return g, f


def test_phase4_frozen_equal_wallclock_boundaries():
    fs = folds()
    assert len(fs) == 5
    for i, f in enumerate(fs):
        assert f["train_end"] == f["validation_start"]
        assert f["validation_end"] == f["test_start"]
        assert f["test_end"] - f["test_start"] == 2340 * 10**9
        assert f["train_end"] - f["train_start"] == (4 + i) * 2340 * 10**9


@pytest.mark.parametrize("embargo", [0, 10])
def test_phase4_purge_uses_target_time_not_row_count(sample, embargo):
    g, f = sample
    g["target_timestamp_1S"][10] = f["validation_start"]
    p, a = partitions(g, f, embargo)
    assert 10 not in p["train"]
    assert a[0]["purged"] == 2 + embargo
    assert g["target_timestamp_1S"][p["train"]].max() < f["validation_start"] - embargo * 10**9
    assert g["target_timestamp_1S"][p["validation"]].max() < f["test_start"] - embargo * 10**9
    assert len(set(np.concatenate(list(p.values())))) == sum(map(len, p.values()))


def test_phase4_training_only_imputer_scaler():
    train = np.array([[1.0, np.nan], [3.0, 10.0], [5.0, 20.0]])
    p = Preprocess().fit(train)
    np.testing.assert_array_equal(p.median, [3, 15])
    np.testing.assert_array_equal(p.mean, [3, 15])
    np.testing.assert_allclose(p.scale, np.std([[1, 15], [3, 10], [5, 20]], axis=0))
    before = p.mean.copy()
    p.transform(np.array([[1e20, np.nan]]))
    np.testing.assert_array_equal(before, p.mean)
    np.testing.assert_array_equal(p.transform(np.array([[np.nan, np.nan]])), [[0, 0]])


def test_phase4_selection_ties_and_validation_only():
    assert select([0.1, 0.1, 0.09], [0.1, 1, 10], True) == 1
    assert select([0.1, 0.1, 0.09], [0.1, 1, 10], False) == 0
    rng = np.random.default_rng(51)
    x = rng.normal(size=(100, 2))
    y = x[:, 0] + rng.normal(size=100)
    _, _, a = fit_model(x[:80], y[:80], x[80:], y[80:], ["a", "b"])
    scores = [v["score"] for v in a["validation_scores"]]
    assert (
        a["parameter"]
        == a["validation_scores"][
            select(scores, [v["parameter"] for v in a["validation_scores"]], True)
        ]["parameter"]
    )


def test_phase4_test_label_isolation_and_reproducibility(sample):
    g, f = sample
    with threadpool_limits(limits=1):
        original = run_fold(g, f, variants=False)
        altered = run_fold(mutate_test(g, f), f, variants=False)
        compare(original, altered)
        repeat = run_fold(g, f, variants=False)
        compare(original, repeat, True, g, g)
    te = original["test_indices"]
    assert (
        regression_metrics(g["GRID_FWD_MID_RET_1S"][te], original["regression"]["Ridge"])["mse"]
        < 0.01
    )


def test_phase4_future_feature_and_row_order_isolation(sample):
    g, f = sample
    f = dict(f, test_end=2100 * 10**9)
    with threadpool_limits(limits=1):
        original = run_fold(g, f, variants=False)
        changed = {k: v.copy() for k, v in g.items()}
        for name in FEATURES:
            changed[name][2100:] = 1e30
        altered = run_fold(changed, f, variants=False)
        compare(original, altered, True, g, changed)
        idx = np.random.default_rng(51).permutation(len(g["grid_timestamp"]))
        shuffled = run_fold({k: v[idx] for k, v in g.items()}, f, variants=False)
        compare(original, shuffled)


def test_phase4_random_labels_only_from_training(sample):
    g, f = sample
    with threadpool_limits(limits=1):
        result = run_fold(g, f)
        p, _ = partitions(g, f)
        x = np.column_stack([g[n] for n in FEATURES])
        _, _, a = fit_model(
            x[p["train"]],
            np.roll(g["GRID_FWD_MID_RET_1S"][p["train"]], 600),
            x[p["validation"]],
            g["GRID_FWD_MID_RET_1S"][p["validation"]],
            FEATURES,
            fixed=result["artifacts"]["Ridge"]["parameter"],
        )
    assert a["coefficient"] == result["artifacts"]["RandomLabel"]["coefficient"]
    assert result["artifacts"]["RandomLabel"]["label_rotation_seconds"] == 600


def test_phase4_confusion_natural_classes_and_probability_metrics():
    y = np.array(["DOWN", "FLAT", "FLAT", "UP"])
    p = np.tile([0.0, 1.0, 0.0], (4, 1))
    m = classification_metrics(y, p)
    assert m["accuracy"] == 0.5
    assert m["balanced_accuracy"] == pytest.approx(1 / 3)
    assert m["macro_f1"] == pytest.approx(2 / 9)
    assert np.isfinite(m["log_loss"])
    assert CLASSES.tolist() == ["DOWN", "FLAT", "UP"]


@pytest.mark.parametrize("value", [0.1, 0.0000017])
def test_phase4_constant_baseline_pearson_is_undefined(value):
    y = np.linspace(-0.001, 0.001, 2339)
    p = np.full(len(y), value)
    metrics = regression_metrics(y, p)
    assert np.isnan(metrics["pearson"])
    assert np.isnan(metrics["spearman"])


def test_phase4_duplicate_grid_refused(sample):
    g, _ = sample
    g["grid_timestamp"][1] = g["grid_timestamp"][0]
    with pytest.raises(ValueError, match="duplicate"):
        canonical(g)


def test_phase4_saved_prediction_tagging(tmp_path, monkeypatch, sample):
    from lob_research.phase4 import storage

    g, f = sample
    monkeypatch.setattr(storage, "OUT", tmp_path)
    with threadpool_limits(limits=1):
        result = run_fold(g, f, variants=False)
    storage.persist(g, [result])
    for task in ("regression", "classification"):
        for (_, name), d in storage.prediction_groups(task).items():
            assert (d["grid_timestamp"] >= f["test_start"]).all()
            assert (d["target_timestamp"] < f["test_end"]).all()
            assert set(d["fold_id"]) == {1}
            assert set(d["feature_version"]) == {"4.0.0"}
            assert set(d["model_name"]) == {name}
