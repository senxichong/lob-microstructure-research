"""Actual reruns for isolation, chronology and repeatability, not mocked flags."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lob_research.phase3.artifacts import configuration, load_phase2
from lob_research.phase3.grid import build_grid, epoch_ns

from .core import FEATURES, canonical, classification_metrics, regression_metrics, run_fold


def compare(left, right, include_metrics=False, g_left=None, g_right=None):
    checks = 0
    assert left["artifacts"] == right["artifacts"]
    checks += 1
    np.testing.assert_array_equal(left["test_indices"], right["test_indices"])
    checks += 1
    for task in ("regression", "classification"):
        assert left[task].keys() == right[task].keys()
        for name in left[task]:
            np.testing.assert_allclose(left[task][name], right[task][name], rtol=0, atol=1e-12)
            checks += 1
            if include_metrics:
                key = "GRID_FWD_MID_RET_1S" if task == "regression" else "GRID_DIRECTION_1S"
                metric = regression_metrics if task == "regression" else classification_metrics
                a = metric(g_left[key][left["test_indices"]], left[task][name])
                b = metric(g_right[key][right["test_indices"]], right[task][name])
                np.testing.assert_allclose(
                    list(a.values()), list(b.values()), rtol=0, atol=1e-12, equal_nan=True
                )
                checks += 1
    return checks


def mutate_test(g, fold):
    changed = {k: v.copy() for k, v in g.items()}
    mask = (g["grid_timestamp"] >= fold["test_start"]) & (g["grid_timestamp"] < fold["test_end"])
    changed["GRID_FWD_MID_RET_1S"][mask] = -g["GRID_FWD_MID_RET_1S"][mask] + 0.01
    original = g["GRID_DIRECTION_1S"][mask]
    changed["GRID_DIRECTION_1S"][mask] = np.where(original == "UP", "DOWN", "UP")
    return changed


def real_audits(g, earlier):
    fold = earlier["fold"]
    cfg = configuration()
    source = load_phase2(cfg)
    start, end = epoch_ns(cfg["session_start_utc"]), epoch_ns(cfg["session_end_utc"])
    rebuilt = build_grid(source, FEATURES, start, end, horizons=(1000,))
    rebuilt["RANDOM_CONTROL"] = g["RANDOM_CONTROL"].copy()
    for f in FEATURES + ["GRID_FWD_MID_RET_1S"]:
        np.testing.assert_equal(rebuilt[f], g[f])
    # Read ALL Phase 2 columns. Mutate every later row's payload; preserve only
    # authoritative timestamp/source_order/event_index keys for valid chronology.
    raw = pq.read_table(cfg["input_matrix"])
    future = raw["timestamp"].cast(pa.int64()).to_numpy() > fold["test_end"]
    changed_cols = {}
    touched = []
    for name in raw.column_names:
        col = raw[name]
        if name in {"timestamp", "source_order", "event_index"}:
            changed_cols[name] = col
            continue
        values = col.to_pylist()
        typ = col.type
        if pa.types.is_timestamp(typ):
            ints = col.cast(pa.int64()).to_pylist()
            for i in np.flatnonzero(future):
                ints[i] = end + 10**9
            changed_cols[name] = pa.array(ints, type=pa.int64()).cast(typ)
        else:
            for i in np.flatnonzero(future):
                old = values[i]
                if pa.types.is_boolean(typ):
                    values[i] = not bool(old)
                elif pa.types.is_integer(typ):
                    values[i] = 17 if old is None else old + 17
                elif pa.types.is_floating(typ):
                    values[i] = 123.456 if old is None else -old * 7 + 123.456
                else:
                    values[i] = "MUTATED_FUTURE"
            changed_cols[name] = pa.array(values, type=typ)
        touched.append(name)
    mutated = pa.table(changed_cols)
    changed_source = {}
    for name in source:
        col = mutated[name]
        changed_source[name] = (
            col.cast(pa.int64()).to_numpy() if pa.types.is_timestamp(col.type) else col.to_numpy()
        )
    future_grid = build_grid(changed_source, FEATURES, start, end, horizons=(1000,))
    future_grid["RANDOM_CONTROL"] = g["RANDOM_CONTROL"].copy()
    future_result = run_fold(future_grid, fold)
    checks = compare(earlier, future_result, True, g, future_grid)
    test_changed = mutate_test(g, fold)
    test_result = run_fold(test_changed, fold)
    checks += compare(earlier, test_result)
    idx = earlier["test_indices"]
    assert regression_metrics(
        g["GRID_FWD_MID_RET_1S"][idx], earlier["regression"]["Ridge"]
    ) != regression_metrics(
        test_changed["GRID_FWD_MID_RET_1S"][idx], test_result["regression"]["Ridge"]
    )
    assert classification_metrics(
        g["GRID_DIRECTION_1S"][idx], earlier["classification"]["Logistic"]
    ) != classification_metrics(
        test_changed["GRID_DIRECTION_1S"][idx], test_result["classification"]["Logistic"]
    )
    rng = np.random.default_rng(404035)
    indices = rng.permutation(len(source["timestamp"]))
    shuffled_source = canonical({k: v[indices] for k, v in source.items()}, source=True)
    shuffled_grid = build_grid(shuffled_source, FEATURES, start, end, horizons=(1000,))
    shuffled_grid["RANDOM_CONTROL"] = g["RANDOM_CONTROL"].copy()
    indices = rng.permutation(len(g["grid_timestamp"]))
    shuffled_result = run_fold({k: v[indices] for k, v in shuffled_grid.items()}, fold)
    checks += compare(earlier, shuffled_result, True, g, shuffled_grid)
    return dict(
        status="PASS",
        numerical_checks=checks,
        future_mutated_rows=int(future.sum()),
        future_mutated_columns=len(touched),
        preserved_keys=["timestamp", "source_order", "event_index"],
        future_mutation="PASS",
        test_target_mutation="PASS",
        test_metrics_changed=True,
        row_order_invariance="PASS",
        shuffle_seed=404035,
        tolerance=1e-12,
    )
