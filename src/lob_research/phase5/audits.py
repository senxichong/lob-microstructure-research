import numpy as np

from lob_research.phase3.artifacts import configuration, load_phase2
from lob_research.phase3.grid import build_grid, epoch_ns
from lob_research.phase4.audits import mutate_test
from lob_research.phase4.core import FEATURES, canonical

from .core import compare, run_fold, score, whitelist


def real_audits(g, original):
    fold = original["fold"]
    cfg = configuration()
    source = load_phase2(cfg)
    future = source["timestamp"] > fold["test_end"]
    altered = {k: v.copy() for k, v in source.items()}
    changed = []
    for name, v in altered.items():
        if name in ["timestamp", "source_order", "event_index"]:
            continue
        if v.dtype.kind == "b":
            v[future] = ~v[future]
        elif v.dtype.kind in "iu":
            v[future] += 17
        elif v.dtype.kind == "f":
            v[future] = -7 * np.nan_to_num(v[future]) + 123.456
        else:
            v[future] = "MUTATED"
        changed.append(name)
    args = (FEATURES, epoch_ns(cfg["session_start_utc"]), epoch_ns(cfg["session_end_utc"]))
    rebuilt = build_grid(altered, *args, horizons=(1000,))
    rebuilt["RANDOM_CONTROL"] = g["RANDOM_CONTROL"].copy()
    a = run_fold(rebuilt, fold)
    n = compare(original, a, g, rebuilt)
    mutated = mutate_test(g, fold)
    b = run_fold(mutated, fold)
    n += compare(original, b)
    for task, key in [
        ("regression", "GRID_FWD_MID_RET_1S"),
        ("classification", "GRID_DIRECTION_1S"),
    ]:
        te = original["partitions"]["test"]
        # Regression Spearman alone is invariant to affine target changes, so
        # audit squared error as well as classifier Macro F1.
        p = original["results"][task]["LightGBM"]["prediction"]
        if task == "regression":
            assert np.mean((g[key][te] - p) ** 2) != np.mean((mutated[key][te] - p) ** 2)
        else:
            assert score(task, g[key][te], p) != score(task, mutated[key][te], p)
    idx = np.random.default_rng(505034).permutation(len(source["timestamp"]))
    s = canonical({k: v[idx] for k, v in source.items()}, source=True)
    rg = build_grid(s, *args, horizons=(1000,))
    rg["RANDOM_CONTROL"] = g["RANDOM_CONTROL"].copy()
    idx = np.random.default_rng(505034).permutation(len(g["grid_timestamp"]))
    c = run_fold({k: v[idx] for k, v in rg.items()}, fold)
    n += compare(original, c, g, rg)
    try:
        whitelist(FEATURES + ["FUTURE_RETURN_1S"])
    except ValueError:
        rejected = True
    else:
        raise AssertionError("future feature was accepted")
    return dict(
        status="PASS",
        checks=n,
        future_rows_mutated=int(future.sum()),
        payload_columns_mutated=changed,
        preserved_keys=["timestamp", "source_order", "event_index"],
        future_mutation="PASS",
        test_label_isolation="PASS",
        evaluation_changed=True,
        row_order_invariance="PASS",
        future_feature_rejected=rejected,
    )
