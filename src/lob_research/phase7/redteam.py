"""Adversarial tests of production entry points; never changes research artifacts."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from .common import CHECKS, arrays, close, dump, js, record, write
from .statistical_audit import FEATURES


def sample():
    rng = np.random.default_rng(707077)
    n = 2400
    t = np.arange(n, dtype=np.int64) * 10**9
    g = {f: rng.normal(size=n) for f in FEATURES + ["RANDOM_CONTROL"]}
    y = np.where(abs(g["OBI_L1"]) > 0.6, np.sign(g["OBI_L1"]) * 0.001, 0)
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


def future_trap():
    from lob_research.phase5 import core

    # Read-only real future-midprice return payload; deliberately invalid as a feature.
    # The whitelist must reject it before either training or model construction.
    g = arrays(js("PHASE3_MANIFEST.json")["output_path"])
    x = np.column_stack([g[k] for k in FEATURES] + [g["GRID_FWD_MID_RET_1S"]])

    def forbidden(*a, **kw):
        raise AssertionError("leaked feature reached model construction")

    with patch.object(core, "make_model", forbidden):
        try:
            core.fit(
                x[:1200],
                g["GRID_FWD_MID_RET_1S"][:1200],
                x[1200:1800],
                g["GRID_FWD_MID_RET_1S"][1200:1800],
                FEATURES + ["FUTURE_PRICE_1S"],
                "LightGBM",
                "regression",
            )
        except ValueError as e:
            if "whitelist" not in str(e):
                raise
        else:
            raise AssertionError("leak trap accepted")
    return True


def model_isolation():
    from lob_research.phase5.core import run_fold

    g, f = sample()
    changed = {k: v.copy() for k, v in g.items()}
    test = g["grid_timestamp"] >= f["test_start"]
    changed["GRID_FWD_MID_RET_1S"][test] = np.random.default_rng(707078).normal(size=test.sum())
    changed["GRID_DIRECTION_1S"][test] = np.random.default_rng(707079).choice(
        ["DOWN", "FLAT", "UP"], test.sum()
    )
    idx = np.random.default_rng(707080).permutation(len(test))
    shuffled = {k: v[idx] for k, v in changed.items()}
    with threadpool_limits(limits=1):
        a = run_fold(g, f, diagnostics=False)
        b = run_fold(shuffled, f, diagnostics=False)
    for task, group in a["results"].items():
        for name, left in group.items():
            right = b["results"][task][name]
            if left["metadata"] != right["metadata"] or left["candidates"] != right["candidates"]:
                raise AssertionError("test mutation affected fitting")
            np.testing.assert_array_equal(left["prediction"], right["prediction"])
    return True


def future_mutations():
    # Production functions are deliberately the subject, not the independent reference.
    from lob_research.phase2.common import load_stream
    from lob_research.phase2.features import build_features
    from lob_research.phase2.labels import build_labels

    cfg = yaml.safe_load(Path("config.yaml").read_text())
    s = load_stream(cfg)
    options = cfg["phase2"]
    a, _ = build_features(s, options)
    la, _ = build_labels(s, options)
    rng = np.random.default_rng(707008)
    cuts = np.sort(rng.choice(np.arange(20000, len(s.time) - 20000), 5, replace=False))
    rows = []
    for c in cuts:
        for variant in ["payload", "drop", "reorder_future_payload"]:
            z = s.copy()
            if variant == "payload":
                shift = rng.integers(100000, 200000)
                z.bid[c + 1 :] += shift
                z.ask[c + 1 :] += shift
                z.bsize[c + 1 :] = rng.integers(1, 10000, z.bsize[c + 1 :].shape)
                z.asize[c + 1 :] = rng.integers(1, 10000, z.asize[c + 1 :].shape)
                z.kind[c + 1 :] = "A"
                z.side[c + 1 :] = "B"
                z.size[c + 1 :] = rng.integers(1, 10000, len(s.time) - c - 1)
            elif variant == "drop":
                z = z.take(slice(0, c + 1))
            else:
                order = np.r_[np.arange(c + 1), rng.permutation(np.arange(c + 1, len(s.time)))]
                for key in ["bid", "ask", "bsize", "asize", "kind", "side", "size"]:
                    setattr(z, key, getattr(z, key)[order])
            b, _ = build_features(z, options)
            lb, _ = build_labels(z, options)
            for key in a:
                close(
                    "production_mutation", f"{c}/{variant}/{key}", a[key][: c + 1], b[key][: c + 1]
                )
            for tag, h in [("100ms", 100), ("1s", 1000), ("5s", 5000), ("10s", 10000)]:
                settled = s.time[: c + 1] + h * 10**6 < s.time[c]
                for key in la:
                    if not key.lower().endswith("_" + tag):
                        continue
                    aa, bb = la[key], lb[key]
                    for j, (v, w) in enumerate(
                        zip(aa, bb) if isinstance(aa, tuple) else [(aa, bb)]
                    ):
                        close(
                            "production_mutation",
                            f"{c}/{variant}/settled/{key}/{j}",
                            v[: c + 1][settled],
                            w[: c + 1][settled],
                        )
            rows.append(
                dict(
                    cutoff_index=int(c),
                    variant=variant,
                    past_feature_columns=len(a),
                    settled_labels_compared=True,
                )
            )
    # Physical row order corruptions must fail, then restore using both causal keys.
    order = np.random.default_rng(707081).permutation(len(s.time))
    bad = s.take(order)
    try:
        build_features(bad, options)
    except ValueError as e:
        record("redteam", "physical unordered source rejected", "unordered" in str(e))
    else:
        record("redteam", "physical unordered source rejected", False)
    restored = bad.take(np.lexsort((bad.order, bad.time)))
    rebuilt, _ = build_features(restored, options)
    for k in a:
        close("redteam", "restored real source " + k, a[k], rebuilt[k])
    # A later member of a real timestamp tie may never enter its predecessor's feature prefix.
    ties = np.flatnonzero(np.diff(s.time) == 0) + 1
    chosen = ties[np.linspace(0, len(ties) - 1, 8, dtype=int)]
    for i in chosen:
        prefix, _ = build_features(s.take(slice(0, i)), options)
        for k in a:
            close("redteam", f"tie {i} prefix " + k, prefix[k], a[k][:i])
        malformed = s.take(np.r_[np.arange(i - 1), i, i - 1, np.arange(i + 1, len(s.time))])
        try:
            build_features(malformed, options)
        except ValueError:
            record("redteam", f"tie {i} reordered source keys rejected", True)
        else:
            record("redteam", f"tie {i} reordered source keys rejected", False)
    write("production_future_mutations", rows)


def run():
    record("redteam", "future leaking column rejected before construction", future_trap())
    record(
        "redteam",
        "test labels plus physical grid shuffle leave fitted metadata/candidates/predictions identical",
        model_isolation(),
        detail="Synthetic folds only; both tasks, RF/LightGBM/validation-selected families, all frozen candidate grids.",
    )
    future_mutations()
    dump("redteam_checks.json", CHECKS)
    write("adversarial_controls", CHECKS)
