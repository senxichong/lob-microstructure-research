"""Adversarial regressions for the independent audit and frozen research entry points."""

import numpy as np
import pytest
from phase2_helpers import settings, stream

from lob_research.phase2.features import build_features
from lob_research.phase6.core import Book, replay
from lob_research.phase7.common import arrays, corr, js, predictions, readcsv, sha
from lob_research.phase7.execution_audit import MS, NS, Quotes, simulate, validation_thresholds
from lob_research.phase7.redteam import future_trap, model_isolation


def fixture_books():
    t = np.array([0, 990, 1000, 1000, 1001, 2000, 2999]) * MS
    b = np.array([100, 100, 101, 102, 90, 103, 104]) * 10000
    source = dict(
        timestamp=t,
        source_row=np.arange(7),
        bid_price_units_1=b,
        ask_price_units_1=b + 20000,
        bid_size_1=np.full(7, 10),
        ask_size_1=np.full(7, 10),
    )
    production = Book(
        dict(
            timestamp=t,
            source_order=np.arange(7),
            best_bid_price=b / 10000,
            best_ask_price=(b + 20000) / 10000,
            best_bid_size=np.full(7, 10),
            best_ask_size=np.full(7, 10),
            book_valid=np.ones(7, bool),
        ),
        0,
        3 * NS,
    )
    return Quotes(source), production


def test_phase7_strict_validation_threshold_provenance():
    a = arrays("reports/phase6/validation_predictions.parquet")
    different = 0
    for r in readcsv("reports/phase6/thresholds.csv"):
        f = int(r["fold_id"])
        tail = float(r["tail"])
        name = r["model_name"]
        v = a["prediction"][(a["model_name"] == name) & (a["fold_id"] == f)]
        observed = validation_thresholds(v, tail)
        np.testing.assert_allclose(
            observed, [float(r["q_low"]), float(r["q_high"])], atol=1e-15, rtol=0
        )
        p = predictions(5, "regression", name)
        test = p["prediction"][p["fold_id"] == f]
        different += not np.allclose(
            observed, np.quantile(test, [tail, 1 - tail]), atol=1e-15, rtol=0
        )
    assert different == 60


@pytest.mark.parametrize(
    "scores,delta", [([1, -1, 0], [1, -2, 1, 0]), ([-1, 1, 0], [-1, 2, -1, 0])]
)
def test_phase7_quote_side_flip_and_cash(scores, delta):
    q, b = fixture_books()
    r = simulate(q, [0, NS, 2 * NS], scores, (-0.5, 0.5), 0, 3 * NS, 0)
    p = replay(b, [0, NS, 2 * NS], scores, -0.5, 0.5, 0, 3 * NS, 0)
    assert [x["delta_position"] for x in r["ledger"]] == delta
    cash = 0
    for x in r["ledger"]:
        d = x["delta_position"]
        if d:
            assert x["fill_price"] == (x["ask"] if d > 0 else x["bid"])
        cash -= d * x["fill_price"]
        assert cash == x["cash"]
        assert abs(x["inventory"]) <= 1
    assert r["position_flips"] == 1 and p["flips"] == 1
    assert r["quote_pnl"] == p["pnl"] == cash
    assert r["gross_traded_notional"] == p["notional"]


def test_phase7_terminal_quote_flattening_and_failure():
    q, b = fixture_books()
    r = simulate(q, [NS], [1], (-0.5, 0.5), 0, 3 * NS, 0)
    assert r["ledger"][-1]["delta_position"] == -1
    assert r["ledger"][-1]["fill_price"] == 104
    assert r["ledger"][-1]["inventory"] == 0
    q.valid[-1] = False
    b.valid[-1] = False
    with pytest.raises(ValueError, match="terminal"):
        simulate(q, [NS], [1], (-0.5, 0.5), 0, 3 * NS, 0)
    with pytest.raises(ValueError, match="terminal"):
        replay(b, [NS], [1], -0.5, 0.5, 0, 3 * NS, 0)


def test_phase7_before_target_tie_and_no_future_quote():
    q, b = fixture_books()
    # At 1.000s the final same-time quote is 102/104; next millisecond is 90/92.
    for latency in [0, 10]:
        time = NS - latency * MS
        r = simulate(q, [time, 2 * NS], [1, 0], (-0.5, 0.5), 0, 3 * NS, latency)
        p = replay(b, [time, 2 * NS], [1, 0], -0.5, 0.5, 0, 3 * NS, latency)
        assert r["ledger"][0]["fill_price"] == 104 == p["ledger"]["fill_price"][0]
        assert r["ledger"][0]["source_order"] == 3
        assert all(
            x["actual_book_state_timestamp"] <= x["requested_execution_time"] for x in r["ledger"]
        )


def test_phase7_same_timestamp_feature_prefix():
    s = stream((0, 100, 100, 200, 1000))
    original, _ = build_features(s, settings())
    z = s.copy()
    z.bsize[2:] *= 19
    z.asize[2:] *= 37
    changed, _ = build_features(z, settings())
    for k in original:
        np.testing.assert_array_equal(original[k][:2], changed[k][:2])
    assert original["EVENT_COUNT_100MS"][1] == 1 and original["EVENT_COUNT_100MS"][2] == 2
    assert original["OBI_L1"][2] != changed["OBI_L1"][2]


def test_phase7_future_feature_trap():
    assert future_trap()


def test_phase7_model_test_labels_and_physical_shuffle_isolation():
    assert model_isolation()


def test_phase7_frozen_artifact_hash_integrity():
    assert (
        sha(js("PHASE2_MANIFEST.json")["output_path"])
        == "a5a493636d9f6d592e2d229921533940ec7cfffcdc5da8f1c9fb636445d34e96"
    )
    for phase in [4, 5, 6]:
        m = js(f"PHASE{phase}_MANIFEST.json")
        for file, digest in m["evidence_sha256"].items():
            if file not in ["RESEARCH_LOG.md", ".gitignore"]:
                assert sha(file) == digest, file


def test_phase7_reported_metrics_consistent_with_predictions_and_cash():
    p = predictions(5, "regression", "LightGBM")
    s = next(
        r for r in readcsv("reports/phase5/regression_metrics.csv") if r["model_name"] == "LightGBM"
    )
    assert corr(p["prediction"], p["target"]) == pytest.approx(float(s["spearman"]), abs=1e-15)
    a = arrays("reports/phase6/primary_schedule.parquet")
    d = a["delta_position"]
    px = np.where(d > 0, a["ask"], a["bid"])
    pnl = float(np.sum(-d * px))
    report = readcsv("reports/phase6/primary_metrics.csv")[0]
    assert pnl == pytest.approx(float(report["quote_pnl"]), abs=1e-8)
    assert int(np.sum(d != 0)) == int(report["trades"])
    assert float(np.sum(abs(d) * px)) == pytest.approx(
        float(report["gross_traded_notional"]), abs=1e-8
    )
