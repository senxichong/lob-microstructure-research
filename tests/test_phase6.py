import numpy as np
import pytest

from lob_research.phase6.core import (
    MS,
    NS,
    Book,
    break_even,
    equity_curve,
    random_scores,
    replay,
    targets,
    thresholds,
)
from lob_research.phase6.evaluate import bootstrap, clock_path


def book():
    t = np.array([0, 1, 1, 2, 3, 4, 5, 6]) * NS
    return Book(
        dict(
            timestamp=t,
            source_order=np.arange(8),
            best_bid_price=np.array([99, 100, 101, 102, 103, 104, 105, 106.0]),
            best_ask_price=np.array([101, 102, 103, 104, 105, 106, 107, 108.0]),
            best_bid_size=np.full(8, 100.0),
            best_ask_size=np.full(8, 100.0),
            book_valid=np.ones(8, bool),
        ),
        0,
        7 * NS,
    )


@pytest.mark.parametrize(
    "path,delta",
    [
        ([1, 1, 0], [1, 0, -1, 0]),
        ([-1, -1, 0], [-1, 0, 1, 0]),
        ([1, -1, 0], [1, -2, 1, 0]),
        ([-1, 1, 0], [-1, 2, -1, 0]),
    ],
)
def test_transitions(path, delta):
    r = replay(book(), np.arange(3) * NS, path, -0.5, 0.5, 0, 4 * NS, 0)
    a = r["ledger"]
    np.testing.assert_array_equal(a["delta_position"], delta)
    assert abs(a["inventory"]).max() <= 1
    for d, p, b, ask in zip(a["delta_position"], a["fill_price"], a["bid"], a["ask"]):
        if d:
            assert p == (ask if d > 0 else b)
    assert r["notional"] == sum(abs(d) * p for d, p in zip(delta, a["fill_price"]))
    assert r["pnl"] == pytest.approx(r["gross"] - r["spread"])


def test_validation_only_threshold_api_and_inversion():
    v = np.arange(100.0)
    lo, hi = thresholds(v)
    assert (lo, hi) == (9.9, 89.10000000000001)
    test = np.array([-100, 50, 500.0])
    np.testing.assert_array_equal(targets(test, lo, hi), [-1, 0, 1])
    il, ih = thresholds(-v)
    np.testing.assert_array_equal(targets(-test, il, ih), -targets(test, lo, hi))
    assert thresholds(v) == (lo, hi)  # Test scores/targets never enter threshold API.
    with pytest.raises(ValueError):
        thresholds(np.zeros(10))


def test_latency_asof_ties_and_missing():
    b = book()
    i, valid = b.lookup(np.array([-MS, NS, NS + 50 * MS, 7 * NS]))
    np.testing.assert_array_equal(i, [-1, 2, 2, 7])
    np.testing.assert_array_equal(valid, [False, True, True, False])
    r = replay(b, [NS - 10 * MS, 2 * NS], [1, 0], -0.5, 0.5, 0, 4 * NS, 10)
    assert r["ledger"]["fill_price"][0] == 103
    b.valid[2] = False
    r = replay(b, [NS, 2 * NS], [1, 0], -0.5, 0.5, 0, 4 * NS, 0)
    assert r["failed"] == 1 and r["trades"] == 0


def test_fold_and_session_terminal_liquidation():
    for end in [4 * NS, 7 * NS]:
        r = replay(book(), [NS], [1], -0.5, 0.5, 0, end, 10)
        a = r["ledger"]
        assert a["requested_execution_time"][-1] == end - MS
        assert a["delta_position"][-1] == -1 and a["inventory"][-1] == 0
        assert a["actual_book_state_timestamp"][-1] < end
    b = book()
    b.valid[-1] = False
    with pytest.raises(ValueError, match="terminal"):
        replay(b, [NS], [1], -0.5, 0.5, 0, 7 * NS)


def test_flip_episodes_cost_cash_spread_break_even():
    b = book()
    r = replay(b, [0, NS, 2 * NS], [1, -1, 0], -0.5, 0.5, 0, 4 * NS, 0, 2)
    a = r["ledger"]
    assert r["trades"] == 3 and r["flips"] == 1
    assert len(r["episodes"]) == 2
    assert r["notional"] == 101 + 2 * 101 + 104
    assert r["spread"] == 4
    assert r["extra_cost"] == pytest.approx(r["notional"] * 0.0002)
    assert r["pnl"] == pytest.approx(-101 + 202 - 104 - r["extra_cost"])
    np.testing.assert_allclose(
        a["cash"], np.cumsum(-a["delta_position"] * a["fill_price"] - a["extra_cost"])
    )
    np.testing.assert_array_equal(a["inventory"], np.cumsum(a["delta_position"]))
    assert sum(e["net_pnl"] for e in r["episodes"]) == pytest.approx(r["pnl"])
    assert break_even(2, 1000) == 20
    assert break_even(-2, 1000) == -20
    assert sum(r["exposure"].values()) == 4


def test_depth_including_flip_and_retry():
    b = book()
    b.a["best_bid_size"][2] = 1
    r = replay(b, [0, NS, 2 * NS], [1, -1, 0], -0.5, 0.5, 0, 4 * NS, 0)
    assert r["depth_failed"] == 1
    np.testing.assert_array_equal(r["ledger"]["inventory"], [1, 1, 0, 0])


def test_future_independence_and_row_order():
    b = book()
    r = replay(b, [0, NS, 2 * NS], [1, -1, 0], -0.5, 0.5, 0, 4 * NS, 10)
    d = {k: v.copy() for k, v in b.a.items()}
    for k in ["best_bid_price", "best_ask_price"]:
        d[k][d["timestamp"] >= 4 * NS] += 1000
    changed = replay(Book(d, 0, 7 * NS), [0, NS, 2 * NS], [1, -1, 0], -0.5, 0.5, 0, 4 * NS, 10)
    assert r["pnl"] == changed["pnl"]
    for k in r["ledger"]:
        np.testing.assert_equal(r["ledger"][k], changed["ledger"][k])
    d = {k: v[::-1] for k, v in b.a.items()}
    shuffled = replay(Book(d, 0, 7 * NS), [2 * NS, 0, NS], [0, 1, -1], -0.5, 0.5, 0, 4 * NS, 10)
    for k in r["ledger"]:
        np.testing.assert_equal(r["ledger"][k], shuffled["ledger"][k])


def test_random_reproducibility_and_event_equity():
    a = random_scores(100, 200, np.random.default_rng(606043))
    b = random_scores(100, 200, np.random.default_rng(606043))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    bk = book()
    r = replay(bk, [0, 2 * NS], [1, 0], -0.5, 0.5, 0, 4 * NS, 0)
    curve = equity_curve(bk, r, 0, 4 * NS, 0.5)
    assert curve["equity"][-1] == pytest.approx(r["pnl"] - r["notional"] * 0.00005)
    assert np.any(curve["timestamp"] == NS)


def test_full_clock_reconciliation_and_bootstrap_literal():
    bk = book()
    r = replay(bk, [0, 2 * NS], [1, 0], -0.5, 0.5, 0, 4 * NS, 0)
    path = clock_path([r], bk, [dict(test_start=0, test_end=4 * NS)])
    assert len(path["timestamp"]) == 4
    assert path["pnl_increment"].sum() == pytest.approx(r["pnl"])
    assert path["episode_pnl"].sum() == pytest.approx(r["pnl"])
    assert path["notional"].sum() == pytest.approx(r["notional"])
    _, draws = bootstrap({"x": path}, 100, n=1)
    rng = np.random.default_rng(606060)
    start = int(rng.integers(0, 4, size=1)[0])
    i = (start + np.arange(4)) % 4
    assert draws[0]["normalized_pnl"] == pytest.approx(path["pnl_increment"][i].sum() / 100)


def test_event_drawdown_sees_between_rebalances():
    bk = book()
    bk.mid[2] = 90.0  # Intrasecond marked dip not visible at exit.
    r = replay(bk, [0, 2 * NS], [1, 0], -0.5, 0.5, 0, 4 * NS, 0)
    c = equity_curve(bk, r, 0, 4 * NS)
    assert c["equity"].min() <= -11
