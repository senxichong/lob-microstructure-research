"""One-unit state machine and cash ledger. No training APIs or outcome labels."""

import numpy as np

NS = 10**9
MS = 10**6


def thresholds(validation_scores, tail=0.1):
    v = np.asarray(validation_scores, float)
    if not len(v) or not np.isfinite(v).all() or not 0 < tail < 0.5:
        raise ValueError("finite validation scores and nondegenerate tail required")
    low, high = np.quantile(v, [tail, 1 - tail], method="linear")
    if low >= high:
        raise ValueError("degenerate thresholds")
    return float(low), float(high)


def targets(scores, low, high):
    if low >= high:
        raise ValueError("degenerate thresholds")
    p = np.asarray(scores, float)
    return np.where(np.isfinite(p), np.where(p >= high, 1, np.where(p <= low, -1, 0)), 0).astype(
        int
    )


def random_scores(nval, ntest, rng):
    return rng.uniform(size=nval), rng.uniform(size=ntest)


class Book:
    def __init__(self, states, session_start, session_end):
        i = np.lexsort((states["source_order"], states["timestamp"]))
        self.a = {k: np.asarray(v)[i].copy() for k, v in states.items()}
        a = self.a
        if not len(i) or np.any(np.diff(a["source_order"]) <= 0):
            raise ValueError("non-authoritative/duplicate source order")
        self.start, self.end = int(session_start), int(session_end)
        self.t = a["timestamp"]
        if self.t.min() < self.start or self.t.max() >= self.end:
            raise ValueError("source states outside session")
        self.bid, self.ask = a["best_bid_price"], a["best_ask_price"]
        self.mid = (self.bid + self.ask) / 2
        self.valid = (
            a["book_valid"]
            & np.isfinite(self.bid)
            & np.isfinite(self.ask)
            & (self.bid > 0)
            & (self.ask > self.bid)
        )
        self.last_valid = np.maximum.accumulate(np.where(self.valid, np.arange(len(i)), -1))
        changes = np.r_[False, (np.diff(self.bid) != 0) | (np.diff(self.ask) != 0)]
        self.change_count = np.cumsum(changes)
        self.tie_count = np.searchsorted(self.t, self.t, side="right") - np.searchsorted(
            self.t, self.t, side="left"
        )

    def lookup(self, t):
        t = np.asarray(t, dtype=np.int64)
        idx = np.searchsorted(self.t, t, side="right") - 1
        safe = np.maximum(idx, 0)
        valid = (idx >= 0) & (t >= self.start) & (t < self.end) & self.valid[safe]
        return idx, valid


def replay(book, times, scores, low, high, start, end, latency_ms=10, cost_bps=0.0, detailed=True):
    """Process targets in clock order; end-1ms close is a separate risk instruction."""
    times, scores = np.asarray(times, np.int64), np.asarray(scores, float)
    order = np.argsort(times)
    times, scores = times[order], scores[order]
    if len(times) != len(scores) or not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("unique nonempty signal clock required")
    if times[0] < start or times[-1] + int(latency_ms * MS) >= end - MS:
        raise ValueError("signal execution collides with terminal boundary")
    if cost_bps < 0 or latency_ms < 0:
        raise ValueError("negative cost or latency")
    requests = np.r_[times + int(latency_ms * MS), end - MS]
    signal_times = np.r_[times, end - MS]
    desired = np.r_[targets(scores, low, high), 0]
    idx, valid = book.lookup(requests)
    safe = np.maximum(idx, 0)
    bid, ask, mid = book.bid[safe], book.ask[safe], book.mid[safe]
    n = len(requests)
    inventory = np.zeros(n, int)
    delta = np.zeros(n, int)
    failed = np.zeros(n, bool)
    depth_failed = np.zeros(n, bool)
    current = 0
    for j in range(n):
        need = desired[j] - current
        if need:
            depth = book.a["best_ask_size" if need > 0 else "best_bid_size"][safe[j]]
            depth_failed[j] = bool(valid[j] and (not np.isfinite(depth) or depth < abs(need)))
            if not valid[j] or depth_failed[j]:
                failed[j] = True
            else:
                delta[j] = need
                current += need
        inventory[j] = current
    if inventory[-1] != 0:
        raise ValueError("terminal liquidation failed; cannot certify economic result")
    assert np.abs(inventory).max() <= 1
    executed = delta != 0
    price = np.where(executed, np.where(delta > 0, ask, bid), 0.0)
    notional = np.abs(delta) * price
    fees = notional * cost_bps / 10000
    cash_delta = -delta * price - fees
    cash = np.cumsum(cash_delta)
    # Avoid 0*NaN at unavailable nontrading states.
    gross_delta = -delta * np.where(executed, mid, 0.0)
    gross_cash = np.cumsum(gross_delta)
    spread = np.abs(delta) * np.where(executed, (ask - bid) / 2, 0.0)
    marks_idx = book.last_valid[safe]
    mark = np.where((idx >= 0) & (marks_idx >= 0), book.mid[np.maximum(marks_idx, 0)], np.nan)
    equity = cash + np.where(inventory != 0, inventory * mark, 0.0)
    gross_equity = gross_cash + np.where(inventory != 0, inventory * mark, 0.0)
    before = np.r_[0, inventory[:-1]]
    time_weights = np.diff(np.r_[start, requests, end]) / NS
    held = np.r_[0, inventory]
    exposure = {v: float(time_weights[held == v].sum()) for v in [-1, 0, 1]}
    out = dict(
        pnl=float(cash[-1]),
        gross=float(gross_cash[-1]),
        spread=float(spread.sum()),
        extra_cost=float(fees.sum()),
        notional=float(notional.sum()),
        trades=int(executed.sum()),
        flips=int(((before * inventory) == -1).sum()),
        failed=int(failed.sum()),
        depth_failed=int(depth_failed.sum()),
        exposure=exposure,
    )
    assert abs(out["gross"] - out["spread"] - out["extra_cost"] - out["pnl"]) < 1e-8
    if not detailed:
        return out
    baseidx, basevalid = book.lookup(signal_times)
    baseidx = np.maximum(baseidx, 0)
    book_changes = book.change_count[safe] - book.change_count[baseidx]
    quote_changed = (book.bid[baseidx] != bid) | (book.ask[baseidx] != ask)
    out["ledger"] = dict(
        signal_timestamp=signal_times,
        requested_execution_time=requests,
        actual_book_state_timestamp=np.where(idx >= 0, book.t[safe], -1),
        source_order=np.where(idx >= 0, book.a["source_order"][safe], -1),
        execution_state_age_ms=np.where(idx >= 0, (requests - book.t[safe]) / MS, np.nan),
        prediction=np.r_[scores, np.nan],
        q_low=np.full(n, low),
        q_high=np.full(n, high),
        target_position=desired,
        previous_position=before,
        inventory=inventory,
        delta_position=delta,
        bid=bid,
        ask=ask,
        midprice=mid,
        visible_bid_size=book.a["best_bid_size"][safe],
        visible_ask_size=book.a["best_ask_size"][safe],
        fill_price=price,
        traded_notional=notional,
        extra_cost=fees,
        cash_delta=cash_delta,
        cash_before=np.r_[0.0, cash[:-1]],
        cash=cash,
        gross_cash=gross_cash,
        equity=equity,
        gross_equity=gross_equity,
        spread_paid=spread,
        missed_execution=failed,
        depth_violation=depth_failed,
        valid_quote=valid,
        stale_mark=~valid,
        terminal=np.arange(n) == n - 1,
        bbo_price_changes_during_delay=book_changes,
        quote_changed_during_delay=quote_changed & basevalid,
        same_timestamp_states=book.tie_count[safe],
    )
    episodes = []
    entry = None
    for j in np.flatnonzero(executed):
        if before[j] != 0:
            assert entry is not None
            k = entry
            direction = int(before[j])
            quote = direction * (price[j] - price[k])
            gross = direction * (mid[j] - mid[k])
            costs = (price[j] + price[k]) * cost_bps / 10000
            episodes.append(
                dict(
                    entry_time=int(requests[k]),
                    exit_time=int(requests[j]),
                    direction=direction,
                    entry_signal_time=int(signal_times[k]),
                    entry_prediction=float(scores[k]),
                    entry_quote=float(price[k]),
                    exit_quote=float(price[j]),
                    entry_mid=float(mid[k]),
                    exit_mid=float(mid[j]),
                    duration_seconds=float((requests[j] - requests[k]) / NS),
                    gross_mid_pnl=float(gross),
                    quote_pnl=float(quote),
                    additional_cost=float(costs),
                    net_pnl=float(quote - costs),
                    exit_is_flip=bool(inventory[j] != 0),
                    entry_row=int(k),
                    exit_row=int(j),
                )
            )
            entry = None
        if inventory[j] != 0:
            entry = j
    assert entry is None
    assert abs(sum(e["net_pnl"] for e in episodes) - out["pnl"]) < 1e-8
    out["episodes"] = episodes
    return out


def equity_curve(book, result, start, end, cost_bps=0):
    """Mark every source state, requests after all source ties at the same time."""
    a = result["ledger"]
    req = a["requested_execution_time"]
    src = np.flatnonzero((book.t >= start) & (book.t < end))
    ts = book.t[src]
    before_request = np.searchsorted(req, ts, side="left") - 1
    ix = np.maximum(before_request, 0)
    cash = a["cash"][ix] - np.cumsum(a["traded_notional"])[ix] * cost_bps / 10000
    inv = a["inventory"][ix]
    cash = np.where(before_request >= 0, cash, 0.0)
    inv = np.where(before_request >= 0, inv, 0)
    marks = book.mid[np.maximum(book.last_valid[src], 0)]
    eq = cash + inv * marks
    geq = np.where(before_request >= 0, a["gross_cash"][ix], 0.0) + inv * marks
    notion = np.where(before_request >= 0, np.cumsum(a["traded_notional"])[ix], 0.0)
    # Stable sort source rows before request marks at a timestamp.
    times = np.r_[start, ts, req, end - 1]
    notion = np.r_[0.0, notion, np.cumsum(a["traded_notional"]), a["traded_notional"].sum()]
    equity = np.r_[
        0.0,
        eq,
        a["equity"] - np.cumsum(a["traded_notional"]) * cost_bps / 10000,
        result["pnl"] - result["notional"] * cost_bps / 10000,
    ]
    gross = np.r_[0.0, geq, a["gross_equity"], result["gross"]]
    i = np.argsort(times, kind="stable")
    return dict(
        timestamp=times[i], equity=equity[i], gross_equity=gross[i], cumulative_notional=notion[i]
    )


def drawdown(equity):
    e = np.r_[0.0, equity]
    return float(np.max(np.maximum.accumulate(e) - e))


def break_even(pnl, notional):
    return float(10000 * pnl / notional) if notional > 0 else float("nan")
