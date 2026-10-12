import numpy as np

from .core import NS, break_even, drawdown, equity_curve

COSTS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
LATENCIES = [0, 10, 50, 100, 250]


def summarize(results, book, folds, capital, model, tail, latency, cost=0):
    pnl = sum(r["pnl"] for r in results)
    gross = sum(r["gross"] for r in results)
    notional = sum(r["notional"] for r in results)
    fee = notional * cost / 10000
    episodes = [e for r in results for e in r["episodes"]]
    epnet = np.array(
        [e["quote_pnl"] - (e["entry_quote"] + e["exit_quote"]) * cost / 10000 for e in episodes]
    )
    duration = sum((f["test_end"] - f["test_start"]) / NS for f in folds)
    exposure = {v: sum(r["exposure"][v] for r in results) / duration for v in [-1, 0, 1]}
    curves = []
    cum = 0.0
    for r, f in zip(results, folds):
        c = equity_curve(book, r, f["test_start"], f["test_end"], cost)
        curves.extend((cum + c["equity"]).tolist())
        cum += r["pnl"] - r["notional"] * cost / 10000
    a = [r["ledger"] for r in results]
    spreads = np.concatenate([(x["ask"] - x["bid"])[x["delta_position"] != 0] for x in a])
    return dict(
        model_name=model,
        tail=tail,
        latency_ms=latency,
        cost_bps=cost,
        quote_pnl=pnl,
        gross_mid_pnl=gross,
        spread_drag=sum(r["spread"] for r in results),
        additional_cost=fee,
        net_pnl=pnl - fee,
        normalized_return=(pnl - fee) / capital,
        pnl_per_notional_bps=break_even(pnl - fee, notional),
        break_even_extra_bps=break_even(pnl, notional),
        nonnegative_fee_capacity=bool(pnl > 0),
        gross_traded_notional=notional,
        trades=sum(r["trades"] for r in results),
        position_changes=sum(r["trades"] for r in results),
        position_flips=sum(r["flips"] for r in results),
        episodes=len(episodes),
        mean_episode_pnl=float(epnet.mean()) if len(epnet) else np.nan,
        median_episode_pnl=float(np.median(epnet)) if len(epnet) else np.nan,
        episode_win_rate=float(np.mean(epnet > 0)) if len(epnet) else np.nan,
        average_holding_seconds=float(np.mean([e["duration_seconds"] for e in episodes]))
        if episodes
        else np.nan,
        long_fraction=exposure[1],
        flat_fraction=exposure[0],
        short_fraction=exposure[-1],
        time_in_market=1 - exposure[0],
        max_drawdown=drawdown(curves),
        pnl_per_transaction=(pnl - fee) / sum(r["trades"] for r in results),
        traded_notional_per_second=notional / duration,
        missed_executions=sum(r["failed"] for r in results),
        depth_violations=sum(r["depth_failed"] for r in results),
        terminal_inventory=0,
        mean_execution_spread=float(spreads.mean()),
        median_execution_spread=float(np.median(spreads)),
        mean_execution_half_spread=float(spreads.mean() / 2),
    )


def clock_path(results, book, folds):
    """Complete clock additive path, including the purged final second in each fold."""
    store = {
        k: []
        for k in [
            "timestamp",
            "pnl_increment",
            "gross_increment",
            "notional",
            "episode_pnl",
            "episode_count",
            "equity",
            "gross_equity",
        ]
    }
    cumulative, cumgross = 0.0, 0.0
    for r, f in zip(results, folds):
        start, end = f["test_start"], f["test_end"]
        clock = np.arange(start, end, NS, dtype=np.int64)
        curve = equity_curve(book, r, start, end)
        i = np.searchsorted(curve["timestamp"], clock + NS - 1, side="right") - 1
        eq, gr = curve["equity"][i], curve["gross_equity"][i]
        notional = np.zeros(len(clock))
        eps = np.zeros(len(clock))
        epn = np.zeros(len(clock))
        a = r["ledger"]
        cells = (a["requested_execution_time"] - start) // NS
        np.add.at(notional, cells, a["traded_notional"])
        for e in r["episodes"]:
            cell = (e["exit_time"] - start) // NS
            eps[cell] += e["net_pnl"]
            epn[cell] += 1
        vals = [
            clock,
            np.diff(np.r_[0.0, eq]),
            np.diff(np.r_[0.0, gr]),
            notional,
            eps,
            epn,
            cumulative + eq,
            cumgross + gr,
        ]
        for k, v in zip(store, vals):
            store[k].extend(v.tolist())
        cumulative += r["pnl"]
        cumgross += r["gross"]
    return {k: np.asarray(v) for k, v in store.items()}


def bootstrap(paths, capital, n=1000):
    rng = np.random.default_rng(606060)
    size = len(next(iter(paths.values()))["timestamp"])
    weights = np.empty((n, size))
    for j in range(n):
        starts = rng.integers(0, size, size=int(np.ceil(size / 60)))
        idx = ((starts[:, None] + np.arange(60)) % size).ravel()[:size]
        weights[j] = np.bincount(idx, minlength=size)
    rows, draws = [], []
    for name, p in paths.items():
        totals = weights @ np.column_stack(
            [p[k] for k in ["pnl_increment", "notional", "episode_pnl", "episode_count"]]
        )
        values = np.column_stack(
            [
                totals[:, 0] / capital,
                10000 * totals[:, 0] / totals[:, 1],
                totals[:, 2] / totals[:, 3],
            ]
        )
        observed = [
            p["pnl_increment"].sum() / capital,
            10000 * p["pnl_increment"].sum() / p["notional"].sum(),
            p["episode_pnl"].sum() / p["episode_count"].sum(),
        ]
        for j, key in enumerate(["normalized_pnl", "pnl_per_notional_bps", "mean_episode_pnl"]):
            lo, hi = np.quantile(values[:, j], [0.025, 0.975])
            rows.append(
                dict(
                    model_name=name,
                    metric=key,
                    observed=float(observed[j]),
                    ci_lower=float(lo),
                    ci_upper=float(hi),
                    block_seconds=60,
                    replicates=n,
                    seed=606060,
                )
            )
        draws.extend(
            dict(
                model_name=name,
                replicate=j,
                normalized_pnl=float(v[0]),
                pnl_per_notional_bps=float(v[1]),
                mean_episode_pnl=float(v[2]),
            )
            for j, v in enumerate(values)
        )
    return rows, draws


def concentrations(name, results, path, folds):
    total = sum(r["pnl"] for r in results)
    minute_ids = (path["timestamp"] // (60 * NS)).astype(np.int64)
    mins = np.array([path["pnl_increment"][minute_ids == i].sum() for i in np.unique(minute_ids)])
    episodes = np.sort([e["net_pnl"] for r in results for e in r["episodes"]])[::-1]
    time, trade = [], []
    for k in [1, 5, 10]:
        for values, out, kind in [
            (np.sort(mins)[::-1], time, "minute"),
            (episodes, trade, "episode"),
        ]:
            contribution = float(values[:k].sum())
            out.append(
                dict(
                    model_name=name,
                    kind=kind,
                    top_k=k,
                    contribution=contribution,
                    total_pnl=total,
                    signed_fraction=contribution / total if total else np.nan,
                    remaining_pnl=total - contribution,
                    interpretation="NON-IMPLEMENTABLE EX-POST FRAGILITY TEST"
                    if kind == "episode"
                    else "descriptive time concentration",
                )
            )
    for r, f in zip(results, folds):
        time.append(
            dict(
                model_name=name,
                kind="leave_one_fold_out",
                fold_id=f["fold_id"],
                total_pnl=total,
                remaining_pnl=total - r["pnl"],
            )
        )
    return time, trade


def adverse(book, results, folds):
    rows = []
    bad = np.cumsum(~book.valid)
    for horizon in [10, 100, 1000]:
        moves, weights = [], []
        eligible = 0
        total = 0
        for r, f in zip(results, folds):
            a = r["ledger"]
            fills = np.flatnonzero(a["delta_position"])
            t = a["requested_execution_time"][fills]
            future = t + horizon * 10**6
            j, v = book.lookup(future)
            i, _ = book.lookup(t)
            keep = v & (future < f["test_end"]) & (bad[np.maximum(j, 0)] == bad[np.maximum(i, 0)])
            move = np.sign(a["delta_position"][fills]) * (
                book.mid[np.maximum(j, 0)] - a["midprice"][fills]
            )
            moves.extend(move[keep])
            weights.extend(np.abs(a["delta_position"][fills][keep]))
            eligible += int(keep.sum())
            total += len(fills)
        m, w = np.array(moves), np.array(weights)
        rows.append(
            dict(
                horizon_ms=horizon,
                eligible_transactions=eligible,
                unavailable_transactions=total - eligible,
                mean_signed_mid_move=float(m.mean()),
                quantity_weighted_signed_mid_move=float(np.average(m, weights=w)),
                median_signed_mid_move=float(np.median(m)),
                positive_fraction=float(np.mean(m > 0)),
            )
        )
    return rows


def strength(results, threshold_by_fold):
    bins = {
        k: []
        for k in ["short_0_1", "short_1_5", "short_5_10", "long_90_95", "long_95_99", "long_99_100"]
    }
    for fid, r in enumerate(results, 1):
        q = threshold_by_fold[fid]
        for e in r["episodes"]:
            p = e["entry_prediction"]
            if e["direction"] > 0:
                name = (
                    "long_99_100"
                    if p >= q["q99"]
                    else ("long_95_99" if p >= q["q95"] else "long_90_95")
                )
            else:
                name = (
                    "short_0_1"
                    if p <= q["q01"]
                    else ("short_1_5" if p <= q["q05"] else "short_5_10")
                )
            bins[name].append(e)
    return [
        dict(
            strength_bin=k,
            episodes=len(v),
            gross_mid_pnl=sum(e["gross_mid_pnl"] for e in v),
            quote_pnl=sum(e["quote_pnl"] for e in v),
            mean_gross_mid_pnl=float(np.mean([e["gross_mid_pnl"] for e in v])) if v else np.nan,
            mean_quote_pnl=float(np.mean([e["quote_pnl"] for e in v])) if v else np.nan,
            win_rate=float(np.mean([e["quote_pnl"] > 0 for e in v])) if v else np.nan,
        )
        for k, v in bins.items()
    ]
