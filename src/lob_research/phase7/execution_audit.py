"""Independent integer-price cash ledger. No Phase 6 imports."""

import numpy as np

from .common import CHECKS, arrays, blocks, close, dump, predictions, readcsv, write
from .statistical_audit import folds

NS = 10**9
MS = 10**6


class Quotes:
    def __init__(self, b):
        ix = np.lexsort((b["source_row"], b["timestamp"]))
        self.t = b["timestamp"][ix]
        self.order = b["source_row"][ix]
        if np.any(np.diff(self.order) <= 0):
            raise ValueError("invalid source order")
        self.bid = b["bid_price_units_1"][ix]
        self.ask = b["ask_price_units_1"][ix]
        self.qb = b["bid_size_1"][ix]
        self.qa = b["ask_size_1"][ix]
        self.valid = (self.bid > 0) & (self.ask > self.bid) & (self.qb > 0) & (self.qa > 0)
        if "book_valid" in b:
            self.valid &= b["book_valid"][ix]

    def lookup(self, time):
        i = np.searchsorted(self.t, time, side="right") - 1
        return i, (i >= 0) & self.valid[np.maximum(i, 0)]


def validation_thresholds(scores, tail=0.1):
    if not np.isfinite(scores).all() or not len(scores):
        raise ValueError("invalid validation scores")
    low, high = np.quantile(scores, [tail, 1 - tail])
    if not low < high:
        raise ValueError("degenerate thresholds")
    return float(low), float(high)


def simulate(q, times, scores, threshold, start, end, latency=10, detailed=True):
    order = np.argsort(times)
    times = np.asarray(times)[order]
    scores = np.asarray(scores)[order]
    if np.any(np.diff(times) <= 0) or times[0] < start or times[-1] + latency * MS >= end - MS:
        raise ValueError("bad signal clock")
    lo, hi = threshold
    if lo >= hi:
        raise ValueError("degenerate thresholds")
    requests = np.r_[times + latency * MS, end - MS]
    desired = np.zeros(len(requests), int)
    desired[:-1] = np.where(
        np.isfinite(scores), np.where(scores >= hi, 1, np.where(scores <= lo, -1, 0)), 0
    )
    idx, valid = q.lookup(requests)
    safe = np.maximum(idx, 0)
    pos = 0
    cash = 0
    gross_twice = 0
    notional = 0
    spread_twice = 0
    flips = 0
    trades = 0
    miss = 0
    depthmiss = 0
    exposure = {-1: 0, 0: 0, 1: 0}
    prior = start
    ledger = []
    episodes = []
    entry = None
    for j, (when, target, k, ok) in enumerate(zip(requests, desired, safe, valid)):
        exposure[pos] += int(when - prior)
        prior = when
        before = pos
        need = int(target - pos)
        delta = 0
        fill = 0
        dep = bool(need and ok and (q.qa[k] if need > 0 else q.qb[k]) < abs(need))
        failed = bool(need and (not ok or dep))
        if failed:
            miss += 1
            depthmiss += int(dep)
        if need and not failed:
            delta = need
            fill = int(q.ask[k] if delta > 0 else q.bid[k])
            pos = int(target)
            cash -= delta * fill
            gross_twice -= delta * int(q.bid[k] + q.ask[k])
            notional += abs(delta) * fill
            spread_twice += abs(delta) * int(q.ask[k] - q.bid[k])
            trades += 1
            flips += int(before * pos == -1)
            if detailed:
                if before:
                    ep = dict(
                        entry_time=entry[0],
                        exit_time=int(when),
                        direction=before,
                        duration_seconds=(when - entry[0]) / NS,
                        quote_pnl=before * (fill - entry[1]) / 10000,
                        gross_mid_pnl=before * (int(q.bid[k] + q.ask[k]) - entry[2]) / 20000,
                    )
                    episodes.append(ep)
                    entry = None
                if pos:
                    entry = (int(when), fill, int(q.bid[k] + q.ask[k]))
        if detailed:
            mid = (q.bid[k] + q.ask[k]) / 20000
            ledger.append(
                dict(
                    requested_execution_time=int(when),
                    signal_timestamp=int(times[j]) if j < len(times) else end - MS,
                    actual_book_state_timestamp=int(q.t[k]) if idx[j] >= 0 else -1,
                    source_order=int(q.order[k]) if idx[j] >= 0 else -1,
                    execution_state_age_ms=(when - q.t[k]) / MS if idx[j] >= 0 else np.nan,
                    target_position=int(target),
                    previous_position=before,
                    inventory=pos,
                    delta_position=delta,
                    bid=q.bid[k] / 10000,
                    ask=q.ask[k] / 10000,
                    midprice=mid,
                    fill_price=fill / 10000,
                    traded_notional=abs(delta) * fill / 10000,
                    cash_delta=-delta * fill / 10000,
                    cash=cash / 10000,
                    gross_cash=gross_twice / 20000,
                    equity=cash / 10000 + pos * mid,
                    gross_equity=gross_twice / 20000 + pos * mid,
                    spread_paid=abs(delta) * (q.ask[k] - q.bid[k]) / 20000,
                    missed_execution=failed,
                    depth_violation=dep,
                    valid_quote=bool(ok),
                    terminal=j == len(requests) - 1,
                )
            )
    exposure[pos] += int(end - prior)
    if pos:
        raise ValueError("failed terminal flattening")
    result = dict(
        quote_pnl=cash / 10000,
        gross_mid_pnl=gross_twice / 20000,
        spread_drag=spread_twice / 20000,
        gross_traded_notional=notional / 10000,
        trades=trades,
        position_flips=flips,
        missed_executions=miss,
        depth_violations=depthmiss,
        terminal_inventory=pos,
        terminal_trades=int(bool(ledger and ledger[-1]["delta_position"])),
        exposure={k: v / NS for k, v in exposure.items()},
    )
    if detailed:
        result.update(ledger=ledger, episodes=episodes)
    return result


def clock(q, results, fs):
    total = 0.0
    out = {
        k: []
        for k in [
            "timestamp",
            "pnl_increment",
            "notional",
            "episode_pnl",
            "episode_count",
            "equity",
        ]
    }
    for r, f in zip(results, fs):
        start, end = f["test_start"], f["test_end"]
        t = np.arange(start, end, NS)
        sample = t + NS - 1
        ledger = r["ledger"]
        req = np.array([x["requested_execution_time"] for x in ledger])
        last = np.searchsorted(req, sample, side="right") - 1
        pos = np.r_[0, [x["inventory"] for x in ledger]][last + 1]
        cash = np.r_[0, [x["cash"] for x in ledger]][last + 1]
        k, valid = q.lookup(sample)
        if not valid.all():
            raise ValueError("real-data audit clock requires valid marks")
        eq = cash + pos * (q.bid[k] + q.ask[k]) / 20000
        notion = np.zeros(len(t))
        eps = np.zeros(len(t))
        epn = np.zeros(len(t))
        for x in ledger:
            notion[(x["requested_execution_time"] - start) // NS] += x["traded_notional"]
        for x in r["episodes"]:
            cell = (x["exit_time"] - start) // NS
            eps[cell] += x["quote_pnl"]
            epn[cell] += 1
        for key, v in dict(
            timestamp=t,
            pnl_increment=np.diff(np.r_[0, eq]),
            notional=notion,
            episode_pnl=eps,
            episode_count=epn,
            equity=eq + total,
        ).items():
            out[key].extend(v)
        total += r["quote_pnl"]
    return {k: np.asarray(v) for k, v in out.items()}


def summary(results):
    result = {
        k: sum(r[k] for r in results)
        for k in [
            "quote_pnl",
            "gross_mid_pnl",
            "spread_drag",
            "gross_traded_notional",
            "trades",
            "position_flips",
            "missed_executions",
            "depth_violations",
            "terminal_trades",
        ]
    }
    ep = [e for r in results for e in r.get("episodes", [])]
    result.update(
        pnl_per_notional_bps=result["quote_pnl"] / result["gross_traded_notional"] * 10000,
        normalized_return=result["quote_pnl"] / 53.6,
        time_in_market=sum(r["exposure"][-1] + r["exposure"][1] for r in results) / 11700,
    )
    if ep:
        result.update(
            episodes=len(ep),
            average_holding_seconds=float(np.mean([e["duration_seconds"] for e in ep])),
        )
    return result


def run(books, g):
    q = Quotes(books)
    fs = folds(g)
    val = arrays("reports/phase6/validation_predictions.parquet")
    thresholds = readcsv("reports/phase6/thresholds.csv")
    source = {name: predictions(5, "regression", name) for name in ["Ridge", "RF", "LightGBM"]}
    prepared = {}
    threshold_rows = []
    for name, a in source.items():
        prepared[name] = []
        for f in fs:
            fid = f["fold_id"]
            m = a["fold_id"] == fid
            v = val["prediction"][(val["model_name"] == name) & (val["fold_id"] == fid)]
            for tail in [0.05, 0.1, 0.2, 0.3]:
                low, high = validation_thresholds(v, tail)
                saved = next(
                    r
                    for r in thresholds
                    if r["model_name"] == name
                    and int(r["fold_id"]) == fid
                    and float(r["tail"]) == tail
                )
                close(
                    "thresholds",
                    name + f" fold{fid} tail{tail}",
                    [low, high],
                    [float(saved["q_low"]), float(saved["q_high"])],
                )
                test = np.quantile(a["prediction"][m], [tail, 1 - tail])
                threshold_rows.append(
                    dict(
                        model=name,
                        fold=fid,
                        tail=tail,
                        validation_low=low,
                        validation_high=high,
                        test_low=test[0],
                        test_high=test[1],
                        matches_validation=True,
                    )
                )
            prepared[name].append((a["grid_timestamp"][m], a["prediction"][m], v))
    write("threshold_provenance", threshold_rows)
    # The validation predictions themselves are reconstructed independently in provenance5/4.
    rng = np.random.default_rng(606043)
    for name in ["RandomSignal", "Inversion", "Delayed600", "DelayMatchedOriginal"]:
        prepared[name] = []
        for times, p, v in prepared["LightGBM"]:
            if name == "RandomSignal":
                vv = rng.uniform(size=len(v))
                pp = rng.uniform(size=len(p))
            elif name == "Inversion":
                vv = -v
                pp = -p
            elif name == "Delayed600":
                vv = v
                pp = np.r_[np.full(600, np.nan), p[:-600]]
            else:
                vv = v
                pp = np.r_[np.full(600, np.nan), p[600:]]
            prepared[name].append((times, pp, vv))
    scenarios = readcsv("reports/phase6/scenario_metrics.csv")
    schedule = arrays("reports/phase6/scheduled_execution.parquet")
    foldsaved = readcsv("reports/phase6/fold_economic_metrics.csv")
    combinations = list(
        dict.fromkeys((r["model_name"], float(r["tail"]), int(r["latency_ms"])) for r in scenarios)
    )
    base = {}
    replay_rows = []
    cost_rows = []
    paths = {}
    for name, tail, latency in combinations:
        rr = []
        for f, (times, p, v) in zip(fs, prepared[name]):
            r = simulate(
                q, times, p, validation_thresholds(v, tail), f["test_start"], f["test_end"], latency
            )
            rr.append(r)
            m = (
                (schedule["model_name"] == name)
                & (schedule["tail"] == tail)
                & (schedule["latency_ms"] == latency)
                & (schedule["fold_id"] == f["fold_id"])
            )
            for k in r["ledger"][0]:
                close(
                    "execution",
                    f"{name}/{tail}/{latency}/fold{f['fold_id']}/{k}",
                    [x[k] for x in r["ledger"]],
                    schedule[k][m],
                    1e-8 if k in ["cash", "gross_cash", "equity", "gross_equity"] else 1e-9,
                )
            for saved in [
                x
                for x in foldsaved
                if x["model_name"] == name
                and float(x["tail"]) == tail
                and int(x["latency_ms"]) == latency
                and int(x["fold_id"]) == f["fold_id"]
                and float(x["cost_bps"]) == 0
            ]:
                for k in [
                    "quote_pnl",
                    "gross_mid_pnl",
                    "spread_drag",
                    "gross_traded_notional",
                    "trades",
                    "position_flips",
                ]:
                    close(
                        "execution",
                        f"{name}/{tail}/{latency}/fold{f['fold_id']}/{k}",
                        r[k],
                        float(saved[k]),
                        1e-8,
                    )
        s = summary(rr)
        base[(name, tail, latency)] = (rr, s)
        replay_rows.append(dict(model_name=name, tail=tail, latency_ms=latency, **s))
        for saved in [
            r
            for r in scenarios
            if r["model_name"] == name
            and float(r["tail"]) == tail
            and int(r["latency_ms"]) == latency
        ]:
            cost = float(saved["cost_bps"])
            fee = s["gross_traded_notional"] * cost / 10000
            net = s["quote_pnl"] - fee
            for k, v in dict(
                net_pnl=net,
                additional_cost=fee,
                normalized_return=net / 53.6,
                pnl_per_notional_bps=net / s["gross_traded_notional"] * 10000,
            ).items():
                close("cost", f"{name}/{tail}/{latency}/{cost}/{k}", v, float(saved[k]), 1e-8)
            if cost == 0:
                for k, v in s.items():
                    if k in saved:
                        close(
                            "execution",
                            f"{name}/{tail}/{latency}/summary/{k}",
                            v,
                            float(saved[k]),
                            1e-8,
                        )
            cost_rows.append(
                dict(
                    model_name=name,
                    tail=tail,
                    latency_ms=latency,
                    cost_bps=cost,
                    quote_pnl=s["quote_pnl"],
                    gross_mid_pnl=s["gross_mid_pnl"],
                    spread_drag=s["spread_drag"],
                    additional_cost=fee,
                    net_pnl=net,
                    identity_error=s["gross_mid_pnl"] - s["spread_drag"] - fee - net,
                )
            )
        if tail == 0.1 and latency == 10:
            paths[name] = clock(q, rr, fs)
    write("execution_replay", replay_rows)
    write("cost_reconciliation", cost_rows)
    # Independently reconstruct every cost-ledger row from the new zero-fee path.
    feeledger = arrays("reports/phase6/execution_cost_ledger.parquet")
    for name, tail, latency in combinations:
        rr, _ = base[name, tail, latency]
        for fid, r in enumerate(rr, 1):
            fills = [x for x in r["ledger"] if x["delta_position"] or x["missed_execution"]]
            for cost in sorted(set(feeledger["cost_bps"])):
                mask = (
                    (feeledger["model_name"] == name)
                    & (feeledger["tail"] == tail)
                    & (feeledger["latency_ms"] == latency)
                    & (feeledger["fold_id"] == fid)
                    & (feeledger["cost_bps"] == cost)
                )
                if not mask.any():
                    continue
                notion = np.asarray([x["traded_notional"] for x in fills])
                fees = notion * cost / 10000
                deltas = np.asarray([x["cash_delta"] for x in fills]) - fees
                for key, value in dict(
                    extra_cost=fees,
                    cumulative_extra_cost=np.cumsum(fees),
                    cash_delta=deltas,
                    cash=np.cumsum(deltas),
                    equity=np.asarray([x["equity"] for x in fills]) - np.cumsum(fees),
                ).items():
                    close(
                        "cost_ledger",
                        f"{name}/{tail}/{latency}/{fid}/{cost}/{key}",
                        value,
                        feeledger[key][mask],
                        1e-8,
                    )
    primary, main = base["LightGBM", 0.1, 10]
    prior = 0.0
    funding = []
    for r in primary:
        funding.extend(53.6 + prior + x["cash"] for x in r["ledger"])
        prior += r["quote_pnl"]
    main["minimum_cash_if_53_60_initial_funding"] = min(funding)
    pc = paths["LightGBM"]
    savedclock = arrays("reports/phase6/clock_pnl.parquet")
    for name, pp in paths.items():
        m = savedclock["model_name"] == name
        for k, v in pp.items():
            close("clock", name + " " + k, v, savedclock[k][m], 1e-8)
    # Conditional uncertainty on additive accounting cells, without restitching inventory.
    savedboot = readcsv("reports/phase6/bootstrap_uncertainty.csv")
    bootrows = []
    for name, pp in paths.items():
        rng = np.random.default_rng(606060)
        draw = []
        for j in range(1000):
            ix = blocks(11700, 60, rng)
            pnl = pp["pnl_increment"][ix].sum()
            notional = pp["notional"][ix].sum()
            draw.append(
                [
                    pnl / 53.6,
                    pnl / notional * 10000,
                    pp["episode_pnl"][ix].sum() / pp["episode_count"][ix].sum(),
                ]
            )
        for j, key in enumerate(["normalized_pnl", "pnl_per_notional_bps", "mean_episode_pnl"]):
            ci = np.quantile(np.asarray(draw)[:, j], [0.025, 0.975])
            s = next(r for r in savedboot if r["model_name"] == name and r["metric"] == key)
            close(
                "execution_bootstrap",
                name + " " + key,
                ci,
                [float(s["ci_lower"]), float(s["ci_upper"])],
                1e-10,
            )
            bootrows.append(dict(model_name=name, metric=key, ci_lower=ci[0], ci_upper=ci[1]))
    write("execution_bootstrap", bootrows)
    # Re-run every rotation through this independent state machine.
    rng = np.random.default_rng(606042)
    null = []
    nullsaved = readcsv("reports/phase6/circular_shift_null.csv")
    quality = readcsv("reports/phase6/null_execution_quality.csv")
    for rep in range(500):
        rr = []
        offsets = []
        for f, (times, p, v) in zip(fs, prepared["LightGBM"]):
            offset = int(rng.integers(600, len(p) - 600 + 1))
            offsets.append(offset)
            rr.append(
                simulate(
                    q,
                    times,
                    np.roll(p, offset),
                    validation_thresholds(v),
                    f["test_start"],
                    f["test_end"],
                    10,
                    False,
                )
            )
        s = summary(rr)
        old = nullsaved[rep]
        for k in ["quote_pnl", "pnl_per_notional_bps", "trades"]:
            close("null6", f"replicate{rep} {k}", s[k], float(old[k]), 1e-8)
        close(
            "null6",
            f"replicate{rep} depth misses",
            s["depth_violations"],
            int(quality[rep]["missed_depth_instructions"]),
            0,
        )
        close(
            "null6",
            f"replicate{rep} actual fills",
            s["trades"],
            int(quality[rep]["actual_fills"]),
            0,
        )
        close(
            "null6",
            f"replicate{rep} offsets",
            offsets,
            [int(old[f"offset_fold{i}"]) for i in range(1, 6)],
            0,
        )
        null.append(dict(replicate=rep, **s))
    write("independent_null", null)
    nullp = np.array([r["quote_pnl"] for r in null])
    nullbps = np.array([r["pnl_per_notional_bps"] for r in null])
    main.update(
        null_percentile=float(np.mean(nullp < main["quote_pnl"])),
        null_p=float((1 + (nullp >= main["quote_pnl"]).sum()) / 501),
        null_bps_p=float((1 + (nullbps >= main["pnl_per_notional_bps"]).sum()) / 501),
        null_mean=float(nullp.mean()),
        null_ci=np.quantile(nullp, [0.025, 0.975]).tolist(),
        null_depth_misses=sum(r["depth_violations"] for r in null),
        fold_pnl=[r["quote_pnl"] for r in primary],
    )
    # Adverse selection and spread use dollars per share, with transaction weighting.
    moves = []
    half = []
    for r, f in zip(primary, fs):
        for x in r["ledger"]:
            if x["delta_position"]:
                when = x["requested_execution_time"] + NS
                k, ok = q.lookup(when)
                if ok and when < f["test_end"]:
                    moves.append(
                        np.sign(x["delta_position"])
                        * ((q.bid[k] + q.ask[k]) / 20000 - x["midprice"])
                    )
                half.append((x["ask"] - x["bid"]) / 2)
    main.update(
        mean_signed_1s_move=float(np.mean(moves)),
        mean_half_spread=float(np.mean(half)),
        half_spread_to_signed_move=float(np.mean(half) / np.mean(moves)),
    )
    adverse = next(
        r for r in readcsv("reports/phase6/adverse_selection.csv") if r["horizon_ms"] == "1000"
    )
    close(
        "spread",
        "signed 1s move",
        main["mean_signed_1s_move"],
        float(adverse["mean_signed_mid_move"]),
    )
    minute = pc["pnl_increment"].reshape(-1, 60).sum(1)
    episodes = np.sort([e["quote_pnl"] for r in primary for e in r["episodes"]])[::-1]
    main.update(
        positive_minutes=int((minute > 0).sum()),
        minutes=len(minute),
        top_episode_contributions={str(k): float(episodes[:k].sum()) for k in [1, 5, 10]},
        leave_one_fold_out=[main["quote_pnl"] - r["quote_pnl"] for r in primary],
    )
    for filename, kind, vals in [
        ("time_concentration", "minute", np.sort(minute)[::-1]),
        ("trade_concentration", "episode", episodes),
    ]:
        for row in readcsv("reports/phase6/" + filename + ".csv"):
            if row["model_name"] == "LightGBM" and row["kind"] == kind:
                close(
                    "concentration",
                    filename + " top" + row["top_k"],
                    vals[: int(row["top_k"])].sum(),
                    float(row["contribution"]),
                    1e-8,
                )
    controls = [
        dict(control=name, **base[name, 0.1, 10][1])
        for name in ["RandomSignal", "Inversion", "Delayed600", "DelayMatchedOriginal"]
    ]
    write("control_summary", controls)
    dump(
        "execution_summary.json",
        dict(primary=main, bootstrap=bootrows, controls=controls, scenarios=replay_rows),
    )
    dump("execution_checks.json", CHECKS)
