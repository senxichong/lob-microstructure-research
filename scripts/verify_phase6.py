"""Independent integer-price replay. Does not import the Phase 6 engine/metrics."""

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # Needed only to unpickle inherited model classes.
OUT = ROOT / "reports/phase6"
NS = 10**9
checks = []


def js(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def csvrows(p):
    with Path(p).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def data(p, columns=None):
    t = pq.read_table(p, columns=columns)
    return {
        k: t[k].cast(pa.int64()).to_numpy() if pa.types.is_timestamp(t[k].type) else t[k].to_numpy()
        for k in t.column_names
    }


def evidence(phase, suffix):
    p = [
        k for k in js(ROOT / f"PHASE{phase}_MANIFEST.json")["evidence_sha256"] if k.endswith(suffix)
    ]
    assert len(p) == 1
    return ROOT / p[0]


def check(name, value, error=0.0):
    checks.append(dict(check=name, passed=bool(value), max_discrepancy=float(error)))


def close(name, x, y, tol=1e-8):
    x, y = np.asarray(x), np.asarray(y)
    error = float(np.nanmax(np.abs(x - y))) if x.size else 0.0
    check(name, np.allclose(x, y, atol=tol, rtol=0, equal_nan=True), error)


def write(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def main():
    u = js(OUT / "upstream_freeze.json")
    for p, h in u["files"].items():
        b = (ROOT / p).read_bytes()
        if p in u["prefix_bytes"]:
            b = b[: u["prefix_bytes"][p]]
        check("upstream " + p, hashlib.sha256(b).hexdigest() == h)
    p = js(OUT / "protocol_freeze.json")
    check("protocol hash", sha(ROOT / "PHASE6_PROTOCOL.md") == p["protocol_sha256"])
    check(
        "hypotheses prefix",
        hashlib.sha256((ROOT / "RESEARCH_LOG.md").read_bytes()[: p["log_bytes"]]).hexdigest()
        == p["log_sha256"],
    )
    for p, h in js(OUT / "signal_freeze.json")["files"].items():
        check("signal freeze " + p, sha(ROOT / p) == h)
    for p, h in js(OUT / "execution_manifest.json")["code_sha256"].items():
        check("execution code " + p, sha(ROOT / p) == h)
    check(
        "primary schedule freeze",
        sha(OUT / "primary_schedule.parquet")
        == js(OUT / "primary_result_freeze.json")["primary_schedule_sha256"],
    )
    p2 = js(ROOT / "PHASE2_MANIFEST.json")
    p3 = js(ROOT / "PHASE3_MANIFEST.json")
    check(
        "phase2 exact hash",
        sha(ROOT / p2["output_path"])
        == p2["output_sha256"]
        == "a5a493636d9f6d592e2d229921533940ec7cfffcdc5da8f1c9fb636445d34e96",
    )
    q = data(
        ROOT / p2["output_path"],
        [
            "timestamp",
            "source_order",
            "best_bid_price",
            "best_ask_price",
            "best_bid_size",
            "best_ask_size",
            "book_valid",
        ],
    )
    ix = np.lexsort((q["source_order"], q["timestamp"]))
    q = {k: v[ix] for k, v in q.items()}
    t = q["timestamp"]
    bu = np.rint(q["best_bid_price"] * 10000).astype(np.int64)
    au = np.rint(q["best_ask_price"] * 10000).astype(np.int64)
    check("source order strictly increasing", np.all(np.diff(q["source_order"]) > 0))
    check(
        "price unit round trip",
        np.allclose(bu / 10000, q["best_bid_price"], atol=1e-12, rtol=0)
        and np.allclose(au / 10000, q["best_ask_price"], atol=1e-12, rtol=0),
    )
    check("all real books valid", np.all(q["book_valid"]) and np.all(au > bu))
    fs = csvrows(evidence(4, "/fold_manifest.csv"))

    def ns(s):
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()) * NS

    fs = [
        {
            k: int(v) if k == "fold_id" else ns(v)
            for k, v in f.items()
            if k == "fold_id" or k.endswith(("_start", "_end"))
        }
        for f in fs
    ]
    g = data(ROOT / p3["output_path"])
    val = data(OUT / "validation_predictions.parquet")
    th = csvrows(OUT / "thresholds.csv")
    threshold = {(r["model_name"], int(r["fold_id"]), float(r["tail"])): r for r in th}
    saved = {}
    for name in ["Ridge", "RF", "LightGBM"]:
        a = data(evidence(4 if name == "Ridge" else 5, "/regression_oos_predictions.parquet"))
        mask = a["model_name"] == name
        if name == "Ridge":
            mask &= a["embargo_seconds"] == 0
        ix = np.flatnonzero(mask)
        ix = ix[np.argsort(a["grid_timestamp"][ix])]
        saved[name] = {k: v[ix] for k, v in a.items()}
        check(name + " 11695 rows", len(ix) == 11695)
    models = js(evidence(5, "/model_artifacts.json"))
    linear = js(evidence(4, "/fitted_models.json"))
    for name, s in saved.items():
        close(
            name + " identical OOS clock",
            s["grid_timestamp"],
            saved["LightGBM"]["grid_timestamp"],
            0,
        )
        for f in fs:
            fid = f["fold_id"]
            tag = f"{name} fold{fid}"
            vmask = (
                (g["grid_timestamp"] >= f["validation_start"])
                & (g["grid_timestamp"] < f["validation_end"])
                & g["target_available_1S"]
                & g["grid_state_available"]
                & (g["target_timestamp_1S"] < f["validation_end"])
            )
            vi = np.flatnonzero(vmask)
            xval = val["prediction"][(val["model_name"] == name) & (val["fold_id"] == fid)]
            if name == "Ridge":
                m = next(
                    r
                    for r in linear
                    if r["model_name"] == name and r["fold_id"] == fid and r["embargo_seconds"] == 0
                )
                x = np.column_stack([g[k][vi] for k in m["features"]])
                actual = (
                    (np.where(np.isfinite(x), x, m["median"]) - m["mean"]) / m["scale"]
                ) @ np.array(m["coefficient"]) + m["intercept"]
            else:
                r = next(
                    r
                    for r in models
                    if r["task"] == "regression" and r["model_name"] == name and r["fold_id"] == fid
                )
                check(tag + " model hash", sha(ROOT / r["path"]) == r["sha256"])
                bundle = joblib.load(ROOT / r["path"])
                m = bundle["metadata"]
                x = np.column_stack([g[k][vi] for k in m["features"]])
                x = np.where(np.isfinite(x), x, m["median"])
                model = bundle["model"]
                actual = (
                    model.booster_.predict(x, num_threads=1)
                    if hasattr(model, "booster_")
                    else model.predict(x)
                )
            close(tag + " validation reconstruction", actual, xval, 1e-12)
            close(
                tag + " validation times",
                g["grid_timestamp"][vi],
                val["grid_timestamp"][(val["model_name"] == name) & (val["fold_id"] == fid)],
                0,
            )
            check(tag + " validation before test", g["grid_timestamp"][vi].max() < f["test_start"])
            for tail in [0.05, 0.1, 0.2, 0.3]:
                row = threshold[(name, fid, tail)]
                close(
                    tag + f" quantiles {tail}",
                    np.quantile(actual, [tail, 1 - tail]),
                    [float(row["q_low"]), float(row["q_high"])],
                    1e-12,
                )
            sm = s["fold_id"] == fid
            check(
                tag + " test boundaries",
                np.all(s["grid_timestamp"][sm] >= f["test_start"])
                and np.all(s["target_timestamp"][sm] < f["test_end"]),
            )
            check(
                tag + " causal source",
                np.all(s["source_state_timestamp"][sm] <= s["grid_timestamp"][sm]),
            )
    paired = csvrows(evidence(5, "/nonlinear_vs_ridge_bootstrap.csv"))
    for r in paired:
        if r["model_name"] in ["RF", "LightGBM"]:
            check(
                "prior " + r["model_name"] + " improvement",
                int(r["fold_wins"]) == 5 and float(r["ci_lower"]) > 0,
            )
    prod = data(OUT / "scheduled_execution.parquet")
    episodes = csvrows(OUT / "episode_ledger.csv")
    allmetrics = csvrows(OUT / "scenario_metrics.csv")
    cap = js(OUT / "capital_convention.json")["capital_usd"]
    close("capital", (bu[0] + au[0]) / 20000, cap)
    rng = np.random.default_rng(606043)
    ctrl = {}
    control_th = {}
    for f in fs:
        fid = f["fold_id"]
        p = saved["LightGBM"]["prediction"][saved["LightGBM"]["fold_id"] == fid]
        vp = val["prediction"][(val["model_name"] == "LightGBM") & (val["fold_id"] == fid)]
        rv = rng.uniform(size=len(vp))
        rp = rng.uniform(size=len(p))
        ctrl[("RandomSignal", fid)] = rp
        control_th[("RandomSignal", fid)] = np.quantile(rv, [0.1, 0.9])
        ctrl[("Inversion", fid)] = -p
        control_th[("Inversion", fid)] = np.quantile(-vp, [0.1, 0.9])
        ctrl[("Delayed600", fid)] = np.r_[np.full(600, np.nan), p[:-600]]
        ctrl[("DelayMatchedOriginal", fid)] = np.r_[np.full(600, np.nan), p[600:]]

    def replay_independent(name, tail, latency, f, scores=None):
        fid = f["fold_id"]
        s = saved.get(name, saved["LightGBM"])
        m = s["fold_id"] == fid
        p = s["prediction"][m] if name in saved else ctrl[(name, fid)]
        if scores is not None:
            p = scores
        qr = threshold[(name if name in saved else "LightGBM", fid, tail)]
        low, high = control_th.get((name, fid), (float(qr["q_low"]), float(qr["q_high"])))
        st = s["grid_timestamp"][m]
        requests = np.r_[st + latency * 10**6, f["test_end"] - 10**6]
        desired = np.r_[
            np.where(np.isfinite(p), np.where(p >= high, 1, np.where(p <= low, -1, 0)), 0), 0
        ]
        qi = np.searchsorted(t, requests, side="right") - 1
        previous = 0
        cash = 0
        gross = 0
        rows = []
        ep = []
        entry = None
        for j, (at, want, idx) in enumerate(zip(requests, desired, qi)):
            before = previous
            d = int(want - before)
            valid = idx >= 0 and q["book_valid"][idx] and au[idx] > bu[idx] > 0
            depth = q["best_ask_size" if d > 0 else "best_bid_size"][idx] if idx >= 0 else 0
            fail = bool(d and (not valid or depth < abs(d)))
            if fail:
                d = 0
            price = int(au[idx] if d > 0 else bu[idx]) if d else 0
            cb = cash
            cash -= d * price
            gross -= d * int(bu[idx] + au[idx])
            previous += d
            r = dict(
                requested_execution_time=int(at),
                actual_book_state_timestamp=int(t[idx]),
                source_order=int(q["source_order"][idx]),
                target_position=int(want),
                previous_position=before,
                inventory=previous,
                delta_position=d,
                fill_price=price / 10000,
                cash=cash / 10000,
                cash_before=cb / 10000,
                cash_delta=-d * price / 10000,
                gross_cash=gross / 20000,
                spread_paid=abs(d) * (au[idx] - bu[idx]) / 20000,
                traded_notional=abs(d) * price / 10000,
                missed_execution=fail,
                equity=(2 * cash + previous * int(bu[idx] + au[idx])) / 20000,
                bid=bu[idx] / 10000,
                ask=au[idx] / 10000,
                mid=(bu[idx] + au[idx]) / 20000,
                score=float(p[j]) if j < len(p) else np.nan,
                low=low,
                high=high,
                quote_idx=int(idx),
                terminal=j == len(requests) - 1,
            )
            rows.append(r)
            if d:
                if before:
                    k = entry
                    rr = rows[k]
                    ep.append(
                        dict(
                            entry_row=k,
                            exit_row=j,
                            direction=before,
                            entry_time=rr["requested_execution_time"],
                            exit_time=int(at),
                            entry_quote=rr["fill_price"],
                            exit_quote=r["fill_price"],
                            quote_pnl=before
                            * (price - int(round(rr["fill_price"] * 10000)))
                            / 10000,
                            gross_mid_pnl=before * (r["mid"] - rr["mid"]),
                            duration_seconds=(at - rr["requested_execution_time"]) / NS,
                        )
                    )
                    entry = None
                if previous:
                    entry = j
        assert previous == 0
        return rows, ep

    replays = {}
    max_cash = 0.0
    max_inventory = 0.0
    manual = []
    configurations = sorted(
        set((r["model_name"], float(r["tail"]), int(r["latency_ms"])) for r in allmetrics)
    )
    for name, tail, lat in configurations:
        combined = []
        combined_ep = []
        for f in fs:
            fid = f["fold_id"]
            tag = f"{name}/{tail}/{lat}/fold{fid}"
            rows, ep = replay_independent(name, tail, lat, f)
            replays[(name, tail, lat, fid)] = (rows, ep)
            mask = (
                (prod["model_name"] == name)
                & (prod["tail"] == tail)
                & (prod["latency_ms"] == lat)
                & (prod["fold_id"] == fid)
            )
            pi = np.flatnonzero(mask)
            pi = pi[np.argsort(prod["requested_execution_time"][pi])]
            check(tag + " schedule size", len(pi) == len(rows))
            for k in [
                "requested_execution_time",
                "actual_book_state_timestamp",
                "source_order",
                "target_position",
                "previous_position",
                "inventory",
                "delta_position",
                "fill_price",
                "cash",
                "cash_before",
                "cash_delta",
                "gross_cash",
                "spread_paid",
                "traded_notional",
                "missed_execution",
                "equity",
            ]:
                x = np.array([r[k] for r in rows])
                y = prod[k][pi]
                if x.dtype == bool:
                    check(tag + " " + k, np.array_equal(x, y))
                else:
                    close(tag + " " + k, x, y)
            max_cash = max(
                max_cash,
                float(np.max(np.abs(np.array([r["cash"] for r in rows]) - prod["cash"][pi]))),
            )
            max_inventory = max(max_inventory, max(abs(r["inventory"]) for r in rows))
            check(
                tag + " causal quotes",
                all(
                    r["actual_book_state_timestamp"]
                    <= r["requested_execution_time"]
                    < f["test_end"]
                    for r in rows
                ),
            )
            check(
                tag + " buys ask sells bid",
                all(
                    r["fill_price"] == (r["ask"] if r["delta_position"] > 0 else r["bid"])
                    for r in rows
                    if r["delta_position"]
                ),
            )
            check(
                tag + " depth",
                all(
                    q["best_ask_size" if r["delta_position"] > 0 else "best_bid_size"][
                        r["quote_idx"]
                    ]
                    >= abs(r["delta_position"])
                    for r in rows
                    if r["delta_position"]
                ),
            )
            check(
                tag + " flat terminal",
                rows[-1]["inventory"] == 0
                and rows[-1]["requested_execution_time"] == f["test_end"] - 10**6,
            )
            eprod = [
                e
                for e in episodes
                if e["model_name"] == name
                and float(e["tail"]) == tail
                and int(e["latency_ms"]) == lat
                and int(e["fold_id"]) == fid
            ]
            check(tag + " episode count", len(ep) == len(eprod))
            for k in [
                "entry_time",
                "exit_time",
                "direction",
                "entry_quote",
                "exit_quote",
                "quote_pnl",
                "gross_mid_pnl",
                "duration_seconds",
            ]:
                close(tag + " episodes " + k, [e[k] for e in ep], [float(e[k]) for e in eprod])
            combined.append(rows)
            combined_ep += ep
        for m in [
            r
            for r in allmetrics
            if r["model_name"] == name and float(r["tail"]) == tail and int(r["latency_ms"]) == lat
        ]:
            cost = float(m["cost_bps"])
            flat = [r for rows in combined for r in rows]
            notional = sum(r["traded_notional"] for r in flat)
            pnl = sum(rows[-1]["cash"] for rows in combined)
            gross = sum(rows[-1]["gross_cash"] for rows in combined)
            spread = sum(r["spread_paid"] for r in flat)
            net = pnl - notional * cost / 10000
            eps = [
                e["quote_pnl"] - (e["entry_quote"] + e["exit_quote"]) * cost / 10000
                for e in combined_ep
            ]
            exposure = {-1: 0.0, 0: 0.0, 1: 0.0}
            for f, rows in zip(fs, combined):
                positions = [0] + [r["inventory"] for r in rows]
                durations = (
                    np.diff(
                        [f["test_start"]]
                        + [r["requested_execution_time"] for r in rows]
                        + [f["test_end"]]
                    )
                    / NS
                )
                for position, duration in zip(positions, durations):
                    exposure[position] += duration
            seconds = sum(exposure.values())
            expected = dict(
                quote_pnl=pnl,
                gross_mid_pnl=gross,
                spread_drag=spread,
                additional_cost=notional * cost / 10000,
                net_pnl=net,
                gross_traded_notional=notional,
                trades=sum(r["delta_position"] != 0 for r in flat),
                position_flips=sum(r["previous_position"] * r["inventory"] == -1 for r in flat),
                episodes=len(combined_ep),
                mean_episode_pnl=np.mean(eps),
                median_episode_pnl=np.median(eps),
                episode_win_rate=np.mean(np.array(eps) > 0),
                normalized_return=net / cap,
                pnl_per_notional_bps=10000 * net / notional,
                break_even_extra_bps=10000 * pnl / notional,
                long_fraction=exposure[1] / seconds,
                short_fraction=exposure[-1] / seconds,
                flat_fraction=exposure[0] / seconds,
                time_in_market=(exposure[1] + exposure[-1]) / seconds,
                average_holding_seconds=np.mean([e["duration_seconds"] for e in combined_ep]),
                missed_executions=sum(r["missed_execution"] for r in flat),
            )
            for k, v in expected.items():
                close(f"{name}/{tail}/{lat}/{cost} metric {k}", v, float(m[k]))
            close(
                f"{name}/{tail}/{lat}/{cost} reconciliation",
                gross - spread - notional * cost / 10000,
                net,
            )
    check("inventory bound", max_inventory <= 1)
    # Independent source-event and execution-time marks, plus 1-second additive clock.
    independent_clock = {}
    curves = {}
    for name in list(saved) + ["RandomSignal", "Inversion", "Delayed600", "DelayMatchedOriginal"]:
        seq = []
        curv = []
        cum = 0.0
        for f in fs:
            fid = f["fold_id"]
            rows, ep = replays[(name, 0.1, 10, fid)]
            rt = np.array([r["requested_execution_time"] for r in rows])
            inv = np.array([r["inventory"] for r in rows])
            cash = np.array([r["cash"] for r in rows])
            nt = np.cumsum([r["traded_notional"] for r in rows])
            src = np.flatnonzero((t >= f["test_start"]) & (t < f["test_end"]))
            pos = np.searchsorted(rt, t[src], side="left") - 1
            safe = np.maximum(pos, 0)
            ev = np.where(pos >= 0, cash[safe] + inv[safe] * (bu[src] + au[src]) / 20000, 0.0)
            en = np.where(pos >= 0, nt[safe], 0.0)
            tt = np.r_[f["test_start"], t[src], rt, f["test_end"] - 1]
            ee = np.r_[0.0, ev, [r["equity"] for r in rows], cash[-1]]
            nn = np.r_[0.0, en, nt, nt[-1]]
            ix = np.argsort(tt, kind="stable")
            tt, ee, nn = tt[ix], ee[ix], nn[ix]
            curv.append((tt, ee, nn))
            seconds = np.arange(f["test_start"], f["test_end"], NS)
            si = np.searchsorted(tt, seconds + NS - 1, side="right") - 1
            incr = np.diff(np.r_[0.0, ee[si]])
            trade = np.bincount(
                ((rt - f["test_start"]) // NS).astype(int),
                weights=[r["traded_notional"] for r in rows],
                minlength=len(seconds),
            )
            epn = np.zeros(len(seconds))
            epp = np.zeros(len(seconds))
            for e in ep:
                i = (e["exit_time"] - f["test_start"]) // NS
                epn[i] += 1
                epp[i] += e["quote_pnl"]
            seq.extend(zip(seconds, incr, trade, epp, epn))
            cum += cash[-1]
        independent_clock[name] = np.array(seq)
        curves[name] = curv
    clock = data(OUT / "clock_pnl.parquet")
    for name, seq in independent_clock.items():
        mask = clock["model_name"] == name
        for j, k in enumerate(
            ["timestamp", "pnl_increment", "notional", "episode_pnl", "episode_count"]
        ):
            close(name + " clock " + k, seq[:, j], clock[k][mask])
        for m in [
            r
            for r in allmetrics
            if r["model_name"] == name and float(r["tail"]) == 0.1 and int(r["latency_ms"]) == 10
        ]:
            cost = float(m["cost_bps"])
            total = 0.0
            curve = [0.0]
            for tt, ee, nn in curves[name]:
                net = ee - nn * cost / 10000
                curve.extend(total + net)
                total += net[-1]
            e = np.array(curve)
            dd = np.max(np.maximum.accumulate(e) - e)
            close(name + f" drawdown {cost}", dd, float(m["max_drawdown"]))
    # Literal bootstrap sampling independently reconstructs all primary draws.
    seq = independent_clock["LightGBM"]
    rng = np.random.default_rng(606060)
    actual = []
    for rep in range(1000):
        starts = rng.integers(0, len(seq), size=195)
        idx = np.concatenate([(int(s) + np.arange(60)) % len(seq) for s in starts])
        total = seq[idx, 1:].sum(axis=0)
        actual.append([total[0] / cap, 10000 * total[0] / total[1], total[2] / total[3]])
    draws = [r for r in csvrows(OUT / "bootstrap_draws.csv") if r["model_name"] == "LightGBM"]
    for j, k in enumerate(["normalized_pnl", "pnl_per_notional_bps", "mean_episode_pnl"]):
        values = np.array(actual)[:, j]
        close("bootstrap draws " + k, values, [float(r[k]) for r in draws])
        r = next(
            r
            for r in csvrows(OUT / "bootstrap_uncertainty.csv")
            if r["model_name"] == "LightGBM" and r["metric"] == k
        )
        close(
            "bootstrap interval " + k,
            np.quantile(values, [0.025, 0.975]),
            [float(r["ci_lower"]), float(r["ci_upper"])],
        )
    # Independently rebuild all null offsets and execution paths, including misses.
    null = csvrows(OUT / "circular_shift_null.csv")
    rng = np.random.default_rng(606042)
    null_quality = []
    for j, r in enumerate(null):
        pnl = 0.0
        notional = 0.0
        misses = 0
        fills = 0
        for f in fs:
            fid = f["fold_id"]
            p = saved["LightGBM"]["prediction"][saved["LightGBM"]["fold_id"] == fid]
            offset = int(rng.integers(600, len(p) - 600 + 1))
            check(f"null{j} fold{fid} offset", offset == int(r[f"offset_fold{fid}"]))
            rows, _ = replay_independent("LightGBM", 0.1, 10, f, np.roll(p, offset))
            pnl += rows[-1]["cash"]
            notional += sum(x["traded_notional"] for x in rows)
            misses += sum(x["missed_execution"] for x in rows)
            fills += sum(x["delta_position"] != 0 for x in rows)
        close(f"null{j} PnL", pnl, float(r["quote_pnl"]))
        close(f"null{j} notional bps", 10000 * pnl / notional, float(r["pnl_per_notional_bps"]))
        close(f"null{j} fills", fills, int(r["trades"]))
        null_quality.append(
            dict(
                replicate=j,
                missed_depth_instructions=misses,
                actual_fills=fills,
                terminal_inventory=0,
            )
        )
    write("null_execution_quality.csv", null_quality)
    shift = js(OUT / "shift_null_summary.json")
    observed = sum(replays[("LightGBM", 0.1, 10, f["fold_id"])][0][-1]["cash"] for f in fs)
    close(
        "null pvalue",
        (1 + sum(float(r["quote_pnl"]) >= observed for r in null)) / 501,
        shift["quote_pnl_pvalue"],
    )
    # Linear-cost ledger is audited against independent fold cash, not totals.
    if (OUT / "execution_cost_ledger.parquet").exists():
        costs = data(OUT / "execution_cost_ledger.parquet")
        for m in allmetrics:
            name, tail, lat, cost = (
                m["model_name"],
                float(m["tail"]),
                int(m["latency_ms"]),
                float(m["cost_bps"]),
            )
            group = (
                (costs["model_name"] == name)
                & (costs["tail"] == tail)
                & (costs["latency_ms"] == lat)
                & (costs["cost_bps"] == cost)
            )
            for f in fs:
                fid = f["fold_id"]
                rows, _ = replays[(name, tail, lat, fid)]
                selected = [r for r in rows if r["delta_position"] or r["missed_execution"]]
                mask = group & (costs["fold_id"] == fid)
                check(
                    f"fee ledger {name}/{tail}/{lat}/{cost}/{fid} length",
                    len(selected) == int(mask.sum()),
                )
                fee = np.array([r["traded_notional"] for r in selected]) * cost / 10000
                close(
                    f"fee ledger {name}/{tail}/{lat}/{cost}/{fid} fee",
                    fee,
                    costs["extra_cost"][mask],
                )
                close(
                    f"fee ledger {name}/{tail}/{lat}/{cost}/{fid} cash",
                    np.array([r["cash"] for r in selected]) - np.cumsum(fee),
                    costs["cash"][mask],
                )
                close(
                    f"fee ledger {name}/{tail}/{lat}/{cost}/{fid} delta",
                    np.array([r["cash_delta"] for r in selected]) - fee,
                    costs["cash_delta"][mask],
                )
    for name, seq in independent_clock.items():
        total = seq[:, 1].sum()
        minute = seq[:, 0].astype(np.int64) // (60 * NS)
        values = np.sort([seq[minute == i, 1].sum() for i in np.unique(minute)])[::-1]
        eps = np.sort(
            [e["quote_pnl"] for f in fs for e in replays[(name, 0.1, 10, f["fold_id"])][1]]
        )[::-1]
        for filename, kind, ranked in [
            ("time_concentration.csv", "minute", values),
            ("trade_concentration.csv", "episode", eps),
        ]:
            for r in csvrows(OUT / filename):
                if r["model_name"] != name or r["kind"] != kind:
                    continue
                top = ranked[: int(r["top_k"])].sum()
                for key, v in [
                    ("contribution", top),
                    ("remaining_pnl", total - top),
                    ("signed_fraction", top / total),
                ]:
                    close(name + f" {kind} concentration {r['top_k']} {key}", v, float(r[key]))
    # Direct post-fill price arithmetic, independently signed and quantity weighted.
    for r in csvrows(OUT / "adverse_selection.csv"):
        moves = []
        weights = []
        total = 0
        for f in fs:
            rows, _ = replays[("LightGBM", 0.1, 10, f["fold_id"])]
            for a in rows:
                if not a["delta_position"]:
                    continue
                total += 1
                future = a["requested_execution_time"] + int(r["horizon_ms"]) * 10**6
                if future >= f["test_end"]:
                    continue
                ix = np.searchsorted(t, future, side="right") - 1
                move = np.sign(a["delta_position"]) * ((bu[ix] + au[ix]) / 20000 - a["mid"])
                moves.append(move)
                weights.append(abs(a["delta_position"]))
        close(
            "adverse " + r["horizon_ms"] + " mean", np.mean(moves), float(r["mean_signed_mid_move"])
        )
        close(
            "adverse " + r["horizon_ms"] + " weighted",
            np.average(moves, weights=weights),
            float(r["quantity_weighted_signed_mid_move"]),
        )
        close(
            "adverse " + r["horizon_ms"] + " available", len(moves), int(r["eligible_transactions"])
        )
    # Explicit hand-formula checks on independently selected real paths.
    pool = []
    for f in fs:
        rows, ep = replays[("LightGBM", 0.1, 10, f["fold_id"])]
        pool.extend((f["fold_id"], rows, e) for e in ep)
    selected = []
    predicates = [
        lambda z: z[2]["direction"] == 1,
        lambda z: z[2]["direction"] == -1,
        lambda z: abs(z[1][z[2]["entry_row"]]["delta_position"]) == 2,
        lambda z: abs(z[1][z[2]["exit_row"]]["delta_position"]) == 2,
        lambda z: z[2]["quote_pnl"] > 0,
        lambda z: z[2]["quote_pnl"] < 0,
    ]
    for pred in predicates:
        match = next((z for z in pool if pred(z)), None)
        if match is not None and match not in selected:
            selected.append(match)
    for i in np.linspace(0, len(pool) - 1, 12, dtype=int):
        if pool[i] not in selected:
            selected.append(pool[i])
    for i, (fid, rows, e) in enumerate(selected):
        a, b = rows[e["entry_row"]], rows[e["exit_row"]]
        direction = e["direction"]
        pnl = direction * (b["fill_price"] - a["fill_price"])
        gross = direction * (b["mid"] - a["mid"])
        spread = ((a["ask"] - a["bid"]) + (b["ask"] - b["bid"])) / 2
        production_episode = next(
            z
            for z in episodes
            if z["model_name"] == "LightGBM"
            and float(z["tail"]) == 0.1
            and int(z["latency_ms"]) == 10
            and int(z["fold_id"]) == fid
            and int(z["entry_time"]) == e["entry_time"]
        )
        error = abs(pnl - float(production_episode["quote_pnl"]))
        check(f"manual path{i}", error < 1e-8 and abs(gross - spread - pnl) < 1e-8)
        manual.append(
            dict(
                path_id=i,
                fold_id=fid,
                direction=direction,
                entry_time=e["entry_time"],
                exit_time=e["exit_time"],
                prediction=a["score"],
                q_low=a["low"],
                q_high=a["high"],
                entry_previous=a["previous_position"],
                entry_target=a["target_position"],
                entry_delta=a["delta_position"],
                exit_previous=b["previous_position"],
                exit_target=b["target_position"],
                exit_delta=b["delta_position"],
                entry_bid=a["bid"],
                entry_ask=a["ask"],
                exit_bid=b["bid"],
                exit_ask=b["ask"],
                entry_cash_delta=-a["delta_position"] * a["fill_price"],
                exit_cash_delta=-b["delta_position"] * b["fill_price"],
                entry_inventory_delta=a["delta_position"],
                exit_inventory_delta=b["delta_position"],
                manual_episode_pnl=pnl,
                production_episode_pnl=float(production_episode["quote_pnl"]),
                gross_mid=gross,
                spread=spread,
                max_discrepancy=error,
                formula="direction*(exit side quote-entry side quote); flip transaction charges abs(delta)=2, episode gets one leg",
                passed=True,
            )
        )
    write("manual_pnl_verification.csv", manual)
    # Deterministic diverse row traces, including no-change and tie groups.
    trace = []
    seen = set()
    coverage = {}
    choices = []
    for lat in [0, 10, 100]:
        for f in fs:
            rows, ep = replays[("LightGBM", 0.1, lat, f["fold_id"])]
            choices.extend((lat, f["fold_id"], j, r) for j, r in enumerate(rows))
    categories = {
        "long_entry": lambda z: z[3]["previous_position"] == 0 and z[3]["inventory"] == 1,
        "long_exit": lambda z: z[3]["previous_position"] == 1 and z[3]["inventory"] == 0,
        "short_entry": lambda z: z[3]["previous_position"] == 0 and z[3]["inventory"] == -1,
        "short_exit": lambda z: z[3]["previous_position"] == -1 and z[3]["inventory"] == 0,
        "flip": lambda z: abs(z[3]["delta_position"]) == 2,
        "unchanged": lambda z: z[3]["delta_position"] == 0,
        "same_timestamp": lambda z: (
            np.searchsorted(t, t[z[3]["quote_idx"]], side="right")
            - np.searchsorted(t, t[z[3]["quote_idx"]], side="left")
            > 1
        ),
        "quote_change": lambda z: (
            z[0] > 0
            and (
                bu[z[3]["quote_idx"]]
                != bu[
                    max(
                        0,
                        np.searchsorted(
                            t, z[3]["requested_execution_time"] - z[0] * 10**6, side="right"
                        )
                        - 1,
                    )
                ]
                or au[z[3]["quote_idx"]]
                != au[
                    max(
                        0,
                        np.searchsorted(
                            t, z[3]["requested_execution_time"] - z[0] * 10**6, side="right"
                        )
                        - 1,
                    )
                ]
            )
        ),
    }
    for label, pred in categories.items():
        matches = [z for z in choices if pred(z)]
        coverage[label] = len(matches)
        for z in matches[:3]:
            key = z[:3]
            if key not in seen:
                trace.append((*z, label))
                seen.add(key)
    for lat in [0, 10, 100]:
        zset = [z for z in choices if z[0] == lat]
        for j in np.linspace(0, len(zset) - 1, 20, dtype=int):
            z = zset[j]
            if z[:3] not in seen:
                trace.append((*z, "clock-spaced"))
                seen.add(z[:3])
    for sign in [1, -1]:
        z = next((x for x in pool if np.sign(x[2]["quote_pnl"]) == sign), None)
        coverage["positive_episode" if sign == 1 else "negative_episode"] = int(z is not None)
        if z:
            fid, rows, e = z
            for j in [e["entry_row"], e["exit_row"]]:
                trace.append((10, fid, j, rows[j], f"episode PnL {e['quote_pnl']:.6f}"))

    def iso(n):
        return str(np.datetime64(int(n), "ns")) + "Z"

    lines = [
        "# Phase 6 deterministic execution traces",
        "",
        f"{len(trace)} rows; local-only real MSFT 2003-01-03 paths. Same-time source order is authoritative.",
        "Cash is fold-local USD; both crossing legs of flips are charged. Zero delta is no fill.",
        "",
    ]
    for i, (lat, fid, j, r, label) in enumerate(trace):
        lines.extend(
            [
                f"## Trace {i + 1}: fold {fid}, {label}",
                f"- Signal {iso(r['requested_execution_time'] - lat * 10**6) if not r['terminal'] else 'risk close'}; score {r['score']:.12g}; validation thresholds [{r['low']:.12g}, {r['high']:.12g}].",
                f"- Position {r['previous_position']} -> target {r['target_position']} -> inventory {r['inventory']}; delta {r['delta_position']}; latency {lat}ms.",
                f"- Request {iso(r['requested_execution_time'])}; book {iso(r['actual_book_state_timestamp'])}; source order {r['source_order']}; age {(r['requested_execution_time'] - r['actual_book_state_timestamp']) / 1e6:.3f}ms.",
                f"- Bid/ask {r['bid']:.4f}/{r['ask']:.4f}; fill {r['fill_price']:.4f}; cash {r['cash_before']:.6f} -> {r['cash']:.6f}; delta {r['cash_delta']:.6f}; equity {r['equity']:.6f}.",
                "",
            ]
        )
    (ROOT / "reports/phase6_execution_traces.md").write_text("\n".join(lines), encoding="utf-8")
    check("at least 50 traces", len(trace) >= 50)
    check("at least 10 real manual paths", len(manual) >= 10)
    for r in csvrows(OUT / "pnl_decomposition.csv"):
        value = (
            float(r["gross_mid_zero"])
            - float(r["spread_zero"])
            - float(r["latency_effect"])
            - float(r["additional_cost"])
        )
        close(
            "decomposition " + str((r["model_name"], r["tail"], r["latency_ms"], r["cost_bps"])),
            value,
            float(r["net_pnl"]),
        )
    check("isolation audit", js(OUT / "isolation_audit.json")["status"] == "PASS")
    write("verification_checks.csv", checks)
    failures = [r for r in checks if not r["passed"]]
    receipt = dict(
        status="PASS" if not failures else "FAIL",
        checks=len(checks),
        critical_failures=len(failures),
        maximum_cash_discrepancy=max_cash,
        maximum_observed_inventory=max_inventory,
        manual_paths=len(manual),
        trace_rows=len(trace),
        trace_coverage=coverage,
        primary_reconstructed_pnl=observed,
        verifier_sha256=sha(__file__),
        failures=failures[:20],
    )
    if (ROOT / "PHASE6_REPORT.md").exists():
        receipt["report_sha256"] = sha(ROOT / "PHASE6_REPORT.md")
    (OUT / "independent_verification.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
