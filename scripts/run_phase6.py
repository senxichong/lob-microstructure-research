import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from lob_research.phase4.core import folds
from lob_research.phase6.core import Book, break_even, random_scores, replay, thresholds
from lob_research.phase6.evaluate import (
    COSTS,
    LATENCIES,
    adverse,
    bootstrap,
    clock_path,
    concentrations,
    strength,
    summarize,
)
from lob_research.phase6.io import (
    OUT,
    arrays,
    csvread,
    csvwrite,
    digest,
    dump,
    frozen,
    parquet,
    predictions,
)


def main():
    u = frozen()
    assert not (OUT / "primary_result_freeze.json").exists(), "refuse overwrite of frozen PnL"
    code = list(Path("src/lob_research/phase6").glob("*.py")) + [
        Path(__file__),
        Path("scripts/prepare_phase6.py"),
    ]
    dump(
        "execution_manifest.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            code_sha256={p.as_posix(): digest(p) for p in code},
            protocol_sha256=digest("PHASE6_PROTOCOL.md"),
            signal_freeze_sha256=digest(OUT / "signal_freeze.json"),
        ),
    )
    fs = folds()
    session_start = fs[0]["train_start"]
    session_end = fs[-1]["test_end"]
    cols = [
        "timestamp",
        "source_order",
        "best_bid_price",
        "best_ask_price",
        "best_bid_size",
        "best_ask_size",
        "book_valid",
    ]
    book = Book(arrays(u["phase2_path"], cols), session_start, session_end)
    first = np.flatnonzero(book.valid)[0]
    capital = float(book.mid[first])
    dump(
        "capital_convention.json",
        dict(
            capital_usd=capital,
            source_timestamp=int(book.t[first]),
            source_order=int(book.a["source_order"][first]),
            convention="one share at first valid regular-session state; fixed, noncompounded across folds",
        ),
    )
    thresholds_rows = csvread(OUT / "thresholds.csv")
    th = {
        (r["model_name"], int(r["fold_id"]), float(r["tail"])): {
            k: float(v) for k, v in r.items() if k not in ["model_name"]
        }
        for r in thresholds_rows
    }
    saved = {n: predictions(n) for n in ["Ridge", "RF", "LightGBM"]}
    vals = arrays(OUT / "validation_predictions.parquet")
    base = {}
    ledgers = []
    episodes = []
    metrics = []
    foldmetrics = []

    def run(name, tail, latency, overrides=None, override_thresholds=None, detailed=True):
        s = saved.get(name, saved["LightGBM"])
        results = []
        for f in fs:
            fid = f["fold_id"]
            mask = s["fold_id"] == fid
            score = s["prediction"][mask] if overrides is None else overrides[fid]
            q = th[(name if name in saved else "LightGBM", fid, tail)]
            lo, hi = (
                (q["q_low"], q["q_high"])
                if override_thresholds is None
                else override_thresholds[fid]
            )
            results.append(
                replay(
                    book,
                    s["grid_timestamp"][mask],
                    score,
                    lo,
                    hi,
                    f["test_start"],
                    f["test_end"],
                    latency,
                    0,
                    detailed,
                )
            )
        return results

    # Primary frozen before any threshold robustness or control is evaluated.
    base[("LightGBM", 0.1, 10)] = run("LightGBM", 0.1, 10)
    primary = base[("LightGBM", 0.1, 10)]
    primary_rows = []
    for f, r in zip(fs, primary):
        for j in range(len(r["ledger"]["inventory"])):
            primary_rows.append(
                dict(fold_id=f["fold_id"], **{k: v[j].item() for k, v in r["ledger"].items()})
            )
    parquet("primary_schedule", primary_rows)
    pm = summarize(primary, book, fs, capital, "LightGBM", 0.1, 10)
    dump(
        "primary_result_freeze.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            primary_schedule_sha256=digest(OUT / "primary_schedule.parquet"),
            metrics=pm,
        ),
    )
    print(
        "Primary result frozen:",
        {k: pm[k] for k in ["gross_mid_pnl", "quote_pnl", "trades", "spread_drag"]},
        flush=True,
    )
    for n in saved:
        for lat in LATENCIES:
            if (n, 0.1, lat) not in base:
                base[(n, 0.1, lat)] = run(n, 0.1, lat)
    for tail in [0.05, 0.2, 0.3]:
        # 0ms counterpart is solely for exact attribution; robustness primary remains 10ms.
        for lat in [0, 10]:
            base[("LightGBM", tail, lat)] = run("LightGBM", tail, lat)
    rng = np.random.default_rng(606043)
    ctrl = {n: {} for n in ["RandomSignal", "Inversion", "Delayed600", "DelayMatchedOriginal"]}
    cth = {n: {} for n in ctrl}
    for f in fs:
        fid = f["fold_id"]
        s = saved["LightGBM"]
        p = s["prediction"][s["fold_id"] == fid]
        v = vals["prediction"][(vals["model_name"] == "LightGBM") & (vals["fold_id"] == fid)]
        rv, rp = random_scores(len(v), len(p), rng)
        ctrl["RandomSignal"][fid] = rp
        cth["RandomSignal"][fid] = thresholds(rv)
        ctrl["Inversion"][fid] = -p
        cth["Inversion"][fid] = thresholds(-v)
        ctrl["Delayed600"][fid] = np.r_[np.full(600, np.nan), p[:-600]]
        ctrl["DelayMatchedOriginal"][fid] = np.r_[np.full(600, np.nan), p[600:]]
        q = th[("LightGBM", fid, 0.1)]
        for name in ["Delayed600", "DelayMatchedOriginal"]:
            cth[name][fid] = (q["q_low"], q["q_high"])
    for n in ctrl:
        base[(n, 0.1, 10)] = run(n, 0.1, 10, ctrl[n], cth[n])
    csvwrite(
        "control_thresholds",
        [
            dict(model_name=n, fold_id=fid, q_low=q[0], q_high=q[1])
            for n, d in cth.items()
            for fid, q in d.items()
        ],
    )
    for (name, tail, lat), results in base.items():
        tag = dict(model_name=name, tail=tail, latency_ms=lat)
        for f, r in zip(fs, results):
            a = r["ledger"]
            for j in range(len(a["inventory"])):
                ledgers.append(
                    dict(**tag, fold_id=f["fold_id"], **{k: v[j].item() for k, v in a.items()})
                )
            episodes.extend(dict(**tag, fold_id=f["fold_id"], **e) for e in r["episodes"])
        costs = COSTS if name in saved else [0.0]
        for cost in costs:
            metrics.append(summarize(results, book, fs, capital, name, tail, lat, cost))
            if tail == 0.1 and lat == 10:
                for f, r in zip(fs, results):
                    foldmetrics.append(
                        dict(
                            fold_id=f["fold_id"],
                            **summarize([r], book, [f], capital, name, tail, lat, cost),
                        )
                    )
    parquet("scheduled_execution", ledgers)
    parquet(
        "execution_ledger", [r for r in ledgers if r["delta_position"] or r["missed_execution"]]
    )
    csvwrite("episode_ledger", episodes)
    csvwrite("scenario_metrics", metrics)
    csvwrite(
        "primary_metrics",
        [
            r
            for r in metrics
            if r["model_name"] == "LightGBM"
            and r["tail"] == 0.1
            and r["latency_ms"] == 10
            and r["cost_bps"] == 0
        ],
    )
    csvwrite("fold_economic_metrics", foldmetrics)
    csvwrite("turnover_metrics", [r for r in metrics if r["cost_bps"] == 0])
    csvwrite(
        "threshold_robustness",
        [
            r
            for r in metrics
            if r["model_name"] == "LightGBM" and r["latency_ms"] == 10 and r["cost_bps"] == 0
        ],
    )
    csvwrite(
        "cost_robustness",
        [
            r
            for r in metrics
            if r["model_name"] == "LightGBM" and r["tail"] == 0.1 and r["latency_ms"] == 10
        ],
    )
    latrows = []
    for r in [
        x
        for x in metrics
        if x["model_name"] == "LightGBM" and x["tail"] == 0.1 and x["cost_bps"] == 0
    ]:
        ref = next(
            x
            for x in metrics
            if x["model_name"] == "LightGBM"
            and x["tail"] == 0.1
            and x["latency_ms"] == 0
            and x["cost_bps"] == 0
        )
        changes = 0
        events = 0
        price_effect = []
        unmatched = 0
        for b, a in zip(base[("LightGBM", 0.1, 0)], base[("LightGBM", 0.1, r["latency_ms"])]):
            x, y = b["ledger"], a["ledger"]
            fill = y["delta_position"] != 0
            changes += int(y["quote_changed_during_delay"][fill].sum())
            events += int(y["bbo_price_changes_during_delay"][fill].sum())
            same = fill & (x["delta_position"] == y["delta_position"])
            unmatched += int((fill & ~same).sum())
            price_effect.extend(
                (
                    np.sign(y["delta_position"][same])
                    * (y["fill_price"][same] - x["fill_price"][same])
                ).tolist()
            )
        latrows.append(
            dict(
                **r,
                delta_pnl_vs_zero=r["net_pnl"] - ref["net_pnl"],
                delta_pnl_per_trade=r["pnl_per_transaction"] - ref["pnl_per_transaction"],
                delta_win_rate=r["episode_win_rate"] - ref["episode_win_rate"],
                quotes_changed_before_fill=changes,
                bbo_price_change_events=events,
                mean_signed_fill_price_degradation=float(np.mean(price_effect)),
                unmatched_fills=unmatched,
            )
        )
    csvwrite("latency_robustness", latrows)
    decomposition = []
    for r in metrics:
        if r["model_name"] not in saved:
            continue
        ref = next(
            x
            for x in metrics
            if x["model_name"] == r["model_name"]
            and x["tail"] == r["tail"]
            and x["latency_ms"] == 0
            and x["cost_bps"] == 0
        )
        ld = ref["quote_pnl"] - r["quote_pnl"]
        resid = ref["gross_mid_pnl"] - ref["spread_drag"] - ld - r["additional_cost"] - r["net_pnl"]
        assert abs(resid) < 1e-8
        decomposition.append(
            dict(
                model_name=r["model_name"],
                tail=r["tail"],
                latency_ms=r["latency_ms"],
                cost_bps=r["cost_bps"],
                gross_mid_zero=ref["gross_mid_pnl"],
                spread_zero=ref["spread_drag"],
                gross_mid_at_latency=r["gross_mid_pnl"],
                spread_at_latency=r["spread_drag"],
                latency_effect=ld,
                latency_mid_effect=ref["gross_mid_pnl"] - r["gross_mid_pnl"],
                latency_spread_effect=r["spread_drag"] - ref["spread_drag"],
                additional_cost=r["additional_cost"],
                net_pnl=r["net_pnl"],
                reconciliation_error=resid,
            )
        )
    csvwrite("pnl_decomposition", decomposition)
    rho = {
        r["model_name"]: float(r["spearman"])
        for r in csvread("reports/phase5/regression_metrics.csv")
    }
    csvwrite(
        "model_economic_comparison",
        [
            dict(**r, predictive_spearman=rho[r["model_name"]])
            for r in metrics
            if r["model_name"] in saved and r["tail"] == 0.1 and r["latency_ms"] == 10
        ],
    )
    paths = {n: clock_path(base[(n, 0.1, 10)], book, fs) for n in list(saved) + list(ctrl)}
    clockrows = []
    tc = []
    ec = []
    for n, p in paths.items():
        clockrows.extend(
            dict(model_name=n, **{k: v[i].item() for k, v in p.items()})
            for i in range(len(p["timestamp"]))
        )
        t, e = concentrations(n, base[(n, 0.1, 10)], p, fs)
        tc += t
        ec += e
    parquet("clock_pnl", clockrows)
    csvwrite("time_concentration", tc)
    csvwrite("trade_concentration", ec)
    boot, draws = bootstrap(paths, capital)
    csvwrite("bootstrap_uncertainty", boot)
    csvwrite("bootstrap_draws", draws)
    csvwrite("adverse_selection", adverse(book, primary, fs))
    csvwrite(
        "prediction_strength",
        strength(primary, {f["fold_id"]: th[("LightGBM", f["fold_id"], 0.1)] for f in fs}),
    )
    for n, file in [
        ("RandomSignal", "random_signal_control"),
        ("Inversion", "sign_inversion_control"),
        ("Delayed600", "delayed_signal_placebo"),
    ]:
        rows = [
            r
            for r in metrics
            if r["model_name"] == n
            or (n == "Delayed600" and r["model_name"] == "DelayMatchedOriginal")
        ]
        csvwrite(file, rows)
    print(
        "Scenarios, reconciliations and bootstrap saved. Running 500 fixed shift-null draws.",
        flush=True,
    )
    null = []
    rng = np.random.default_rng(606042)
    for rep in range(500):
        shifts = {}
        offsets = {}
        for f in fs:
            fid = f["fold_id"]
            p = saved["LightGBM"]["prediction"][saved["LightGBM"]["fold_id"] == fid]
            offset = int(rng.integers(600, len(p) - 600 + 1))
            offsets[f"offset_fold{fid}"] = offset
            shifts[fid] = np.roll(p, offset)
        runs = run("LightGBM", 0.1, 10, shifts, detailed=False)
        pnl = sum(x["pnl"] for x in runs)
        notional = sum(x["notional"] for x in runs)
        null.append(
            dict(
                replicate=rep,
                quote_pnl=pnl,
                pnl_per_notional_bps=break_even(pnl, notional),
                trades=sum(x["trades"] for x in runs),
                **offsets,
            )
        )
        if (rep + 1) % 100 == 0:
            print("Shift null", rep + 1, flush=True)
    csvwrite("circular_shift_null", null)
    obs = pm["quote_pnl"]
    obsbps = pm["pnl_per_notional_bps"]
    dump(
        "shift_null_summary.json",
        dict(
            replicates=500,
            seed=606042,
            observed_quote_pnl=obs,
            quote_pnl_pvalue=(1 + sum(r["quote_pnl"] >= obs for r in null)) / 501,
            quote_pnl_percentile=sum(r["quote_pnl"] < obs for r in null) / 500,
            notional_pvalue=(1 + sum(r["pnl_per_notional_bps"] >= obsbps for r in null)) / 501,
            null_mean_pnl=float(np.mean([r["quote_pnl"] for r in null])),
            null_ci=np.quantile([r["quote_pnl"] for r in null], [0.025, 0.975]).tolist(),
        ),
    )
    # Exact primary repeat plus real future-state mutation and shuffled physical storage.
    repeat = run("LightGBM", 0.1, 10)
    maxerr = max(
        float(np.max(np.abs(a["ledger"]["cash"] - b["ledger"]["cash"])))
        for a, b in zip(primary, repeat)
    )
    assert maxerr == 0
    f = fs[0]
    s = saved["LightGBM"]
    mask = s["fold_id"] == 1
    q = th[("LightGBM", 1, 0.1)]
    future = {k: v.copy() for k, v in book.a.items()}
    later = future["timestamp"] >= f["test_end"]
    for k in ["best_bid_price", "best_ask_price"]:
        future[k][later] += 777
    for k in ["best_bid_size", "best_ask_size"]:
        future[k][later] = 0
    future["book_valid"][later] = False
    changed = replay(
        Book(future, session_start, session_end),
        s["grid_timestamp"][mask],
        s["prediction"][mask],
        q["q_low"],
        q["q_high"],
        f["test_start"],
        f["test_end"],
    )
    shuf = np.random.default_rng(606034).permutation(len(book.t))
    shuffled = replay(
        Book({k: v[shuf] for k, v in book.a.items()}, session_start, session_end),
        s["grid_timestamp"][mask][::-1],
        s["prediction"][mask][::-1],
        q["q_low"],
        q["q_high"],
        f["test_start"],
        f["test_end"],
    )
    for r in [changed, shuffled]:
        for k in primary[0]["ledger"]:
            np.testing.assert_equal(primary[0]["ledger"][k], r["ledger"][k])
    dump(
        "isolation_audit.json",
        dict(
            status="PASS",
            repeat_max_cash_discrepancy=maxerr,
            mutated_future_rows=int(later.sum()),
            future_quote_independence=True,
            row_order_invariance=True,
            terminal_inventory_zero=True,
            max_reconciliation_error=max(abs(r["reconciliation_error"]) for r in decomposition),
            missed_executions=sum(r["failed"] for rs in base.values() for r in rs),
            depth_violations=sum(r["depth_failed"] for rs in base.values() for r in rs),
        ),
    )
    frozen()
    print("Execution research complete; independent audit and report remain.", flush=True)


if __name__ == "__main__":
    main()
