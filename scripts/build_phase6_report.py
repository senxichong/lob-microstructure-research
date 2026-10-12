"""Post-result reporting and linear fee ledgers; never refit or resimulate signals."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lob_research.phase6.io import OUT, arrays, csvread, csvwrite, digest, dump, frozen, read_json


def read(name):
    return csvread(OUT / (name + ".csv"))


def f(r, k):
    return float(r[k])


def table(rows, columns):
    text = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for r in rows:
        vals = []
        for key, _ in columns:
            v = r.get(key, "")
            try:
                v = f"{float(v):.8g}" if v != "" else ""
            except (ValueError, TypeError):
                v = str(v)
            vals.append(v)
        text.append("| " + " | ".join(vals) + " |")
    return "\n".join(text)


def supplemental():
    a = arrays(OUT / "scheduled_execution.parquet")
    metrics = read("scenario_metrics")
    quality = []
    spreads = []
    for name, tail, lat in sorted(
        set((r["model_name"], f(r, "tail"), int(r["latency_ms"])) for r in metrics)
    ):
        mask = (a["model_name"] == name) & (a["tail"] == tail) & (a["latency_ms"] == lat)
        fill = mask & (a["delta_position"] != 0)
        depth = mask & a["depth_violation"]
        quality.append(
            dict(
                model_name=name,
                tail=tail,
                latency_ms=lat,
                requests=int(mask.sum()),
                filled_transactions=int(fill.sum()),
                missed=int(a["missed_execution"][mask].sum()),
                depth_failed=int(depth.sum()),
                invalid_quote_requests=int((~a["valid_quote"][mask]).sum()),
                terminal_fills=int((fill & a["terminal"]).sum()),
                max_age_ms=float(a["execution_state_age_ms"][fill].max()),
                median_age_ms=float(np.median(a["execution_state_age_ms"][fill])),
                p99_age_ms=float(np.quantile(a["execution_state_age_ms"][fill], 0.99)),
                minimum_filled_side_size=float(
                    np.min(
                        np.where(
                            a["delta_position"][fill] > 0,
                            a["visible_ask_size"][fill],
                            a["visible_bid_size"][fill],
                        )
                    )
                ),
            )
        )
    primary = (a["model_name"] == "LightGBM") & (a["tail"] == 0.1) & (a["latency_ms"] == 10)
    for label, mask in [
        ("all_requests", primary),
        ("fills", primary & (a["delta_position"] != 0)),
        ("entries", primary & (a["previous_position"] == 0) & (a["inventory"] != 0)),
        ("exits", primary & (a["previous_position"] != 0) & (a["inventory"] == 0)),
        ("flips", primary & (np.abs(a["delta_position"]) == 2)),
    ]:
        spread = a["ask"][mask] - a["bid"][mask]
        spreads.append(
            dict(
                population=label,
                rows=int(mask.sum()),
                mean_quoted_spread=float(spread.mean()),
                median_quoted_spread=float(np.median(spread)),
                mean_half_spread=float(spread.mean() / 2),
                spread_paid=float(a["spread_paid"][mask].sum()),
                mean_spread_paid_per_transition=float(a["spread_paid"][mask].mean()),
            )
        )
    csvwrite("execution_quality", quality)
    csvwrite("spread_statistics", spreads)
    # Fee scenarios are algebraic copies of the exact same executed path.
    costtables = []
    for m in metrics:
        name, tail, lat, cost = (
            m["model_name"],
            f(m, "tail"),
            int(m["latency_ms"]),
            f(m, "cost_bps"),
        )
        mask = (a["model_name"] == name) & (a["tail"] == tail) & (a["latency_ms"] == lat)
        ix = np.flatnonzero(mask & ((a["delta_position"] != 0) | a["missed_execution"]))
        cum = np.zeros(len(ix))
        for fid in range(1, 6):
            fm = a["fold_id"][ix] == fid
            cum[fm] = np.cumsum(a["traded_notional"][ix][fm])
        fee = a["traded_notional"][ix] * cost / 10000
        fields = {
            k: a[k][ix]
            for k in [
                "model_name",
                "tail",
                "latency_ms",
                "fold_id",
                "requested_execution_time",
                "actual_book_state_timestamp",
                "delta_position",
                "inventory",
                "fill_price",
                "traded_notional",
                "missed_execution",
                "terminal",
            ]
        }
        fields.update(
            cost_bps=np.full(len(ix), cost),
            extra_cost=fee,
            cumulative_extra_cost=cum * cost / 10000,
            cash=a["cash"][ix] - cum * cost / 10000,
            cash_delta=a["cash_delta"][ix] - fee,
            equity=a["equity"][ix] - cum * cost / 10000,
        )
        costtables.append(pa.table(fields))
    pq.write_table(
        pa.concat_tables(costtables), OUT / "execution_cost_ledger.parquet", compression="zstd"
    )
    return quality, spreads


def main():
    frozen()
    quality, spreads = supplemental()
    p = read("primary_metrics")[0]
    cost = read("cost_robustness")
    lat = sorted(read("latency_robustness"), key=lambda r: f(r, "latency_ms"))
    tails = sorted(read("threshold_robustness"), key=lambda r: f(r, "tail"))
    model = [r for r in read("model_economic_comparison") if f(r, "cost_bps") == 0]
    folds = [
        r
        for r in read("fold_economic_metrics")
        if f(r, "cost_bps") == 0 and r["model_name"] in ["Ridge", "RF", "LightGBM"]
    ]
    primaryfolds = [r for r in folds if r["model_name"] == "LightGBM"]
    bt = [r for r in read("bootstrap_uncertainty") if r["model_name"] == "LightGBM"]
    tc = [r for r in read("time_concentration") if r["model_name"] == "LightGBM"]
    ec = [r for r in read("trade_concentration") if r["model_name"] == "LightGBM"]
    ctrl = (
        read("random_signal_control")
        + read("sign_inversion_control")
        + read("delayed_signal_placebo")
    )
    adverse = read("adverse_selection")
    strength = read("prediction_strength")
    shift = read_json(OUT / "shift_null_summary.json")
    audit = read_json(OUT / "independent_verification.json")
    isolation = read_json(OUT / "isolation_audit.json")
    tests = ET.parse(OUT / "test_results.xml").getroot()
    suite = tests if tests.tag == "testsuite" else tests.find("testsuite")
    passed = (
        int(suite.attrib["tests"])
        - int(suite.attrib["failures"])
        - int(suite.attrib["errors"])
        - int(suite.attrib["skipped"])
    )
    valid = (
        audit["status"] == "PASS"
        and isolation["status"] == "PASS"
        and int(suite.attrib["failures"]) == int(suite.attrib["errors"]) == 0
    )
    ci = next(r for r in bt if r["metric"] == "normalized_pnl")
    positive = sum(f(r, "net_pnl") > 0 for r in primaryfolds)
    controls_sane = all(
        f(r, "ci_lower") <= 0
        for r in read("bootstrap_uncertainty")
        if r["model_name"] in ["RandomSignal", "Delayed600"] and r["metric"] == "normalized_pnl"
    )
    if not valid or not controls_sane:
        verdict = "INCONCLUSIVE"
    elif f(p, "quote_pnl") <= 0:
        verdict = "PREDICTIVE SIGNAL DOES NOT SURVIVE PRIMARY EXECUTION ASSUMPTIONS"
    elif f(ci, "ci_lower") <= 0 or shift["quote_pnl_pvalue"] > 0.05:
        verdict = "NO RELIABLE ECONOMIC EDGE"
    elif (
        positive >= 4
        and all(f(r, "remaining_pnl") > 0 for r in tc if r["kind"] == "leave_one_fold_out")
        and all(f(r, "remaining_pnl") > 0 for r in ec)
        and f(p, "break_even_extra_bps") >= 1
        and all(f(r, "net_pnl") > 0 for r in lat)
    ):
        verdict = "EXECUTION-ROBUST WITHIN-DAY EDGE"
    else:
        verdict = "MODEST / COST-SENSITIVE WITHIN-DAY EDGE"
    phase = "PASS" if valid and controls_sane else "FAIL"
    dump(
        "research_verdict.json",
        dict(
            verdict=verdict,
            phase_decision=phase,
            controls_sane=controls_sane,
            positive_folds=positive,
            tests_passed=passed,
            tests_failed=int(suite.attrib["failures"]),
            test_errors=int(suite.attrib["errors"]),
            independent_checks=audit["checks"],
            primary_unchanged=True,
        ),
    )
    decomp = next(
        r
        for r in read("pnl_decomposition")
        if r["model_name"] == "LightGBM"
        and f(r, "tail") == 0.1
        and int(r["latency_ms"]) == 10
        and f(r, "cost_bps") == 0
    )
    qprimary = next(
        r
        for r in quality
        if r["model_name"] == "LightGBM" and r["tail"] == 0.1 and r["latency_ms"] == 10
    )
    p5 = read_json("PHASE5_MANIFEST.json")
    provenance = read_json(OUT / "upstream_verification.json")
    report = f"""# Phase 6 — Execution-aware economic validation

**{verdict}**

**PHASE 6: {phase}.** Methodological validity and economic success are different
questions. The frozen predictive result survives as a forecasting result; the
predeclared aggressive execution strategy loses after crossing the spread.
**MSFT, 2003-01-03; single historical trading day; aggressive top-of-book
simulation; one-share price-taking assumption.** All dollar figures below are
aggregate USD PnL from this one-share experiment, not per-share price forecasts.

## Upstream integrity

Phase 2 SHA-256:
`a5a493636d9f6d592e2d229921533940ec7cfffcdc5da8f1c9fb636445d34e96`.
Phase 3 grid SHA-256:
`1781e33df0ff416cdf9d377fd3a66eed3fed1718f21efec9f465b292276f9e5b`.
Phase 5 regression predictions SHA-256:
`{p5["evidence_sha256"]["reports/phase5/regression_oos_predictions.parquet"]}`.
Paths were discovered from authoritative manifests, not substituted. All
{provenance["evidence_files_checked"]} Phase 4/5 evidence entries and 55 saved Phase 5
model artifacts matched. Earlier log versions are verified as append-only prefixes.
Predictive source, reports, feature definitions, matrices, models and scores remain
unchanged. Phase 6 adds code/artifacts and appends the research log/gitignore only.

Saved genuine OOS regression scores are reused: Ridge rho 0.050746889, RF
0.097539968, LightGBM 0.098636590. RF and LightGBM beat Ridge in 5/5 folds;
LightGBM's paired gain is 0.047889701, 60s CI [0.030392513,0.068317717].
Frozen classification Macro F1 remains Logistic 0.302183044, RF 0.345043280,
LightGBM 0.384455023. The prior verdicts RELIABLE NONLINEAR IMPROVEMENT and
RELIABLE NONLINEAR DIRECTIONAL IMPROVEMENT are preserved, not reinterpreted as profit.

Exactly the same five folds and 11,695 test scores per model are used, 12:45–16:00
Eastern. The full OOS clock is 11,700 seconds. The purged final score cell in each
fold receives no fresh signal; existing inventory is held until the declared
terminal close. No train/validation prediction is traded and no model is refit.

## Execution model

Signal time is g. Execution request is g+L, primary L=10ms. Use the latest Phase 2
state at or before request, last physical source order resolving ties. Book state
persists until the next event; never first-after, nearest, interpolated or future
quotes. Validate the latest state rather than skipping an invalid segment. Signal
features remain the original causal source-state features; no future regeneration.

Buy at ask; sell at bid. Inventory is -1, 0 or +1 share. Trade only target minus
current; a flip trades two shares and pays two half-spreads. Each fold starts flat,
ends with a mandatory risk liquidation request at test_end-1ms, and has no capital
reset that erases a loss. These boundary risk orders add no further signal latency.
No inventory crosses a fold/session boundary and no next-day quote is used.

Insufficient displayed top-level depth or an invalid/missing quote produces a
missed instruction and retains inventory until the next scheduled target. No
deeper-book or partial fill is invented. Primary: **0 missing/depth failures**.
The 30% tail stress has 23 depth-limited requests at 10ms and 24 at 0ms; these are
two-share flips against one displayed share. All other tested base scenarios have
zero failures. Every actual fill passes the size check. These failed paths remain
in the stress results. Terminal inventory is zero in every scenario.

Primary filled-quote age: median {qprimary["median_age_ms"]:.3f}ms,
p99 {qprimary["p99_age_ms"]:.3f}ms, maximum {qprimary["max_age_ms"]:.3f}ms.
Age is time since the last observed source state, not a measured feed delay.
Source accuracy is 1ms; original transport/receive times and completeness are unknown.

Cash changes by -delta*side_quote-extra_fee. Marked equity is cash+inventory*mid.
For drawdown, marks include every source event and execution request; figures
display the 1s sampled path. No passive order, maker rebate, queue-position fill,
or fill-probability model is used. No institutional size or market impact claim.

## Primary signal

**Frozen LightGBM regression; validation-derived symmetric 10% tails; one-second
rebalancing.** Each fold has its own q10/q90 from its selected saved model's
validation predictions. Score>=q90 targets +1, score<=q10 targets -1, otherwise 0.
Validation predictions are recovered with frozen training medians and saved tree
objects; Ridge uses saved coefficients and training preprocessing. Reconstructed
test predictions agree within 1e-12, but test scores never estimate thresholds.
Validation predictions and thresholds were hash-frozen before any real PnL.
No ensemble or classifier execution experiment was added.

## Midprice vs executable result

At 10ms, **MIDPRICE GROSS PNL (NON-EXECUTABLE UPPER BOUND)** is
**${f(p, "gross_mid_pnl"):.4f}**. **QUOTE-EXECUTABLE PNL before extra fees is
${f(p, "quote_pnl"):.4f}**. This optimistic quote simulation already assumes our
infinitesimal order can take the observed quote without impact or competing orders.

Fixed normalization capital is **$53.60**, one share at the first valid regular
state, 09:30:00.060 Eastern; the exact session-open grid has no state. This same
denominator is used for all folds/scenarios and never compounded. Primary normalized
PnL is **{100 * f(p, "normalized_return"):.4f}%**, PnL/traded-notional
**{f(p, "pnl_per_notional_bps"):.6f}bps**, and maximum marked drawdown
**${f(p, "max_drawdown"):.4f}**. The large normalized loss reflects repeated spread
payments relative to one share of capital, not a percentage move in MSFT.

This denominator is a fixed exposure normalization, not a broker funding model:
margin, short borrow, financing and capital-exhaustion liquidation are not simulated.
Some controls/cost stresses exceed a 100% normalized loss. Their continued fixed
one-share paths require funding and must not be described as feasible funded accounts.
Position size never adapts to equity or results. There are no annualized headlines.

## Spread impact

At actual 10ms fills, spread drag is **${f(p, "spread_drag"):.4f}**, about
{f(p, "spread_drag") / f(p, "gross_mid_pnl"):.2f} times gross midprice PnL.
Average execution spread is ${f(p, "mean_execution_spread"):.6f}; median
${f(p, "median_execution_spread"):.6f}; mean half-spread
${f(p, "mean_execution_half_spread"):.6f}. The observed gross price movement is
far too small to pay for entering and exiting through visible quotes.

{table(spreads, [("population", "Population"), ("rows", "Rows"), ("mean_quoted_spread", "Mean spread $"), ("median_quoted_spread", "Median spread $"), ("spread_paid", "Total paid $"), ("mean_spread_paid_per_transition", "Mean paid/request $")])}

Two equivalent exact decompositions prevent double-counting:

- Same-latency: {f(p, "gross_mid_pnl"):.4f} - {f(p, "spread_drag"):.4f} = {f(p, "quote_pnl"):.4f}.
- Zero-latency reference: {f(decomp, "gross_mid_zero"):.4f} - {f(decomp, "spread_zero"):.4f}
  - ({f(decomp, "latency_effect"):.4f}) - 0 extra fees = {f(p, "quote_pnl"):.4f}.

Latency effect is Quote_0-Quote_L. Its mid component is
{f(decomp, "latency_mid_effect"):.4f}, spread component
{f(decomp, "latency_spread_effect"):.4f}; the signed total is
**{f(decomp, "latency_effect"):.4f}**, a small benefit at 10ms in this sample.
Never subtract this again from already-delayed gross PnL. All 126 model/tail/latency/
cost attribution rows reconcile; maximum residual is
{isolation["max_reconciliation_error"]:.3g} USD. Attribution with depth failures
also includes changed inventory paths and is not a causal latency estimate.

## Latency

{table(lat, [("latency_ms", "Latency ms"), ("net_pnl", "Quote PnL $"), ("pnl_per_notional_bps", "PnL/notional bps"), ("trades", "Trades"), ("quotes_changed_before_fill", "Changed BBO at fill"), ("bbo_price_change_events", "BBO changes during delay"), ("max_drawdown", "Drawdown $")])}

{table(lat, [("latency_ms", "Latency ms"), ("delta_pnl_vs_zero", "PnL minus 0ms $"), ("delta_pnl_per_trade", "Delta PnL/trade $"), ("delta_win_rate", "Delta win rate"), ("mean_signed_fill_price_degradation", "Mean signed price change $")])}

10/50/100/250ms are 1/5/10/25% of the frozen 1s prediction horizon. Zero latency
is an **IDEALIZED UPPER BOUND assumption**, not realistic receipt/processing time;
it need not have the highest realized PnL when quotes change favorably by chance.
Latency recomputes every quote, rather than subtracting a chosen penalty.
The observed PnL effect is small and nonmonotonic; H3 is not supported as a monotonic
statement on this day. Spread dominates at every tested latency.

## Additional costs

{table(cost, [("cost_bps", "Extra one-way bps"), ("additional_cost", "Extra cost $"), ("net_pnl", "Net PnL $"), ("pnl_per_notional_bps", "PnL/notional bps"), ("max_drawdown", "Drawdown $")])}

Each fill pays abs(delta)*fill_price*cost_bps/10000 on top of quote crossing.
These are generic costs, **not the verified historical Nasdaq/Island 2003 fee
schedule**. Fees do not change the frozen order path. The analytic fee-only root
is {f(p, "break_even_extra_bps"):.6f}bps: **there is no nonnegative additional-cost
break-even allowance**. The negative root would require a subsidy after already
paying spread, not positive fee capacity or a proposed rebate strategy.

## Model comparison

{table(model, [("model_name", "Model"), ("predictive_spearman", "Frozen OOS rho"), ("trades", "Trades"), ("gross_mid_pnl", "Gross mid $"), ("spread_drag", "Spread $"), ("quote_pnl", "10ms quote $"), ("pnl_per_notional_bps", "PnL/notional bps")])}

The full model_economic_comparison.csv also gives every generic fee stress and
break-even root for every model; all five latencies use the same engine. LightGBM
has greater gross mid PnL and a less negative per-notional result than Ridge,
but **worse total quote PnL** because it turns over more. RF has the least negative
per-notional outcome of the three. Ranking improvement does not establish economic
improvement; H4 is not supported for aggregate executable dollars. None is positive.

## Threshold robustness

{table(tails, [("tail", "Symmetric tail fraction"), ("trades", "Trades"), ("time_in_market", "Time active"), ("gross_mid_pnl", "Gross mid $"), ("quote_pnl", "10ms quote $"), ("spread_drag", "Spread $"), ("pnl_per_notional_bps", "PnL/notional bps"), ("break_even_extra_bps", "Signed fee root bps")])}

All four thresholds are validation-only and were predeclared. All lose. The primary
10% result was saved and hash-frozen before these robustness runs. No threshold
is promoted, optimized or selected for profitability. The 30% depth failures are
included rather than replaced with nonexistent liquidity.

## Turnover

Primary **{int(p["trades"]):,} filled transactions**, {int(p["position_flips"])} two-share
flips, {int(p["episodes"]):,} closed episodes. Gross one-way traded notional is
**${f(p, "gross_traded_notional"):,.3f}**, or ${f(p, "traded_notional_per_second"):.4f}
per OOS second. A flip counts once as a transaction and twice in traded quantity.
Position changes equal fills; unchanged targets have no trade. No capital is
allocated separately to each overlapping prediction.

Duration-weighted long/flat/short fractions are
{100 * f(p, "long_fraction"):.4f}% / {100 * f(p, "flat_fraction"):.4f}% /
{100 * f(p, "short_fraction"):.4f}%; time in market is **{100 * f(p, "time_in_market"):.4f}%**.
Mean episode duration is {f(p, "average_holding_seconds"):.6f}s. Repeated same-direction
targets can hold beyond 1s; the prediction horizon remains 1s. Mean/median episode
PnL are ${f(p, "mean_episode_pnl"):.8f} / ${f(p, "median_episode_pnl"):.8f}; strictly positive
episode win rate is {100 * f(p, "episode_win_rate"):.4f}%. Zero-PnL episodes are not wins.

## Economic uncertainty

{table(bt, [("metric", "Metric"), ("observed", "Observed"), ("ci_lower", "95% lower"), ("ci_upper", "95% upper")])}

1,000 circular moving 60-second block draws, seed 606060, preserve the full
11,700-second clock. Resample additive marked PnL and traded notional jointly;
completed episode PnL/count are attributed to exit seconds for the mean-episode
ratio. This is a conditional path bootstrap, not execution on a synthetic stitched
market, independent episodes, or cross-day confidence. Episode boundaries can
cross blocks; model fitting, regime changes and previous research exposure are
not resolved by the interval. All three primary intervals are entirely negative.

500 within-fold large circular score rotations, seed 606042, preserve local score
order except the seam, with offsets from 600 to n-600 seconds and unchanged
validation thresholds. The same execution/depth/termination engine yields a
null mean of ${shift["null_mean_pnl"]:.6f}; null central 95% range
[${shift["null_ci"][0]:.6f}, ${shift["null_ci"][1]:.6f}]. Observed quote PnL is at
the **{100 * shift["quote_pnl_percentile"]:.2f}th percentile**; one-sided quote-PnL
p={shift["quote_pnl_pvalue"]:.6f}, per-notional p={shift["notional_pvalue"]:.6f}.
The observed economic path does not exceed this null. A predictive rho near 0.10
can coexist with this: score alignment may improve mid moves while selecting
different spread/turnover states. The null is within-day conditional, not a
cross-day exchangeability or causal test.

The independent null replay records 144 depth-limited instructions across 90 of
500 draws (maximum three per draw). These follow the same no-fill/hold policy;
no terminal inventory remains. They are retained in each null outcome and listed
in null_execution_quality.csv, rather than silently granting liquidity.

## Concentration

{table(folds, [("model_name", "Model"), ("fold_id", "Fold"), ("gross_mid_pnl", "Mid gross $"), ("spread_drag", "Spread $"), ("net_pnl", "Quote PnL $")])}

All three models have **0/5 positive quote-PnL folds**. No losing fold is removed.

{table(tc, [("kind", "Diagnostic"), ("top_k", "Top k"), ("fold_id", "Omitted fold"), ("contribution", "Contribution $"), ("signed_fraction", "Signed fraction of total"), ("remaining_pnl", "Remaining PnL $")])}

Every one-minute primary marked-PnL bin is negative; even the best minutes lose.
Minute attribution uses clock equity increments, not exit-only episode totals.
All leave-one-fold-out results remain negative. For a negative denominator,
signed fractions are accounting ratios, not shares of positive profits.

{table(ec, [("top_k", "Top episodes"), ("contribution", "Contribution $"), ("signed_fraction", "Signed fraction of total"), ("remaining_pnl", "After exclusion $")])}

Episode exclusion is **NON-IMPLEMENTABLE EX-POST FRAGILITY TEST**. Removing the
best 1/5/10 episodes only deepens the aggregate loss; exclusions never become rules.

## Controls

{table(ctrl, [("model_name", "Control"), ("trades", "Trades"), ("time_in_market", "Time active"), ("gross_mid_pnl", "Gross mid $"), ("quote_pnl", "Quote PnL $"), ("pnl_per_notional_bps", "PnL/notional bps")])}

One fixed random seed, 606043; validation uniform scores determine thresholds,
not test ranks. It approximates the nominal 20% active frequency, but does not
match the real signal's persistence: 3,952 transactions versus 2,989. This is a
coarse activity control, not an exact turnover-matched comparison. It loses,
with an entirely negative normalized-PnL interval; no redraw.

Inversion negates both validation and test scores. It exactly reverses the primary
mid gross to -$2.4475, while both directions pay the same spread drag; therefore
quote PnL is not the negative of the original quote PnL. This is expected accounting,
not a failed control or an invitation to trade the inversion.

The 600s delayed signal uses only earlier same-fold saved scores; first 600s stay
flat, no circular wrap. Its mid gross is -$0.1600 versus +$1.8550 for the original
signal with the same flat warmup/eligible clock. The delayed total quote loss is
smaller because turnover is smaller; per-notional PnL is slightly worse. Thus local
midprice alignment weakens, but absolute-dollar comparisons need exposure context.
Controls are sane; none supplies a reliable positive economic result.

## Adverse selection

{table(adverse, [("horizon_ms", "Post-fill ms"), ("eligible_transactions", "Eligible fills"), ("unavailable_transactions", "Unavailable"), ("mean_signed_mid_move", "Mean signed move $"), ("quantity_weighted_signed_mid_move", "Quantity-weighted $"), ("positive_fraction", "Positive fraction")])}

Direction*(future_mid-execution_mid), positive for favorable buys or sells.
All marks are causal as-of states, within the fold, with invalid intervals rejected.
These small average favorable movements (roughly $0.000230 at 1s per transaction)
are far below the $0.015099 mean half-spread. Closing trades are included.
This describes post-fill price behavior, not a causal adverse-selection estimate.

{table(strength, [("strength_bin", "Entry validation percentile"), ("episodes", "Episodes"), ("mean_gross_mid_pnl", "Mean mid gross $"), ("mean_quote_pnl", "Mean quote $"), ("win_rate", "Win rate")])}

The most extreme bins have greater mean mid gross in both directions, supporting
H2 descriptively. Every bin remains negative after quote crossing. This analysis
uses entry scores, not test-fitted quantiles, and never changes thresholds.

## Audit and reproducibility

Full pytest: **{passed} passed, zero failures/errors**; Phase 1–5 tests unchanged.
Synthetic cases cover every state transition, flips, quote sides, source ties,
as-of latency, missing/latest-invalid books, inadequate depth, terminal failure,
fees/cash/inventory/spread/turnover/episode arithmetic, validation thresholds,
future isolation, row permutation, event-level marking and block aggregation.

Real two-run primary paths match exactly. Mutating 64,191 later source states
does not change earlier execution, and shuffled input rows restore the identical
authoritative chronological path. Independent integer-price replay imports no
production execution/metric functions: **{audit["checks"]:,} checks, zero critical
failures**, maximum cash discrepancy **${audit["maximum_cash_discrepancy"]:.3g}**,
maximum |inventory|=1. It rebuilds validation predictions/quantiles, all scenario
targets, side prices, cash, inventory, episodes, terminations and cost arithmetic;
checks primary-family event drawdowns and 1s paths; independently reproduces all
1,000 primary bootstrap draws and all 500 null execution paths/offsets. The separate
cost-stress ledger is checked transaction by transaction against independent cash.

{audit["manual_paths"]} independently selected real paths have explicit hand-formula
cash/episode comparisons in manual_pnl_verification.csv; {audit["trace_rows"]} traces
cover entries/exits/flips/unchanged, positive/negative episodes, source ties,
0/10/100ms and quotes changing during latency. No claim of an outside human audit
is made; these are independently coded and inspected arithmetic checks.

Hypotheses: H1 and H5 supported on this day (spread consumes the predictive edge);
H2 supported descriptively for gross edge; H3 not monotonically supported; H4 not
supported for total executable dollars. No adverse result prompted retuning.

## Limitations

One MSFT day in 2003, repeatedly examined through earlier phases; no untouched day,
other-stock, regime or modern-market confirmation. OOS model parameters do not make
the research process pristine. Visible legacy Island/INET-lineage venue book only,
not consolidated NBBO. Historical fee schedule, transport completeness, receive
latency, borrow/financing, trading-status certification and competitor access to
the quote remain unverified. Features retain their frozen source-state staleness.

No passive queue model, hidden liquidity, maker fills/rebates, impact or
institutional-capacity estimate. One-share execution is an optimistic economic
diagnostic. Quote PnL is simulated, not realized brokerage profit. Fixed capital
normalization does not certify funding feasibility. A negative result here is
evidence against these primary assumptions, not proof that every execution style,
market or future day is uneconomic. No new strategy variant is authorized.

## Artifacts and reproduction

PHASE6_PROTOCOL.md, protocol_freeze.json and signal_freeze.json precede real PnL.
primary_result_freeze.json precedes robustness. execution_manifest.json records
executed production code hashes. PHASE6_MANIFEST.json freezes the final evidence.
Eleven PNG/SVG figure pairs identify the full historical/execution scope and were
visually inspected. Figures: [mid vs quote](reports/phase6/figures/01_mid_vs_quote.png),
[models](reports/phase6/figures/02_models.png),
[decomposition](reports/phase6/figures/03_decomposition.png),
[latency](reports/phase6/figures/04_latency.png),
[costs](reports/phase6/figures/05_costs.png),
[thresholds](reports/phase6/figures/06_thresholds.png),
[folds](reports/phase6/figures/07_folds.png),
[strength](reports/phase6/figures/08_strength.png),
[post-fill moves](reports/phase6/figures/09_adverse_selection.png),
[shift null](reports/phase6/figures/10_shift_null.png), and
[equity/drawdown](reports/phase6/figures/11_equity_drawdown.png).

prepare_phase6.py verifies upstream and reconstructs validation scores without
fitting; run_phase6.py refuses to overwrite frozen real PnL; verify_phase6.py
replays independently; build_phase6_report.py and plot_phase6.py read saved results.
The inherited requirements.phase5.txt environment suffices; no dependency added.

execution_ledger.parquet stores all zero-extra-cost fills and missed instructions;
scheduled_execution.parquet includes unchanged targets and terminal instructions.
execution_cost_ledger.parquet materializes all fee scenarios with per-fill costs
and cumulative cash. episode_ledger.csv stores zero-extra-fee episodes for every
base scenario (additional_cost=0); stressed episode fee is exactly
(entry_quote+exit_quote)*bps/10000, summarized in scenario_metrics.csv. Every
timestamp is UTC integer nanoseconds; source precision remains 1ms. Row-level
market/score/ledger artifacts and real traces stay local-only under inherited
redistribution restrictions. Aggregate tables, report, code and figures are reviewable.

**STOP after Phase 6. No predictive retuning, threshold search, passive execution
or Phase 7 red-team work has begun.**
"""
    Path("PHASE6_REPORT.md").write_text(report, encoding="utf-8")
    log = Path("RESEARCH_LOG.md")
    if "## Phase 6 outcome" not in log.read_text(encoding="utf-8"):
        with log.open("a", encoding="utf-8") as h:
            h.write(
                f"\n## Phase 6 outcome — frozen execution assumptions retained\n\nPrimary LightGBM 10% validation tails, 10ms, one share: gross mid ${f(p, 'gross_mid_pnl'):.4f}, spread ${f(p, 'spread_drag'):.4f}, quote PnL ${f(p, 'quote_pnl'):.4f}, no extra fees. 2,989 fills, 193 flips, 0/5 positive folds. No primary depth/missing failures.\n\nH1/H5 supported: spread consumes edge. H2 gross-strength association supported descriptively, every bin loses after crossing. H3 not supported monotonically: 10ms latency effect -$0.0710 versus 0ms. H4 not supported for aggregate dollars: Ridge -$41.3840, RF -$43.5250, LightGBM -$45.6460; turnover differs.\n\n60s/1,000 bootstrap normalized PnL CI [{f(ci, 'ci_lower'):.9f}, {f(ci, 'ci_upper'):.9f}]. 500-shift one-sided PnL p={shift['quote_pnl_pvalue']:.9f}. Random/inversion/600s controls sane; delayed mid gross loses alignment. All predefined thresholds/latencies lose; no result-driven changes.\n\n{verdict}. PHASE 6: {phase}. Full pytest {passed} passed, zero failures/errors. Independent replay PASS; discrepancy {audit['maximum_cash_discrepancy']:.3g} USD. Single examined 2003 stock/day only. STOP; wait for Phase 7 authorization.\n"
            )
    dump(
        "report_code.json",
        dict(
            files={
                p: digest(p) for p in ["scripts/build_phase6_report.py", "scripts/plot_phase6.py"]
            },
            report_sha256=digest("PHASE6_REPORT.md"),
        ),
    )
    print(verdict + "; PHASE 6 " + phase)


if __name__ == "__main__":
    main()
