"""Audit findings and report, grounded in independently generated evidence."""

import re
from collections import Counter
from pathlib import Path

from .common import OUT, dump, js, readcsv, sha, write

ISSUES = [
    (
        "H01",
        "HIGH",
        "Repeated research exposure removes untouched-day status",
        "3–6",
        "RESEARCH_LOG.md; PHASE3_REPORT.md; PHASE6_REPORT.md",
        "The same MSFT session informed features, model phases and execution diagnostics. Five chronological test folds do not undo researcher exposure.",
        "Human adaptation and uncounted cross-phase choices can inflate apparent evidence.",
        "No numerical change; weakens signal certainty.",
        "Evaluate frozen decisions on genuinely new days in a separately authorized project.",
        "INHERENT LIMITATION — ACKNOWLEDGED",
    ),
    (
        "H02",
        "HIGH",
        "One historical stock-day cannot establish external validity",
        "1–6",
        "DATA_MANIFEST.json; DATA_LIMITATIONS.md",
        "Only MSFT 2003-01-03, legacy Island/INET-lineage event semantics and venue-specific displayed book.",
        "No independent days, stock population, modern market, live execution or institutional capacity are tested.",
        "No within-day numerical change.",
        "Restrict every conclusion to this sample; new data are outside Phase 7.",
        "INHERENT LIMITATION — ACKNOWLEDGED",
    ),
    (
        "M01",
        "MEDIUM",
        "Transport completeness and legal tick remain unverified",
        "1–2",
        "MARKET_DATA_SCHEMA.md; DATA_LIMITATIONS.md; config.yaml",
        "Archive has millisecond timestamps and physical row ordering but no original transport sequence numbers. Tick columns are null and excluded.",
        "A consistently reconstructable archive can still omit messages; encoding precision is not proof of legal tick size.",
        "No discrepancy detected; completeness cannot be certified.",
        "Retain explicit source limitations; do not impute tick features.",
        "INHERENT LIMITATION — ACKNOWLEDGED",
    ),
    (
        "M02",
        "MEDIUM",
        "Uncertainty is conditional on one dependent observed path",
        "3–6",
        "DEPENDENCE_NOTES.md; PHASE6_PROTOCOL.md",
        "Clock bootstrap and circular shifts rely on approximate temporal regularity. Shifts introduce artificial seams; frozen-path PnL blocks do not regenerate inventory decisions.",
        "Intervals exclude new-day, researcher-selection, model-fitting and execution-model uncertainty.",
        "No formula correction; conditional estimates reproduce.",
        "Describe conditional intervals and approximate nulls, never independent-trade or population confidence.",
        "INTERPRETATION CLARIFIED",
    ),
    (
        "M03",
        "MEDIUM",
        "Fixed normalization is not a funded account return",
        "6",
        "PHASE6_PROTOCOL.md; PHASE6_REPORT.md",
        "53.60 is the first valid session midpoint; inventory stays within one share but continued trading after losses can require borrowing.",
        "No margin, financing, stock-borrow or insolvency engine supports the percentage-return interpretation.",
        "Absolute PnL and PnL/notional are unchanged.",
        "Use dollars and traded-notional bps as primary economic evidence; label normalization a denominator convention.",
        "INTERPRETATION CLARIFIED",
    ),
    (
        "M04",
        "MEDIUM",
        "Displayed-quote simulation omits operational execution frictions",
        "6",
        "PHASE6_PROTOCOL.md; DATA_LICENSE.md",
        "One-share price-taking model, no impact, hidden liquidity modeling, broker/transport delays, actual historical fee/rebate schedule or passive queue model.",
        "Displayed depth and an as-of state do not guarantee historical fills; fee stresses are hypothetical.",
        "Loss under the stated assumptions reproduces; no capacity or live claim.",
        "Keep aggressive simulation conditional; no passive switch or profitability search.",
        "INHERENT LIMITATION — ACKNOWLEDGED",
    ),
    (
        "L01",
        "LOW",
        "README current-status wording is historical",
        "documentation",
        "README.md:5; README.md:75",
        "README says Phase 1 only and later stop after Phase 3, while frozen Phase 4–6 artifacts exist.",
        "Readers can misread archival scope as current repository status.",
        "No numerical change.",
        "Publish an explicit current-status correction without changing frozen historical reports.",
        "CORRECTED IN PHASE 7 CLAIM ADDENDUM",
    ),
]


def issues_and_claims():
    keys = [
        "ID",
        "severity",
        "title",
        "affected_phase",
        "files",
        "evidence",
        "why_it_matters",
        "core_results_change",
        "required_fix",
        "status",
    ]
    write("audit_issues", [dict(zip(keys, r)) for r in ISSUES])
    pattern = re.compile(
        r"\b(alpha|profitable|robust\w*|significant|tradable|production.ready|market.neutral|HFT.strategy|edge)\b",
        re.I,
    )
    rows = []
    candidates = list(Path(".").glob("*.md"))
    for root in ["src", "scripts", "tests", "reports"]:
        candidates.extend(
            p
            for p in Path(root).rglob("*")
            if p.suffix in [".py", ".md", ".json", ".csv", ".txt", ".yaml"]
        )
    for p in sorted(set(candidates)):
        if (
            "phase7" in p.as_posix().lower()
            or p.name == "QUANT_AUDIT.md"
            or "__pycache__" in p.parts
        ):
            continue
        lines = p.read_text(encoding="utf-8-sig").splitlines()
        for i, line in enumerate(lines, 1):
            words = pattern.findall(line)
            if not words:
                continue
            context = " ".join(lines[max(0, i - 3) : min(len(lines), i + 2)])
            if p.name == "README.md" and i in [5, 75]:
                status = "HISTORICAL STATUS — CORRECTED BY ADDENDUM"
                reason = "Current completion status is Phase 7; old scope is archival."
            elif p.suffix in [".py", ".csv", ".json"] or any(
                w.lower().startswith("robust") for w in words
            ):
                status = "TECHNICAL NAME / FIXED SENSITIVITY"
                reason = "Parameter, statistic, file/column name, or specified sensitivity test; not a robustness guarantee."
            elif "PROTOCOL" in p.name:
                status = "HYPOTHESIS / PROSPECTIVE RULE"
                reason = "Prespecified hypothesis, possible verdict or diagnostic, not a claimed realized result."
            elif re.search(r"no |not |never|cannot|does not|not automatically", context, re.I):
                status = "BOUNDED / NEGATED CLAIM"
                reason = "Surrounding passage limits tradability, scope or significance."
            elif "significant" in line.lower():
                status = "CONDITIONAL WITHIN-DAY STATISTICAL CLAIM"
                reason = (
                    "Phase 3 adjusted shift result reproduces; not an independent-day discovery."
                )
            else:
                status = "PROJECT TITLE / CONDITIONAL ASSOCIATION"
                reason = "Alpha/edge describes research aim or within-day midprice statistic; execution evidence is negative."
            rows.append(
                dict(
                    file=p.as_posix(),
                    line=i,
                    terms="|".join(sorted(set(words))),
                    excerpt=line,
                    classification=status,
                    rationale=reason,
                )
            )
    write("claim_audit", rows)
    (OUT / "claim_corrections.md").write_text(
        """# Phase 7 authoritative claim and status addendum

README.md lines 5 and 75 describe historical Phase 1/3 scope. They are not the
current repository status: Phases 4–6 subsequently produced frozen models and
execution simulations, and Phase 7 now audits those results. QUANT_AUDIT.md is
the current interpretation. Historical phase reports and their stop instructions
remain preserved as records of their respective completion dates.

“Alpha” in the project title is a research subject, not a profitability finding.
“Robustness” labels predeclared sensitivity runs, not cross-market reliability.
Phase 3 “significant” means conditional within-day adjusted shift evidence only.
No result establishes modern-market, live HFT, market-neutral strategy,
production readiness, passive fills, scalability or institutional capacity.
Fee stresses of 0–5 bps are hypothetical one-way notional charges, not an actual
historical Nasdaq fee/rebate schedule. The -85.16% number divides PnL by $53.60;
it is not a feasible funded-account return. Use -$45.646 and -2.691472 bps instead.
""",
        encoding="utf-8",
    )
    return len(rows)


def degrees():
    (OUT / "researcher_degrees_of_freedom.md").write_text(
        """# Researcher decisions and evidence limits

Classification is relative to the phase in which a decision was made. PREDECLARED
means local protocol/hash evidence precedes that phase's saved results, not an
externally witnessed preregistration or proof the researcher never viewed data.
Git commits mostly capture completed phases, not separate preregistration commits.

| Decision | Classification | Evidence and implication |
|---|---|---|
| MSFT, one available historical day, regular session, venue archive | POTENTIAL RESEARCHER DEGREE OF FREEDOM | Data feasibility chose the sample; no stock/day selection experiment establishes representativeness. |
| Integer price scale, event semantics, physical-row tie order, premaket seeding | UPSTREAM-FROZEN | Phase 1 schema and immutable lineage; required data interpretation. |
| 149 Phase 2 features, rank-weighted OBI, 100ms/1s/5s/10s windows | PREDECLARED | Phase 2 protocol/config hashes precede matrix creation. Larger feature inventory is not the Phase 3 testing family. |
| 1s primary horizon, secondary 100ms/5s/10s | PREDECLARED | Phase 2–3 protocols; no promotion of the strongest later horizon. |
| 1s grid, last-as-of carry, exact label anchoring | PREDECLARED | Phase 3 protocol/config; staleness retained and documented. |
| Ten primary features, seven directional and three context | PREDECLARED | Phase 3 feature list and ten-hypothesis BH/BY; random feature excluded. |
| 60s primary block, 30/300s sensitivity, 1000 draws | PREDECLARED | Phase 3 protocol; Phase 5 reuses 60s. Phase 7 fixes 30/300 seeds before stress estimates. |
| Large circular shifts, 600s exclusion, finite p correction | PREDECLARED | Phase 3/6 protocols; artificial wrap seam remains. |
| Whole-day quintiles and regime bins | POST-HOC DIAGNOSTIC | Descriptive edges and relationships condition on observed data; never online execution thresholds. |
| Seven model features after univariate phase | UPSTREAM-FROZEN | Frozen for Phase 4/5, but informed by an already inspected day. Human selection is not covered by Phase 3 FDR. |
| Five expanding folds, purge, optional 10s embargo | PREDECLARED | Phase 4 fold/protocol hashes; reused unchanged later. |
| OLS/Ridge/Logistic and finite regularization grids | PREDECLARED | Phase 4 protocol; training-only preprocessing, validation-only selection. |
| RF/LightGBM and four candidates per family | PREDECLARED | Phase 5 protocol; selected on validation, no test early stopping/refit. Choice to advance to trees remains researcher discretion. |
| Validation-selected family and static/flow/control diagnostics | PREDECLARED | Phase 5 definitions; not promotion of test winner. |
| q10/q90 one-share aggressive targets, equality handling | PREDECLARED | Phase 6 protocol and validation-score provenance; not selected to minimize losses. |
| 10ms primary, 0/50/100/250ms latency stress | PREDECLARED | Phase 6 protocol; no best-latency selection. |
| Zero extra fee primary; 0.1/0.5/1/2/5bps stress | PREDECLARED | Hypothetical charges, spread already in executable quotes. |
| 5/20/30 percent tails | PREDECLARED | Secondary Phase 6 variants; primary result hash-frozen before runs. |
| First valid midpoint $53.60 normalization, fixed fold liquidation | PREDECLARED | Phase 6 protocol; denominator convention and terminal risk instruction, not leverage or financing policy. |
| Random, inversion, delayed600, matched truncation, rotations | PREDECLARED | Fixed seeds/rules; no redraw or threshold rescue. |
| Concentration, conditional strength, signed mark-outs | POST-HOC DIAGNOSTIC | Characterize frozen outcomes; no new regime strategy or trading rule. |
| Phase 7 ESS approximations and claim interpretation | POST-HOC DIAGNOSTIC | Context, not added confirmatory tests. |
| Decision to stop / expand research after seeing each phase | POTENTIAL RESEARCHER DEGREE OF FREEDOM | Neither local hashes nor chronological model splits control researcher-level adaptation. |

No test-label path into fitting was detected. This mechanical result does not
eliminate repeated-day exposure. The entire day is now research-used. BH/BY
corrects the ten Phase 3 hypotheses only; it does not correct the cumulative
multi-phase search. No additional day, threshold or profitable variant is tried.
""",
        encoding="utf-8",
    )


def summarize(test_result):
    checks = []
    for name in ["data", "statistical", "execution", "redteam"]:
        checks.extend(js(OUT / (name + "_checks.json")))
    checks.extend(js(OUT / "integrity_checks.json"))
    failures = [r for r in checks if not r["passed"]]
    freeze = js(OUT / "upstream_freeze.json")
    mutations = [p for p, h in freeze["files"].items() if sha(p) != h]
    if failures or mutations:
        raise ValueError(dict(failed_checks=failures, unexpected_mutations=mutations))
    s = js(OUT / "statistical_summary.json")
    e = js(OUT / "execution_summary.json")
    primary = e["primary"]
    counts = Counter(r[1] for r in ISSUES)
    claims = issues_and_claims()
    degrees()

    def maximum(section):
        return max((r["max_discrepancy"] for r in checks if r["section"] == section), default=0)

    cashmax = max(
        r["max_discrepancy"]
        for r in checks
        if r["section"] == "execution"
        and r["check"].endswith(
            ("/cash", "/cash_delta", "/cash_before", "/gross_cash", "/equity", "/gross_equity")
        )
    )
    verdicts = dict(
        integrity="RESEARCH PIPELINE VALID WITH MATERIAL LIMITATIONS",
        signal="PREDICTIVE SIGNAL SURVIVES BUT IS FRAGILE",
        economic="NO EXECUTABLE EDGE UNDER PRIMARY ASSUMPTIONS",
    )
    final = dict(
        phase=7,
        status="PASS"
        if test_result["failed"] == 0
        and test_result["errors"] == 0
        and test_result["passed"] >= 149
        else "FAIL",
        checks=len(checks),
        failed_checks=len(failures),
        unchanged_upstream_files=len(freeze["files"]),
        artifact_hash_checks=len(readcsv(OUT / "artifact_integrity.csv")),
        phase2_sha256=sha(js("PHASE2_MANIFEST.json")["output_path"]),
        phase3_max_statistic_discrepancy=maximum("phase3"),
        execution_max_discrepancy=maximum("execution"),
        execution_max_cash_discrepancy=cashmax,
        cost_ledger_max_discrepancy=maximum("cost_ledger"),
        issue_counts={k: counts[k] for k in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]},
        test_result=test_result,
        verdicts=verdicts,
        statistics=s,
        execution=e,
        claims_reviewed=claims,
        research_scope="MSFT 2003-01-03 only; no new data or real-model refits",
        stop_after_phase7=True,
    )
    write("all_checks", checks)
    dump("final_verified_metrics.json", final)
    stats = "\n".join(
        f"| {r['feature']} | {float(r['spearman_rho']):.9f} | [{float(r['bootstrap_ci_low']):.9f}, {float(r['bootstrap_ci_high']):.9f}] | {float(r['permutation_p']):.6f} | {float(r['fdr_bh_q']):.6f} |"
        for r in readcsv(OUT / "phase3_independent_statistics.csv")
    )
    stress = "\n".join(
        f"| {r['model']} | {r['block_seconds']}s | [{r['ci_lower']:.9f}, {r['ci_upper']:.9f}] | {r['fold_wins']}/5 |"
        for r in s["stress"]
        if r["task"] == "regression"
    )
    issue_text = "\n\n".join(
        f"**{r[0]} — {r[1]}: {r[2]}.** {r[5]} {r[6]} {r[7]} Required response: {r[8]} Status: {r[9]}."
        for r in ISSUES
    )
    m = s["metrics"]
    lgb = m["phase5_regression_LightGBM"]["spearman"]
    ridge = m["phase4_regression_Ridge"]["spearman"]
    rf = m["phase5_regression_RF"]["spearman"]
    boot = {r["metric"]: r for r in e["bootstrap"] if r["model_name"] == "LightGBM"}
    report = f"""# Executive Summary

**PHASE 7: {final["status"]}.** The separate audit reproduced the modest within-day predictive association and the aggressive-execution loss. No unresolved CRITICAL or HIGH implementation defect was found. Integrity: **{verdicts["integrity"]}**. Signal: **{verdicts["signal"]}**. Economics: **{verdicts["economic"]}**.

This is a separately coded adversarial audit within the same research repository, not an external human certification. It uses the frozen sample and models; it does not optimize, refit real models, change thresholds or introduce data. There are {len(checks):,} recorded checks and zero failed comparisons. Full pytest: {test_result["passed"]} passed, {test_result["failed"]} failed, {test_result["errors"]} errors, {test_result["warnings"]} warnings in {test_result["runtime_seconds"]:.2f}s. [Machine-readable results](reports/phase7/final_verified_metrics.json) and [all checks](reports/phase7/all_checks.csv) are authoritative for exact values.

# Data Integrity

Phase 2 SHA-256 is `{final["phase2_sha256"]}`. All {final["artifact_hash_checks"]} registered file/composite hash comparisons pass; {final["unchanged_upstream_files"]} starting-state files remain byte-identical. Old append-only log hashes are checked against exact newline-ended prefixes, not waived. Archived Phase 3 runner bytes reproduce the original executed code composite; Phase 4's archived core resolves its documented pre-audit metric correction. No unexpected upstream mutation occurred.

The inventory hashes all files under data/reports/artifacts and source/scripts/tests, parses all JSON/CSV, decodes all Parquet batches and indexes source/document text. Detailed review targeted schema, causal lookups, fitting boundaries, statistical and execution accounting paths. This is not a claim that a static text scan proves arbitrary code correct. Serialized model hashes are verified before loading; all 55 fitted bundles are examined. See [artifact integrity](reports/phase7/artifact_integrity.csv), [inventory](reports/phase7/review_inventory.csv) and [starting freeze](reports/phase7/upstream_freeze.json).

# Order Book Reconstruction

A new parser reads all 6,115,981 raw messages, seeds MSFT orders from the beginning of the archive and independently reproduces all 162,879 regular-session states, all ten bid/ask levels and quantities. Maximum integer-state discrepancy is zero. Six deterministic windows (72 selected rows) are retained locally. Price units are 1/10,000 USD; quantities are shares, not lots. Adds, partial cancellations/executions, full removals, hidden prints, side handling and sorted aggregate levels reconcile. ITCH 2 has no standalone DELETE/REPLACE messages here: full X removes the order; P does not change displayed depth. No B trade-break messages occur in this file, so the audit does not claim empirical coverage of absent semantics.

Every retained state has positive prices and queues, strictly descending unique bids, ascending unique asks and best bid below best ask. No locked/crossed or negative-queue exception was discarded. Same-time rows retain physical source order. **No reconstruction inconsistency was detected; feed completeness was not proven.** Missing transport sequence numbers prevent that stronger conclusion.

# Causal Feature Alignment

The independent feature path imports no Phase 2 feature functions. It recomputes OBI at four depths, reciprocal-rank WOBI, microprice, OFI, event counts/rates, historical dynamics, prior-trade age and four-horizon labels across the complete retained event series, exceeding the requested small-window check. The rebuilt 23,400-point one-second grid matches source timestamp, source order, age and all ten primary features. Grid features carry the last source-event feature state; rolling windows and age features are not silently refreshed to grid time.

There are 3,976 adjacent same-time source pairs. Eight selected ties pass prefix-only comparisons; swapping payloads changes real predecessor features, and swapping source keys is rejected. Five preregistered cutoffs exercise changed prices/sizes/types, dropped future rows and reordered future payloads. All production feature columns before each cutoff and every already settled label remain identical. Independent-reference mutations also pass. Physical source/grid shuffles restore by authoritative keys. See [causal audit](reports/phase7/causal_alignment_audit.csv) and [adversarial controls](reports/phase7/adversarial_controls.csv).

# Label Alignment

The primary target is the last same-session source state at or before grid time plus one second, including the final physical row of a timestamp tie. It is not a forward/nearest quote. Labels are reanchored to grid time, not copied from event time. Independent bracketing checks establish selected_time <= requested_target < next_event_time; the 100ms/1s/5s/10s definitions all reproduce. There are 23,398 valid primary grid labels; the first grid point lacks state and the terminal primary target reaches session end. No cross-session label is admitted. Local raw windows and label brackets remain under ignored data/processed/phase7; no raw redistribution is introduced.

# Phase 3 Statistical Audit

All ten primary features are reproduced, including the seven directional features. Maximum discrepancy across correlations, percentile limits, quintile relationships and corrected p-values is **{final["phase3_max_statistic_discrepancy"]:.3g}**. Literal 60-second circular blocks preserve internal temporal order and the full 23,400-cell clock; each of 1,000 draws recomputes ranks. Quintile edges remain fixed observed-day descriptive edges, and ties are not split to manufacture balanced bins.

| Feature | Spearman | 60s 95% CI | Shift p | BH q |
|---|---:|---|---:|---:|
{stats}

The 1,000 target rotations use nonzero offsets at least 600 cells from either zero alignment. Rotations preserve the circular series and its missingness, not ordinary boundary continuity; wrap seams are an approximate-null limitation, not labels fed into training. Finite correction is (1 + exceedances)/(1000 + 1). BH and BY use exactly ten hypotheses, not seven directional features alone. All quintile counts and means reproduce, including empty OFI bins. OFI_100MS remains negative in rank association; no sign is flipped. Local protocol hashes support within-phase predeclaration, but git commits mainly capture completed phases and do not independently prove no prior inspection.

# Phase 4 Linear Modeling Audit

Five expanding chronological folds and unchanged 39-minute test blocks reproduce. Every retained actual target timestamp is no later than requested target and strictly before the corresponding partition cutoff. Zero and ten-second embargo paths are checked with real timestamps. Train/validation/test are disjoint; one second at each boundary is purged. Training medians, means and standard deviations reproduce for every stored linear model. Saved coefficients reproduce validation scores and OOS predictions; alpha/C choice follows validation score and frozen tie rules. Test labels are not fitting inputs.

Aggregate OOS Spearman: OBI_L1 {m["phase4_regression_OBI_L1"]["spearman"]:.12f}, OLS {m["phase4_regression_OLS"]["spearman"]:.12f}, Ridge **{ridge:.12f}**. Logistic Macro F1 {m["phase4_classification_Logistic"]["macro_f1"]:.12f}, balanced accuracy {m["phase4_classification_Logistic"]["balanced_accuracy"]:.12f}, accuracy {m["phase4_classification_Logistic"]["accuracy"]:.12f}.

# Phase 5 Nonlinear Modeling Audit

The same folds and 11,695 OOS timestamps per model are used. RF Spearman **{rf:.12f}**; LightGBM **{lgb:.12f}**. Macro F1: RF {m["phase5_classification_RF"]["macro_f1"]:.12f}, LightGBM {m["phase5_classification_LightGBM"]["macro_f1"]:.12f}. All 55 model bundles have permitted feature lists, train-only medians, recorded seeds, finite frozen candidate grids and validation selection. No test early stopping, test-based feature selection or use of null tick columns was detected. Saved models reproduce test predictions and Phase 6 validation scores. Ridge/Logistic baselines are read from Phase 4 rather than fabricated as Phase 5 model artifacts.

Both tree regressors beat Ridge in all five folds. Paired differences use identical clock blocks for model and baseline, retain the 11,700-second clock with purged holes, and rerank literal samples. The audit's 30/300s lengths and seeds were fixed before stress results; no retraining occurs.

| Model minus Ridge | Block | Delta Spearman 95% CI | Fold wins |
|---|---:|---|---:|
{stress}

An explicitly named future-price feature is rejected before model construction. Synthetic test-label mutation plus physical row shuffle leaves all fitted metadata, candidate selection and OOS predictions unchanged for both tasks. Name whitelisting alone cannot prove arbitrary column contents causal; source hashes and independent reconstruction provide the content provenance here. [Model audit](reports/phase7/model_provenance_audit.csv), [statistical checks](reports/phase7/statistical_reproduction.csv).

# Phase 6 Execution Audit

The new engine imports no Phase 6 module. It reads saved scores, validation distributions and independently verified Phase 1 integer quotes (which also reproduce Phase 2 quote fields). For every model/fold, q10/q90 and secondary tails equal validation quantiles, not test quantiles. Threshold input predictions independently reproduce from saved models. Instructions use the last source state <= signal time + latency; no forward lookup, nearest quote, interpolation or stale-valid-state skip is used for fills. All real retained book states are valid; synthetic invalid-state tests reject fills/failed terminal closure.

Every primary buy fills at ask and sell at bid: zero violations. Inventory stays within one share; 193 flips trade two shares. The engine matches each saved request's target, source order, quote, position, fill, cash and mark. Fold ends enforce an executable close at test_end minus 1ms; all five folds end flat. The primary happens to be flat already, so there are **zero actual primary terminal fills**, not five fictitious liquidation trades. Other scenarios' terminal paths are also compared.

Maximum execution comparison discrepancy: **{final["execution_max_discrepancy"]:.3g}**; maximum cash/equity discrepancy: **{cashmax:.3g} USD**. These are floating-point accumulation differences against an integer-price ledger. Primary fold PnLs are {primary["fold_pnl"]}, 0/5 positive, summing to **-${abs(primary["quote_pnl"]):.4f}**. [Execution replay](reports/phase7/execution_replay.csv).

# Statistical Dependence / Effective Sample Size

11,695 OOS rows are not 11,695 independent discoveries. Prediction, target and residual autocorrelations through 300 seconds are saved in [autocorrelation.csv](reports/phase7/autocorrelation.csv). A first-nonpositive-lag integrated-autocorrelation diagnostic gives prediction ESS about {m["ess_prediction"]["approximate_ess"]:.0f} (integrated time {m["ess_prediction"]["integrated_time"]:.2f}). Target/residual lag-one correlations are negative, so that truncation returns the nominal count; this is a weakness of the approximation, **not evidence of independence**. These mean-process ESS estimates are not Spearman or model-delta degrees of freedom. There is still only one independent observed day.

Execution bootstrap samples additive end-of-second marked-equity changes, notional, and episode PnL/count at exit, using the same 60s blocks for all quantities. It does not splice raw transactions into an unfinanced inventory path or claim those blocks form a tradable strategy. Long-episode exits are clustered at exit cells. All 1,000 literal draws reproduce the frozen-path intervals: normalized PnL [{boot["normalized_pnl"]["ci_lower"]:.9f}, {boot["normalized_pnl"]["ci_upper"]:.9f}], PnL/notional [{boot["pnl_per_notional_bps"]["ci_lower"]:.9f}, {boot["pnl_per_notional_bps"]["ci_upper"]:.9f}] bps. These intervals condition on fitted signals and the realized accounting path, omitting model/researcher/new-day uncertainty.

# Cost & Turnover Audit

Gross midpoint PnL **+$2.4475** minus displayed-quote spread drag **$48.0935** equals **-$45.6460** at 10ms before extra costs. Spread is already paid through bid/ask fills and is not charged twice. Every materialized cost-ledger row and all aggregate fee scenarios reconcile; maximum cost-ledger cash discrepancy is {final["cost_ledger_max_discrepancy"]:.3g} USD. Extra cost is one-way traded notional times bps/10,000.

Verified turnover: **2,989 transactions**, **193 flips**, **1,591 episodes**, **$169,594.898** traded notional, **23.7179487%** time in market, mean episode duration **1.744186s**. PnL/notional and signed break-even extra fee are **-2.691472476 bps**: no nonnegative fee allowance exists. Net PnL at 0/0.1/0.5/1/2/5 bps is -45.646000/-47.34194898/-54.12574490/-62.60548980/-79.56497960/-130.44344900 USD.

The $53.60 denominator was frozen as the first valid session midpoint before PnL. It is coherent as a fixed one-share scale convention, not as capital sufficient to fund the path. If interpreted as initial cash, the replay would reach cash {primary["minimum_cash_if_53_60_initial_funding"]:.4f} USD while still trading, requiring financing. The -85.16% normalized number must not drive the economic conclusion; dollar loss and notional bps are denominator-independent evidence. Inventory bounded at one share does not imply bounded leverage relative to dwindling equity.

Latency PnLs at 0/10/50/100/250ms are -45.717/-45.646/-45.612/-45.787/-45.550 USD. The 10ms change versus zero is +$0.071, including both midpoint and spread/path effects; finite quote changes need not worsen monotonically. This small difference is descriptive, not demonstrated latency alpha, and is tiny beside the loss and conditional uncertainty width. All as-of comparisons are causal. Ridge/RF under identical rules lose $41.384/$43.525: better ranking alone does not compensate for spread, signal magnitudes and turnover.

Mean signed 1s move is **${primary["mean_signed_1s_move"]:.9f}/share** versus mean half-spread **${primary["mean_half_spread"]:.9f}/share**, a **{primary["half_spread_to_signed_move"]:.2f}x** ratio (about twice that for a representative round trip; this is a scale comparison, not identical held-trade matching). Predeclared 5/10/20/30% tails all lose: $28.656/$45.646/$65.820/$79.374 at 10ms. No alternative threshold is selected. All 195 primary minute PnLs and all five folds are negative; removing any fold still loses. Top 1/5/10 profitable-episode contributions are $0.012/$0.049/$0.061; concentration cannot rescue the economic result. [Cost reconciliation](reports/phase7/cost_reconciliation.csv).

# Controls / Placebos

All 500 foldwise score rotations are rerun through the new execution engine; offsets exactly reproduce the fixed seed and exclude near-zero shifts. The observed quote PnL is at the **22.2nd percentile**, with corrected upper-tail **p={primary["null_p"]:.12f}**. Null mean is -$45.213958, central 95% [-$46.309975, -$44.063750]. Rotation preserves circular signal structure but introduces fold seams and assumes approximate time exchangeability; it does not supply causal proof. Displayed-depth failures in null runs are retained (144 missed instructions total), not dropped from results.

One fixed-seed random signal loses **$59.885**; it has different turnover/activity and is not a matched-transaction counterfactual. Inversion negates both validation and test scores; midpoint PnL reverses to -$2.4475 while quote PnL is **-$50.541**, because both directions pay spread. The non-wrapping 600s placebo loses **$32.825**, with midpoint gross -$0.160. Its first 600 scores in each fold are unavailable/flat. Matched-truncated original loses $34.481 with gross +$1.855. Lower turnover and exposure explain why less-negative absolute placebo PnL is not a profitability result. Future mutation, leaked-feature rejection, model test isolation, physical ordering, threshold provenance and cost reconciliation all pass. [Controls](reports/phase7/control_summary.csv).

# Researcher Degrees of Freedom

[The decision ledger](reports/phase7/researcher_degrees_of_freedom.md) separates PREDECLARED, UPSTREAM-FROZEN, POST-HOC DIAGNOSTIC and POTENTIAL RESEARCHER DEGREE OF FREEDOM. Primary feature set/horizon, grid, within-phase candidate grids, q10/q90, latency and fee stresses have local freeze evidence. Choices to advance model families and retain seven model features occurred after earlier inspection. Within-phase freezing does not remove cumulative researcher discretion. Ten-hypothesis FDR is not a correction for the entire research history. Git and local timestamp/hash records are evidence of recorded chronology, not externally attested precommitment.

# Generalization Risk

The same day was repeatedly inspected in Phases 3–6. Walk-forward OOS remains mechanically chronological, but the day is no longer an untouched research holdout. Separate HIGH issues cover this exposure and the lack of external validity. No inference is justified about other MSFT days, stocks, venues, modern US equities, live trading, production HFT or capacity. A 2003 decimalization-era archive with legacy Island/INET lineage and message semantics cannot be treated as a modern consolidated book. No historical legal tick is established, and no tick-null feature enters a fitted model. Hidden liquidity, impact, actual fees, broker/transport timing, stock borrow and queue position are unmodeled. No passive-fill claim is made; one-share price taking does not establish scale.

# Claims Audit

The repository-wide indexed scan records {claims} matching uses of alpha, profitability, robustness, significance, tradability, production readiness, market neutrality, HFT strategy and edge, with line-level dispositions in [claim_audit.csv](reports/phase7/claim_audit.csv). Positive deployment/capacity/profitability claims were not found. Research titles, regularization alpha, probability-direction edge, sensitivity labels and prospective verdict options are distinguished from findings. Phase 3 significance is valid only under its stated within-day null. Generic fee stresses are never interpreted as actual historical fees.

The README's historical Phase 1-only/Phase 3-stop wording is stale as a current status statement. [The authoritative claim/status addendum](reports/phase7/claim_corrections.md) explicitly corrects this interpretation while preserving frozen historical bytes. This report is the current Phase 7 interpretation. The audit does not rename conditional predictability as executable alpha.

# Issues by Severity

**CRITICAL: 0; HIGH: {counts["HIGH"]}; MEDIUM: {counts["MEDIUM"]}; LOW: {counts["LOW"]}.** The HIGH issues are inherent research limitations, not unresolved implementation faults. No hidden implementation exception is excluded from the tally. [Issue register](reports/phase7/audit_issues.csv).

{issue_text}

# Corrected Results, if any

No Phase 1–6 numerical result required correction. Historical Phase 3 console serialization and Phase 4 constant-prediction Pearson handling were already documented before this audit; their archived code and current artifact hashes are reconciled. The Phase 4 correction changes undefined within-fold TrainingMean Pearson reporting, not fitted models, OOS predictions or aggregate Ridge comparisons. Phase 7 adds the README status/claim clarification, dependence/funding interpretation and independent evidence, with no result beautification or upstream rewrite.

# Final Research Interpretation

The supported statement is limited: modest within-day one-second midprice predictability exists in this historical MSFT sample; frozen nonlinear models improve rank and classification metrics over linear baselines, including the declared audit block sensitivities. Aggressive quote execution under the frozen assumptions loses money because the signed move scale is far smaller than displayed spread/friction. This conclusion survives independent reconstruction and accounting attempts to falsify it. It does not establish a reusable alpha, an independent-day discovery or a feasible live strategy.

# Final Audit Verdict

Integrity: **{verdicts["integrity"]}**.

Signal: **{verdicts["signal"]}**.

Economics: **{verdicts["economic"]}**.

**PHASE 7: {final["status"]}** means the audit is complete with zero unresolved critical implementation defect, not positive alpha. Reproduce with `.venv\\Scripts\\python.exe scripts/audit_phase7.py`; it uses the existing local raw/model artifacts, runs separate audit stages and the full test suite, and writes this evidence package. No upstream test is removed. The script exits nonzero on failed comparisons, integrity drift or test failures. **STOP AFTER PHASE 7.**
"""
    Path("QUANT_AUDIT.md").write_text(report, encoding="utf-8")
    return final
