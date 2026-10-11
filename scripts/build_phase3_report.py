"""Report every preregistered hypothesis, including contrary and empty-bin results."""

import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lob_research.download import sha256
from lob_research.phase2.artifacts import write_csv
from lob_research.phase3.artifacts import check_frozen


def main():
    cfg = check_frozen()
    directory = Path(cfg["reports_dir"])

    def read(name):
        with (directory / (name + ".csv")).open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    primary = read("primary_feature_statistics")
    byname = {r["feature"]: r for r in primary}
    tod = read("time_of_day_stability")
    placebo = read("placebo_results")
    horizons = read("horizon_decay")
    bootstrap = read("bootstrap_intervals")
    audit = json.loads((directory / "independent_verification.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path("PHASE3_MANIFEST.json").read_text(encoding="utf-8"))
    freeze = json.loads((directory / "protocol_freeze.json").read_text(encoding="utf-8"))
    tests = ET.parse(directory / "test_results.xml").getroot().find("testsuite")
    pytest_output = (directory / "pytest_output.txt").read_text(encoding="utf-8-sig")
    runtime = re.search(r"in ([0-9.]+)s", pytest_output).group(1)
    warnings_match = re.search(r"(\d+) warnings?", pytest_output)
    warnings = int(warnings_match.group(1)) if warnings_match else 0
    failures = int(tests.attrib["failures"]) + int(tests.attrib["errors"])
    skipped = int(tests.attrib["skipped"])
    control = read("random_control")[0]
    stale = read("grid_staleness")[0]
    names = cfg["features"]
    directional = cfg["directional_features"]
    verdicts = []
    for row in primary:
        name = row["feature"]
        rho = float(row["spearman_rho"])
        q = float(row["fdr_bh_q"])
        lo, hi = float(row["bootstrap_ci_low"]), float(row["bootstrap_ci_high"])
        same = sum((float(r["spearman_rho"]) * rho) > 0 for r in tod if r["feature"] == name)
        pl = next(r for r in placebo if r["feature"] == name)
        expected = name not in directional or rho > 0
        monotonic = name not in directional or (
            int(row["nonempty_quintiles"]) == 5 and float(row["mean_return_monotonicity"]) >= 0.8
        )
        consistent = (
            expected
            and (lo > 0 or hi < 0)
            and q <= 0.05
            and same >= 4
            and pl["absolute_weakened"] == "True"
            and monotonic
        )
        verdict = (
            "CONSISTENT EVIDENCE"
            if consistent
            else "WEAK EVIDENCE"
            if q <= 0.05
            else "NO EVIDENCE"
            if lo <= 0 <= hi
            else "INCONCLUSIVE"
        )
        verdicts.append(
            {
                "feature": name,
                "verdict": verdict,
                "expected_direction_met": expected,
                "same_sign_time_blocks": same,
                "nonempty_quintiles": int(row["nonempty_quintiles"]),
                "monotonicity_requirement_met": monotonic,
                "placebo_weakened": pl["absolute_weakened"] == "True",
            }
        )
    families = {
        "Imbalance": names[:4],
        "Microprice": ["MICROPRICE_MINUS_MID"],
        "Order flow": ["OFI_100MS", "OFI_1S"],
        "Context": names[7:],
    }
    family_rows = []
    for family, columns in families.items():
        vv = [v["verdict"] for v in verdicts if v["feature"] in columns]
        result = (
            "CONSISTENT WITH SHORT-HORIZON PREDICTIVE INFORMATION"
            if "CONSISTENT EVIDENCE" in vv
            else "WEAK / FRAGILE ASSOCIATION"
            if "WEAK EVIDENCE" in vv
            else "NO RELIABLE ASSOCIATION"
            if all(v == "NO EVIDENCE" for v in vv)
            else "INCONCLUSIVE"
        )
        family_rows.append({"family": family, "verdict": result, "features": ";".join(columns)})
    consistent_families = [
        r["family"]
        for r in family_rows
        if r["family"] != "Context"
        and r["verdict"] == "CONSISTENT WITH SHORT-HORIZON PREDICTIVE INFORMATION"
    ]
    large_families = [
        family
        for family in consistent_families
        if any(
            v["verdict"] == "CONSISTENT EVIDENCE"
            and v["feature"] in families[family]
            and abs(float(byname[v["feature"]]["spearman_rho"])) >= 0.1
            for v in verdicts
        )
    ]
    if len(large_families) >= 2:
        overall = "CLEAR WITHIN-DAY MICROSTRUCTURE ASSOCIATION"
    elif len(consistent_families) >= 2:
        overall = "MODEST WITHIN-DAY MICROSTRUCTURE ASSOCIATION"
    elif any(float(r["fdr_bh_q"]) <= 0.05 for r in primary):
        overall = "WEAK / FRAGILE WITHIN-DAY ASSOCIATION"
    elif all(float(r["bootstrap_ci_low"]) <= 0 <= float(r["bootstrap_ci_high"]) for r in primary):
        overall = "NO RELIABLE WITHIN-DAY ASSOCIATION"
    else:
        overall = "INCONCLUSIVE"
    hmap = {(r["feature"], int(r["horizon_ms"])): float(r["spearman_rho"]) for r in horizons}
    imbalance_consistent = sum(v["verdict"] == "CONSISTENT EVIDENCE" for v in verdicts[:4])
    h1 = (
        "SUPPORTED"
        if imbalance_consistent >= 3
        else "PARTIALLY SUPPORTED"
        if any(float(r["spearman_rho"]) > 0 and float(r["fdr_bh_q"]) <= 0.05 for r in primary[:4])
        else "NOT SUPPORTED"
        if all(float(r["spearman_rho"]) <= 0 for r in primary[:4])
        else "INCONCLUSIVE"
    )
    micro = verdicts[4]
    mr = primary[4]
    h2 = (
        "SUPPORTED"
        if micro["verdict"] == "CONSISTENT EVIDENCE"
        else "PARTIALLY SUPPORTED"
        if float(mr["spearman_rho"]) > 0 and float(mr["fdr_bh_q"]) <= 0.05
        else "NOT SUPPORTED"
        if float(mr["spearman_rho"]) <= 0
        else "INCONCLUSIVE"
    )
    stronger = [
        abs(hmap["OFI_1S", 1000]) > abs(hmap["OBI_L1", 1000]),
        abs(hmap["OFI_100MS", 100]) > abs(hmap["OBI_L1", 100]),
    ]
    h3 = (
        "SUPPORTED"
        if all(stronger)
        else "PARTIALLY SUPPORTED"
        if any(stronger)
        else "NOT SUPPORTED"
    )
    decaying = [
        name
        for name in directional
        if all(
            abs(hmap[name, a]) >= abs(hmap[name, b])
            for a, b in [(100, 1000), (1000, 5000), (5000, 10000)]
        )
    ]
    h4 = (
        "SUPPORTED"
        if len(decaying) >= 5
        else "PARTIALLY SUPPORTED"
        if decaying
        else "NOT SUPPORTED"
    )
    hypotheses = [
        {
            "hypothesis": 1,
            "verdict": h1,
            "evidence": f"{imbalance_consistent}/4 imbalance columns satisfy the predeclared consistency rubric; all four have positive full-day rho and positive point estimates in all five time blocks.",
        },
        {
            "hypothesis": 2,
            "verdict": h2,
            "evidence": "Microprice has positive 1s association and monotonic mean returns; its 100ms rank association is approximately zero and slightly negative. Support is limited to the primary 1s result, not all horizons.",
        },
        {
            "hypothesis": 3,
            "verdict": h3,
            "evidence": "At 1s, |OFI_1S rho| is smaller than |OBI_L1 rho|. At 100ms, |OFI_100MS rho| is larger but NEGATIVE, contrary to the directional hypothesis. Magnitude-only partial support is not positive-flow confirmation or a formal dominance test.",
        },
        {
            "hypothesis": 4,
            "verdict": h4,
            "evidence": f"{len(decaying)}/7 signed features show nonincreasing absolute rho across all four horizons. Static imbalance and microprice generally strengthen over longer horizons; OFI changes sign.",
        },
        {
            "hypothesis": 5,
            "verdict": "INCONCLUSIVE",
            "evidence": "Event/grid magnitudes differ in both directions. No naive event-level standard errors were computed, so excess statistical confidence is not directly measured. Dependence remains a design constraint rather than a demonstrated iid comparison.",
        },
    ]
    write_csv(directory / "feature_verdicts.csv", verdicts)
    write_csv(directory / "family_verdicts.csv", family_rows)
    write_csv(directory / "hypothesis_results.csv", hypotheses)
    permitted = (
        audit["status"] in ["CORE_PASS", "PASS"]
        and audit["critical_failures"] == 0
        and failures == 0
        and skipped == 0
        and int(tests.attrib["tests"]) >= 101
        and control["passed"] == "True"
    )
    phase = "PASS" if permitted else "FAIL"

    def table(headers, rows):
        return [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
            "",
        ]

    def fmt(x, digits=6):
        value = float(x)
        return f"{value:.{digits}f}" if math.isfinite(value) else "unavailable"

    text = [
        "# Phase 3 — Univariate Microstructure Signal Research & Statistical Validation",
        "",
        f"**Overall research verdict: {overall}.** Methodology decision: **PHASE 3: {phase}**.",
        "",
        "**One stock, one trading day, 2003.** These results concern only this historical MSFT session. They establish no cross-day or cross-stock stability, modern-market relevance, live tradability, capacity or execution profitability. A feature predicting the midprice is not automatically tradable alpha.",
        "",
        "## Sample and immutable upstream",
        "",
        f"Instrument/date: **MSFT — 2003-01-03**, Nasdaq-hosted historical ITCH 2.0, legacy Island/INET lineage (historically inferred). Session: **[09:30,16:00) America/New_York**, **[14:30,21:00) UTC**. This remains a time-of-day proxy without certified halt/auction states. Phase 2 has {manifest['event_rows']:,} rows, 220 columns and 149 frozen features. Phase 3 uses exactly ten predeclared features.",
        "",
        f"Verified Phase-2 SHA-256: `{cfg['input_sha256']}`. The matrix, Phase-2 code, tests, labels, root config and all original reports are unchanged. Phase-3 grid SHA-256: `{manifest['output_sha256']}`; path `{manifest['output_path']}`.",
        "",
        f"There are **{manifest['grid_rows']:,} grid points**, **{manifest['valid_primary_targets']:,} valid 1s targets** and {int(stale['non_null_count']):,} points with valid current states. The opening grid point precedes the first retained event and is unavailable; the final point's 1s target equals the excluded close. OFI_1S and EVENT_RATE_1S have 23,397 valid primary pairs because one additional source event lacks full rolling history; other primary features have 23,398. FLAT observations are retained.",
        "",
        "At grid time g, select the last Phase-2 state with timestamp<=g, resolving ties by final source order. Features are copied at that post-event source key; their rolling windows and time-since values are NOT recomputed at g. Targets ARE reanchored to g+h, using the final same-session state<=g+h, with exact integer-mid direction, invalid-segment rejection and tau<close. No-update intervals persist. Event-anchored forward labels are never substituted for grid targets.",
        "",
        f"Grid-state age: mean {float(stale['mean']):.3f}ms, median {float(stale['median']):g}ms, p95 {float(stale['p95']):g}ms, p99 {float(stale['p99']):g}ms, maximum {float(stale['max']):g}ms. No staleness cutoff was imposed. Carried source-state features can be materially older than the grid clock; this is part of the frozen estimand, especially for 100ms OFI.",
        "",
        *table(
            ["Horizon", "Available", "Missing", "UP", "FLAT", "DOWN"],
            [
                [r["horizon"], r["valid"], r["unavailable"], r["up"], r["flat"], r["down"]]
                for r in read("grid_label_summary")
            ],
        ),
        "## Preregistration and statistical method",
        "",
        f"The protocol and feature set were hash-frozen at `{freeze['frozen_at_utc']}`, before study execution. Primary horizon remains **1 second**. No feature, direction, seed, block length or boundary was changed after results. `PHASE3_PROTOCOL.md`, `PHASE3_FEATURE_SET.md`, `config.phase3.yaml` and the immutable opening of `RESEARCH_LOG.md` record all decisions.",
        "",
        "Primary statistic: Spearman correlation of average ranks on finite grid pairs. Pearson is secondary. No ordinary iid/asymptotic p-value is reported. Adjacent clock samples remain dependent, and event rows are strongly dependent through clustering, persistent books, overlapping history and labels. Neither 162,879 events nor 23,400 grid points are asserted to be independent observations.",
        "",
        "Primary inference uses **1,000 paired circular moving-block bootstrap draws of 60 seconds**, with exact ranks recalculated in each draw. Seeds are 314219 (60s), 314189 (30s robustness) and 314459 (300s robustness). All share paired clock positions, including missing cells; sample weights implement literal duplication exactly. Percentile 95% intervals are reported for rho and Q5-Q1 mean return. Q5-Q1 intervals condition on fixed observed-day bin edges, not estimated-threshold uncertainty.",
        "",
        "The null uses **1,000 target circular shifts**, seed 271828, offsets uniformly sampled from 600..22,800 seconds inclusive. Target ordering and missingness shift together. Two-sided p=(1+number of |null rho|>=|observed rho|)/1001. Minimum p is 0.000999, not zero. This approximate alignment null preserves circular temporal structure but does not guarantee exact exchangeability under intraday nonstationarity. The circular bootstrap also creates an artificial end/start seam and assumes approximate stationarity.",
        "",
        "BH correction covers exactly the ten primary feature p-values. BY is also reported because positive dependence needed for BH is not guaranteed by correlated features. Both corrections concern these approximate marginal p-values and cannot fix their calibration. Secondary horizon/state/event diagnostics have no discovery-level inferential status. Method conventions: [SciPy Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html), [SciPy BH/BY](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).",
        "",
        "## Primary 1-second results",
        "",
        "Rows remain in preregistered order. Delta return is Q5 minus Q1 mean **mid return**, expressed in basis points (return x10,000); delta edge is the difference in P(UP)-P(DOWN), a probability difference. These are ex-post outcome contrasts, not strategy returns. Mono is the correlation of quintile index with quintile mean return; nan means five nonempty bins do not exist.",
        "",
        "| Feature | N | Spearman rho | 60s-block 95% CI | Shift p | BH q | BY q | Delta return bp | Delta edge | Mono |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    # Format independently duplicated by verify_phase3.py for exact report reconciliation.
    for row in primary:

        def f(key):
            return float(row[key])

        text.append(
            f"| {row['feature']} | {int(row['n']):,} | {f('spearman_rho'):.6f} | [{f('bootstrap_ci_low'):.6f}, {f('bootstrap_ci_high'):.6f}] | {f('permutation_p'):.6f} | {f('fdr_bh_q'):.6f} | {f('fdr_by_q'):.6f} | {f('q5_minus_q1_mean_return') * 10000:.6f} | {f('q5_minus_q1_direction_edge'):.6f} | {f('mean_return_monotonicity'):.3f} |"
        )
    text += [
        "",
        "Seven primary columns pass both BH and BY at .05, but they are not seven independent discoveries. All four imbalance columns and microprice have small positive rank associations. OFI_100MS has a significant **negative** rank association, contrary to the predeclared positive direction. OFI_1S is positive overall but less stable across time and regimes. The three context features have intervals containing zero and no adjusted primary evidence. No sign was flipped.",
        "",
        "### Bootstrap sensitivity and outcome-difference intervals",
        "",
        *table(
            [
                "Feature",
                "30s rho CI",
                "60s rho CI (primary)",
                "300s rho CI",
                "60s Q5-Q1 mean-return CI, bp",
            ],
            [
                [
                    name,
                    *[
                        f"[{fmt(next(r for r in bootstrap if r['feature'] == name and r['metric'] == 'spearman_rho' and int(r['block_seconds']) == b)['ci_low'])}, {fmt(next(r for r in bootstrap if r['feature'] == name and r['metric'] == 'spearman_rho' and int(r['block_seconds']) == b)['ci_high'])}]"
                        for b in [30, 60, 300]
                    ],
                    "["
                    + ", ".join(
                        fmt(
                            float(
                                next(
                                    r
                                    for r in bootstrap
                                    if r["feature"] == name
                                    and r["metric"] == "q5_minus_q1_mean_return"
                                    and int(r["block_seconds"]) == 60
                                )[k]
                            )
                            * 10000
                        )
                        for k in ["ci_low", "ci_high"]
                    )
                    + "]",
                ]
                for name in names
            ],
        ),
        "All reported bootstrap intervals use 1,000 finite rho replicates per feature and length. The sign/exclusion conclusions for primary rho are unchanged at 30s and 300s. This sensitivity does not establish nominal coverage or eliminate slow intraday confounding.",
        "",
        "## Quintiles, direction and price scale",
        "",
        "Full-day empirical cutoffs are **EX-POST DESCRIPTIVE BINS, not executable online thresholds**. Equal values stay together in the lower bin. OBI_L1/L5/WOBI and microprice have mean-return monotonicity 1.0; OBI_L10 has 0.9 with a local reversal. Direction-edge monotonicity is 1.0 for OBI_L1/L5 and 0.9 for OBI_L10/WOBI/microprice; these high five-point summaries are diagnostics, not separate significance tests.",
        "",
        "OFI_100MS has only Q1 and Q5 populated (20,260 and 3,138 rows); Q2-Q4 remain empty because all four quantile cutoffs coincide at zero. OFI_1S has four populated bins with Q3 empty. Their five-bin monotonicity is explicitly unavailable. We did not split ties, relabel bins or retune boundaries. OFI_100MS has positive Q5-Q1 mean return but negative direction-edge difference and negative overall rank association; mean effects and ranked directional behavior need not agree for tied/heavy-tailed distributions.",
        "",
        "OBI_L1 Q1/Q5 mean returns are -0.089308/+0.107138 bp, with P(UP)-P(DOWN) -0.028284/+0.034067 (difference 6.235081 percentage points). The corresponding mean USD movement contrast is $0.001046760. Microprice Q5-Q1 mean-return contrast is 0.173537 bp ($0.000925049); OFI_1S is 0.185554 bp ($0.000989183). These are small conditional outcome differences. No unverified tick conversion is made.",
        "",
        "Every primary feature has per-bin counts, mean/median returns and direction probabilities in `primary_feature_quintiles.csv`; conditional feature p05/p25/median/p75/p95/IQR by UP/FLAT/DOWN are in `primary_direction_statistics.csv`. All populated primary quintile median future returns are zero, reflecting retained FLAT observations. Fixed imbalance bins are separate secondary diagnostics in `extreme_imbalance.csv`. Median absolute 1s mid movement is $0.000 and median spread is $0.029; this is descriptive price-scale context only, not an execution-profit calculation.",
        "",
        "## Predeclared hypothesis outcomes",
        "",
        *table(
            ["Hypothesis", "Verdict", "Evidence"],
            [[r["hypothesis"], r["verdict"], r["evidence"]] for r in hypotheses],
        ),
        "## Horizon decay / persistence",
        "",
        *table(
            ["Feature", "100ms rho", "1s rho — PRIMARY", "5s rho", "10s rho"],
            [[name, *[fmt(hmap[name, h]) for h in [100, 1000, 5000, 10000]]] for name in names],
        ),
        "Association does **not** show the proposed general monotonic decay. Static imbalance strengthens toward 5-10s. Microprice is nearly zero/slightly negative at 100ms and positive at later horizons. OFI_100MS moves from negative at 100ms/1s to positive at 5s/10s; OFI_1S is negative at 100ms, positive at 1s, larger at 5s and somewhat lower at 10s. This does not authorize changing the 1s primary horizon. Horizon-specific quintile direction differences use the original 1s edges and are saved in `horizon_decay.csv` and `horizon_quintiles.csv`.",
        "",
        "## Time-of-day and state stability",
        "",
        "Blocks are fixed 78-minute windows: A 09:30-10:48, B 10:48-12:06, C 12:06-13:24, D 13:24-14:42, E 14:42-16:00 Eastern. Each contains 4,680 grid points; per-feature valid counts are explicitly recorded.",
        "",
        *table(
            ["Feature", "A rho", "B rho", "C rho", "D rho", "E rho", "Interior rho"],
            [
                [
                    name,
                    *[
                        fmt(
                            next(r for r in tod if r["feature"] == name and r["block"] == b)[
                                "spearman_rho"
                            ]
                        )
                        for b in "ABCDE"
                    ],
                    fmt(
                        next(
                            r
                            for r in read("interior_session_robustness")
                            if r["feature"] == name and r["sample"] == "interior"
                        )["spearman_rho"]
                    ),
                ]
                for name in names
            ],
        ),
        "Imbalance and microprice point estimates are positive in all five periods; their size is generally larger near the open and weaker in the middle. This is distributed sign evidence, not proof that each period is significant. OFI_1S is negative in block C and positive in four blocks. OFI_100MS is negative in every block, against the original direction hypothesis. Removing the first/last 30 minutes leaves 19,800 interior pairs and preserves those full-day directional signs; the full session remains primary.",
        "",
        "### EX-POST DESCRIPTIVE REGIME ANALYSIS",
        "",
        *table(
            [
                "Feature",
                "Activity low",
                "Activity medium",
                "Activity high",
                "Spread low",
                "Spread medium",
                "Spread high",
            ],
            [
                [
                    name,
                    *[
                        fmt(
                            next(
                                r
                                for r in read(filename)
                                if r["feature"] == name and r["regime"] == regime
                            )["spearman_rho"]
                        )
                        for filename in ["activity_conditioning", "spread_conditioning"]
                        for regime in ["low", "medium", "high"]
                    ],
                ]
                for name in directional
            ],
        ),
        "Activity terciles contain 7,987 / 8,765 / 6,645 usable source-activity observations because equal integer rates are not split; spread terciles contain 7,806 / 7,802 / 7,790 target-valid observations (OFI_1S has one additional missing value). These ex-post regimes are not online filters. OBI_L1/microprice remain positive across activity states, while deep OBI weakens at high activity. OFI_1S changes from negative low-activity to positive high-activity association; low-spread effects are stronger, and its high-spread estimate is negative. OFI_100MS also changes sign across spread states. These results caution against a universal flow interpretation.",
        "",
        "## Feature redundancy",
        "",
        *table(
            ["Pair", "Grid Spearman"],
            [
                [
                    a + " / " + b,
                    fmt(
                        next(
                            r
                            for r in read("feature_correlation")
                            if r["feature_a"] == a and r["feature_b"] == b
                        )["spearman_rho"]
                    ),
                ]
                for a, b in [
                    ("OBI_L1", "OBI_L5"),
                    ("OBI_L1", "OBI_L10"),
                    ("OBI_L5", "OBI_L10"),
                    ("OBI_L1", "WOBI_L10"),
                    ("OBI_L1", "MICROPRICE_MINUS_MID"),
                    ("OFI_1S", "OFI_100MS"),
                    ("OFI_1S", "EVENT_RATE_1S"),
                ]
            ],
        ),
        "OBI_L1 and microprice deviation have rho approximately 0.945 and are strongly redundant. The two named families satisfy the preregistered overall rubric but are NOT two independent economic discoveries. OFI window variants also overlap. A near-zero OFI/activity rank correlation does not remove the strong conditional activity dependence shown above. All 100 pairwise values are saved in `feature_correlation.csv`.",
        "",
        "## Event-level versus grid",
        "",
        *table(
            ["Feature", "Event N", "Event rho — DESCRIPTIVE", "Grid N", "Grid rho — PRIMARY"],
            [
                [r["feature"], r["n"], fmt(r["spearman_rho"]), r["grid_n"], fmt(r["grid_rho"])]
                for r in read("event_level_secondary")
            ],
        ),
        "Event-level estimates are HIGH-FREQUENCY DESCRIPTIVE ESTIMATES. They are not uniformly larger: OBI_L1, microprice and OFI_1S are larger in magnitude, but several depth imbalances are smaller. Event/grid differences combine activity weighting, repeated information, source-feature age and distinct event/grid target anchors; they cannot be attributed solely to pseudo-replication. The much larger event count is not independent evidence or permission to use iid uncertainty. Primary inference remains grid-based and block-aware.",
        "",
        "## Controls and independent verification",
        "",
        f"Exactly one random feature, seed {control['seed']}: rho {float(control['spearman_rho']):.9f}, 60s bootstrap CI [{float(control['bootstrap_ci_low']):.9f}, {float(control['bootstrap_ci_high']):.9f}], shift p={float(control['permutation_p']):.9f}. **PASS**, no redraws. It is excluded from the ten-feature FDR family. Its 30s/300s intervals also contain zero.",
        "",
        *table(
            [
                "Feature",
                "Aligned rho on same pairs",
                "600s lag placebo rho",
                "Absolute association weakens",
            ],
            [
                [
                    r["feature"],
                    fmt(r["aligned_rho_same_pairs"]),
                    fmt(r["placebo_rho"]),
                    r["absolute_weakened"],
                ]
                for r in placebo
            ],
        ),
        "The fixed +10-minute causal lag weakens all seven directional features on identical surviving pairs. It weakens spread but does not weaken the two context features with already negligible primary associations; those exceptions are reported rather than treated as positive evidence. No alternative placebo offset was tried. Sign inversion passes all seven signed features to 1e-12.",
        "",
        f"Independent audit: **{audit['check_count']:,} checks, {audit['critical_failures']} critical failures**; {audit['manual_feature_analyses']} independent small-sample feature analyses and {audit['manual_comparisons']:,} scalar comparisons, {audit['manual_failures']} mismatches. Grid source/target selections are independently reconstructed with Python bisect; full primary rho is recomputed with SciPy; scalar average ranks on 101 deterministic samples per feature independently reproduce ranks/rho/quintiles. All saved interval bounds and p-values are recomputed from draw distributions, with 21 null draws per feature and three literal bootstrap replicates per feature/block length independently recalculated. This is independent arithmetic/code, not an external dataset replication.",
        "",
        f"**{audit['trace_count']} deterministic real grid traces** cover time, positive/negative/near-zero imbalance, signed flow/microprice, activity extremes and UP/FLAT/DOWN examples. Traces and row-level manual comparisons remain local-only. `scripts/verify_phase3.py` also checks immutable hashes, feature list, primary horizon, seeds, FDR, bin reconciliation, time blocks, controls, event/grid separation and absence of model artifacts; the final report table is reconciled to CSV values.",
        "",
        f"Complete repository suite: **{tests.attrib['tests']} passed, {failures} failures/errors, {skipped} skipped, {warnings} warnings, {runtime}s**. All 49 Phase-1 and 32 Phase-2 tests remain unchanged; 20 Phase-3 cases cover grid/as-of/ties/invalid intervals, bins, ranks, literal versus weighted block resampling, shifts/seeds, FDR, time blocks, random controls, lagging/sign inversion and event/grid target separation.",
        "",
        "## Figures",
        "",
        *[
            f"{i}. [{label}](reports/phase3/figures/{file}.png) ([SVG](reports/phase3/figures/{file}.svg))."
            for i, (label, file) in enumerate(
                [
                    ("OBI_L1 quintile mean return", "01_obi_quintile_return"),
                    ("OFI_1S quintile mean return", "02_ofi_quintile_return"),
                    ("Microprice quintile direction difference", "03_microprice_direction_edge"),
                    ("Primary rho with block intervals", "04_primary_spearman"),
                    ("Horizon persistence/decay", "05_horizon_decay"),
                    ("UP/FLAT/DOWN by quintile", "06_direction_probabilities"),
                    ("Time-of-day stability", "07_time_of_day"),
                    ("Feature correlation heatmap", "08_feature_correlation"),
                ],
                1,
            )
        ],
        "",
        "All eight figures explicitly state MSFT, single historical trading day and 2003-01-03. PNG and editable SVG exports were visually inspected; empty bins are visible and not interpolated into fake observations.",
        "",
        "## Family and overall research verdicts",
        "",
        *table(["Family", "Verdict"], [[r["family"], r["verdict"]] for r in family_rows]),
        *table(
            ["Feature", "Evidence rubric", "Same-sign blocks", "Nonempty quintiles"],
            [
                [
                    r["feature"],
                    r["verdict"],
                    str(r["same_sign_time_blocks"]) + "/5",
                    r["nonempty_quintiles"],
                ]
                for r in verdicts
            ],
        ),
        f"**{overall}**. The positive book-state associations are small and highly redundant; flow relationships depend on horizon/time/state and partly oppose the original hypothesis. This is within-session microstructure evidence only. The rubric intentionally marks flow fragile because its signed/five-quintile stability requirements are not met, even though its two-sided tests reject the approximate alignment null.",
        "",
        "## Limitations",
        "",
        "Single MSFT day; 2003 market structure; historically inferred Island/INET venue lineage; transport completeness unverified; visible book only; hidden liquidity and hidden queues absent despite observed P executions; unverified historical tick size; no independent exchange snapshot; regular-session proxy without halt/auction certification; millisecond timestamps without receive time. There is no queue-position model, no execution costs, no latency model, no cross-stock replication, no cross-day replication, no modern-market validation and no strategy PnL.",
        "",
        "Within-day bootstrap and circular-shift evidence rely on approximate temporal regularity and cannot create independent days. Nonstationarity, zero-heavy returns/OFI, carried source-state features, finite permutation resolution and dependence between hypotheses limit interpretation. BY robustness does not cure null misspecification. No whole-day quantile boundary is an online decision rule. Raw redistribution rights remain unestablished; raw data, row-level grid/traces and manual records stay excluded from Git.",
        "",
        "## Engineering provenance and stopping point",
        "",
        f"Measured single numerical study runtime: {manifest['runtime_seconds']:.3f}s; peak working set {manifest['peak_working_set_bytes']:,} bytes. `PHASE3_MANIFEST.json` records grid/CSV hashes, package versions, seeds and the original executed code digest. `requirements.phase3.txt` pins this environment without changing upstream requirements.",
        "",
        "One presentation-only defect occurred after all numerical outputs, manifest and complete status were saved: the final console JSON print could not serialize a NumPy boolean. That print was amended; no analytic module or saved result changed, and the study was not rerun. Original runner bytes, original/current code digests and the exact console-only change are independently verified in `console_fix_audit.json`. This is not a protocol or statistical-method amendment.",
        "",
        f"Methodology gates: frozen upstream/protocol/features; primary 1s clock inference; 60s bootstrap plus 30/300s robustness; circular-shift null and BH/BY; all horizons/time/state diagnostics; random/placebo/sign controls; independent scalar checks and reproducible report; full tests; zero models and zero execution simulations. **PHASE 3: {phase}** means the methodology gate is satisfied, not that tradable alpha is established. Phase 4 has not begun. Stop and await the next instruction.",
        "",
        f"PHASE 3: {phase}",
    ]
    Path("PHASE3_REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    results = {
        "phase_decision": phase,
        "overall_research_verdict": overall,
        "primary_feature_verdicts": verdicts,
        "family_verdicts": family_rows,
        "hypotheses": hypotheses,
        "tests": {
            "passed": int(tests.attrib["tests"]),
            "failed_errors": failures,
            "skipped": skipped,
            "warnings": warnings,
            "runtime_seconds": float(runtime),
        },
        "report_sha256": sha256("PHASE3_REPORT.md"),
        "report_builder_sha256": sha256(__file__),
        "source_sha256": cfg["input_sha256"],
    }
    (directory / "research_verdict.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    log = Path("RESEARCH_LOG.md")
    logtext = log.read_text(encoding="utf-8")
    if "## Phase 3 frozen results" not in logtext:
        with log.open("a", encoding="utf-8") as f:
            f.write(
                "\n## Phase 3 frozen results\n\n"
                + overall
                + ". No hypotheses, seeds, features or horizons were changed.\n\n"
            )
            for row in hypotheses:
                f.write(f"- Hypothesis {row['hypothesis']}: {row['verdict']}. {row['evidence']}\n")
            f.write(
                "\nFull numerical evidence: PHASE3_REPORT.md and reports/phase3/*.csv. Original preregistration entry above remains verbatim. No Phase 4 work.\n"
            )
    print(
        json.dumps(
            {"phase": phase, "research_verdict": overall, "feature_verdicts": verdicts}, indent=2
        )
    )


if __name__ == "__main__":
    main()
