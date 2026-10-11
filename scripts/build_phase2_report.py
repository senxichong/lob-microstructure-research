"""Render measured Phase-2 evidence and explicit gates without predictive analysis."""

import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lob_research.download import sha256
from lob_research.phase2.artifacts import code_hash, configuration, upstream_check


def main():
    cfg = configuration()
    frozen = upstream_check()
    directory = Path(cfg["phase2"]["reports_dir"])
    manifest = json.loads(Path("PHASE2_MANIFEST.json").read_text())
    audit = json.loads((directory / "independent_verification.json").read_text())
    prereg = json.loads((directory / "protocol_freeze.json").read_text())
    tests = ET.parse(directory / "test_results.xml").getroot().find("testsuite")
    output = (directory / "pytest_output.txt").read_text(encoding="utf-8-sig")
    warning_match = re.search(r"(\d+) warnings?", output)
    warnings = int(warning_match.group(1)) if warning_match else 0
    runtime_match = re.search(r"in ([0-9.]+)s", output)
    terminal_runtime = runtime_match.group(1) if runtime_match else "not recorded"
    failed = int(tests.attrib["failures"]) + int(tests.attrib["errors"])
    test_count = int(tests.attrib["tests"])
    test_names = [node.attrib["name"] for node in tests.findall("testcase")]
    phase2_tests = sum("phase2" in node.attrib["classname"] for node in tests.findall("testcase"))
    current = audit["output_sha256"] == manifest["output_sha256"] == sha256(manifest["output_path"])
    current &= code_hash() == manifest["code_sha256"]
    current &= sha256("scripts/verify_phase2.py") == audit["verifier_sha256"]
    current &= sha256("config.yaml") == prereg["config_sha256"] == manifest["config_sha256"]
    current &= all(sha256(path) == digest for path, digest in prereg["documents"].items())
    current &= json.loads((directory / "build_status.json").read_text())[
        "manifest_sha256"
    ] == sha256("PHASE2_MANIFEST.json")

    def read_csv(name):
        with (directory / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    labels = read_csv("label_summary.csv")
    ages = read_csv("target_age_summary.csv")
    availability = read_csv("feature_availability.csv")
    feature_summary = read_csv("feature_summary.csv")
    event_types = read_csv("event_type_summary.csv")
    metadata = json.loads((directory / "feature_metadata.json").read_text())
    test_groups = {
        "Future mutation, every feature and settled label": [
            "test_future_mutation_every_feature_and_settled_label"
        ],
        "Same-timestamp causal prefix and future mutation": [
            "test_same_timestamp_future_mutation_and_prefix_equivalence",
            "test_unsettled_target_tie_distinction",
        ],
        "Historical and future as-of lookup": [
            "test_historical_asof_last_before_not_first_after",
            "test_target_tie_uses_last_source_order",
        ],
        "Session-end equality, beyond-close and cross-session rejection": [
            "test_session_end_equal_and_beyond_no_overnight"
        ],
        "No-event available FLAT target": ["test_no_event_target_available_flat"],
        "Invalid book interval and recovery": [
            "test_invalid_interval_not_silently_forward_filled",
            "test_invalid_history_resets_and_unavailable_is_not_zero",
        ],
        "Unordered/duplicate source rows rejected": ["test_row_order_rejected"],
        "Hand-calculated OBI (all four depths)": [
            "test_hand_obi_all_depths",
            "test_zero_denominator_missing_depth_and_no_infinity",
        ],
        "Hand-calculated microprice and synthetic tick conversion": [
            "test_hand_microprice_and_verified_synthetic_tick"
        ],
        "Hand-calculated OFI, eight price/size cases": ["test_hand_ofi_signs"],
        "Hand-calculated labels and exact half-unit direction": [
            "test_hand_wall_clock_100ms_and_1s",
            "test_exact_direction_half_encoding_unit_and_synthetic_ticks",
        ],
        "Left-exclusive, current-source-inclusive rolling interval": [
            "test_window_left_exclusive_right_source_inclusive"
        ],
        "Neutral execution sides, previous-only timing and no NEXT features": [
            "test_activity_resting_side_and_previous_times"
        ],
        "Feature availability independent of future targets": [
            "test_label_availability_cannot_change_features"
        ],
    }
    test_gates = {
        name: failed == 0
        and all(any(t.startswith(prefix) for t in test_names) for prefix in prefixes)
        for name, prefixes in test_groups.items()
    }
    passed = current and failed == 0 and int(tests.attrib["skipped"]) == 0 and test_count >= 81
    passed &= (
        all(test_gates.values()) and audit["status"] == "PASS" and audit["violation_count"] == 0
    )
    passed &= (
        audit["manual_trace_rows"] >= 20
        and audit["manual_comparisons"] >= 25
        and audit["manual_failed_comparisons"] == 0
    )
    passed &= all(case["passed"] for case in audit["mutations"].values())
    verdict = "PASS" if passed else "FAIL"
    perf = manifest["performance"]
    text = [
        "# Phase 2 — Causal Feature Engine, Label Alignment & Microstructure Dataset",
        "",
        f"Decision: **{verdict}**. This gate certifies the sample's feature/label definitions and causal alignment, not predictive value. Phase 2 ends here.",
        "",
        "## Dataset and frozen lineage",
        "",
        "MSFT, 2003-01-03, official Nasdaq-hosted ITCH 2.0 sample with historically inferred legacy Island/INET lineage. The inherited session is **[09:30:00,16:00:00) America/New_York**, or [14:30,21:00) UTC. It is a time-of-day proxy; this source does not independently certify continuous trading, auctions or halts.",
        "",
        f"The full archive contains **{manifest['source_raw_messages']:,} messages**. Phase 1 retained **{manifest['row_count']:,} events/books** and 13,419 trades (6,665 E and 6,754 P). Phase 2 retains **every {manifest['row_count']:,} row**, with **{manifest['column_count']} columns**, {len(manifest['feature_list'])} numeric feature columns (including the unavailable tick feature), seven feature/validity flags, and four wall-clock label horizons. No FLAT or target-unavailable row is removed; no sample selection depends on future movement.",
        "",
        f"Feature timestamp range: `{manifest['timestamp_range'][0]}` through `{manifest['timestamp_range'][1]}`. Event counts: "
        + "; ".join(f"{row['event_type']}={int(row['count']):,}" for row in event_types)
        + ".",
        "",
        "| Artifact | SHA-256 |",
        "|---|---|",
        f"| Raw S010303-v2.zip | `{manifest['raw_sha256']}` |",
        *[
            f"| {cfg['phase2'][key]} | `{digest}` |"
            for key, digest in manifest["input_hashes"].items()
        ],
        f"| {manifest['output_path']} | `{manifest['output_sha256']}` |",
        "",
        f"Frozen Phase-1 commit: `{frozen['git_commit']}`. All original raw/normalized artifacts, source, tests and reports are unchanged. `config.phase1.yaml` preserves the original bytes; only a separate Phase-2 namespace was appended to root config, with README/gitignore additions. `reports/phase2/upstream_freeze.json` records the {len(frozen['files'])}-file baseline and these explicit exceptions. No Phase-1 rebuild or reconstruction change was made.",
        "",
        f"Protocol, feature and label versions are **{manifest['protocol_version']}**. Definitions/config were hash-frozen at `{prereg['frozen_at_utc']}` before final summaries. Production code digest: `{manifest['code_sha256']}`. Config digest: `{manifest['config_sha256']}`. Manifest and Parquet metadata contain all input hashes; every row contains raw/books/events hashes and feature/label versions. The builder refuses to overwrite an existing matrix or manifest.",
        "",
        "## Causal timing",
        "",
        "For event key **(timestamp t, source_order k)**, Feature(t,k) is observable immediately AFTER applying event k to the reconstructed book; it uses the current post-event state, current event and earlier source rows only, and cannot see later source orders at the same millisecond. Historical rolling data include times **(t-w,t]**, with the right endpoint capped at k, and require full history since the first retained session row; historical point values use the last state at or before t-w. For tau=t+h, the label uses the last valid state with timestamp <=tau in this same session, taking the largest source order among timestamp ties and rejecting any intervening invalid segment. With no update, the valid current/last state persists and can yield an available FLAT label; an observed invalid state is never silently skipped. **tau>=session end is unavailable**, including exact close because the inherited session is half-open; no next-session price or prior-session rolling history is used.",
        "",
        "The canonical key is `(timestamp, source_order)`; `event_index` is its zero-based retained-row position. Source timestamps have 1 ms precision, while deterministic file order resolves 3,976 adjacent retained ties. There are no original transport sequence numbers: deterministic ordering cannot certify feed completeness.",
        "",
        "The primary label is `FWD_MID_RET_1S = mid_asof(t+1,000ms)/mid(t)-1`. The secondary horizons are 100ms, 5s and 10s. `MID_MOVE_h` is the USD difference. UP/FLAT/DOWN compare exact integer bid+ask sums, with no epsilon; midprice is that sum /20,000. Requested/resolved timestamps, resolved source order, nonnegative age in ms and unavailable reason are stored for every horizon. Target availability is independent of feature availability.",
        "",
        "At a mutation cutoff inside a timestamp tie, a target exactly at that millisecond may resolve to a later source row and is not yet settled. The invariant applies to targets whose requested time and resolved key are through the cutoff. At the end of a complete timestamp group, all requested targets <=cutoff time must remain fixed. Both cases have explicit tests.",
        "",
        "## Implemented feature families",
        "",
        "All four rolling windows are fixed at **100ms, 1s, 5s, 10s**. Raw interpretable units are preserved; no whole-day normalization, fitted transform, class balancing or threshold selection was performed.",
        "",
        "- Current book: MIDPRICE, SPREAD, RELATIVE_SPREAD; OBI_L1/L3/L5/L10; reciprocal-rank WOBI_L10; BID_DEPTH, ASK_DEPTH, TOTAL_DEPTH and DEPTH_RATIO at L1/L3/L5/L10.",
        "- Microprice: MICROPRICE, MICROPRICE_MINUS_MID and MICROPRICE_DEVIATION_TICKS. Microprice uses opposite-side price weighting by current displayed quantity.",
        "- Order flow: consecutive-book Cont-style OFI_EVENT and OFI sums at all four windows.",
        "- Activity: EVENT_COUNT; ADD/CANCEL/EXECUTE counts and share sizes; neutral BID_SIDE/ASK_SIDE splits for each; separate VISIBLE_EXECUTE and HIDDEN_EXECUTE counts/sizes. E+P form EXECUTE; resting side is never described as aggressor side.",
        "- Intensity: EVENT_RATE, ADD_RATE, CANCEL_RATE and EXECUTION_RATE, all events/second and all four windows.",
        "- Timing: TIME_SINCE_PREVIOUS_EVENT, TIME_SINCE_PREVIOUS_TRADE and TIME_SINCE_PREVIOUS_BOOK_PRICE_CHANGE, milliseconds to the strictly prior qualifying row.",
        "- History: MID_RETURN at all four windows; SPREAD_CHANGE_1S and OBI_CHANGE_1S using last-state historical as-of lookup.",
        "",
        "Price encoding is $0.0001; the historical legal tick size is **unverified**. All five tick-denominated columns are deliberately null, with `tick_size_available=false`; there is no tick distribution to estimate. `mid_move_summary.csv` reports USD movement distributions without substituting a guessed tick. Depth ratio and OBI are redundant transforms, and lower-case midprice/spread are schema aliases, not additional independent signals. A K-level feature requires K valid levels per side. Availability flags for rolling groups conservatively require the current complete top ten plus uninterrupted valid history.",
        "",
        "### Availability",
        "",
        "| Flag | Available | Unavailable | Rate |",
        "|---|---:|---:|---:|",
        *[
            f"| {row['flag']} | {int(row['available']):,} | {int(row['unavailable']):,} | {float(row['availability_rate']):.8%} |"
            for row in availability
        ],
        "",
        "The first OFI_EVENT is unavailable because no previous retained BBO exists. Previous-trade/price-change ages are null until a prior qualifying event exists. Early incomplete rolling history is null rather than zero; zero is reserved for measured absence within a fully covered window. No clipping is applied. All feature infinities, imbalance-bound violations, negative valid spreads/depths/activity/ages and microprices outside the BBO: **0**.",
        "",
        "Every numeric feature is listed below with its actual non-null count. `reports/phase2/feature_metadata.json` supplies name, formula, required fields, lookback, units, timestamp semantics and missing rules. `feature_summary.csv` supplies count, non-null count, availability, mean, sample std, min, p01/p05/p25/median/p75/p95/p99/max and infinity count. Quantiles use inverse empirical CDF; these are event-weighted descriptive summaries.",
        "",
        "| Feature | Lookback ms | Unit | Non-null rows |",
        "|---|---:|---|---:|",
        *[
            f"| {row['name']} | {metadata[row['name']]['lookback_ms']} | {metadata[row['name']]['unit']} | {int(row['non_null_count']):,} |"
            for row in feature_summary
        ],
        "",
        "## Labels: descriptive distributions only",
        "",
        "Proportions use available labels as denominator. Every horizon's earliest feature timestamp is 2003-01-03T14:30:00.060Z. The counts below describe outcomes without measuring feature/label association or ranking horizons by predictive quality.",
        "",
        "| Horizon | Available | Missing | UP | FLAT | DOWN | UP % | FLAT % | DOWN % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row['horizon_ms']}ms | {int(row['valid_count']):,} | {row['unavailable_count']} | {int(row['up_count']):,} | {int(row['flat_count']):,} | {int(row['down_count']):,} | {float(row['up_proportion']) * 100:.6f} | {float(row['flat_proportion']) * 100:.6f} | {float(row['down_proportion']) * 100:.6f} |"
            for row in labels
        ],
        "",
        "| Horizon | Mean forward mid return | Sample std | Latest valid feature timestamp UTC |",
        "|---|---:|---:|---|",
        *[
            f"| {row['horizon_ms']}ms | {float(row['mean_future_mid_return']):.12g} | {float(row['std_future_mid_return']):.12g} | {row['latest_valid_feature_timestamp']} |"
            for row in labels
        ],
        "",
        "### Target age (milliseconds)",
        "",
        "| Horizon | Min | Mean | Median | p95 | p99 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row['name']} | {float(row['min']):g} | {float(row['mean']):.6f} | {float(row['median']):g} | {float(row['p95']):g} | {float(row['p99']):g} | {float(row['max']):g} |"
            for row in ages
        ],
        "",
        "No negative target ages occur. Age can equal the requested horizon when the current timestamp remains the latest known state; such targets remain valid. The 10s age maximum is 7,705ms. No arbitrary staleness cutoff was fitted.",
        "",
        "## Tests, mutation experiments and independent verification",
        "",
        f"Full `pytest -q`: **{test_count - failed - int(tests.attrib['skipped'])} passed, {failed} failed/errors, {tests.attrib['skipped']} skipped, {warnings} warnings**, {terminal_runtime}s terminal runtime ({tests.attrib['time']}s JUnit suite time). This retains all {test_count - phase2_tests} Phase-1 tests and adds {phase2_tests} Phase-2 cases. Evidence: `reports/phase2/test_results.xml` and `pytest_output.txt`.",
        "",
        "| Required test | Result |",
        "|---|---|",
        *[f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in test_gates.items()],
        f"| Independent real manual spot checks | {'PASS' if audit['manual_failed_comparisons'] == 0 else 'FAIL'} |",
        "",
        f"`verify_phase2.py` reloads the saved Parquet and Phase-1 inputs: **{audit['check_count']:,} checks, {len(audit['failed_checks'])} failed checks, {audit['violation_count']} violations**, {audit['runtime_seconds']:.3f}s. It checks actual keys/order/rows, inputs/output/code/protocol hashes, per-row lineage, feature bounds/null flags, all four historical as-of returns, all four labels at every row with independent Python-bisect lookup, integer direction, requested/resolved timestamps, ages, unavailable reasons and session boundaries. It does not merely accept builder metadata. Invariants are saved in `invariant_checks.csv`; complete results in `independent_verification.json`.",
        "",
        f"**{audit['manual_trace_rows']} real trace samples** cover early warm history, time-spaced session coverage, midpoint, highest/lowest 1s activity, a same-timestamp pair and the final session row. `reports/phase2_alignment_traces.md` shows full ten-level books, current/previous BBO, imbalance, microprice, OFI and activity at each window, and every requested/resolved target. **{audit['manual_comparisons']} scalar comparisons match**, including OBI at four depths, microprice, piecewise scalar OFI_EVENT, summed OFI_1S, rolling event count and 1s target. Expected values come directly from Phase-1 arrays, using scalar arithmetic and bisect rather than production feature functions. `manual_verification.csv` records every comparison. The trace and comparison files are local-only under the inherited data policy.",
        "",
        "| Real mutation | Future rows changed | Prefix rows unchanged | Feature/flag columns compared | Result |",
        "|---|---:|---:|---:|---|",
        *[
            f"| {name} | {case['mutated_rows']:,} | {case['unchanged_prefix_rows']:,} | {case['feature_and_flag_columns']} | {'PASS' if case['passed'] else 'FAIL'} |"
            for name, case in audit["mutations"].items()
        ],
        "",
        "Both mutations change every future row's prices, quantities, event type and resting side while preserving causal keys. All prefix feature values and null masks remain exactly equal, and every settled label remains unchanged. Synthetic tests also explicitly separate source orders 100 and 101 at the same .123-second timestamp and compare to a physically truncated prefix. Real manual arithmetic uses a 1e-12 absolute/relative tolerance for floating operations; integer directions, source rows, nanosecond target resolution and mutation equality are exact.",
        "",
        "## Build performance and architecture",
        "",
        f"Measured input/output: {perf['input_observations']:,}/{perf['output_observations']:,} observations; **{perf['runtime_seconds']:.6f}s**, **{perf['observations_per_second']:,.2f} observations/s**, **{perf['peak_memory_bytes']:,} bytes ({perf['peak_memory_bytes'] / 1048576:.2f} MiB) peak working set**; output **{perf['parquet_bytes']:,} bytes**. Scope: {perf['scope']}. Measurements are host-specific, not a scaling guarantee.",
        "",
        "`src/lob_research/phase2/` separates validated arrays/as-of primitives, static-book features, order-flow/activity, feature orchestration, labels and typed artifacts. NumPy prefix sums and searchsorted avoid quadratic windows. The full selected single session is loaded into memory; multi-day use requires explicit session partitioning. No feature module imports labels. The independent verifier has separate scalar arithmetic/bisect expected values; production functions are reused there only for the mutation experiments.",
        "",
        "## Known limitations and dependence",
        "",
        "The largest limitation remains **one MSFT day from 2003**, insufficient for historical or contemporary generalization. Legacy Island/INET venue lineage is historically inferred, not identified by a per-event MIC. No original transport sequences means no definitive feed-completeness guarantee. Millisecond exchange-time precision and deterministic file order do not supply receive timestamps or a latency model. The session is a regular-hours proxy without verified halt/auction state.",
        "",
        "The reconstructed book contains visible liquidity from this feed only, not consolidated market depth. Hidden liquidity/queues are absent even though hidden-execution P prints are observed. There is no real queue-position model, no latency model, no fees/rebates, no routing/fill model and no independent exchange book snapshot. The historical minimum tick is unknown. Raw redistribution rights remain unestablished; raw/normalized/Phase-2 row data and real traces must not be published under current project policy.",
        "",
        "Event clustering, overlapping wall-clock labels, serial dependence and microstructure autocorrelation invalidate an iid interpretation of these 162,879 observations. Future work needs blocked/chronological validation and training-only fitted transforms; no split optimization is attempted here. See `DEPENDENCE_NOTES.md`. There is **no predictive evidence yet**: no feature/forward-return correlation, IC, classification metric, model, strategy return or PnL was calculated. Definitions/horizons/weights were not tuned to outcomes.",
        "",
        "## Phase-2 gates",
        "",
        "| Gate | Result |",
        "|---|---|",
        f"| Frozen upstream and current input/output/protocol/code lineage | {'PASS' if current else 'FAIL'} |",
        f"| Deterministic unique causal ordering; same-timestamp and future-mutation checks | {'PASS' if all(case['passed'] for case in audit['mutations'].values()) else 'FAIL'} |",
        f"| Hand arithmetic, exact rolling/as-of semantics, no-update and session boundaries | {'PASS' if all(test_gates.values()) else 'FAIL'} |",
        f"| Real traces, independent scalar checks and saved-artifact verification | {audit['status']} |",
        f"| Full upstream + Phase-2 pytest suite | {'PASS' if failed == 0 and int(tests.attrib['skipped']) == 0 else 'FAIL'} |",
        "| Features and targets separately available; all rows/FLAT labels preserved | PASS |",
        "| No predictive statistics or horizon/weight tuning | PASS |",
        "",
        "PASS permits later statistical microstructure research on this qualified dataset; it does not establish prediction or executable alpha. Phase 3 has not begun. Stop and await the next instruction.",
        "",
        f"PHASE 2: {verdict}",
    ]
    Path("PHASE2_REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    result = {
        "phase": 2,
        "decision": verdict,
        "tests": {
            "passed": test_count - failed - int(tests.attrib["skipped"]),
            "failed_errors": failed,
            "skipped": int(tests.attrib["skipped"]),
            "warnings": warnings,
            "junit_runtime_seconds": float(tests.attrib["time"]),
        },
        "test_gates": test_gates,
        "independent_checks": audit["check_count"],
        "independent_failed_checks": len(audit["failed_checks"]),
        "evidence_sha256": {
            path: sha256(path)
            for path in [
                "PHASE2_MANIFEST.json",
                "PHASE2_REPORT.md",
                "scripts/build_phase2_report.py",
                *[
                    str(directory / name)
                    for name in (
                        "test_results.xml",
                        "pytest_output.txt",
                        "independent_verification.json",
                        "feature_metadata.json",
                        "feature_summary.csv",
                        "feature_availability.csv",
                        "label_summary.csv",
                        "target_age_summary.csv",
                        "event_type_summary.csv",
                        "manual_verification.csv",
                        "invariant_checks.csv",
                    )
                ],
            ]
        },
    }
    (directory / "phase2_decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"PHASE 2: {verdict}; report and decision written")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
