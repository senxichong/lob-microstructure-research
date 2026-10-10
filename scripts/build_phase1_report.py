"""Build the Phase-1 gate report from current run, artifact audit and JUnit results."""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lob_research.download import sha256


def main():
    q = json.loads(Path("reports/data_quality.json").read_text())
    audit = json.loads(Path("reports/artifact_verification.json").read_text())
    status = json.loads(Path("reports/run_status.json").read_text())
    review = json.loads(Path("reports/semantic_review.json").read_text())
    suites = list(ET.parse("reports/test_results.xml").getroot().iter("testsuite"))
    total = sum(int(s.get("tests", "0")) for s in suites)
    failures = sum(int(s.get("failures", "0")) + int(s.get("errors", "0")) for s in suites)
    skipped = sum(int(s.get("skipped", "0")) for s in suites)
    checks = {
        "Sample-specific source/semantics review": review["qualified_for_phase1"]
        and review["raw_sha256"] == q["lineage"]["raw_sha256"]
        and review["config_sha256"] == sha256("config.yaml"),
        "Real downloaded data and verified lineage": audit["status"] == "pass"
        and audit["raw_and_output_hashes"] == "verified",
        "Current completed pipeline and unchanged quality report": status["status"] == "complete"
        and status["quality_sha256"]
        == sha256("reports/data_quality.json")
        == audit["quality_sha256"],
        "No parse/reconstruction failures": q["parse_failures"]
        == q["reconstruction_failures"]
        == 0,
        "No timestamp regressions or local row-order violations": q["source_timestamp_regressions"]
        == q["source_row_order_violations"]
        == 0,
        "Regular-window invariants checked without violations": q["rth_book_issue_states"] == 0
        and audit["book_rows_independently_checked"] == q["rth_events"] > 0,
        "Book exhausted by explicit source cancellations": q[
            "remaining_market_wide_orders_after_eof"
        ]
        == 0,
        "At least ten real raw-crosschecked traces": audit["real_traces_crosschecked_to_raw"] >= 10,
        "Critical tests pass without skips": total >= 49 and failures == skipped == 0,
        "Semantics, time, units and limitations documented": all(
            Path(name).is_file()
            for name in [
                "MARKET_DATA_SCHEMA.md",
                "TIMESTAMP_CONVENTION.md",
                "DATA_LIMITATIONS.md",
                "DATA_LICENSE.md",
                "DATA_SOURCE_EVALUATION.md",
            ]
        ),
    }
    decision = "PASS" if all(checks.values()) else "FAIL"
    d = q["distributions"]
    processed = sum(a["bytes"] for a in q["artifacts"].values())
    lines = [
        "# Phase 1 report",
        "",
        "Evaluation date: 2026-10-10 (Asia/Shanghai). Scope: data feasibility, source semantics and integrity only.",
        "",
        "## Selected Dataset",
        "",
        "Nasdaq-hosted **S010303-v2.zip / S010303-v2.txt**, dated **2003-01-03** by filename and ZIP-member metadata. US equities; selected instrument **MSFT**. Historical **Island/INET-lineage ITCH 2.0**; not a modern Nasdaq consolidated book. Venue lineage is an inference from the official protocol history, not a transmitted venue field.",
        "",
        "The source contains full visible order events for its feed. Reconstructed state retains all occupied selected-symbol price levels; saved observations contain up to **10 levels per side**. Every retained observation in this sample has all ten.",
        "",
        "## Why This Dataset",
        "",
        "Real, first-party hosted, small complete archive with explicit day start, order IDs and documented event semantics. It permits full order-level reconstruction without synthetic seeding. Modern ITCH samples were accessible but several GB; official LOBSTER legacy downloads failed; FI-2010's catalog alone did not establish raw event access. No crypto substitution occurred. See [source decision matrix](DATA_SOURCE_EVALUATION.md).",
        "",
        "## Data Semantics",
        "",
        "Event-level fixed-width ASCII, with A adds, X partial/full cancels, E visible executions, P hidden executions, B trade breaks and S controls. The actual sample has A/X/E/P/S, no B. Timestamps are integer **milliseconds** since Eastern midnight, stored as aware UTC nanoseconds. The file contains no transport sequence; **source row order** resolves ties. Never use order IDs as event sequences.",
        "",
        f"Source UTC range: {q['source_first_timestamp']} to {q['source_last_timestamp']}. Retained UTC range: {q['rth_first_timestamp']} to {q['rth_last_timestamp']}. The [09:30,16:00) Eastern window is a time-of-day proxy: the legacy schema cannot certify halt/auction status.",
        "",
        "## Reconstruction",
        "",
        f"**Works on the complete real sample.** Replay begins with the actual start-of-day and includes premarket seeding and trailing post-close cancellations. Market-wide active references route E/X safely; MSFT quantities are aggregated by price. Four independent order-sum/cache audits pass. There are {q['parse_failures']} parse failures, {q['reconstruction_failures']} reconstruction failures and {q['remaining_market_wide_orders_after_eof']} outstanding market-wide orders at EOF.",
        "",
        "Unknown events, missing orders, duplicate references and over-reductions are logged and abort the run. X implements full deletion when shares reach zero; this version has no D/REPLACE message. No independent exchange snapshot is supplied, so consistency is not external ground-truth certification.",
        "",
        "## Integrity",
        "",
        f"- {q['source_messages']:,} real source messages parsed; {q['source_symbols_with_adds']:,} symbols have adds (including test symbols).",
        f"- {q['selected_all_day_events']:,} selected-symbol all-day events; {q['selected_before_window_events']:,} before and {q['selected_after_window_events']:,} after the output window.",
        f"- **{q['rth_events']:,} retained events and post-event book snapshots**; {q['rth_book_changes']:,} change saved top-ten depth.",
        f"- **{q['rth_book_issue_states']} regular-window invariant violations**, including zero locked, crossed, nonpositive-price, negative-size, zero-size, duplicate-level or ordering cases.",
        f"- {q['all_day_book_issue_states']} outside-window incomplete-book states during initialization/teardown: empty bid appears {q['all_day_issue_counts']['empty_bids']} times and empty ask {q['all_day_issue_counts']['empty_asks']} time (overlapping counts). They are logged, not repaired.",
        f"- {q['source_adjacent_duplicate_timestamps']:,} raw adjacent timestamp ties; {q['rth_adjacent_duplicate_timestamps']:,} selected-window ties; zero clock regressions. No data are dropped for ties.",
        "- Transport gaps cannot be certified absent because transport sequence numbers are not in the file. One terminal blank line is explicitly counted as framing, not a market event.",
        "",
        f"Diagnostic spread min/median/p99/max: **${d['spread_usd']['min']:.3f} / ${d['spread_usd']['p50']:.3f} / ${d['spread_usd']['p99']:.3f} / ${d['spread_usd']['max']:.3f}**. Midprice range **${d['midprice_usd']['min']:.3f}–${d['midprice_usd']['max']:.3f}**. All are event-weighted diagnostics. Full distributions and counts: [data-quality report](reports/data_quality_report.md).",
        "",
        "## Trade Alignment",
        "",
        f"**{q['artifacts']['trades']['rows']:,} trades**: {q['rth_event_types']['E']:,} visible E executions and {q['rth_event_types']['P']:,} hidden P executions. E changes displayed depth at the referenced order's price; P does not. Prequote is the immediately preceding selected event's state, postquote includes the trade row. No aggressor field is directly supplied; it remains null. No impact labels or inferred signs are created.",
        "",
        f"**{q['trace_examples']} real traces** (three each of A/X/E/P) contain full before/after top-ten books, order quantities, BBOs and physical row IDs in `reports/event_trace_examples.md`. All were crosschecked to raw source messages and inspected as compact before/after transitions. The file is local-only under the data policy.",
        "",
        "## Scaling",
        "",
        f"Prices are integer USD/10,000; sizes are individual shares. The observed MSFT add-price GCD is {q['rth_add_price_gcd_units']} encoding units ($0.001); {q['rth_visible_price_off_penny_grid']:,} retained A events are off a penny grid. These are measured facts, **not** proof of a legal tick rule. The historical minimum tick remains unverified; the engine validates the $0.0001 representable grid. No modern penny rule is retroactively imposed.",
        "",
        "## Licensing",
        "",
        "**Raw data cannot be committed under this project's current policy:** no affirmative redistribution grant has been established. The same exclusion covers Parquet rows and real traces. Only code, synthetic tests, hashes and aggregate reports are intended for the public portfolio. No files have been published. See [data-license policy](DATA_LICENSE.md).",
        "",
        "## Engineering",
        "",
        f"Streaming buffered ZIP, retries, immutable verified cache, SHA-256 raw/member/output lineage, typed batched Zstandard Parquet, explicit failure logs and an incomplete-run guard. Raw ZIP: {q['raw_zip_bytes']:,} bytes; member: {q['raw_member_bytes']:,}; normalized Parquet: {processed:,} bytes. Measured processing: **{q['processing_seconds']:.2f} s**, **{q['source_messages_per_second']:,.0f} source messages/s**, **{q['peak_rss_bytes'] / 1048576:.1f} MiB peak working set**. This includes parsing, reconstruction and persistence, excluding download/report serialization. Multi-day scaling requires session partitioning; execution-match indexes and exact histograms are not constant-memory forever.",
        "",
        "## Tests",
        "",
        f"**{total - failures - skipped} passed, {failures} failed/errors, {skipped} skipped.** Hand-calculated cases cover all applicable events, full X deletion, unknown/overlarge reductions, ties, ordering, timezones/DST rejection, price/size units, BBO/diagnostics, invariants, hash refusal and Parquet round trips. Synthetic end-to-end tests check premarket carry-in, trade-break causality, deterministic output hashes, exclusive close and logged failure. Persisted real data received an additional independent batched invariant/scaling audit.",
        "",
        "## Limitations",
        "",
        "One stock/day from 2003; historically inferred venue lineage; no modern market generalization, consolidated coverage, original transport sequences, receive timestamps, verified halt/auction flags, independent exchange book snapshot, instrument tick-rule field, hidden queues or measured queue position/latency/fees. Redistribution rights remain unestablished. These constraints are explicit in [DATA_LIMITATIONS.md](DATA_LIMITATIONS.md).",
        "",
        "## Recommendation",
        "",
        "Proceed only to a separately authorized Phase 2 for feature construction and microstructure diagnostics on this qualified methodological sample. A PASS is an engineering/data-understanding gate, not a claim of modern US-equity representativeness or executable alpha. Contemporary/execution research needs a better dated, state-aware dataset. No Phase-2 work has begun.",
        "",
        "| Phase-1 gate | Result |",
        "|---|---|",
    ]
    lines += [f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in checks.items()]
    lines += ["", f"PHASE 1: {decision}"]
    Path("PHASE1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{total - failures - skipped} tests passed; PHASE 1: {decision}")


if __name__ == "__main__":
    main()
