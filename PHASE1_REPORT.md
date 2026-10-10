# Phase 1 report

Evaluation date: 2026-10-10 (Asia/Shanghai). Scope: data feasibility, source semantics and integrity only.

## Selected Dataset

Nasdaq-hosted **S010303-v2.zip / S010303-v2.txt**, dated **2003-01-03** by filename and ZIP-member metadata. US equities; selected instrument **MSFT**. Historical **Island/INET-lineage ITCH 2.0**; not a modern Nasdaq consolidated book. Venue lineage is an inference from the official protocol history, not a transmitted venue field.

The source contains full visible order events for its feed. Reconstructed state retains all occupied selected-symbol price levels; saved observations contain up to **10 levels per side**. Every retained observation in this sample has all ten.

## Why This Dataset

Real, first-party hosted, small complete archive with explicit day start, order IDs and documented event semantics. It permits full order-level reconstruction without synthetic seeding. Modern ITCH samples were accessible but several GB; official LOBSTER legacy downloads failed; FI-2010's catalog alone did not establish raw event access. No crypto substitution occurred. See [source decision matrix](DATA_SOURCE_EVALUATION.md).

## Data Semantics

Event-level fixed-width ASCII, with A adds, X partial/full cancels, E visible executions, P hidden executions, B trade breaks and S controls. The actual sample has A/X/E/P/S, no B. Timestamps are integer **milliseconds** since Eastern midnight, stored as aware UTC nanoseconds. The file contains no transport sequence; **source row order** resolves ties. Never use order IDs as event sequences.

Source UTC range: 2003-01-03T12:00:00.001000000Z to 2003-01-04T01:00:31.329000000Z. Retained UTC range: 2003-01-03T14:30:00.060000000Z to 2003-01-03T20:59:59.520000000Z. The [09:30,16:00) Eastern window is a time-of-day proxy: the legacy schema cannot certify halt/auction status.

## Reconstruction

**Works on the complete real sample.** Replay begins with the actual start-of-day and includes premarket seeding and trailing post-close cancellations. Market-wide active references route E/X safely; MSFT quantities are aggregated by price. Four independent order-sum/cache audits pass. There are 0 parse failures, 0 reconstruction failures and 0 outstanding market-wide orders at EOF.

Unknown events, missing orders, duplicate references and over-reductions are logged and abort the run. X implements full deletion when shares reach zero; this version has no D/REPLACE message. No independent exchange snapshot is supplied, so consistency is not external ground-truth certification.

## Integrity

- 6,115,981 real source messages parsed; 6,116 symbols have adds (including test symbols).
- 166,809 selected-symbol all-day events; 1,912 before and 2,018 after the output window.
- **162,879 retained events and post-event book snapshots**; 138,197 change saved top-ten depth.
- **0 regular-window invariant violations**, including zero locked, crossed, nonpositive-price, negative-size, zero-size, duplicate-level or ordering cases.
- 4 outside-window incomplete-book states during initialization/teardown: empty bid appears 4 times and empty ask 1 time (overlapping counts). They are logged, not repaired.
- 787,137 raw adjacent timestamp ties; 3,976 selected-window ties; zero clock regressions. No data are dropped for ties.
- Transport gaps cannot be certified absent because transport sequence numbers are not in the file. One terminal blank line is explicitly counted as framing, not a market event.

Diagnostic spread min/median/p99/max: **$0.001 / $0.029 / $0.069 / $0.260**. Midprice range **$52.885–$53.785**. All are event-weighted diagnostics. Full distributions and counts: [data-quality report](reports/data_quality_report.md).

## Trade Alignment

**13,419 trades**: 6,665 visible E executions and 6,754 hidden P executions. E changes displayed depth at the referenced order's price; P does not. Prequote is the immediately preceding selected event's state, postquote includes the trade row. No aggressor field is directly supplied; it remains null. No impact labels or inferred signs are created.

**12 real traces** (three each of A/X/E/P) contain full before/after top-ten books, order quantities, BBOs and physical row IDs in `reports/event_trace_examples.md`. All were crosschecked to raw source messages and inspected as compact before/after transitions. The file is local-only under the data policy.

## Scaling

Prices are integer USD/10,000; sizes are individual shares. The observed MSFT add-price GCD is 10 encoding units ($0.001); 59,893 retained A events are off a penny grid. These are measured facts, **not** proof of a legal tick rule. The historical minimum tick remains unverified; the engine validates the $0.0001 representable grid. No modern penny rule is retroactively imposed.

## Licensing

**Raw data cannot be committed under this project's current policy:** no affirmative redistribution grant has been established. The same exclusion covers Parquet rows and real traces. Only code, synthetic tests, hashes and aggregate reports are intended for the public portfolio. No files have been published. See [data-license policy](DATA_LICENSE.md).

## Engineering

Streaming buffered ZIP, retries, immutable verified cache, SHA-256 raw/member/output lineage, typed batched Zstandard Parquet, explicit failure logs and an incomplete-run guard. Raw ZIP: 58,907,174 bytes; member: 212,330,490; normalized Parquet: 11,308,061 bytes. Measured processing: **462.08 s**, **13,236 source messages/s**, **201.6 MiB peak working set**. This includes parsing, reconstruction and persistence, excluding download/report serialization. Multi-day scaling requires session partitioning; execution-match indexes and exact histograms are not constant-memory forever.

## Tests

**49 passed, 0 failed/errors, 0 skipped.** Hand-calculated cases cover all applicable events, full X deletion, unknown/overlarge reductions, ties, ordering, timezones/DST rejection, price/size units, BBO/diagnostics, invariants, hash refusal and Parquet round trips. Synthetic end-to-end tests check premarket carry-in, trade-break causality, deterministic output hashes, exclusive close and logged failure. Persisted real data received an additional independent batched invariant/scaling audit.

## Limitations

One stock/day from 2003; historically inferred venue lineage; no modern market generalization, consolidated coverage, original transport sequences, receive timestamps, verified halt/auction flags, independent exchange book snapshot, instrument tick-rule field, hidden queues or measured queue position/latency/fees. Redistribution rights remain unestablished. These constraints are explicit in [DATA_LIMITATIONS.md](DATA_LIMITATIONS.md).

## Recommendation

Proceed only to a separately authorized Phase 2 for feature construction and microstructure diagnostics on this qualified methodological sample. A PASS is an engineering/data-understanding gate, not a claim of modern US-equity representativeness or executable alpha. Contemporary/execution research needs a better dated, state-aware dataset. No Phase-2 work has begun.

| Phase-1 gate | Result |
|---|---|
| Sample-specific source/semantics review | PASS |
| Real downloaded data and verified lineage | PASS |
| Current completed pipeline and unchanged quality report | PASS |
| No parse/reconstruction failures | PASS |
| No timestamp regressions or local row-order violations | PASS |
| Regular-window invariants checked without violations | PASS |
| Book exhausted by explicit source cancellations | PASS |
| At least ten real raw-crosschecked traces | PASS |
| Critical tests pass without skips | PASS |
| Semantics, time, units and limitations documented | PASS |

PHASE 1: PASS
