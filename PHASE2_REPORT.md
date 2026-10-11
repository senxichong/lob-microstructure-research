# Phase 2 — Causal Feature Engine, Label Alignment & Microstructure Dataset

Decision: **PASS**. This gate certifies the sample's feature/label definitions and causal alignment, not predictive value. Phase 2 ends here.

## Dataset and frozen lineage

MSFT, 2003-01-03, official Nasdaq-hosted ITCH 2.0 sample with historically inferred legacy Island/INET lineage. The inherited session is **[09:30:00,16:00:00) America/New_York**, or [14:30,21:00) UTC. It is a time-of-day proxy; this source does not independently certify continuous trading, auctions or halts.

The full archive contains **6,115,981 messages**. Phase 1 retained **162,879 events/books** and 13,419 trades (6,665 E and 6,754 P). Phase 2 retains **every 162,879 row**, with **220 columns**, 149 numeric feature columns (including the unavailable tick feature), seven feature/validity flags, and four wall-clock label horizons. No FLAT or target-unavailable row is removed; no sample selection depends on future movement.

Feature timestamp range: `2003-01-03T14:30:00.060000000Z` through `2003-01-03T20:59:59.520000000Z`. Event counts: A=76,722; E=6,665; P=6,754; X=72,738.

| Artifact | SHA-256 |
|---|---|
| Raw S010303-v2.zip | `eb67a239cf09b7de1843f6b0ede3c473616cc5cac777b293bbe15608bf51794d` |
| data/processed/books.parquet | `2ec94c8d6405e391ed3369a6a6244e2e5a13e58fcf1c074859e2f116a1fdf800` |
| data/processed/events.parquet | `d31e6bc889a923c536ef60074cc7b643f5a865f90e8385bcd08dd3053594acb7` |
| data/processed/phase2/msft_20030103_phase2_v2.0.0.parquet | `a5a493636d9f6d592e2d229921533940ec7cfffcdc5da8f1c9fb636445d34e96` |

Frozen Phase-1 commit: `f4f606ceceacd7cf129ebea6849bde6191c6b02b`. All original raw/normalized artifacts, source, tests and reports are unchanged. `config.phase1.yaml` preserves the original bytes; only a separate Phase-2 namespace was appended to root config, with README/gitignore additions. `reports/phase2/upstream_freeze.json` records the 47-file baseline and these explicit exceptions. No Phase-1 rebuild or reconstruction change was made.

Protocol, feature and label versions are **2.0.0**. Definitions/config were hash-frozen at `2026-10-11T03:12:19.008973+00:00` before final summaries. Production code digest: `a5bac7b0ab9e51d709c5b9f91ff30cdb057b1dc7b05490451c7d7ecb22da1cc4`. Config digest: `43065a637b00853c2680b1e2b0935459ec2697636eed77d0213f09f2dc334922`. Manifest and Parquet metadata contain all input hashes; every row contains raw/books/events hashes and feature/label versions. The builder refuses to overwrite an existing matrix or manifest.

## Causal timing

For event key **(timestamp t, source_order k)**, Feature(t,k) is observable immediately AFTER applying event k to the reconstructed book; it uses the current post-event state, current event and earlier source rows only, and cannot see later source orders at the same millisecond. Historical rolling data include times **(t-w,t]**, with the right endpoint capped at k, and require full history since the first retained session row; historical point values use the last state at or before t-w. For tau=t+h, the label uses the last valid state with timestamp <=tau in this same session, taking the largest source order among timestamp ties and rejecting any intervening invalid segment. With no update, the valid current/last state persists and can yield an available FLAT label; an observed invalid state is never silently skipped. **tau>=session end is unavailable**, including exact close because the inherited session is half-open; no next-session price or prior-session rolling history is used.

The canonical key is `(timestamp, source_order)`; `event_index` is its zero-based retained-row position. Source timestamps have 1 ms precision, while deterministic file order resolves 3,976 adjacent retained ties. There are no original transport sequence numbers: deterministic ordering cannot certify feed completeness.

The primary label is `FWD_MID_RET_1S = mid_asof(t+1,000ms)/mid(t)-1`. The secondary horizons are 100ms, 5s and 10s. `MID_MOVE_h` is the USD difference. UP/FLAT/DOWN compare exact integer bid+ask sums, with no epsilon; midprice is that sum /20,000. Requested/resolved timestamps, resolved source order, nonnegative age in ms and unavailable reason are stored for every horizon. Target availability is independent of feature availability.

At a mutation cutoff inside a timestamp tie, a target exactly at that millisecond may resolve to a later source row and is not yet settled. The invariant applies to targets whose requested time and resolved key are through the cutoff. At the end of a complete timestamp group, all requested targets <=cutoff time must remain fixed. Both cases have explicit tests.

## Implemented feature families

All four rolling windows are fixed at **100ms, 1s, 5s, 10s**. Raw interpretable units are preserved; no whole-day normalization, fitted transform, class balancing or threshold selection was performed.

- Current book: MIDPRICE, SPREAD, RELATIVE_SPREAD; OBI_L1/L3/L5/L10; reciprocal-rank WOBI_L10; BID_DEPTH, ASK_DEPTH, TOTAL_DEPTH and DEPTH_RATIO at L1/L3/L5/L10.
- Microprice: MICROPRICE, MICROPRICE_MINUS_MID and MICROPRICE_DEVIATION_TICKS. Microprice uses opposite-side price weighting by current displayed quantity.
- Order flow: consecutive-book Cont-style OFI_EVENT and OFI sums at all four windows.
- Activity: EVENT_COUNT; ADD/CANCEL/EXECUTE counts and share sizes; neutral BID_SIDE/ASK_SIDE splits for each; separate VISIBLE_EXECUTE and HIDDEN_EXECUTE counts/sizes. E+P form EXECUTE; resting side is never described as aggressor side.
- Intensity: EVENT_RATE, ADD_RATE, CANCEL_RATE and EXECUTION_RATE, all events/second and all four windows.
- Timing: TIME_SINCE_PREVIOUS_EVENT, TIME_SINCE_PREVIOUS_TRADE and TIME_SINCE_PREVIOUS_BOOK_PRICE_CHANGE, milliseconds to the strictly prior qualifying row.
- History: MID_RETURN at all four windows; SPREAD_CHANGE_1S and OBI_CHANGE_1S using last-state historical as-of lookup.

Price encoding is $0.0001; the historical legal tick size is **unverified**. All five tick-denominated columns are deliberately null, with `tick_size_available=false`; there is no tick distribution to estimate. `mid_move_summary.csv` reports USD movement distributions without substituting a guessed tick. Depth ratio and OBI are redundant transforms, and lower-case midprice/spread are schema aliases, not additional independent signals. A K-level feature requires K valid levels per side. Availability flags for rolling groups conservatively require the current complete top ten plus uninterrupted valid history.

### Availability

| Flag | Available | Unavailable | Rate |
|---|---:|---:|---:|
| book_valid | 162,879 | 0 | 100.00000000% |
| feature_static_available | 162,879 | 0 | 100.00000000% |
| tick_size_available | 0 | 162,879 | 0.00000000% |
| feature_100ms_available | 162,878 | 1 | 99.99938605% |
| feature_1s_available | 162,877 | 2 | 99.99877209% |
| feature_5s_available | 162,833 | 46 | 99.97175818% |
| feature_10s_available | 162,791 | 88 | 99.94597216% |

The first OFI_EVENT is unavailable because no previous retained BBO exists. Previous-trade/price-change ages are null until a prior qualifying event exists. Early incomplete rolling history is null rather than zero; zero is reserved for measured absence within a fully covered window. No clipping is applied. All feature infinities, imbalance-bound violations, negative valid spreads/depths/activity/ages and microprices outside the BBO: **0**.

Every numeric feature is listed below with its actual non-null count. `reports/phase2/feature_metadata.json` supplies name, formula, required fields, lookback, units, timestamp semantics and missing rules. `feature_summary.csv` supplies count, non-null count, availability, mean, sample std, min, p01/p05/p25/median/p75/p95/p99/max and infinity count. Quantiles use inverse empirical CDF; these are event-weighted descriptive summaries.

| Feature | Lookback ms | Unit | Non-null rows |
|---|---:|---|---:|
| MIDPRICE | 0 | USD | 162,879 |
| SPREAD | 0 | USD | 162,879 |
| RELATIVE_SPREAD | 0 | ratio | 162,879 |
| MICROPRICE | 0 | USD | 162,879 |
| MICROPRICE_MINUS_MID | 0 | USD | 162,879 |
| MICROPRICE_DEVIATION_TICKS | 0 | ticks | 0 |
| BID_DEPTH_L1 | 0 | shares | 162,879 |
| ASK_DEPTH_L1 | 0 | shares | 162,879 |
| TOTAL_DEPTH_L1 | 0 | shares | 162,879 |
| DEPTH_RATIO_L1 | 0 | ratio | 162,879 |
| OBI_L1 | 0 | ratio | 162,879 |
| BID_DEPTH_L3 | 0 | shares | 162,879 |
| ASK_DEPTH_L3 | 0 | shares | 162,879 |
| TOTAL_DEPTH_L3 | 0 | shares | 162,879 |
| DEPTH_RATIO_L3 | 0 | ratio | 162,879 |
| OBI_L3 | 0 | ratio | 162,879 |
| BID_DEPTH_L5 | 0 | shares | 162,879 |
| ASK_DEPTH_L5 | 0 | shares | 162,879 |
| TOTAL_DEPTH_L5 | 0 | shares | 162,879 |
| DEPTH_RATIO_L5 | 0 | ratio | 162,879 |
| OBI_L5 | 0 | ratio | 162,879 |
| BID_DEPTH_L10 | 0 | shares | 162,879 |
| ASK_DEPTH_L10 | 0 | shares | 162,879 |
| TOTAL_DEPTH_L10 | 0 | shares | 162,879 |
| DEPTH_RATIO_L10 | 0 | ratio | 162,879 |
| OBI_L10 | 0 | ratio | 162,879 |
| WOBI_L10 | 0 | ratio | 162,879 |
| OFI_EVENT | 0 | shares | 162,878 |
| OFI_100MS | 100 | shares | 162,878 |
| EVENT_COUNT_100MS | 100 | events | 162,878 |
| ADD_COUNT_100MS | 100 | events | 162,878 |
| ADD_SIZE_100MS | 100 | shares | 162,878 |
| ADD_BID_SIDE_COUNT_100MS | 100 | events | 162,878 |
| ADD_BID_SIDE_SIZE_100MS | 100 | shares | 162,878 |
| ADD_ASK_SIDE_COUNT_100MS | 100 | events | 162,878 |
| ADD_ASK_SIDE_SIZE_100MS | 100 | shares | 162,878 |
| CANCEL_COUNT_100MS | 100 | events | 162,878 |
| CANCEL_SIZE_100MS | 100 | shares | 162,878 |
| CANCEL_BID_SIDE_COUNT_100MS | 100 | events | 162,878 |
| CANCEL_BID_SIDE_SIZE_100MS | 100 | shares | 162,878 |
| CANCEL_ASK_SIDE_COUNT_100MS | 100 | events | 162,878 |
| CANCEL_ASK_SIDE_SIZE_100MS | 100 | shares | 162,878 |
| EXECUTE_COUNT_100MS | 100 | events | 162,878 |
| EXECUTE_SIZE_100MS | 100 | shares | 162,878 |
| EXECUTE_BID_SIDE_COUNT_100MS | 100 | events | 162,878 |
| EXECUTE_BID_SIDE_SIZE_100MS | 100 | shares | 162,878 |
| EXECUTE_ASK_SIDE_COUNT_100MS | 100 | events | 162,878 |
| EXECUTE_ASK_SIDE_SIZE_100MS | 100 | shares | 162,878 |
| VISIBLE_EXECUTE_COUNT_100MS | 100 | events | 162,878 |
| VISIBLE_EXECUTE_SIZE_100MS | 100 | shares | 162,878 |
| HIDDEN_EXECUTE_COUNT_100MS | 100 | events | 162,878 |
| HIDDEN_EXECUTE_SIZE_100MS | 100 | shares | 162,878 |
| EVENT_RATE_100MS | 100 | events/second | 162,878 |
| ADD_RATE_100MS | 100 | events/second | 162,878 |
| CANCEL_RATE_100MS | 100 | events/second | 162,878 |
| EXECUTION_RATE_100MS | 100 | events/second | 162,878 |
| MID_RETURN_100MS | 100 | return | 162,878 |
| OFI_1S | 1000 | shares | 162,877 |
| EVENT_COUNT_1S | 1000 | events | 162,877 |
| ADD_COUNT_1S | 1000 | events | 162,877 |
| ADD_SIZE_1S | 1000 | shares | 162,877 |
| ADD_BID_SIDE_COUNT_1S | 1000 | events | 162,877 |
| ADD_BID_SIDE_SIZE_1S | 1000 | shares | 162,877 |
| ADD_ASK_SIDE_COUNT_1S | 1000 | events | 162,877 |
| ADD_ASK_SIDE_SIZE_1S | 1000 | shares | 162,877 |
| CANCEL_COUNT_1S | 1000 | events | 162,877 |
| CANCEL_SIZE_1S | 1000 | shares | 162,877 |
| CANCEL_BID_SIDE_COUNT_1S | 1000 | events | 162,877 |
| CANCEL_BID_SIDE_SIZE_1S | 1000 | shares | 162,877 |
| CANCEL_ASK_SIDE_COUNT_1S | 1000 | events | 162,877 |
| CANCEL_ASK_SIDE_SIZE_1S | 1000 | shares | 162,877 |
| EXECUTE_COUNT_1S | 1000 | events | 162,877 |
| EXECUTE_SIZE_1S | 1000 | shares | 162,877 |
| EXECUTE_BID_SIDE_COUNT_1S | 1000 | events | 162,877 |
| EXECUTE_BID_SIDE_SIZE_1S | 1000 | shares | 162,877 |
| EXECUTE_ASK_SIDE_COUNT_1S | 1000 | events | 162,877 |
| EXECUTE_ASK_SIDE_SIZE_1S | 1000 | shares | 162,877 |
| VISIBLE_EXECUTE_COUNT_1S | 1000 | events | 162,877 |
| VISIBLE_EXECUTE_SIZE_1S | 1000 | shares | 162,877 |
| HIDDEN_EXECUTE_COUNT_1S | 1000 | events | 162,877 |
| HIDDEN_EXECUTE_SIZE_1S | 1000 | shares | 162,877 |
| EVENT_RATE_1S | 1000 | events/second | 162,877 |
| ADD_RATE_1S | 1000 | events/second | 162,877 |
| CANCEL_RATE_1S | 1000 | events/second | 162,877 |
| EXECUTION_RATE_1S | 1000 | events/second | 162,877 |
| MID_RETURN_1S | 1000 | return | 162,877 |
| SPREAD_CHANGE_1S | 1000 | USD | 162,877 |
| OBI_CHANGE_1S | 1000 | ratio | 162,877 |
| OFI_5S | 5000 | shares | 162,833 |
| EVENT_COUNT_5S | 5000 | events | 162,833 |
| ADD_COUNT_5S | 5000 | events | 162,833 |
| ADD_SIZE_5S | 5000 | shares | 162,833 |
| ADD_BID_SIDE_COUNT_5S | 5000 | events | 162,833 |
| ADD_BID_SIDE_SIZE_5S | 5000 | shares | 162,833 |
| ADD_ASK_SIDE_COUNT_5S | 5000 | events | 162,833 |
| ADD_ASK_SIDE_SIZE_5S | 5000 | shares | 162,833 |
| CANCEL_COUNT_5S | 5000 | events | 162,833 |
| CANCEL_SIZE_5S | 5000 | shares | 162,833 |
| CANCEL_BID_SIDE_COUNT_5S | 5000 | events | 162,833 |
| CANCEL_BID_SIDE_SIZE_5S | 5000 | shares | 162,833 |
| CANCEL_ASK_SIDE_COUNT_5S | 5000 | events | 162,833 |
| CANCEL_ASK_SIDE_SIZE_5S | 5000 | shares | 162,833 |
| EXECUTE_COUNT_5S | 5000 | events | 162,833 |
| EXECUTE_SIZE_5S | 5000 | shares | 162,833 |
| EXECUTE_BID_SIDE_COUNT_5S | 5000 | events | 162,833 |
| EXECUTE_BID_SIDE_SIZE_5S | 5000 | shares | 162,833 |
| EXECUTE_ASK_SIDE_COUNT_5S | 5000 | events | 162,833 |
| EXECUTE_ASK_SIDE_SIZE_5S | 5000 | shares | 162,833 |
| VISIBLE_EXECUTE_COUNT_5S | 5000 | events | 162,833 |
| VISIBLE_EXECUTE_SIZE_5S | 5000 | shares | 162,833 |
| HIDDEN_EXECUTE_COUNT_5S | 5000 | events | 162,833 |
| HIDDEN_EXECUTE_SIZE_5S | 5000 | shares | 162,833 |
| EVENT_RATE_5S | 5000 | events/second | 162,833 |
| ADD_RATE_5S | 5000 | events/second | 162,833 |
| CANCEL_RATE_5S | 5000 | events/second | 162,833 |
| EXECUTION_RATE_5S | 5000 | events/second | 162,833 |
| MID_RETURN_5S | 5000 | return | 162,833 |
| OFI_10S | 10000 | shares | 162,791 |
| EVENT_COUNT_10S | 10000 | events | 162,791 |
| ADD_COUNT_10S | 10000 | events | 162,791 |
| ADD_SIZE_10S | 10000 | shares | 162,791 |
| ADD_BID_SIDE_COUNT_10S | 10000 | events | 162,791 |
| ADD_BID_SIDE_SIZE_10S | 10000 | shares | 162,791 |
| ADD_ASK_SIDE_COUNT_10S | 10000 | events | 162,791 |
| ADD_ASK_SIDE_SIZE_10S | 10000 | shares | 162,791 |
| CANCEL_COUNT_10S | 10000 | events | 162,791 |
| CANCEL_SIZE_10S | 10000 | shares | 162,791 |
| CANCEL_BID_SIDE_COUNT_10S | 10000 | events | 162,791 |
| CANCEL_BID_SIDE_SIZE_10S | 10000 | shares | 162,791 |
| CANCEL_ASK_SIDE_COUNT_10S | 10000 | events | 162,791 |
| CANCEL_ASK_SIDE_SIZE_10S | 10000 | shares | 162,791 |
| EXECUTE_COUNT_10S | 10000 | events | 162,791 |
| EXECUTE_SIZE_10S | 10000 | shares | 162,791 |
| EXECUTE_BID_SIDE_COUNT_10S | 10000 | events | 162,791 |
| EXECUTE_BID_SIDE_SIZE_10S | 10000 | shares | 162,791 |
| EXECUTE_ASK_SIDE_COUNT_10S | 10000 | events | 162,791 |
| EXECUTE_ASK_SIDE_SIZE_10S | 10000 | shares | 162,791 |
| VISIBLE_EXECUTE_COUNT_10S | 10000 | events | 162,791 |
| VISIBLE_EXECUTE_SIZE_10S | 10000 | shares | 162,791 |
| HIDDEN_EXECUTE_COUNT_10S | 10000 | events | 162,791 |
| HIDDEN_EXECUTE_SIZE_10S | 10000 | shares | 162,791 |
| EVENT_RATE_10S | 10000 | events/second | 162,791 |
| ADD_RATE_10S | 10000 | events/second | 162,791 |
| CANCEL_RATE_10S | 10000 | events/second | 162,791 |
| EXECUTION_RATE_10S | 10000 | events/second | 162,791 |
| MID_RETURN_10S | 10000 | return | 162,791 |
| TIME_SINCE_PREVIOUS_EVENT | None | milliseconds | 162,878 |
| TIME_SINCE_PREVIOUS_TRADE | None | milliseconds | 162,878 |
| TIME_SINCE_PREVIOUS_BOOK_PRICE_CHANGE | None | milliseconds | 162,876 |

## Labels: descriptive distributions only

Proportions use available labels as denominator. Every horizon's earliest feature timestamp is 2003-01-03T14:30:00.060Z. The counts below describe outcomes without measuring feature/label association or ranking horizons by predictive quality.

| Horizon | Available | Missing | UP | FLAT | DOWN | UP % | FLAT % | DOWN % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100ms | 162,879 | 0 | 12,384 | 138,150 | 12,345 | 7.603190 | 84.817564 | 7.579246 |
| 1000ms | 162,877 | 2 | 42,930 | 77,348 | 42,599 | 26.357313 | 47.488596 | 26.154092 |
| 5000ms | 162,832 | 47 | 66,879 | 28,323 | 67,630 | 41.072394 | 17.394001 | 41.533605 |
| 10000ms | 162,758 | 121 | 74,169 | 15,324 | 73,265 | 45.570110 | 9.415205 | 45.014684 |

| Horizon | Mean forward mid return | Sample std | Latest valid feature timestamp UTC |
|---|---:|---:|---|
| 100ms | -2.57013953699e-08 | 5.16892929032e-05 | 2003-01-03T20:59:59.520000000Z |
| 1000ms | -1.12623609246e-07 | 0.000116828268402 | 2003-01-03T20:59:58.869000000Z |
| 5000ms | -2.36880735676e-06 | 0.000244736971682 | 2003-01-03T20:59:54.826000000Z |
| 10000ms | -3.03279005655e-06 | 0.00035354322781 | 2003-01-03T20:59:49.545000000Z |

### Target age (milliseconds)

| Horizon | Min | Mean | Median | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| 100ms | 0 | 65.753946 | 80 | 100 | 100 | 100 |
| 1s | 0 | 182.202607 | 106 | 616 | 1000 | 1000 |
| 5s | 0 | 273.670341 | 123 | 1087 | 2244 | 5000 |
| 10s | 0 | 278.419899 | 124 | 1102 | 2301 | 7705 |

No negative target ages occur. Age can equal the requested horizon when the current timestamp remains the latest known state; such targets remain valid. The 10s age maximum is 7,705ms. No arbitrary staleness cutoff was fitted.

## Tests, mutation experiments and independent verification

Full `pytest -q`: **81 passed, 0 failed/errors, 0 skipped, 0 warnings**, 2.28s terminal runtime (2.219s JUnit suite time). This retains all 49 Phase-1 tests and adds 32 Phase-2 cases. Evidence: `reports/phase2/test_results.xml` and `pytest_output.txt`.

| Required test | Result |
|---|---|
| Future mutation, every feature and settled label | PASS |
| Same-timestamp causal prefix and future mutation | PASS |
| Historical and future as-of lookup | PASS |
| Session-end equality, beyond-close and cross-session rejection | PASS |
| No-event available FLAT target | PASS |
| Invalid book interval and recovery | PASS |
| Unordered/duplicate source rows rejected | PASS |
| Hand-calculated OBI (all four depths) | PASS |
| Hand-calculated microprice and synthetic tick conversion | PASS |
| Hand-calculated OFI, eight price/size cases | PASS |
| Hand-calculated labels and exact half-unit direction | PASS |
| Left-exclusive, current-source-inclusive rolling interval | PASS |
| Neutral execution sides, previous-only timing and no NEXT features | PASS |
| Feature availability independent of future targets | PASS |
| Independent real manual spot checks | PASS |

`verify_phase2.py` reloads the saved Parquet and Phase-1 inputs: **1,079 checks, 0 failed checks, 0 violations**, 13.579s. It checks actual keys/order/rows, inputs/output/code/protocol hashes, per-row lineage, feature bounds/null flags, all four historical as-of returns, all four labels at every row with independent Python-bisect lookup, integer direction, requested/resolved timestamps, ages, unavailable reasons and session boundaries. It does not merely accept builder metadata. Invariants are saved in `invariant_checks.csv`; complete results in `independent_verification.json`.

**25 real trace samples** cover early warm history, time-spaced session coverage, midpoint, highest/lowest 1s activity, a same-timestamp pair and the final session row. `reports/phase2_alignment_traces.md` shows full ten-level books, current/previous BBO, imbalance, microprice, OFI and activity at each window, and every requested/resolved target. **324 scalar comparisons match**, including OBI at four depths, microprice, piecewise scalar OFI_EVENT, summed OFI_1S, rolling event count and 1s target. Expected values come directly from Phase-1 arrays, using scalar arithmetic and bisect rather than production feature functions. `manual_verification.csv` records every comparison. The trace and comparison files are local-only under the inherited data policy.

| Real mutation | Future rows changed | Prefix rows unchanged | Feature/flag columns compared | Result |
|---|---:|---:|---:|---|
| future_mutation | 81,439 | 81,440 | 156 | PASS |
| same_timestamp_mutation | 84,069 | 78,810 | 156 | PASS |

Both mutations change every future row's prices, quantities, event type and resting side while preserving causal keys. All prefix feature values and null masks remain exactly equal, and every settled label remains unchanged. Synthetic tests also explicitly separate source orders 100 and 101 at the same .123-second timestamp and compare to a physically truncated prefix. Real manual arithmetic uses a 1e-12 absolute/relative tolerance for floating operations; integer directions, source rows, nanosecond target resolution and mutation equality are exact.

## Build performance and architecture

Measured input/output: 162,879/162,879 observations; **11.644835s**, **13,987.23 observations/s**, **752,959,488 bytes (718.08 MiB) peak working set**; output **32,744,455 bytes**. Scope: load + features + labels + Parquet; summaries excluded. Measurements are host-specific, not a scaling guarantee.

`src/lob_research/phase2/` separates validated arrays/as-of primitives, static-book features, order-flow/activity, feature orchestration, labels and typed artifacts. NumPy prefix sums and searchsorted avoid quadratic windows. The full selected single session is loaded into memory; multi-day use requires explicit session partitioning. No feature module imports labels. The independent verifier has separate scalar arithmetic/bisect expected values; production functions are reused there only for the mutation experiments.

## Known limitations and dependence

The largest limitation remains **one MSFT day from 2003**, insufficient for historical or contemporary generalization. Legacy Island/INET venue lineage is historically inferred, not identified by a per-event MIC. No original transport sequences means no definitive feed-completeness guarantee. Millisecond exchange-time precision and deterministic file order do not supply receive timestamps or a latency model. The session is a regular-hours proxy without verified halt/auction state.

The reconstructed book contains visible liquidity from this feed only, not consolidated market depth. Hidden liquidity/queues are absent even though hidden-execution P prints are observed. There is no real queue-position model, no latency model, no fees/rebates, no routing/fill model and no independent exchange book snapshot. The historical minimum tick is unknown. Raw redistribution rights remain unestablished; raw/normalized/Phase-2 row data and real traces must not be published under current project policy.

Event clustering, overlapping wall-clock labels, serial dependence and microstructure autocorrelation invalidate an iid interpretation of these 162,879 observations. Future work needs blocked/chronological validation and training-only fitted transforms; no split optimization is attempted here. See `DEPENDENCE_NOTES.md`. There is **no predictive evidence yet**: no feature/forward-return correlation, IC, classification metric, model, strategy return or PnL was calculated. Definitions/horizons/weights were not tuned to outcomes.

## Phase-2 gates

| Gate | Result |
|---|---|
| Frozen upstream and current input/output/protocol/code lineage | PASS |
| Deterministic unique causal ordering; same-timestamp and future-mutation checks | PASS |
| Hand arithmetic, exact rolling/as-of semantics, no-update and session boundaries | PASS |
| Real traces, independent scalar checks and saved-artifact verification | PASS |
| Full upstream + Phase-2 pytest suite | PASS |
| Features and targets separately available; all rows/FLAT labels preserved | PASS |
| No predictive statistics or horizon/weight tuning | PASS |

PASS permits later statistical microstructure research on this qualified dataset; it does not establish prediction or executable alpha. Phase 3 has not begun. Stop and await the next instruction.

PHASE 2: PASS
